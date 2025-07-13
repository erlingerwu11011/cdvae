import logging
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
from torch import nn
from torch.utils.data import Dataset
from torch_ema import ExponentialMovingAverage

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.rep_est.rep_est import EstHeadAutoreg, RepEncoder
from src.models.utils import BRTreatmentOutcomeHead
from src.models.utils_causal_cpc import CPC, decoder

logger = logging.getLogger(__name__)


class Causal_CPC_Encoder(RepEncoder):
    """
    Pytorch-Lightning implementation of Causal Contrastive Predictive Coding (Causal CPC) model
    An end to end version.
    """

    def __init__(
        self,
        args: DictConfig,
        dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
        autoregressive: bool = True,
        has_vitals: bool = None,
        bce_weights: np.array = None,
        **kwargs,
    ):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        self.input_size += args.model.rep_encoder.dim_random_vitals
        logger.info("%s", f"Input size of {self.model_type}: {self.input_size}")
        self.alpha = args.exp.alpha
        self.test_robustness = args.exp.test_robustness
        self.update_alpha = args.exp.update_alpha
        logger.info("%s", f"alpha of {self.model_type}: {self.alpha}")
        logger.info("%s", f"test_robustness: {self.test_robustness}")

        self._init_specific(args.model.rep_encoder)

        self.save_hyperparameters(args)

    def _init_specific(self, sub_args: DictConfig):
        try:
            self.br_size = sub_args.br_size  # balanced representation size
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate
            self.num_layer = sub_args.num_layer
            self.genc_hidden = sub_args.genc_hidden
            self.context_latent_dim = sub_args.context_latent_dim
            self.prediction_step = self.hparams.dataset.projection_horizon + 1
            self.downsampling_factor = sub_args.downsampling_factor
            self.subsample_win_ratio = sub_args.subsample_win_ratio
            self.use_attention = sub_args.use_attention
            self.seq_len = self.hparams.dataset.max_seq_length - 1
            self.use_causalconv = sub_args.use_causalconv
            self.rnn_type = sub_args.rnn_type
            self.weighting = sub_args.weighting

            self.dim_random_vitals = sub_args.dim_random_vitals
            self.alpha_recons = sub_args.alpha_recons
            self.alpha_mse = sub_args.alpha_mse
            self.alpha_infonce = sub_args.alpha_infonce
            self.label_smoothing = sub_args.label_smoothing
            self.use_spectral_norm = sub_args.use_spectral_norm
            self.activation = sub_args.activation
            self.use_instance_noise = sub_args.use_instance_noise

            self.balancing = sub_args.balancing
            self.cpc_lb = sub_args.cpc_lb
            self.infomax_lb = sub_args.infomax_lb

            if self.use_causalconv:
                self.input_channels = sub_args.input_channels
                self.hidden_channels = sub_args.hidden_channels
                self.kernel_size = sub_args.kernel_size
                self.dilation = sub_args.dilation

            if self.br_size is None or self.fc_hidden_units is None or self.dropout_rate is None:
                raise MissingMandatoryValue()

            self.map_br_to_c = nn.Linear(self.br_size, self.context_latent_dim, bias=False)

            self.br_treatment_outcome_head = BRTreatmentOutcomeHead(
                seq_hidden_units=self.context_latent_dim,
                br_size=self.br_size,
                fc_hidden_units=self.fc_hidden_units,
                dim_treatments=self.dim_treatments,
                dim_outcome=self.dim_outcome,
                dim_static_feat=0,
                alpha=self.alpha,
                update_alpha=self.update_alpha,
                balancing=self.balancing,
                use_spectral_norm=self.use_spectral_norm,
                activation=self.activation,
            )

            self.cpc = CPC(
                self.input_size,
                self.genc_hidden,
                self.context_latent_dim,
                self.num_layer,
                self.dropout_rate,
                self.prediction_step,
                self.downsampling_factor,
                use_attention=self.use_attention,
                activation=self.activation,
                rnn_type=self.rnn_type,
                weighting=self.weighting,
            )

        except MissingMandatoryValue:
            logger.warning(
                f"{self.model_type} not fully initialised - some mandatory args are missing! "
                f"(It's ok, if one will perform hyperparameters search afterward)."
            )

        # weight initialization
        for m in self.modules():

            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(self.activation))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(self.activation))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU) or isinstance(m, nn.RNN):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param.data)

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Used for hyperparameter tuning and model reinitialisation
        :param model_args: Sub DictConfig, with encoder/decoder parameters
        :param new_args: New hyperparameters
        :param input_size: Input size of the model
        :param model_type: Submodel specification
        """
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args["learning_rate"]
        sub_args.batch_size = new_args["batch_size"]
        if "seq_hidden_units" in new_args:  # Only relevant for encoder
            sub_args.seq_hidden_units = int(input_size * new_args["seq_hidden_units"])
        sub_args.br_size = int(input_size * new_args["br_size"])
        sub_args.fc_hidden_units = int(sub_args.br_size * new_args["fc_hidden_units"])
        sub_args.dropout_rate = new_args["dropout_rate"]
        sub_args.num_layer = new_args["num_layer"]

    def _encode(
        self,
        batch,
        active_entries,
        return_flatten=False,
        return_comp_reps=False,
    ):
        input = self.build_input(batch)
        enc = self.build_br(input)

        if return_comp_reps:
            enc, comp_reps = enc
            rep_list = [enc] + comp_reps
            if return_flatten:
                return [x[active_entries.squeeze(-1) == 1] for x in rep_list]
            else:
                return rep_list
        else:
            if return_flatten:
                return enc[active_entries.squeeze(-1) == 1]
            else:
                return enc

    def encode(self, batch, return_flatten=False, return_comp_reps=False):
        active_entries = batch["active_entries"]
        return self._encode(
            batch,
            active_entries,
            return_flatten,
            return_comp_reps,
        )

    def build_br(self, x, get_c=False):
        z_xy = self.cpc.encoder(x)
        if self.use_causalconv:
            c = self.cpc.g_ar(z_xy)
        else:
            c = self.cpc.g_ar(z_xy)[0]
        if self.use_attention:
            c = self.cpc.attention(c, c)
        return c

    def build_input(self, batch):
        prev_treatments = batch["prev_treatments"]
        vitals_or_prev_outputs = []

        if self.has_vitals:
            vitals = batch["vitals"]
            if self.test_robustness:
                remove_indices = [2, 5]
                # Create a mask for the indices to keep
                keep_indices = torch.tensor(
                    [i for i in range(vitals.shape[2]) if i not in remove_indices]
                ).to(vitals.device)
                # Remove the specified dimensions
                vitals = vitals.index_select(2, keep_indices)

            vitals_or_prev_outputs.append(vitals)

        else:
            vitals = None

        if self.autoregressive:
            prev_outputs = batch["prev_outputs"]
            vitals_or_prev_outputs.append(prev_outputs)

        vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)
        x = torch.cat((prev_treatments, vitals_or_prev_outputs), dim=-1)

        if self.dim_static_features > 0:
            static_features = batch["static_features"]
            static_features = static_features / torch.norm(
                static_features, p="fro", dim=1, keepdim=True
            )
            x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)

        if self.dim_random_vitals > 0:
            random_vitals = torch.randn(
                (x.size(0), x.size(1), self.dim_random_vitals), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, random_vitals), dim=-1)
        return x

    def forward(self, batch):
        x = self.build_input(batch)
        br = self.build_br(x)
        return br

    def training_step(self, batch, batch_ind, optimizer_idx=0):

        x = self.build_input(batch)
        encoder_samples, predictions, active_entries_samples, c_future, c_future_predictions = (
            self.cpc(x, active_entries=batch["active_entries"])
        )
        info_nce_loss = self.alpha_infonce * self.cpc.loss(
            encoder_samples, predictions, active_entries_samples, cpc_lb=self.cpc_lb
        )
        recons_loss = self.alpha_recons * self.cpc.infomax_loss(
            c_future, c_future_predictions, active_entries_samples, infomax_lb=self.infomax_lb
        )

        loss = info_nce_loss + recons_loss

        self.log(
            f"{self.model_type}_train/loss", loss, on_epoch=True, on_step=False, sync_dist=True
        )
        self.log(
            f"{self.model_type}_train/info_nce_loss",
            info_nce_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            f"{self.model_type}_train/recons_loss",
            recons_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            f"{self.model_type}_alpha",
            self.br_treatment_outcome_head.alpha,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_ind, **kwargs):

        x = self.build_input(batch)
        encoder_samples, predictions, active_entries_samples, c_future, c_future_predictions = (
            self.cpc(x, active_entries=batch["active_entries"])
        )
        info_nce_loss = self.alpha_infonce * self.cpc.loss(
            encoder_samples, predictions, active_entries_samples, cpc_lb=self.cpc_lb
        )
        recons_loss = self.alpha_recons * self.cpc.infomax_loss(
            c_future, c_future_predictions, active_entries_samples, infomax_lb=self.infomax_lb
        )

        loss = info_nce_loss + recons_loss

        subset_name = self.val_dataloader().dataset.subset_name
        self.log(
            f"{self.model_type}_{subset_name}/info_nce_loss",
            info_nce_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            f"{self.model_type}_{subset_name}/recons_loss",
            recons_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log("val/loss", loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_ind, **kwargs):

        x = self.build_input(batch)
        encoder_samples, predictions, active_entries_samples, c_future, c_future_predictions = (
            self.cpc(x, active_entries=batch["active_entries"])
        )
        info_nce_loss = self.cpc.loss(
            encoder_samples, predictions, active_entries_samples, cpc_lb=self.cpc_lb
        )
        recons_loss = self.cpc.infomax_loss(
            c_future, c_future_predictions, active_entries_samples, infomax_lb=self.infomax_lb
        )
        info_nce_loss = (
            self.alpha_infonce
            * self.br_treatment_outcome_head.alpha
            * self.alpha_infonce
            * info_nce_loss
        )
        recons_loss = (
            self.alpha_recons
            * self.br_treatment_outcome_head.alpha
            * self.alpha_recons
            * recons_loss
        )

        loss = info_nce_loss + recons_loss

        subset_name = self.test_dataloader().dataset.subset_name
        self.log(
            f"{self.model_type}_{subset_name}/info_nce_loss",
            info_nce_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            f"{self.model_type}_{subset_name}/recons_loss",
            recons_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def _get_optimizer(self, param_optimizer: list, treat_head: bool = False):
        no_decay = ["bias", "layer_norm"]
        sub_args = self.hparams.model[self.model_type]
        head_type = "treatment_head" if treat_head else "non_treatment_head"
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": sub_args["optimizer"][head_type]["weight_decay"],
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        lr = sub_args["optimizer"][head_type]["learning_rate"]
        optimizer_cls = sub_args["optimizer"][head_type]["optimizer_cls"]

        if optimizer_cls.lower() == "adamw":
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == "adam":
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == "sgd":
            optimizer = optim.SGD(
                optimizer_grouped_parameters,
                lr=lr,
                momentum=sub_args["optimizer"][head_type]["momentum"],
            )
        else:
            raise NotImplementedError()
        return optimizer

    def configure_optimizers(self):
        treatment_head_params = [
            "br_treatment_outcome_head." + s
            for s in self.br_treatment_outcome_head.treatment_head_params
        ]
        treatment_head_params = [
            k
            for k in dict(self.named_parameters())
            for param in treatment_head_params
            if k.startswith(param)
        ]
        non_treatment_head_params = [
            k for k in dict(self.named_parameters()) if k not in treatment_head_params
        ]

        assert len(treatment_head_params + non_treatment_head_params) == len(
            list(self.named_parameters())
        )

        treatment_head_params = [
            (k, v) for k, v in dict(self.named_parameters()).items() if k in treatment_head_params
        ]
        non_treatment_head_params = [
            (k, v)
            for k, v in dict(self.named_parameters()).items()
            if k in non_treatment_head_params
        ]

        non_treatment_head_optimizer = self._get_optimizer(
            non_treatment_head_params, treat_head=False
        )

        if self.hparams.model[self.model_type]["optimizer"]["lr_scheduler"]:
            return self._get_lr_schedulers([non_treatment_head_optimizer])

        return [non_treatment_head_optimizer]


class RNNEstHead(EstHeadAutoreg):
    def __init__(
        self,
        args,
        rep_encoder,
        dataset_collection,
        autoregressive=None,
        has_vitals=None,
        bce_weights=None,
        prefix="",
        **kwargs,
    ):
        super().__init__(
            args,
            rep_encoder,
            dataset_collection,
            autoregressive,
            has_vitals,
            bce_weights,
            prefix=prefix,
        )

        self.alpha = args.exp.alpha
        self.update_alpha = args.exp.update_alpha
        logger.info("%s", f"alpha of {self.model_type}: {self.alpha}")

        self._init_specific(args.model.est_head)
        self.save_hyperparameters(args)

    def _init_specific(self, sub_args):
        super()._init_specific(sub_args)

        try:
            self.br_size = sub_args.br_size  # balanced representation size
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate
            self.num_layer = sub_args.num_layer
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.y_dist_type = sub_args.y_dist_type
            self.teacher_forcing = sub_args.teacher_forcing
            self.treat_hidden_dim = sub_args.treat_hidden_dim
            sub_args.seq_hidden_units = self.rep_encoder.br_size
            self.label_smoothing = sub_args.label_smoothing
            self.use_spectral_norm = sub_args.use_spectral_norm
            self.activation = sub_args.activation
            self.use_instance_noise = sub_args.use_instance_noise
            self.finetune_rep_encoder = sub_args.finetune_rep_encoder
            self.retrain_rep_encoder = sub_args.retrain_rep_encoder
            self.likelihood_training = sub_args.likelihood_training
            self.random_indices = sub_args.random_indices
            self.percentage_to_keep = sub_args.percentage_to_keep
            self.rnn_type = sub_args.rnn_type

            self.historical_avg = sub_args.historical_avg
            self.balancing = sub_args.balancing
            self.alpha_recons = sub_args.alpha_recons
            self.alpha_mse = sub_args.alpha_mse
            self.alpha_infonce = sub_args.alpha_infonce

            if (
                self.seq_hidden_units is None
                or self.br_size is None
                or self.fc_hidden_units is None
                or self.dropout_rate is None
            ):
                raise MissingMandatoryValue()

            if self.teacher_forcing:
                self.br_treatment_outcome_head = BRTreatmentOutcomeHead(
                    self.seq_hidden_units,
                    self.br_size,
                    self.fc_hidden_units,
                    self.dim_treatments,
                    self.dim_outcome,
                    self.dim_static_features,
                    self.alpha,
                    self.update_alpha,
                    self.balancing,
                    use_spectral_norm=self.use_spectral_norm,
                    activation=self.activation,
                    use_instance_noise=self.use_instance_noise,
                )
                self.gru = nn.GRU(
                    input_size=self.input_size,
                    hidden_size=self.seq_hidden_units,
                    batch_first=True,
                    num_layers=self.num_layer,
                    dropout=self.dropout_rate,
                )
            else:
                self.br_treatment_outcome_head = decoder(
                    treat_size=self.dim_treatments,
                    treat_hidden_dim=self.treat_hidden_dim,
                    dim_outcome=self.dim_outcome,
                    seq_hidden_units=self.seq_hidden_units,
                    dim_static_features=self.dim_static_features,
                    br_size=self.br_size,
                    num_layers_dec=self.num_layer,
                    rnn_dropout_dec=self.dropout_rate,
                    y_dist_type=self.y_dist_type,
                    alpha=self.alpha,
                    update_alpha=self.update_alpha,
                    use_spectral_norm=self.use_spectral_norm,
                    activation=self.activation,
                    likelihood_training=self.likelihood_training,
                    rnn_type=self.rnn_type,
                )
                flattened_params = torch.cat(
                    [
                        param.view(-1)
                        for param in self.br_treatment_outcome_head.state_dict().values()
                    ]
                )
                self.register_buffer("avg_weights", torch.zeros_like(flattened_params))

            if not self.finetune_rep_encoder:
                self.rep_encoder.freeze()

            if not self.finetune_rep_encoder and self.retrain_rep_encoder:
                raise Exception("Model cannot be frozen and retrained at the same time!")

        except MissingMandatoryValue:
            logger.warning(
                f"{self.model_type} not fully initialised - some mandatory args are missing! "
                f"(It's ok, if one will perform hyperparameters search afterward)."
            )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(self.activation))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(self.activation))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU) or isinstance(m, nn.RNN):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param.data)

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Used for hyperparameter tuning and model reinitialisation
        :param model_args: Sub DictConfig, with encoder/decoder parameters
        :param new_args: New hyperparameters
        :param input_size: Input size of the model
        :param model_type: Submodel specification
        """
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args["learning_rate"]
        sub_args.batch_size = new_args["batch_size"]
        if "seq_hidden_units" in new_args:  # Only relevant for encoder
            sub_args.seq_hidden_units = int(input_size * new_args["seq_hidden_units"])
        sub_args.br_size = int(input_size * new_args["br_size"])
        sub_args.fc_hidden_units = int(sub_args.br_size * new_args["fc_hidden_units"])
        sub_args.dropout_rate = new_args["dropout_rate"]
        sub_args.num_layer = new_args["num_layer"]

    def forward(
        self,
        batch,
        return_rep=False,
        return_br=False,
        mean_only=False,
        one_step=False,
        selected_indices=None,
    ):

        prev_treatments = batch["prev_treatments"]
        current_treatments = batch["current_treatments"]
        prev_outputs = batch["prev_outputs"]
        x_enc = self.rep_encoder.encode(batch)  # [B, T, D]
        batch_size, step_num, dim_treat = prev_treatments.shape
        unrolled_prev_treatments = self._unroll_horizon_random_indices(
            batch["prev_treatments"], selected_indices
        )
        unrolled_current_treatments = self._unroll_horizon_random_indices(
            batch["current_treatments"], selected_indices
        )
        unrolled_prev_outputs = self._unroll_horizon_random_indices(
            batch["prev_outputs"], selected_indices
        )

        y_init = unrolled_prev_outputs[:, 0, :]  # no leakage
        if selected_indices is not None:
            init_states = x_enc[selected_indices[:, 0], selected_indices[:, 1], :].reshape(
                -1, x_enc.shape[-1]
            )
        else:
            init_states = x_enc.reshape(-1, x_enc.shape[-1])  # [:, selected_indices ,:]

        if self.dim_static_features > 0:
            static_features = batch["static_features"]
            static_features = static_features.unsqueeze(1)

        if self.teacher_forcing:
            x = torch.cat((unrolled_prev_treatments, unrolled_prev_outputs), dim=-1)

            x, _ = self.gru(x, init_states.unsqueeze(0).repeat(self.num_layer, 1, 1))
            br = self.br_treatment_outcome_head.build_br(x)
            outcome_pred = self.br_treatment_outcome_head.build_outcome(
                br, unrolled_current_treatments, static_features
            )

        else:
            if selected_indices is not None:
                if self.dim_static_features > 0:
                    static_features = static_features.repeat(1, step_num, 1)
                    static_features = static_features[
                        selected_indices[:, 0], selected_indices[:, 1], :
                    ].reshape(-1, self.dim_static_features)
                else:
                    static_features = None
            else:
                if self.dim_static_features > 0:
                    static_features = static_features.repeat(1, step_num, 1).reshape(
                        -1, self.dim_static_features
                    )
                else:
                    static_features = None

            outcome_pred, br = self.br_treatment_outcome_head(
                w_init=unrolled_prev_treatments[:, 0, :],
                y_init=y_init,  # unrolled_prev_outputs
                c_init=init_states,
                static_features=static_features,
                w_intended=unrolled_current_treatments,
            )

        if not self.likelihood_training:
            outcome_pred = outcome_pred.reshape(batch_size, step_num, self.output_horizon, -1)

        if self.likelihood_training and mean_only:
            outcome_pred = outcome_pred.mean.reshape(batch_size, step_num, self.output_horizon, -1)

        # tmax = 1 if one_step else self.output_horizon
        if one_step:
            if self.likelihood_training:
                outcome_pred = outcome_pred.mean.reshape(
                    batch_size, step_num, self.output_horizon, -1
                )[:, :, 0, :].unsqueeze(2)
            else:
                outcome_pred = outcome_pred[:, :, 0].unsqueeze(2)
        if return_br:
            return outcome_pred, br
        if return_rep:
            return outcome_pred, x_enc  #! or br ???
        else:
            return outcome_pred

    def _calc_mse_loss(self, outcome_pred, unrolled_outputs, unrolled_active_entries):

        mse_loss = F.mse_loss(outcome_pred, unrolled_outputs, reduce=False)
        mse_loss = torch.mean((unrolled_active_entries * mse_loss).sum(dim=(-2, -1)))
        return mse_loss

    def _calc_ll_loss(self, outcome_pred, unrolled_outputs, unrolled_active_entries):

        log_prob = outcome_pred.log_prob(unrolled_outputs)
        log_prob = torch.mean((unrolled_active_entries * log_prob).sum(dim=(-2, -1)))

        loss_y = -log_prob
        return loss_y

    @staticmethod
    def _generate_indices(percentage_to_keep, BS, T):

        num_elements_to_keep = int(percentage_to_keep * T)
        selected_indices = torch.randint(0, T, (BS, num_elements_to_keep))
        batch_indices = torch.arange(BS).unsqueeze(1).expand_as(selected_indices)
        indices = torch.stack([batch_indices, selected_indices], dim=-1)
        flat_indices = indices.view(-1, 2)

        return flat_indices

    def _unroll_horizon_random_indices(self, input, selected_indices):
        unrolled_input = self._unroll_horizon(input, self.output_horizon)
        if selected_indices is not None:
            unrolled_input = unrolled_input[selected_indices[:, 0], selected_indices[:, 1], :, :]
        else:
            unrolled_input = unrolled_input.reshape(
                -1, self.output_horizon, unrolled_input.shape[-1]
            )

        return unrolled_input

    def training_step(self, batch, batch_ind, optimizer_idx=0):

        if self.random_indices:
            selected_indices = self._generate_indices(
                self.percentage_to_keep,
                BS=batch["active_entries"].shape[0],
                T=batch["active_entries"].shape[1],
            )
        else:
            selected_indices = None

        unrolled_active_entries = self._unroll_horizon_random_indices(
            batch["active_entries"], selected_indices
        )
        unrolled_current_treatments = self._unroll_horizon_random_indices(
            batch["current_treatments"], selected_indices
        )
        unrolled_outputs = self._unroll_horizon_random_indices(batch["outputs"], selected_indices)

        if optimizer_idx == 0:
            if self.hparams.exp.weights_ema:
                with self.ema_treatment.average_parameters():
                    outcome_pred, br = self(
                        batch, return_br=True, selected_indices=selected_indices
                    )
                    treatment_pred = self.br_treatment_outcome_head.build_treatment(
                        br, detached=False
                    )
            else:
                outcome_pred, br = self(batch, return_br=True, selected_indices=selected_indices)
                treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detached=False)

            if self.likelihood_training:
                loss_y = self._calc_ll_loss(
                    outcome_pred, unrolled_outputs, unrolled_active_entries
                )
            else:
                loss_y = self._calc_mse_loss(
                    outcome_pred, unrolled_outputs, unrolled_active_entries
                )
                loss_y = self.alpha_mse * loss_y

            unrolled_active_entries = unrolled_active_entries.reshape(-1, self.output_horizon)

            if self.balancing == "domain_confusion":
                bce_loss = self.bce_loss(
                    treatment_pred, unrolled_current_treatments.double(), kind="confuse"
                )
            elif self.balancing == "mutual_info":
                bce_loss = self.bce_loss(
                    treatment_pred, unrolled_current_treatments.double(), kind="MI"
                )
            else:
                raise NotImplementedError()
            bce_loss = (unrolled_active_entries * bce_loss).sum() / unrolled_active_entries.sum()
            bce_loss = bce_loss * self.br_treatment_outcome_head.alpha
            loss = bce_loss + loss_y

            if self.retrain_rep_encoder:
                x = self.rep_encoder.build_input(batch)
                (
                    encoder_samples,
                    predictions,
                    active_entries_samples,
                    c_future,
                    c_future_predictions,
                ) = self.rep_encoder.cpc(x, batch["active_entries"])
                info_nce_loss = self.rep_encoder.cpc.loss(
                    encoder_samples, predictions, active_entries_samples
                )
                recons_loss = self.rep_encoder.cpc.infonce_loss_recons(
                    c_future, c_future_predictions, active_entries_samples
                )
                info_nce_loss = (
                    self.br_treatment_outcome_head.alpha * self.alpha_infonce * info_nce_loss
                )
                recons_loss = (
                    self.br_treatment_outcome_head.alpha * self.alpha_recons * recons_loss
                )
                loss += info_nce_loss + recons_loss

                self.log(
                    f"{self.model_type}_train/recons_loss",
                    recons_loss,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                    prog_bar=False,
                )
                self.log(
                    f"{self.model_type}_train/info_nce_loss",
                    info_nce_loss,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                    prog_bar=False,
                )

            if self.historical_avg:
                regularization_loss = self.update_avg_weights()
                loss = loss + 50 * regularization_loss
            self.log(
                f"{self.model_type}_train/loss",
                loss,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                prog_bar=False,
            )
            self.log(
                f"{self.model_type}_train/bce_loss",
                bce_loss,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                prog_bar=False,
            )
            self.log(
                f"{self.model_type}_train/loss_y",
                loss_y,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                prog_bar=True,
            )
            return loss

        elif optimizer_idx == 1:  # domain classifier update
            if self.hparams.exp.weights_ema:
                with self.ema_non_treatment.average_parameters():
                    _, br = self(batch, return_br=True, selected_indices=selected_indices)
                    treatment_pred = self.br_treatment_outcome_head.build_treatment(
                        br, detached=True
                    )
            else:
                _, br = self(batch, return_br=True, selected_indices=selected_indices)
                treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detached=True)

            unrolled_active_entries = unrolled_active_entries.reshape(-1, self.output_horizon)
            bce_loss = self.bce_loss(
                treatment_pred,
                unrolled_current_treatments.double(),
                kind="predict",
                label_smoothing=self.label_smoothing,
            )
            bce_loss = (unrolled_active_entries * bce_loss).sum() / unrolled_active_entries.sum()
            bce_loss = bce_loss * self.br_treatment_outcome_head.alpha

            self.log(
                f"{self.model_type}_train/bce_loss_cl",
                bce_loss,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                prog_bar=False,
            )

            return bce_loss

    def update_avg_weights(self):
        l = []
        for param in self.br_treatment_outcome_head.state_dict().values():
            param.requires_grad = True
            l.append(param.view(-1))

        current_weights_gen = torch.cat(l)
        t = self.global_step + 1  # assuming global_step is used for counting steps

        if self.global_step > 100:
            regularization_loss = F.mse_loss(current_weights_gen, self.avg_weights)
            print("regularization_loss", regularization_loss, regularization_loss.requires_grad)
        else:
            regularization_loss = 0

        # Update avg_weights buffer
        self.avg_weights = ((t - 1) * self.avg_weights + current_weights_gen) / t

        return regularization_loss

    def _eval_step(self, batch, batch_ind, subset_name):

        active_entries = batch["active_entries"]
        unrolled_active_entries = self._unroll_horizon_random_indices(
            batch["active_entries"], selected_indices=None
        )
        unrolled_current_treatments = self._unroll_horizon_random_indices(
            batch["current_treatments"], selected_indices=None
        )
        unrolled_outputs = self._unroll_horizon_random_indices(
            batch["outputs"], selected_indices=None
        )

        if self.hparams.exp.weights_ema:
            with self.ema_treatment.average_parameters():
                outcome_pred, br = self(batch, return_br=True, selected_indices=None)
                treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detached=False)
        else:
            outcome_pred, br = self(batch, return_br=True, selected_indices=None)
            treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detached=False)

        if self.likelihood_training:
            mse_loss = self._calc_mse_loss(
                outcome_pred.mean, unrolled_outputs, unrolled_active_entries
            )
            loss_y = self._calc_ll_loss(outcome_pred, unrolled_outputs, unrolled_active_entries)
            self.log(
                f"{self.model_type}_{subset_name}/std",
                outcome_pred.stddev.mean(),
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                "val/mse_loss",
                mse_loss,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                prog_bar=True,
            )

        else:
            loss_y = self._calc_mse_loss(outcome_pred, unrolled_outputs, unrolled_active_entries)
            loss_y = self.alpha_mse * loss_y

        unrolled_active_entries = unrolled_active_entries.reshape(-1, self.output_horizon)
        bce_loss = self.bce_loss(treatment_pred, unrolled_current_treatments.double(), kind="MI")
        bce_loss = (unrolled_active_entries * bce_loss).sum() / unrolled_active_entries.sum()
        bce_loss = self.br_treatment_outcome_head.alpha * bce_loss
        loss = bce_loss + loss_y

        self.log(
            f"{self.model_type}_{subset_name}/loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            f"{self.model_type}_{subset_name}/bce_loss",
            bce_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        if subset_name == self.val_dataloader().dataset.subset_name:
            self.log(
                "val/loss", loss_y, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True
            )

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        outcome_pred = self(batch, mean_only=True, selected_indices=None)
        return outcome_pred.cpu()

    def _get_optimizer(self, param_optimizer: list, head_type: str):
        no_decay = ["bias", "layer_norm"]
        sub_args = self.hparams.model[self.model_type]
        if head_type == "non_treatment_head":

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay) and "rep_encoder" not in n
                    ],
                    "weight_decay": sub_args["optimizer"][head_type]["weight_decay"],
                },
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay) and "rep_encoder" in n
                    ],
                    "weight_decay": sub_args["optimizer"][head_type]["weight_decay"],
                    "lr": sub_args["optimizer"][head_type]["rep_encoder"]["learning_rate"],
                },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": sub_args["optimizer"][head_type]["weight_decay"],
                },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

        lr = sub_args["optimizer"][head_type]["learning_rate"]
        optimizer_cls = sub_args["optimizer"][head_type]["optimizer_cls"]
        if optimizer_cls.lower() == "adamw":
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == "adam":
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                optimizer_grouped_parameters,
                lr=lr,
                momentum=sub_args["optimizer"][head_type]["momentum"],
            )
        elif optimizer_cls.lower() == "sgd":
            optimizer = optim.SGD(
                optimizer_grouped_parameters,
                lr=lr,
                momentum=sub_args["optimizer"][head_type]["momentum"],
            )
        else:
            raise NotImplementedError()
        return optimizer

    def configure_optimizers(self):

        treatment_head_params = [
            "br_treatment_outcome_head." + s
            for s in self.br_treatment_outcome_head.treatment_head_params
        ]
        treatment_head_params = [
            k
            for k in dict(self.named_parameters())
            for param in treatment_head_params
            if k.startswith(param)
        ]

        non_treatment_head_params = [
            k for k in dict(self.named_parameters()) if k not in treatment_head_params
        ]

        assert len(treatment_head_params + non_treatment_head_params) == len(
            list(self.named_parameters())
        )

        treatment_head_params = [
            (k, v) for k, v in dict(self.named_parameters()).items() if k in treatment_head_params
        ]
        non_treatment_head_params = [
            (k, v)
            for k, v in dict(self.named_parameters()).items()
            if k in non_treatment_head_params
        ]

        if self.hparams.exp.weights_ema:
            self.ema_treatment = ExponentialMovingAverage(
                [par[1] for par in treatment_head_params], decay=self.hparams.exp.beta
            )
            self.ema_non_treatment = ExponentialMovingAverage(
                [par[1] for par in non_treatment_head_params], decay=self.hparams.exp.beta
            )

        treatment_head_optimizer = self._get_optimizer(
            treatment_head_params, head_type="treatment_head"
        )
        non_treatment_head_optimizer = self._get_optimizer(
            non_treatment_head_params, head_type="non_treatment_head"
        )

        if self.hparams.model[self.model_type]["optimizer"]["lr_scheduler"]:
            return self._get_lr_schedulers(
                [non_treatment_head_optimizer, treatment_head_optimizer]
            )

        return [non_treatment_head_optimizer, treatment_head_optimizer]

    def get_pehe_one_step(self, dataset: Dataset):
        logger.info(f"PEHE calculation for {dataset.subset_name}.")
        percentage = self.hparams.exp.percentage_rmse

        output_stds, output_means = (
            dataset.scaling_params["output_stds"],
            dataset.scaling_params["output_means"],
        )

        num_samples, time_dim, output_dim = dataset.data["active_entries"].shape
        last_entries = dataset.data["active_entries"] - np.concatenate(
            [dataset.data["active_entries"][:, 1:, :], np.zeros((num_samples, 1, output_dim))],
            axis=1,
        )

        dataset.data["current_treatments"][:, -1, :] = np.ones((num_samples, 1))
        yt_1 = self.get_predictions(dataset)
        yt_1 = yt_1 * output_stds + output_means

        dataset.data["current_treatments"][:, -1, :] = np.zeros((num_samples, 1))
        yt_0 = self.get_predictions(dataset)
        yt_0 = yt_0 * output_stds + output_means

        ites_pred = yt_1 - yt_0
        ite_real = dataset.data["ITE"]

        if len(ite_real.shape) == 2:
            ite_real = ite_real[:, :, np.newaxis]

        pehe_last = ((ites_pred[:, :, 0, :] - ite_real) ** 2) * last_entries

        pehe_last = pehe_last.sum() / last_entries.sum()
        rmse_normalised_last = np.sqrt(pehe_last) / dataset.norm_const

        if percentage:
            rmse_normalised_last *= 100.0

        return rmse_normalised_last
