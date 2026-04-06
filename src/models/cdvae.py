import logging
from collections import namedtuple
from typing import Union

import numpy as np
import torch
import torch.distributions as D
import torch.optim as optim
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
from torch.utils.data import DataLoader, Dataset

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.inference_net import Inference_Net
from src.models.time_varying_model import BRCausalModel
from src.models.utils import *
from src.models.utils_cdvae import GMMprior, deviance_loss, wasserstein

logger = logging.getLogger(__name__)


class WRep_encoder(BRCausalModel):
    """
    PyTorch-Lightning implementation of Causal Dynamical Variational Autoencoding (CDVAE) model
    """

    model_type = "wrep_encoder"

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
            autoregressive: Flag for including previous outcomes in modeling
            has_vitals: Flag indicating if vitals are present in the dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Additional arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.input_size = self.dim_treatments + self.dim_static_features
        self.input_size += self.dim_vitals if self.has_vitals else 0
        self.input_size += self.dim_outcome if self.autoregressive else 0
        logger.info(f"Input size of {self.model_type}: {self.input_size}")

        self._init_specific(args.model.wrep_encoder)
        self.save_hyperparameters(args)

    def _init_specific(self, sub_args: DictConfig):
        """
        Initializes model-specific parameters.
        """
        self.ForwardOutputs = namedtuple(
            "ForwardOutputs",
            [
                "treatment_pred",
                "br",
            ],
        )
        self.IndustrialForwardOutputs = namedtuple(
            "IndustrialForwardOutputs",
            [
                "y_factual",
                "x_recon",
                "background_latent",
                "health_states_base",
                "fault_strength",
                "latent_stats",
            ],
        )
        try:
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.z_latent_dim = sub_args.z_latent_dim
            self.context_latent_dim = sub_args.context_latent_dim
            self.br_size = sub_args.br_size
            self.weighting_method = sub_args.weighting_method
            self.dropout_rate = sub_args.dropout_rate
            self.num_layer = sub_args.num_layer
            self.prediction_step = self.hparams.dataset.projection_horizon + 1
            self.seq_len = self.hparams.dataset.max_seq_length - 1  # max_seq_length
            self.activation = sub_args.activation

            if any(
                param is None for param in [self.br_size, self.fc_hidden_units, self.dropout_rate]
            ):
                raise MissingMandatoryValue()

            self.br_treatment_outcome_head = WRTreatmentOutcomeHead(
                self.context_latent_dim,
                self.br_size,
                self.fc_hidden_units,
                self.z_latent_dim,
                self.dim_treatments,
                self.dim_outcome,
                self.alpha,
                self.update_alpha,
                activation=self.activation,
            )

            self.treat_normalizer = nn.Sigmoid()  # Multilabel assmp

            self.gru = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.context_latent_dim,
                num_layers=self.num_layer,
                batch_first=True,
                dropout=self.dropout_rate,
            )

        except MissingMandatoryValue:
            logger.warning(
                f"{self.model_type} not fully initialized - some mandatory args are missing! "
                f"(It's ok if one will perform hyperparameter search afterward)."
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes weights for different layers.
        """
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

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Sets hyperparameters for model reinitialization.
        """
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args["learning_rate"]
        sub_args.batch_size = new_args["batch_size"]
        if "seq_hidden_units" in new_args:
            sub_args.seq_hidden_units = int(input_size * new_args["seq_hidden_units"])
        sub_args.br_size = int(input_size * new_args["br_size"])
        sub_args.fc_hidden_units = int(sub_args.br_size * new_args["fc_hidden_units"])
        sub_args.dropout_rate = new_args["dropout_rate"]
        sub_args.num_layer = new_args["num_layer"]

    def prepare_data(self) -> None:
        """
        Prepares dataset by normalizing and processing.
        """
        if (
            self.dataset_collection is not None
            and not self.dataset_collection.processed_data_multi
        ):
            self.dataset_collection.process_data_multi()

        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def build_input(self, batch: dict) -> torch.Tensor:
        """
        Builds the input tensor for the model from the batch data.
        """
        prev_treatments = batch["prev_treatments"]
        vitals_or_prev_outputs = []
        if self.has_vitals:
            vitals_or_prev_outputs.append(batch["vitals"])
        if self.autoregressive:
            vitals_or_prev_outputs.append(batch["prev_outputs"])
        vitals_or_prev_outputs = (
            torch.cat(vitals_or_prev_outputs, dim=-1) if vitals_or_prev_outputs else None
        )
        x = torch.cat((prev_treatments, vitals_or_prev_outputs), dim=-1)

        if self.dim_static_features > 0:
            static_features = batch["static_features"]
            x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        return x

    def build_br(self, x: torch.Tensor) -> torch.Tensor:
        """
        Builds the balanced representation (BR) for the given input.
        """
        c = self.gru(x)[0]
        br = self.br_treatment_outcome_head.build_br(c)
        return br

    def forward(self, batch: dict, detach_treatment: bool = False):
        """
        Forward pass through the model.
        """
        x = self.build_input(batch)

        br = self.build_br(x)

        treatment_pred = self.br_treatment_outcome_head.build_treatment(
            br, detached=detach_treatment
        )

        return self.ForwardOutputs(treatment_pred, br)

    def _shared_step(self, batch: dict, stage: str):
        """
        Shared computation for training and validation steps to reduce redundancy.

        Args:
            batch (dict): The input batch.
            stage (str): The stage of training (e.g., 'train', 'val').

        Returns:
            torch.Tensor: The total loss for the given stage.
        """
        treatment_pred, br = self(batch)
        curr_treatments = batch["current_treatments"]

        p_w_x = self.treat_normalizer(treatment_pred)
        weights = self.weighting(curr_treatments, p_w_x)

        bce_loss = self.bce_loss(treatment_pred, curr_treatments.double())
        total_loss, bce_loss = self._aggregate_losses(weights, bce_loss, batch)
        self.log_metrics(stage, total_loss, bce_loss)

        return total_loss

    def _aggregate_losses(self, weights, bce_loss, batch):
        """
        Aggregates different losses to form the total loss.
        """

        active_entries = batch["active_entries"].squeeze(-1)
        weights = weights.squeeze(-1)

        bce_loss = (active_entries * bce_loss).sum(dim=1)
        bce_loss = bce_loss.mean()

        total_loss = bce_loss

        return total_loss, bce_loss

    def training_step(
        self, batch: dict, batch_idx: int, optimizer_idx: int = 0
    ):  # ? batch_idx: int, optimizer_idx: int = 0
        """
        Training step to calculate and log loss.
        """
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: dict, batch_idx: int, **kwargs
    ):  # ? , batch_idx: int, **kwargs
        """
        Validation step to calculate and log loss.
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int, **kwargs):  # ? , batch_idx: int, **kwargs
        """
        Test step to calculate and log loss.
        """
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        """
        Generates normalised output predictions
        """
        ForwardOutputs = self(batch)
        br = ForwardOutputs.br

        return br.cpu()

    def log_metrics(
        self,
        stage: str,
        loss: torch.Tensor,
        bce_loss: torch.Tensor,
    ):
        """
        Logs training metrics in a dictionary format to avoid redundant log lines.

        The reason for using `loss_y` instead of `loss` for the validation stage (`val`) is to prioritize tracking
        the main target variable loss during validation. Since `loss_y` represents the predictive performance
        on the target outcome, it is often more indicative of model performance compared to the total loss,
        which may include additional regularization terms.

        Args:
            stage (str): The stage of training (e.g., 'train', 'val', 'test').
            metrics_dict (dict): A dictionary containing metric names as keys and their corresponding values.
            early_stopping_metric (bool): Whether to log the early stopping metric specifically. self.log('val_y_scale', self.y_scale.item(), prog_bar=True, on_epoch=True)
        """
        metrics = {
            f"{stage}/loss": loss,
            f"{stage}/bce_loss": bce_loss,
        }

        for key, value in metrics.items():
            self.log(
                key,
                value,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                prog_bar=key.endswith("_loss"),
            )
        self.log("alpha", self.br_treatment_outcome_head.alpha, on_epoch=True, on_step=False)

    @staticmethod
    def wasserstein_distance_gauss(mu_1, std_1, mu_2, std_2):
        r"""
        Wasserstein distance betwen two Gaussiance with diagonal covarance matrirx.
        We use the formula where the covariance matrices commute i.e. $\Sigma_1\Sigma_2=\Sigma_2\Sigma_1$.
        The formula is:

        $W_2(N(m_1,\Sigma_1);N(m_2,\Sigma_2))2=∥m_1-m_2∥_2^2+∥\Sigma^{1/2}_1-\Sigma^{1/2}_2∥^2_{F}$

        Reference: https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/#eqWG
        """

        wass = torch.norm(mu_2 - mu_1, dim=-1) + torch.norm(std_2 - std_1, dim=-1)
        wass = wass.reshape(wass.shape[0], -1)

        return wass

    def weighting(self, w: torch.Tensor, p_w_x: torch.Tensor) -> torch.Tensor:
        """
        Computes weighting for the given treatment and predicted treatment probabilities.
        """
        methods = {
            "IPTW": lambda w, p_w_x: w / p_w_x + (1 - w) / (1 - p_w_x),
            "context_aware": lambda w, p_w_x: (
                torch.mean(w, dim=0).view(1, -1).repeat((w.shape[0], 1))
                / (1 - torch.mean(w, dim=0)).view(1, -1).repeat((w.shape[0], 1))
            )
            * ((1 - p_w_x) / p_w_x)
            + 1,
            "overlap": lambda w, p_w_x: p_w_x * (1 - p_w_x) / (w * p_w_x + (1 - w) * (1 - p_w_x)),
            "none": lambda w, p_w_x: torch.ones(w.shape, dtype=w.dtype).to(self.device),
        }
        return methods[self.weighting_method](w, p_w_x)

    def _get_optimizer(self, param_optimizer: list):
        no_decay = ["bias", "layer_norm"]
        sub_args = self.hparams.model.wrep_encoder
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": sub_args["optimizer"]["weight_decay"],
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        lr = sub_args["optimizer"]["learning_rate"]
        optimizer_cls = sub_args["optimizer"]["optimizer_cls"]
        if optimizer_cls.lower() == "adamw":
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == "adam":
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == "sgd":
            optimizer = optim.SGD(
                optimizer_grouped_parameters, lr=lr, momentum=sub_args["optimizer"]["momentum"]
            )
        else:
            raise NotImplementedError()

        return optimizer

    def configure_optimizers(self):
        optimizer = self._get_optimizer(list(self.named_parameters()))

        if self.hparams.model.cdvae["optimizer"]["lr_scheduler"]:
            return self._get_lr_schedulers(optimizer)

        return optimizer


class CDVAE(BRCausalModel):
    """
    PyTorch-Lightning implementation of Causal Dynamical Variational Autoencoding (CDVAE) model
    """

    model_type = "cdvae"

    def __init__(
        self,
        args: DictConfig,
        wrep_encoder,
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
            autoregressive: Flag for including previous outcomes in modeling
            has_vitals: Flag indicating if vitals are present in the dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Additional arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        # self.wrep_encoder = wrep_encoder
        self.input_size = self.dim_treatments + self.dim_static_features
        self.input_size += self.dim_vitals if self.has_vitals else 0
        self.input_size += self.dim_outcome if self.autoregressive else 0
        logger.info(f"Input size of {self.model_type}: {self.input_size}")
        self._init_specific(args.model.cdvae)
        self.save_hyperparameters(args)

        self.br_treatment_outcome_head = wrep_encoder.br_treatment_outcome_head
        self.gru = wrep_encoder.gru

        self.treat_normalizer = nn.Sigmoid()  # Multilabel assmp

        self.inference_re_given_yxw = Inference_Net(
            input_size=self.input_size,
            hidden_size=self.fc_hidden_units,
            z_latent_dim=self.z_latent_dim,
            num_layer=self.num_layer,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            background_latent_dim=self.z_latent_dim,
            health_latent_dim=self.health_state_dim,
            fault_latent_dim=self.fault_state_dim,
            num_regimes=getattr(self, "num_regimes", None),
            regime_embed_dim=getattr(self, "regime_embed_dim", 0),
        )
        self.gmm_prior = GMMprior(
            n_clusters=self.n_clusters,
            z_latent_dim=self.z_latent_dim,
            cov_type_p_z_given_c=self.cov_type_p_z_given_c,
            to_fix_pi_p_c=self.to_fix_pi_p_c,
            init_type_p_z_given_c=self.init_type_p_z_given_c,
            device=self.device,
        )
        self._init_industrial_modules()

    def _init_industrial_modules(self):
        self.industrial_input_projection = nn.LazyLinear(self.input_size)
        self.health_input_projection = nn.LazyLinear(
            self.dim_vitals + self.dim_treatments + self.z_latent_dim
        )
        self.regime_embedding = nn.Embedding(self.regime_vocab_size, self.context_latent_dim)
        self.health_transition = nn.GRUCell(
            self.dim_vitals + self.dim_treatments + self.z_latent_dim,
            self.health_state_dim,
        )
        self.fault_adapter = nn.Sequential(
            nn.Linear(
                self.health_state_dim
                + self.dim_treatments
                + self.z_latent_dim
                + self.fault_state_dim
                + self.context_latent_dim,
                self.health_state_dim,
            ),
            nn.ELU(),
            nn.Linear(self.health_state_dim, self.health_state_dim),
        )
        self.x_decoder = nn.Linear(self.health_state_dim + self.z_latent_dim, self.dim_vitals)
        self.y_head = nn.Linear(self.health_state_dim + self.z_latent_dim, self.dim_outcome)
        self.s_prior_loc = nn.Embedding(self.regime_vocab_size, self.fault_state_dim)
        self.s_prior_logvar = nn.Embedding(self.regime_vocab_size, self.fault_state_dim)
        nn.init.zeros_(self.s_prior_loc.weight)
        nn.init.zeros_(self.s_prior_logvar.weight)

    def get_last_posterior_heads(self):
        """Returns the most recent three-head posterior bundle for diagnostics/debugging."""
        return getattr(self, "latest_inference_outputs", None)

    def _init_specific(self, sub_args: DictConfig):
        """
        Initializes model-specific parameters.
        """
        self.ForwardOutputs = namedtuple(
            "ForwardOutputs",
            [
                "treatment_pred",
                "outcome_pred",
                "br",
                "mu_RE",
                "re",
                "q_re_given_yxw",
                "RE_hidden_states",
            ],
        )
        try:
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.z_latent_dim = sub_args.z_latent_dim
            self.context_latent_dim = sub_args.context_latent_dim
            self.br_size = sub_args.br_size
            self.weighting_method = sub_args.weighting_method
            self.dropout_rate = sub_args.dropout_rate
            self.num_layer = sub_args.num_layer
            self.prediction_step = self.hparams.dataset.projection_horizon + 1
            self.seq_len = self.hparams.dataset.max_seq_length - 1  # max_seq_length
            self.kld_weight = sub_args.kld_weight
            self.use_deviance = sub_args.use_deviance
            self.y_dist_type = sub_args.y_dist_type
            self.activation = sub_args.activation
            self.min_timestep = sub_args.min_timestep

            self.percentage_steps_ipm = sub_args.percentage_steps_ipm
            self.y_scale_require_grad = sub_args.y_scale_require_grad

            self.n_clusters = sub_args.gmm_prior.n_clusters
            self.cov_type_p_z_given_c = sub_args.gmm_prior.cov_type_p_z_given_c
            self.to_fix_pi_p_c = sub_args.gmm_prior.to_fix_pi_p_c
            self.init_type_p_z_given_c = sub_args.gmm_prior.init_type_p_z_given_c

            self.lambda_ipm = sub_args.lambda_ipm
            self.lambda_mm = sub_args.lambda_mm
            self.lambda_y = sub_args.lambda_y

            self.mc_sample_size = sub_args.batch_size // 10
            self.enable_industrial_soft_sensing = getattr(
                sub_args, "enable_industrial_soft_sensing", False
            )
            self.health_state_dim = getattr(sub_args, "health_state_dim", self.context_latent_dim)
            self.fault_state_dim = getattr(sub_args, "fault_state_dim", self.context_latent_dim)
            self.regime_vocab_size = getattr(sub_args, "regime_vocab_size", 32)
            self.num_regimes = getattr(sub_args, "num_regimes", None)
            self.regime_embed_dim = getattr(sub_args, "regime_embed_dim", 0)

            self.log_y_scale = nn.Parameter(
                torch.tensor(0.0), requires_grad=self.y_scale_require_grad
            )

            self.with_mu_RE = False

            if any(
                param is None for param in [self.br_size, self.fc_hidden_units, self.dropout_rate]
            ):
                raise MissingMandatoryValue()

        except MissingMandatoryValue:
            logger.warning(
                f"{self.model_type} not fully initialized - some mandatory args are missing! "
                f"(It's ok if one will perform hyperparameter search afterward)."
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes weights for different layers.
        """
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

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Sets hyperparameters for model reinitialization.
        """
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args["learning_rate"]
        sub_args.batch_size = new_args["batch_size"]
        if "seq_hidden_units" in new_args:
            sub_args.seq_hidden_units = int(input_size * new_args["seq_hidden_units"])
        sub_args.br_size = int(input_size * new_args["br_size"])
        sub_args.fc_hidden_units = int(sub_args.br_size * new_args["fc_hidden_units"])
        sub_args.dropout_rate = new_args["dropout_rate"]
        sub_args.num_layer = new_args["num_layer"]

    @property
    def y_scale(self):
        """
        Returns the positive y_scale by applying the exponential transformation to log_y_scale.
        """
        return torch.exp(self.log_y_scale)

    def prepare_data(self) -> None:
        """
        Prepares dataset by normalizing and processing.
        """
        if (
            self.dataset_collection is not None
            and not self.dataset_collection.processed_data_multi
        ):
            self.dataset_collection.process_data_multi()

        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def build_input(self, batch: dict) -> torch.Tensor:
        """
        Builds the input tensor for the model from the batch data.
        """
        prev_treatments = batch["prev_treatments"]
        prev_treatments_and_outputs = []
        prev_treatments_and_outputs.append(prev_treatments)

        if self.autoregressive:
            prev_treatments_and_outputs.append(batch["prev_outputs"])

        x = torch.cat(prev_treatments_and_outputs, dim=-1)
        x_posterior = torch.cat(prev_treatments_and_outputs, dim=-1)

        if self.has_vitals:
            x = torch.cat((batch["vitals"], x), dim=-1)
            x_posterior = torch.cat((batch["prev_vitals"], x_posterior), dim=-1)

        if self.dim_static_features > 0:
            static_features = batch["static_features"]
            static_features_expanded = static_features.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat((x, static_features_expanded), dim=-1)
            x_posterior = torch.cat((x_posterior, static_features_expanded), dim=-1)

        return x, x_posterior

    def build_br(self, x: torch.Tensor) -> torch.Tensor:
        """
        Builds the balanced representation (BR) for the given input.
        """
        c = self.gru(x)[0]
        br = self.br_treatment_outcome_head.build_br(c)
        return br

    @staticmethod
    def _is_industrial_batch(batch: dict) -> bool:
        return all(
            k in batch for k in ["process_vars", "controls", "outputs", "regime_id", "active_entries"]
        )

    def build_industrial_input(self, batch: dict) -> torch.Tensor:
        process_vars = batch["process_vars"]
        controls = batch["controls"]
        outputs = batch["outputs"]
        regime_id = batch["regime_id"]

        prev_x = torch.zeros_like(process_vars)
        prev_x[:, 1:, :] = process_vars[:, :-1, :]
        prev_y = torch.zeros_like(outputs)
        prev_y[:, 1:, :] = outputs[:, :-1, :]
        regime_embed = self.regime_embedding(regime_id).unsqueeze(1).expand(-1, process_vars.size(1), -1)
        return torch.cat([prev_x, controls, prev_y, regime_embed], dim=-1)

    def _sample_gaussian(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _run_industrial_dynamics(
        self,
        process_vars: torch.Tensor,
        controls: torch.Tensor,
        regime_embed: torch.Tensor,
        b_latent: torch.Tensor,
        h0_init: torch.Tensor,
        s_latent: torch.Tensor,
    ):
        batch_size, seq_len, _ = process_vars.shape
        h_prev = h0_init
        h_fault_seq, h_base_seq, x_seq, y_seq = [], [], [], []
        b_seq = b_latent.unsqueeze(1).expand(-1, seq_len, -1)

        for t in range(seq_len):
            health_in_raw = torch.cat([process_vars[:, t, :], controls[:, t, :], b_latent], dim=-1)
            health_in = self.health_input_projection(health_in_raw)
            h0_t = self.health_transition(health_in, h_prev)
            adapter_in = torch.cat([h0_t, controls[:, t, :], b_latent, s_latent[:, t, :], regime_embed], dim=-1)
            delta_t = self.fault_adapter(adapter_in)
            h_t = h0_t + delta_t
            state_and_b = torch.cat([h_t, b_latent], dim=-1)
            x_seq.append(self.x_decoder(state_and_b))
            y_seq.append(self.y_head(state_and_b))
            h_base_seq.append(h0_t)
            h_fault_seq.append(h_t)
            h_prev = h0_t

        return {
            "x_recon": torch.stack(x_seq, dim=1),
            "y_factual": torch.stack(y_seq, dim=1),
            "h_base_seq": torch.stack(h_base_seq, dim=1),
            "h_fault_seq": torch.stack(h_fault_seq, dim=1),
            "b_seq": b_seq,
        }

    @staticmethod
    def _kl_diag_gaussian(mu_q, logvar_q, mu_p, logvar_p):
        var_ratio = torch.exp(logvar_q - logvar_p)
        kl = 0.5 * (
            logvar_p
            - logvar_q
            + var_ratio
            + ((mu_q - mu_p) ** 2) / torch.exp(logvar_p)
            - 1.0
        )
        return kl.sum(dim=-1)

    def _build_industrial_priors(self, regime_id: torch.Tensor, seq_len: int):
        b_mu = torch.zeros(
            regime_id.shape[0], self.z_latent_dim, device=regime_id.device, dtype=torch.float32
        )
        b_logvar = torch.zeros_like(b_mu)
        h0_mu = torch.zeros(
            regime_id.shape[0], self.health_state_dim, device=regime_id.device, dtype=torch.float32
        )
        h0_logvar = torch.zeros_like(h0_mu)

        s_mu_regime = self.s_prior_loc(regime_id)
        s_logvar_regime = self.s_prior_logvar(regime_id)
        s_mu = s_mu_regime.unsqueeze(1).expand(-1, seq_len, -1)
        s_logvar = s_logvar_regime.unsqueeze(1).expand(-1, seq_len, -1)
        return {
            "b_mu": b_mu,
            "b_logvar": b_logvar,
            "h0_mu": h0_mu,
            "h0_logvar": h0_logvar,
            "s_mu": s_mu,
            "s_logvar": s_logvar,
        }

    def forward(self, batch: dict, detach_treatment: bool = False):
        """
        Forward pass through the model.
        """
        if self._is_industrial_batch(batch):
            return self.forward_industrial(batch)

        x, x_posterior = self.build_input(batch)

        posterior_heads = self.inference_re_given_yxw(x_posterior, return_dict=True)
        mu_RE = posterior_heads["compat"]["re_loc"]
        log_var_RE = posterior_heads["compat"]["re_logvar"]
        RE_hidden_states = posterior_heads["enc_feat"]
        self.latest_inference_outputs = posterior_heads
        if torch.isnan(mu_RE).any():
            print("The mu_RE contains NaN values.")

        if torch.isnan(log_var_RE).any():
            print("The log_var_RE contains NaN values.")

        mu_RE[torch.isnan(mu_RE)] = 0.0
        log_var_RE[torch.isnan(log_var_RE)] = 0.0

        q_re_given_yxw = D.Independent(
            D.Normal(loc=mu_RE, scale=torch.exp(0.5 * log_var_RE)),
            reinterpreted_batch_ndims=1,  # interpret dim 1 (d_re diemion as a single event)
        )  # ? Do we need a permutation ?
        re = q_re_given_yxw.rsample()
        re_extended = re.unsqueeze(1).repeat(1, x.shape[1], 1)

        br = self.build_br(x)
        treatment_pred = self.br_treatment_outcome_head.build_treatment(
            br, detached=detach_treatment
        )

        curr_treatments = batch["current_treatments"]
        outcome_pred = self.br_treatment_outcome_head.build_outcome(
            br, re_extended, curr_treatments, y_dist_type=self.y_dist_type
        )

        return self.ForwardOutputs(
            treatment_pred, outcome_pred, br, mu_RE, re, q_re_given_yxw, RE_hidden_states
        )

    def forward_industrial(self, batch: dict):
        inference_in = self.industrial_input_projection(self.build_industrial_input(batch))
        inference_outputs = self.inference_re_given_yxw(
            inference_in, return_dict=True, regime_context=batch.get("regime_id", None)
        )
        self.latest_inference_outputs = inference_outputs
        latent_stats = inference_outputs["posterior_params"]
        b_latent = inference_outputs["samples"]["b"]
        h0_init = inference_outputs["samples"]["h0"]
        s_latent = inference_outputs["samples"]["s"]
        priors = self._build_industrial_priors(batch["regime_id"], seq_len=batch["process_vars"].shape[1])
        kl_b = self._kl_diag_gaussian(
            latent_stats["b_loc"], latent_stats["b_logvar"], priors["b_mu"], priors["b_logvar"]
        ).mean()
        kl_h0 = self._kl_diag_gaussian(
            latent_stats["h0_loc"], latent_stats["h0_logvar"], priors["h0_mu"], priors["h0_logvar"]
        ).mean()
        kl_s = self._kl_diag_gaussian(
            latent_stats["s_loc"], latent_stats["s_logvar"], priors["s_mu"], priors["s_logvar"]
        ).mean()

        regime_embed = self.regime_embedding(batch["regime_id"])
        rollout = self._run_industrial_dynamics(
            process_vars=batch["process_vars"],
            controls=batch["controls"],
            regime_embed=regime_embed,
            b_latent=b_latent,
            h0_init=h0_init,
            s_latent=s_latent,
        )
        return self.IndustrialForwardOutputs(
            y_factual=rollout["y_factual"],
            x_recon=rollout["x_recon"],
            background_latent=b_latent,
            health_states_base=rollout["h_base_seq"],
            fault_strength=s_latent,
            latent_stats={**latent_stats, "kl_b": kl_b, "kl_h0": kl_h0, "kl_s": kl_s},
        )

    def _abduct_industrial_latents(self, batch: dict):
        inference_in = self.industrial_input_projection(self.build_industrial_input(batch))
        inference_outputs = self.inference_re_given_yxw(
            inference_in, return_dict=True, regime_context=batch.get("regime_id", None)
        )
        return (
            inference_outputs,
            inference_outputs["samples"]["b"],
            inference_outputs["samples"]["h0"],
            inference_outputs["samples"]["s"],
        )

    def _apply_s_intervention(
        self,
        s: torch.Tensor,
        action: str = "zero",
        intervention_s: torch.Tensor = None,
        intervention_mask: torch.Tensor = None,
        regime_override: torch.Tensor = None,
        regime_seen_mask: torch.Tensor = None,
        use_unknown_prior: bool = True,
    ) -> torch.Tensor:
        _ = regime_seen_mask
        if intervention_mask is None:
            intervention_mask = torch.ones_like(s, dtype=torch.bool)

        if action == "zero":
            s_target = torch.zeros_like(s)
        elif action == "replace":
            if intervention_s is None:
                raise ValueError("`intervention_s` must be provided when action='replace'.")
            s_target = intervention_s
        elif action == "prior_mean":
            if regime_override is None:
                if use_unknown_prior:
                    regime_override = torch.zeros(
                        s.shape[0], dtype=torch.long, device=s.device
                    )  # default no-fault center
                else:
                    raise ValueError(
                        "`regime_override` is required for action='prior_mean' when use_unknown_prior=False."
                    )
            priors = self._build_industrial_priors(regime_override, seq_len=s.shape[1])
            s_target = priors["s_mu"]
        else:
            raise ValueError(f"Unsupported intervention action: {action}")

        return torch.where(intervention_mask, s_target, s)

    def counterfactual_query(
        self,
        batch: dict,
        intervention_s: torch.Tensor = None,
        intervention_mask: torch.Tensor = None,
        action: str = "zero",
        regime_override: torch.Tensor = None,
        use_unknown_prior: bool = True,
        return_latents: bool = False,
    ):
        _, b_latent, h0_init, s_latent = self._abduct_industrial_latents(batch)

        factual_rollout = self._run_industrial_dynamics(
            process_vars=batch["process_vars"],
            controls=batch["controls"],
            regime_embed=self.regime_embedding(batch["regime_id"]),
            b_latent=b_latent,
            h0_init=h0_init,
            s_latent=s_latent,
        )

        if regime_override is None:
            regime_override = torch.zeros_like(batch["regime_id"], dtype=torch.long)
        elif isinstance(regime_override, int):
            regime_override = torch.full_like(batch["regime_id"], fill_value=int(regime_override))

        regime_embed = self.regime_embedding(regime_override)
        s_cf = self._apply_s_intervention(
            s=s_latent,
            action=action,
            intervention_s=intervention_s,
            intervention_mask=intervention_mask,
            regime_override=regime_override,
            use_unknown_prior=use_unknown_prior,
        )
        rollout_cf = self._run_industrial_dynamics(
            process_vars=batch["process_vars"],
            controls=batch["controls"],
            regime_embed=regime_embed,
            b_latent=b_latent,
            h0_init=h0_init,
            s_latent=s_cf,
        )
        y_cf = rollout_cf["y_factual"]
        y_factual = factual_rollout["y_factual"]
        result = {
            "y_factual": y_factual,
            "y_counterfactual_normal": y_cf,
            "delta_quality": y_factual - y_cf,
            "applied_action": action,
            "regime_override": regime_override,
        }
        if return_latents:
            result.update(
                {
                    "background_latent": b_latent,
                    "healthy_initial_state": h0_init,
                    "fault_strength": s_latent,
                    "fault_strength_counterfactual": s_cf,
                }
            )
        return result

    def industrial_latent_diagnostics(self, batch: dict, normal_regime_id: int = 0):
        outputs = self.counterfactual_query(
            batch,
            action="zero",
            regime_override=normal_regime_id,
            return_latents=True,
        )
        s_abs_mean = outputs["fault_strength"].abs().mean(dim=(1, 2))
        delta_direction = outputs["delta_quality"].mean(dim=(1, 2))
        b_var = outputs["background_latent"].var(dim=-1)
        return {
            "s_abs_mean": s_abs_mean,
            "delta_quality_mean": delta_direction,
            "background_variance": b_var,
        }

    def _get_y_dist(self, outcome_pred: torch.Tensor):
        """
        Returns the distribution of the predicted outcome.
        """
        outcome_pred[torch.isnan(outcome_pred)] = 0.0
        if self.y_dist_type == "discrete":
            return Poisson(rate=outcome_pred)
        elif self.y_dist_type == "continuous":
            return Normal(loc=outcome_pred, scale=self.y_scale)
        else:
            raise ValueError("Unsupported y_dist_type")

    def _shared_step(self, batch: dict, stage: str):
        """
        Shared computation for training and validation steps to reduce redundancy.

        Args:
            batch (dict): The input batch.
            stage (str): The stage of training (e.g., 'train', 'val').

        Returns:
            torch.Tensor: The total loss for the given stage.
        """
        treatment_pred, outcome_pred, br, _, re, q_re_given_yxw, RE_hidden_states = self(batch)
        curr_treatments = batch["current_treatments"]

        p_w_x = self.treat_normalizer(treatment_pred)

        weights = self.weighting(curr_treatments, p_w_x)
        if torch.isnan(weights).any():
            print("The weights contains NaN values.")
        y_dist = self._get_y_dist(outcome_pred)
        loss_y = self.y_loss(y_dist, batch["outputs"], weights, self.use_deviance)

        loss_1mm = self.reg_matching(RE_hidden_states)

        kld_z, kld_c = self.kld_loss(q_re_given_yxw)
        if torch.isnan(kld_z).any():
            print("The kld_z contains NaN values.")

        if torch.isnan(kld_c).any():
            print("The kld_c contains NaN values.")

        total_loss, loss_y, loss_1mm, kld_z, kld_c = self._aggregate_losses(
            loss_y, weights, loss_1mm, kld_z, kld_c, batch
        )
        self.log_metrics(stage, total_loss, kld_z, kld_c, loss_y, loss_1mm)
        self.log("weights", torch.mean(weights.sum(dim=-1)), on_epoch=True, on_step=False)

        return total_loss

    def _aggregate_losses(self, loss_y, weights, loss_1mm, kld_z, kld_c, batch):
        """
        Aggregates different losses to form the total loss.
        """

        active_entries = batch["active_entries"].squeeze(-1)
        weights = weights.squeeze(-1)

        loss_y = (active_entries * loss_y.mean(dim=2)).sum(dim=1)

        loss_y = loss_y.mean()
        loss_1mm = (active_entries[:, self.min_timestep : -1] * loss_1mm).sum(dim=-1)

        loss_1mm = loss_1mm.mean()

        kld_re = kld_z + kld_c
        kld_re = kld_re.mean()

        loss_1mm *= self.lambda_mm
        loss_y *= self.lambda_y
        kld_re *= self.kld_weight

        total_loss = kld_re + loss_y + loss_1mm

        return total_loss, loss_y, loss_1mm, kld_z, kld_c

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0):
        """
        Training step to calculate and log loss.
        """
        curr_treatments = batch["current_treatments"]
        active_entries = batch["active_entries"].squeeze(-1)

        if optimizer_idx == 0:
            loss = self._shared_step(batch, "train")
            return loss

        elif optimizer_idx == 1:
            ForwardOutputs = self(batch, detach_treatment=True)
            treatment_pred = ForwardOutputs.treatment_pred
            bce_loss = self.bce_loss(treatment_pred, curr_treatments.double())
            bce_loss = (active_entries * bce_loss).sum(dim=1)
            bce_loss = bce_loss.mean()

            br = ForwardOutputs.br
            p_w_x = self.treat_normalizer(treatment_pred)
            weights = self.weighting(curr_treatments, p_w_x)

            loss_ipm = 0
            if self.lambda_ipm > 0:
                num_steps_ipm = int(self.percentage_steps_ipm * self.seq_len)
                time_samples = torch.randint(low=1, high=self.seq_len, size=(num_steps_ipm,))
                for timestep in time_samples:
                    if 0 < curr_treatments[:, timestep, :].mean() < 1:
                        ipm_timestep = wasserstein(
                            br[:, timestep, :],
                            curr_treatments[:, timestep, :],
                            active_entries[:, timestep],
                            weights[:, timestep, :],
                        )
                        loss_ipm += ipm_timestep
            loss_ipm *= self.lambda_ipm

            self.log(
                "train/bce_loss_cl",
                bce_loss,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                prog_bar=False,
            )

            self.log(
                "train/loss_ipm",
                loss_ipm,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                prog_bar=False,
            )

            loss = bce_loss + loss_ipm

        return loss

    def validation_step(
        self, batch: dict, batch_idx: int, **kwargs
    ):  # ? , batch_idx: int, **kwargs
        """
        Validation step to calculate and log loss.
        """

        val_loss = self._shared_step(batch, "val")

        return val_loss

    def test_step(self, batch: dict, batch_idx: int, **kwargs):  # ? , batch_idx: int, **kwargs
        """
        Validation step to calculate and log loss.
        """
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        """
        Generates normalised output predictions
        """
        ForwardOutputs = self(batch)
        outcome_pred, br = ForwardOutputs.outcome_pred, ForwardOutputs.br

        if not self.with_mu_RE:
            return outcome_pred.cpu(), br.cpu()
        else:
            mu_RE = ForwardOutputs.mu_RE
            return outcome_pred.cpu(), br.cpu(), mu_RE.cpu()

    def kld_loss(self, q_re_given_yxw: torch.distributions.Normal) -> torch.Tensor:
        """
        Computes the Kullback-Leibler Divergence losses.
        """

        if self.cov_type_p_z_given_c == "diag":
            sigma_square_p_z_given_c = self.gmm_prior.sigma_square_p_z_given_c
            p_z_given_c = D.Independent(
                D.Normal(
                    loc=self.gmm_prior.mu_p_z_given_c.permute(1, 0),
                    scale=torch.sqrt(sigma_square_p_z_given_c.permute(1, 0)),
                ),
                1,
            )
        elif self.cov_type_p_z_given_c == "full":
            l_mat_p_z_given_c = self.gmm_prior.l_mat_p_z_given_c
            p_z_given_c = D.MultivariateNormal(
                loc=self.gmm_prior.mu_p_z_given_c.permute(
                    1, 0
                ),  # ! permutation gives (n_clusters, z_latent_dim)
                scale_tril=l_mat_p_z_given_c.permute(
                    2, 0, 1
                ),  # !  (z_latent_dim, z_latent_dim, n_clusters) - >   permutation gives (n_clusters, z_latent_dim, z_latent_dim)
            )

        re = q_re_given_yxw.rsample((self.mc_sample_size,))

        re_pad = torch.unsqueeze(re, -2)
        log_prob_p_z_given_c = p_z_given_c.log_prob(
            re_pad
        )  #! shape [mc_sample, batch_size, n_clsuters]
        log_prob_p_c = torch.log(self.gmm_prior.pi_p_c)  #! shape (n_clsuters)
        prob_p_z = torch.exp(log_prob_p_z_given_c + log_prob_p_c.unsqueeze(0).unsqueeze(0)).sum(
            dim=2
        )  # sum over clsuter dim -> [mc_samples, batch_size]
        log_prob_p_z = torch.log(prob_p_z.clamp(min=1e-8))
        log_prob_q_re_given_yxw = q_re_given_yxw.log_prob(re)  # [mc_samples, batch_size]
        kld_z = torch.mean(
            log_prob_q_re_given_yxw - log_prob_p_z, dim=0
        )  # mean over mc sample -> (batch_size)

        kld_q_re_given_yxw_vs_p_z_given_c = torch.mean(
            log_prob_q_re_given_yxw.unsqueeze(-1) - log_prob_p_z_given_c, dim=0
        )  # mean over mc sample -> (batch_size, n_clusters)

        if torch.isnan(kld_q_re_given_yxw_vs_p_z_given_c).any():
            print("The kld_q_re_given_yxw_vs_p_z_given_c contains NaN values.")

        if torch.isnan(log_prob_p_c).any():
            print("The log_prob_p_c contains NaN values.")

        if torch.isnan(torch.exp(kld_z)).any():
            print("The torch.exp(kld_z) contains NaN values.")

        logsum = torch.logsumexp(
            -kld_q_re_given_yxw_vs_p_z_given_c + log_prob_p_c.unsqueeze(0), dim=1
        )

        if torch.isnan(logsum).any():
            print("The logsum contains NaN values.")

        kld_c = -kld_z - logsum  # -torch.log(Z_q_re_given_yxw)

        return kld_z, kld_c

    def y_loss(
        self,
        y_dist: torch.distributions.Distribution,
        y: torch.Tensor,
        weights: torch.Tensor,
        use_deviance: bool,
    ) -> torch.Tensor:
        """
        Computes the loss for the target variable `y` based on log-probability or deviance.
        """
        if use_deviance:
            weighted_dev = deviance_loss(y_dist, y, self.y_dist_type) * weights
            return weighted_dev  # torch.cumsum(weighted_dev, dim=1)
        else:
            weighted_log_prob = y_dist.log_prob(y) * weights
            return -weighted_log_prob  # torch.cumsum(weighted_log_prob, dim=1)

    def log_metrics(
        self,
        stage: str,
        loss: torch.Tensor,
        kld_z: torch.Tensor,
        kld_c: torch.Tensor,
        loss_y: torch.Tensor,
        loss_1mm: torch.Tensor,
    ):
        """
        Logs training metrics in a dictionary format to avoid redundant log lines.

        The reason for using `loss_y` instead of `loss` for the validation stage (`val`) is to prioritize tracking
        the main target variable loss during validation. Since `loss_y` represents the predictive performance
        on the target outcome, it is often more indicative of model performance compared to the total loss,
        which may include additional regularization terms.

        Args:
            stage (str): The stage of training (e.g., 'train', 'val', 'test').
            metrics_dict (dict): A dictionary containing metric names as keys and their corresponding values.
            early_stopping_metric (bool): Whether to log the early stopping metric specifically. self.log('val_y_scale', self.y_scale.item(), prog_bar=True, on_epoch=True)
        """
        metrics = {
            f"{stage}/loss": loss_y if stage == "val" else loss,
            f"{stage}/y_scale": self.y_scale.item(),
            f"{stage}/kld_z": kld_z,
            f"{stage}/kld_c": kld_c,
            f"{stage}/loss_1mm": loss_1mm,
        }
        if stage != "val":
            metrics[f"{stage}/loss_y"] = loss_y

        for key, value in metrics.items():
            self.log(
                key,
                value,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                prog_bar=key.endswith("_loss"),
            )
        self.log("alpha", self.br_treatment_outcome_head.alpha, on_epoch=True, on_step=False)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterizes to obtain a sample from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def reg_matching(self, RE_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Computes a regularization loss for matching hidden states.
        """
        mu_hidden_states = self.inference_re_given_yxw.fc_mu(RE_hidden_states)
        log_var_hidden_states = self.inference_re_given_yxw.fc_var(RE_hidden_states)
        std_hidden_states = torch.exp(0.5 * log_var_hidden_states)

        dist_matching = self.wasserstein_distance_gauss(
            mu_1=mu_hidden_states[:, self.min_timestep + 1 :, :],
            std_1=std_hidden_states[:, self.min_timestep + 1 :, :],
            mu_2=mu_hidden_states[:, self.min_timestep : -1, :],
            std_2=std_hidden_states[:, self.min_timestep : -1, :],
        )

        return dist_matching

    @staticmethod
    def wasserstein_distance_gauss(mu_1, std_1, mu_2, std_2):
        r"""
        Wasserstein distance betwen two Gaussiance with diagonal covarance matrirx.
        We use the formula where the covariance matrices commute i.e. $\Sigma_1\Sigma_2=\Sigma_2\Sigma_1$.
        The formula is:

        $W_2(N(m_1,\Sigma_1);N(m_2,\Sigma_2))2=∥m_1-m_2∥_2^2+∥\Sigma^{1/2}_1-\Sigma^{1/2}_2∥^2_{F}$

        Reference: https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/#eqWG
        """

        wass = torch.norm(mu_2 - mu_1, dim=-1) + torch.norm(std_2 - std_1, dim=-1)
        wass = wass.reshape(wass.shape[0], -1)

        return wass

    def weighting(self, w: torch.Tensor, p_w_x: torch.Tensor) -> torch.Tensor:
        """
        Computes weighting for the given treatment and predicted treatment probabilities.
        """
        methods = {
            "IPTW": lambda w, p_w_x: w / p_w_x + (1 - w) / (1 - p_w_x),
            "context_aware": lambda w, p_w_x: (
                torch.mean(w, dim=0).view(1, -1).repeat((w.shape[0], 1))
                / (1 - torch.mean(w, dim=0)).view(1, -1).repeat((w.shape[0], 1))
            )
            * ((1 - p_w_x) / p_w_x)
            + 1,
            "overlap": lambda w, p_w_x: p_w_x * (1 - p_w_x) / (w * p_w_x + (1 - w) * (1 - p_w_x)),
            "none": lambda w, p_w_x: torch.ones(w.shape, dtype=w.dtype).to(self.device),
        }
        return methods[self.weighting_method](w, p_w_x)

    def _get_optimizer(self, param_optimizer: list, head_type: str):
        no_decay = ["bias", "layer_norm"]
        sub_args = self.hparams.model[self.model_type]
        if head_type == "non_treatment_head":

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay) 
                    ],
                    "weight_decay": sub_args["optimizer"][head_type]["weight_decay"],
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

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f"Autoregressive Prediction for {dataset.subset_name}.")

        predicted_outputs = np.zeros(
            (len(dataset), self.hparams.dataset.projection_horizon, self.dim_outcome)
        )
        for t in range(self.hparams.dataset.projection_horizon):
            logger.info(f"t = {t + 2}")

            outputs_scaled = self.get_predictions(dataset)
            predicted_outputs[:, t] = outputs_scaled[:, t]

            if t < (self.hparams.dataset.projection_horizon - 1):
                dataset.data["prev_outputs"][:, t + 1, :] = outputs_scaled[:, t, :]

        return predicted_outputs

    def get_clusters_RE(self, data_loader):

        all_re_labels = []

        for batch in data_loader:
            re_labels = batch["re_labels"]
            all_re_labels.append(re_labels)

        all_re_labels = torch.cat(all_re_labels, dim=0)

        return all_re_labels.numpy()

    def get_mu_RE(self, data) -> np.array:

        # If input is a dataset, wrap it in a DataLoader
        if not isinstance(data, DataLoader):
            data_loader = DataLoader(
                data, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
            )
        else:
            data_loader = data

        # update flag
        self.with_mu_RE = True
        predictions = self.trainer.predict(self, data_loader)
        _, _, mu_RE = (torch.cat(arrs) for arrs in zip(*predictions))

        # Turn off the flag
        self.with_mu_RE = False
        return mu_RE.numpy()

    def get_predictive_check_p_values(self, data_loader):
        self.eval()

        p_values_mean = np.zeros(self.seq_len + 1)
        self.mc_sample_size = 4 * self.mc_sample_size

        batch_init = next(iter(data_loader))
        data_keys = batch_init.keys()
        data = {
            key: torch.concat([batch[key] for batch in data_loader], dim=0) for key in data_keys
        }
        x, x_posterior = self.build_input(data)
        br = self.build_br(x)

        inference_outputs = self.inference_re_given_yxw(x_posterior, return_dict=True)
        self.latest_inference_outputs = inference_outputs
        mu_RE = inference_outputs["compat"]["re_loc"]
        log_var_RE = inference_outputs["compat"]["re_logvar"]
        if torch.isnan(mu_RE).any():
            print("The mu_RE contains NaN values.")
        if torch.isnan(log_var_RE).any():
            print("The log_var_RE contains NaN values.")
        mu_RE[torch.isnan(mu_RE)] = 0.0
        log_var_RE[torch.isnan(log_var_RE)] = 0.0

        q_re_given_yxw = D.Independent(
            D.Normal(loc=mu_RE, scale=torch.exp(0.5 * log_var_RE)),
            reinterpreted_batch_ndims=1,  # interpret dim 1 (d_re diemion as a single event)
        )  # ? Do we need a permutation ?

        curr_treatments = data["current_treatments"]
        y_observed = data["outputs"]
        re = q_re_given_yxw.sample()

        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detached=False)
        p_w_x = self.treat_normalizer(treatment_pred)

        weights = self.weighting(curr_treatments, p_w_x)
        weights = weights.unsqueeze(0).repeat(self.mc_sample_size, 1, 1, 1)
        re_extended = re.unsqueeze(1).repeat(1, br.shape[1], 1)

        y_observed = y_observed.unsqueeze(0).repeat(self.mc_sample_size, 1, 1, 1)

        outcome_pred = self.br_treatment_outcome_head.build_outcome(
            br, re_extended, curr_treatments, y_dist_type=self.y_dist_type
        )

        y_dist = self._get_y_dist(outcome_pred)
        y_replicated = y_dist.sample((self.mc_sample_size,))

        y_mean = y_dist.mean.unsqueeze(0).repeat(self.mc_sample_size, 1, 1, 1)
        statistic_observed = (y_observed - y_mean) ** 2
        statistic_replicated = (y_replicated - y_mean) ** 2

        p_values = (statistic_replicated > statistic_observed).double()
        p_values = p_values.mean(dim=[0, 1]).numpy()

        return p_values.reshape(-1)
