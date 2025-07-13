
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.utils import spectral_norm, weight_norm


class ICLUB(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self, treatment_pred: torch.Tensor, current_treatments: torch.Tensor, mode: str
    ) -> torch.Tensor:
        sample_size = current_treatments.shape[0]
        permutation = torch.randperm(sample_size)
        shuffled_treatments = current_treatments[permutation]
        if mode == "multiclass":
            positive = -F.cross_entropy(
                treatment_pred.permute(0, 2, 1),
                current_treatments.permute(0, 2, 1),
                reduce=False,
            )
            negative = -F.cross_entropy(
                treatment_pred.permute(0, 2, 1),
                shuffled_treatments.permute(0, 2, 1),
                reduce=False,
            )

            return positive - negative

        elif mode == "multilabel":
            positive = -F.binary_cross_entropy_with_logits(
                treatment_pred,
                current_treatments,
                reduce=False,
            )
            negative = -F.binary_cross_entropy_with_logits(
                treatment_pred,
                shuffled_treatments,
                reduce=False,
            )

            return (positive - negative).mean(dim=-1)

        else:
            raise NotImplementedError()


class CPC(nn.Module):
    def __init__(
        self,
        input_size: int,
        genc_hidden: int,
        context_latent_dim,
        num_layers,
        dropout_rate,
        prediction_steps,
        downsampling_factor,
        use_attention,
        activation="elu",
        rnn_type="gru",
        weighting="uniform",
    ):

        super().__init__()
        self.ts = prediction_steps
        self.downsampling_factor = downsampling_factor
        self.genc_hidden = genc_hidden
        self.use_attention = use_attention
        self.activation = activation
        self.rnn_type = rnn_type.lower()
        self.weighting = weighting

        self.encoder = nn.Sequential(
            weight_norm(nn.Linear(input_size, 3 * self.genc_hidden // 2)),
            self.get_activation(),
            weight_norm(nn.Linear(3 * self.genc_hidden // 2, self.genc_hidden)),
        )
        self.map_ct_to_future = nn.Sequential(
            nn.Linear(context_latent_dim, context_latent_dim, bias=False),
        )

        if self.rnn_type == "gru":

            self.g_ar = nn.GRU(
                input_size=self.genc_hidden,
                hidden_size=context_latent_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate,
            )

        elif self.rnn_type == "lstm":

            self.g_ar = nn.LSTM(
                input_size=self.genc_hidden,
                hidden_size=context_latent_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate,
            )

        self.wk = nn.ModuleList(
            [
                nn.Linear(context_latent_dim, self.genc_hidden, bias=False)
                for _ in range(prediction_steps)
            ]
        )

    def get_activation(self) -> nn.Module:
        if self.activation == "elu":
            return nn.ELU()
        elif self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        elif self.activation == "selu":
            return nn.SELU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def forward(self, x: torch.Tensor, active_entries: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]
        batch_size = x.shape[0]
        hindex = int(sequence_length / self.downsampling_factor) - self.ts
        encoder_samples = torch.empty(
            (self.ts, batch_size, self.genc_hidden), device=x.device, dtype=torch.double
        )
        active_entries_samples = torch.empty(
            (self.ts, batch_size, 1), device=x.device, dtype=torch.double
        )
        predictions = torch.empty(
            (self.ts, batch_size, self.genc_hidden), device=x.device, dtype=torch.double
        )

        time_sample = torch.randint(low=1, high=hindex, size=(1,)).long()
        z = self.encoder(x)
        for k in range(1, self.ts + 1):
            encoder_samples[k - 1] = z[:, time_sample + k, :].view(-1, self.genc_hidden)
            active_entries_samples[k - 1] = active_entries[:, time_sample + k, :].view(
                -1, 1
            )  # active_entries for current

        prior_sequence = z[:, : time_sample + 1, :]
        output = self.g_ar(prior_sequence)[0]
        if self.use_attention:
            output = self.attention(output, output)

        ct = output[:, time_sample, :].view(batch_size, -1)
        for k in range(self.ts):
            predictions[k] = self.wk[k](ct)

        time_sample_ = torch.randint(
            low=1, high=time_sample[0].item() + 1, size=(1,)
        ).long()  # +1 to avoid high=low
        past_prior_sequence = prior_sequence[:, :time_sample_, :]
        future_prior_sequence = prior_sequence[:, time_sample_:, :]
        output_past_prior = self.g_ar(past_prior_sequence)[0]
        output_future = self.g_ar(future_prior_sequence)[0]
        c_future = output_future[:, -1, :].view(batch_size, -1)
        c_future_predictions = self.map_ct_to_future(
            output_past_prior[:, -1, :].view(batch_size, -1)
        )

        return encoder_samples, predictions, active_entries_samples, c_future, c_future_predictions

    @staticmethod
    def infomax_infonce(
        c_future: torch.Tensor,
        c_future_predictions: torch.Tensor,
        active_entries_samples: torch.Tensor,
    ) -> torch.Tensor:

        log_softmax = torch.nn.LogSoftmax(dim=-1)
        attention = torch.mm(c_future, torch.transpose(c_future_predictions, 0, 1))
        InfoNCE = torch.sum(
            torch.diag(log_softmax(attention)) * active_entries_samples[0].squeeze(-1)
        )
        InfoNCE /= -1.0 * active_entries_samples[0].sum()
        return InfoNCE

    def infomax_nwj(
        self,
        c_future: torch.Tensor,
        c_future_predictions: torch.Tensor,
        active_entries_samples: torch.Tensor,
    ) -> torch.Tensor:

        attention = torch.mm(c_future, torch.transpose(c_future_predictions, 0, 1))
        attention = attention - 1
        joint_term = (attention.diag() * active_entries_samples[0].squeeze(-1)).mean()
        marg_term = self.logmeanexp_nodiag(attention).exp()
        nwj = 1.0 + joint_term - marg_term
        nwj /= -1.0
        return nwj

    def infomax_mine(
        self,
        c_future: torch.Tensor,
        c_future_predictions: torch.Tensor,
        active_entries_samples: torch.Tensor,
    ) -> torch.Tensor:

        attention = torch.mm(c_future, torch.transpose(c_future_predictions, 0, 1))
        mine = self.mine_lower_bound(attention, active_entries_samples[0])[0]
        mine /= -1.0
        return mine

    def infomax_loss(
        self,
        c_future: torch.Tensor,
        c_future_predictions: torch.Tensor,
        active_entries_samples: torch.Tensor,
        infomax_lb: str,
    ) -> torch.Tensor:

        if infomax_lb.lower() == "infonce":
            loss = self.infomax_infonce(c_future, c_future_predictions, active_entries_samples)
        if infomax_lb.lower() == "nwj":
            loss = self.infomax_nwj(c_future, c_future_predictions, active_entries_samples)
        if infomax_lb.lower() == "mine":
            loss = self.infomax_mine(c_future, c_future_predictions, active_entries_samples)
        return loss

    def loss(
        self,
        encoder_samples: torch.Tensor,
        predictions: torch.Tensor,
        active_entries_samples: torch.Tensor,
        cpc_lb: str,
    ) -> torch.Tensor:

        if cpc_lb.lower() == "infonce":
            loss = self.infonce_loss(encoder_samples, predictions, active_entries_samples)
        elif cpc_lb.lower() == "nwj":
            loss = self.nwj_loss(encoder_samples, predictions, active_entries_samples)
        elif cpc_lb.lower() == "mine":
            loss = self.mine_loss(encoder_samples, predictions, active_entries_samples)
        else:
            raise NotImplementedError()
        return loss

    def get_weights(self):
        if self.weighting.lower() == "uniform":
            weights = [1 for j in range(1, self.ts + 1)]
        elif self.weighting.lower() == "short-term":
            weights = [(self.ts + 1 - j) / (1 + self.ts) for j in range(1, self.ts + 1)]
        else:
            raise NotImplementedError()
        weights = torch.tensor(weights, dtype=torch.double, requires_grad=False)
        return weights

    def infonce_loss(
        self,
        encoder_samples: torch.Tensor,
        predictions: torch.Tensor,
        active_entries_samples: torch.Tensor,
    ) -> torch.Tensor:
        time_steps = encoder_samples.shape[0]
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        InfoNCE = 0
        weights = self.get_weights()
        for i in np.arange(0, time_steps):
            attention = torch.mm(encoder_samples[i], torch.transpose(predictions[i], 0, 1))
            InfoNCE += (
                torch.sum(
                    torch.diag(log_softmax(attention)) * active_entries_samples[i].squeeze(-1)
                )
                * weights[i]
            )
        InfoNCE /= -1.0 * active_entries_samples.sum() * time_steps
        return InfoNCE

    def nwj_loss(
        self,
        encoder_samples: torch.Tensor,
        predictions: torch.Tensor,
        active_entries_samples: torch.Tensor,
    ) -> torch.Tensor:

        time_steps = encoder_samples.shape[0]
        nwj = 0
        for i in np.arange(0, time_steps):
            attention = torch.mm(encoder_samples[i], torch.transpose(predictions[i], 0, 1))
            attention = attention - 1
            joint_term = (attention.diag() * active_entries_samples[i].squeeze(-1)).mean()
            marg_term = self.logmeanexp_nodiag(attention).exp()
            nwj += 1.0 + joint_term - marg_term

        nwj /= -1.0 * time_steps
        return nwj

    def mine_loss(
        self,
        encoder_samples: torch.Tensor,
        predictions: torch.Tensor,
        active_entries_samples: torch.Tensor,
    ) -> torch.Tensor:

        time_steps = encoder_samples.shape[0]
        mine = 0
        for i in np.arange(0, time_steps):
            attention = torch.mm(encoder_samples[i], torch.transpose(predictions[i], 0, 1))
            mine += self.mine_lower_bound(attention, active_entries_samples[i])[0]

        mine /= -1.0 * time_steps
        return mine

    def mine_lower_bound(self, f, active_entries_samples, buffer=None, momentum=0.9):
        """
        MINE lower bound based on DV inequality.
        https://github.com/ermongroup/smile-mi-estimator/blob/master/estimators.py

        """
        if buffer is None:
            buffer = torch.tensor(1.0).to(f.device)
        first_term = (f.diag() * active_entries_samples.squeeze(-1)).mean()

        buffer_update = self.logmeanexp_nodiag(f).exp()
        with torch.no_grad():
            second_term = self.logmeanexp_nodiag(f)
            buffer_new = buffer * momentum + buffer_update * (1 - momentum)
            buffer_new = torch.clamp(buffer_new, min=1e-4)
            third_term_no_grad = buffer_update / buffer_new

        third_term_grad = buffer_update / buffer_new

        return first_term - second_term - third_term_grad + third_term_no_grad, buffer_update

    @staticmethod
    def logmeanexp_nodiag(x, dim=None):
        """
        https://github.com/ermongroup/smile-mi-estimator/blob/master/estimators.py
        """
        batch_size = x.size(0)
        if dim is None:
            dim = (0, 1)
        logsumexp = torch.logsumexp(
            x - torch.diag(np.inf * torch.ones(batch_size).to(x.device)), dim=dim
        )
        try:
            if len(dim) == 1:
                num_elem = batch_size - 1.0
            else:
                num_elem = batch_size * (batch_size - 1.0)
        except ValueError:
            num_elem = batch_size - 1

        return logsumexp - torch.log(torch.tensor(num_elem)).to(x.device)


class decoder(nn.Module):
    def __init__(
        self,
        treat_size,
        treat_hidden_dim,
        dim_outcome,
        seq_hidden_units,
        dim_static_features,
        br_size,
        num_layers_dec,
        rnn_dropout_dec,
        y_dist_type,
        alpha=0.0,
        update_alpha=True,
        use_spectral_norm=False,
        activation="elu",
        use_instance_noise=False,
        likelihood_training=False,
        rnn_type="gru",
    ):
        super().__init__()

        self.y_dist_type = y_dist_type
        self.treat_size = treat_size
        self.num_layers_dec = num_layers_dec
        self.alpha = alpha if not update_alpha else 0.0
        self.alpha_max = alpha
        self.activation = activation
        self.use_instance_noise = use_instance_noise
        self.likelihood_training = likelihood_training
        self.seq_hidden_units = seq_hidden_units
        self.rnn_type = rnn_type.lower()

        self.map_treat_to_continuous = weight_norm(nn.Linear(treat_size, treat_hidden_dim))

        if self.rnn_type == "gru":
            self.gru_treat = nn.GRU(
                input_size=treat_hidden_dim,
                hidden_size=treat_hidden_dim,
                batch_first=True,
                num_layers=1,
                dropout=rnn_dropout_dec,
            )
            self.gru = nn.GRU(
                input_size=treat_hidden_dim + dim_outcome,
                hidden_size=seq_hidden_units,
                batch_first=True,
                num_layers=1,
                dropout=rnn_dropout_dec,
            )

        elif self.rnn_type == "lstm":

            self.gru_treat = nn.LSTM(
                input_size=treat_hidden_dim,
                hidden_size=treat_hidden_dim,
                batch_first=True,
                num_layers=1,
                dropout=rnn_dropout_dec,
            )

            self.gru = nn.LSTM(
                input_size=treat_hidden_dim + dim_outcome,
                hidden_size=seq_hidden_units,
                batch_first=True,
                num_layers=1,
                dropout=rnn_dropout_dec,
            )
        else:
            raise Exception("Only GRU or LSTM for seq modeling!")

        self.linear1 = nn.Linear(br_size, br_size)
        self.linear2 = nn.Linear(br_size, treat_size)
        self.treatment_head_params = ["linear1", "linear2"]

        if use_spectral_norm:
            self.linear2 = spectral_norm(self.linear2)
            self.linear1 = spectral_norm(self.linear1)

        self.treatment_head = nn.Sequential(self.linear1, self.get_activation(), self.linear2)

        if dim_static_features > 0:
            self.static_features_transform = nn.Sequential(
                weight_norm(nn.Linear(dim_static_features, dim_static_features // 2)), nn.SELU()
            )

        self.outcome_head = nn.Sequential(
            weight_norm(nn.Linear(br_size + dim_static_features // 2 + treat_hidden_dim, br_size)),
            self.get_activation(),
            weight_norm(nn.Linear(br_size, dim_outcome)),
        )
        self.build_br = nn.Sequential(weight_norm(nn.Linear(seq_hidden_units, br_size)), nn.SELU())

    def get_activation(self) -> nn.Module:
        if self.activation == "elu":
            return nn.ELU()
        elif self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        elif self.activation == "selu":
            return nn.SELU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def build_treatment(self, br: torch.Tensor, detached: bool = False) -> torch.Tensor:
        if detached:
            br = br.detach()
        if self.use_instance_noise:
            br = self.instance_noise(br)
        p_w = self.treatment_head(br)
        return p_w

    def forward(self, w_init, y_init, c_init, w_intended, static_features):
        projection_horizon = w_intended.shape[1]
        y = y_init.unsqueeze(1)
        y_list = []
        c_list = []
        h_state = c_init
        w_intended_init = torch.cat((w_init.unsqueeze(1), w_intended), dim=1)
        w_intended_init = self.map_treat_to_continuous(w_intended_init)
        w_embed = self.gru_treat(w_intended_init)[0]
        w = w_embed[:, 0, :].unsqueeze(1)
        c_0 = torch.zeros(self.num_layers_dec, h_state.size(0), h_state.size(1)).to(h_state.device)
        if static_features is not None:
            static_features = self.static_features_transform(static_features)

        for t in range(projection_horizon):
            x = torch.cat((w, y), dim=-1)

            if self.rnn_type == "gru":
                h_n = self.gru(x, h_state.unsqueeze(0).repeat(self.num_layers_dec, 1, 1))[0]
            elif self.rnn_type == "lstm":
                h_n = self.gru(x, (h_state.unsqueeze(0).repeat(self.num_layers_dec, 1, 1), c_0))[0]
            h_state = h_n[:, -1, :]

            br = self.build_br(h_state)

            if static_features is not None:
                x_y = torch.cat((br, static_features), dim=-1)
                x_y = torch.cat(
                    (
                        x_y,
                        self.map_treat_to_continuous(w_intended[:, t, :]),
                    ),
                    dim=-1,
                )
            else:
                x_y = torch.cat(
                    (
                        br,
                        self.map_treat_to_continuous(w_intended[:, t, :]),
                    ),
                    dim=-1,
                )

            y_mean = self.outcome_head(x_y).unsqueeze(1)
            y_list.append(y_mean)

            w = w_embed[:, t + 1, :].unsqueeze(1)
            if t < projection_horizon:
                c_list.append(br.unsqueeze(1))

            y = y_mean
            h_state = h_state

        c = torch.cat(c_list, dim=1)
        y_pred = torch.cat(y_list, dim=1)
        if self.likelihood_training:
            y_pred = Normal(loc=y_pred, scale=1)

        return y_pred, c
