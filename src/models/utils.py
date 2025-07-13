from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
from torch.autograd import Function
from torch.nn.utils import spectral_norm, weight_norm


def grad_reverse(x, scale=1.0):

    class ReverseGrad(Function):
        """
        Gradient reversal layer
        """

        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad_output):
            return scale * grad_output.neg()

    return ReverseGrad.apply(x)


class FilteringMlFlowLogger(MLFlowLogger):
    def __init__(self, filter_submodels: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.filter_submodels = filter_submodels

    @rank_zero_only
    def log_hyperparams(self, params) -> None:
        params = deepcopy(params)
        [
            params.model.pop(filter_submodel)
            for filter_submodel in self.filter_submodels
            if filter_submodel in params.model
        ]
        super().log_hyperparams(params)


def bce(treatment_pred, current_treatments, mode, weights=None, label_smoothing=0):
    if mode == "multiclass":
        return F.cross_entropy(
            treatment_pred.permute(0, 2, 1),
            current_treatments.permute(0, 2, 1),
            reduce=False,
            weight=weights,
            label_smoothing=label_smoothing,
        )
    elif mode == "multilabel":
        return F.binary_cross_entropy_with_logits(
            treatment_pred, current_treatments, reduce=False, weight=weights
        ).mean(dim=-1)
    else:
        raise NotImplementedError()


class InstanceNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        else:
            return x


class BRTreatmentOutcomeHead(nn.Module):
    def __init__(
        self,
        seq_hidden_units,
        br_size,
        fc_hidden_units,
        dim_treatments,
        dim_outcome,
        dim_static_feat=0,
        alpha=0.0,
        update_alpha=True,
        balancing="grad_reverse",
        use_spectral_norm=False,
        activation="elu",
    ):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.br_size = br_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome
        self.alpha = alpha if not update_alpha else 0.0
        self.alpha_max = alpha
        self.balancing = balancing
        self.activation = activation

        self.linear1 = nn.Linear(self.seq_hidden_units, self.br_size)
        self.activation1 = self.get_activation(activation)

        self.linear2 = nn.Linear(self.br_size, self.fc_hidden_units)
        self.activation2 = self.get_activation(activation)
        self.linear3 = nn.Linear(self.fc_hidden_units, self.dim_treatments)

        if dim_static_feat > 0:
            self.static_features_transform = nn.Sequential(
                weight_norm(nn.Linear(dim_static_feat, dim_static_feat // 2)),
                self.get_activation(activation),
            )

        self.linear4 = nn.Linear(
            self.br_size + dim_static_feat // 2 + self.dim_treatments, self.fc_hidden_units
        )
        self.activation3 = self.get_activation(activation)
        self.linear5 = nn.Linear(self.fc_hidden_units, self.dim_outcome)

        if use_spectral_norm:
            # spectral_norm  applied to treatment subnetowrks for CPC
            self.linear2 = spectral_norm(self.linear2)
            self.linear3 = spectral_norm(self.linear3)

        self.treatment_head_params = ["linear2", "linear3"]

    def get_activation(self, activation):
        if activation == "elu":
            return nn.ELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        elif activation == "selu":
            return nn.SELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def build_treatment(self, br, detached=False):
        if detached:
            br = br.detach()

        if self.balancing == "grad_reverse":
            br = grad_reverse(br, self.alpha)

        br = self.activation2(self.linear2(br))
        treatment = self.linear3(br)  # Softmax is encapsulated into F.cross_entropy()
        return treatment

    def build_outcome(self, br, current_treatment, static_features=None):
        if static_features is not None:
            static_features = self.static_features_transform(static_features)
            br = torch.cat((br, static_features.unsqueeze(1).expand(-1, br.size(1), -1)), dim=-1)
        x = torch.cat((br, current_treatment), dim=-1)
        x = self.activation3(self.linear4(x))
        outcome = self.linear5(x)
        return outcome

    def build_br(self, seq_output):
        if self.activation == "elu":
            br = self.activation1(self.linear1(seq_output))
        else:
            br = nn.SELU()(self.linear1(seq_output))
        return br


class WRTreatmentOutcomeHead(nn.Module):
    def __init__(
        self,
        context_latent_dim,
        br_size,
        fc_hidden_units,
        z_latent_dim,
        dim_treatments,
        dim_outcome,
        alpha=0.0,
        update_alpha=False,
        activation="leaky_relu",
    ):
        super().__init__()

        self.context_latent_dim = context_latent_dim
        self.br_size = br_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome
        self.alpha = alpha if not update_alpha else 0.0
        self.alpha_max = alpha
        self.activation = activation
        self.br_normalizer = nn.Tanh()

        self.linear1 = nn.Linear(self.context_latent_dim, self.br_size)
        self.activation1 = self.get_activation(activation)

        self.linear2 = nn.Linear(self.br_size, self.fc_hidden_units)
        self.activation2 = self.get_activation(activation)
        self.linear3 = nn.Linear(self.fc_hidden_units, self.dim_treatments)

        dim_input_outc_model = z_latent_dim + br_size
        dim_hidden_outc_model = (dim_input_outc_model) // 2
        self.learn_mean_1 = nn.Sequential(
            weight_norm(nn.Linear(dim_input_outc_model, dim_hidden_outc_model)),
            self.get_activation(activation),
            weight_norm(nn.Linear(dim_hidden_outc_model, dim_outcome)),
        )

        self.learn_mean_0 = nn.Sequential(
            weight_norm(nn.Linear(dim_input_outc_model, dim_hidden_outc_model)),
            self.get_activation(activation),
            weight_norm(nn.Linear(dim_hidden_outc_model, dim_outcome)),
        )
        self.outcome_head_params = ["learn_mean_1", "learn_mean_0"]

        self.treatment_head_params = ["linear2", "linear3"]

    def get_activation(self, activation):
        if activation == "elu":
            return nn.ELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        elif activation == "selu":
            return nn.SELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def build_treatment(self, br, detached=False):
        if detached:
            br = br.detach()

        br = self.activation2(self.linear2(br))
        treatment = self.linear3(br)  # Softmax is encapsulated into F.cross_entropy()
        return treatment

    def build_outcome(self, br, re, current_treatment, y_dist_type="continuous", detached=False):
        if detached:
            br = br.detach()
        z_re_x = torch.cat((br, re), dim=-1)
        mean_0 = self.learn_mean_0(z_re_x)
        mean_1 = self.learn_mean_1(z_re_x)
        if y_dist_type == "discrete":
            # Ensure positivity of Poisson parameter
            mean_0 = torch.exp(mean_0)
            mean_1 = torch.exp(mean_1)

        outcome = mean_1 * current_treatment + mean_0 * (1 - current_treatment)
        return outcome

    def build_br(self, seq_output):

        br = self.br_normalizer(self.linear1(seq_output))

        return br


class ROutcomeVitalsHead(nn.Module):
    """Used by G-Net"""

    def __init__(
        self,
        seq_hidden_units,
        r_size,
        fc_hidden_units,
        dim_outcome,
        dim_vitals,
        dim_static_features,
        num_comp,
        comp_sizes,
    ):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.r_size = r_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_outcome = dim_outcome
        self.dim_vitals = dim_vitals
        self.num_comp = num_comp
        self.comp_sizes = comp_sizes

        self.linear1 = nn.Linear(self.seq_hidden_units, self.r_size)
        self.elu1 = nn.ELU()

        # Conditional distribution networks init
        self.cond_nets = []

        self.static_features_transform = nn.Linear(
            dim_static_features, dim_static_features // 2, nn.ELU()
        )
        self.rep_transform = nn.Sequential(nn.Linear(self.r_size, self.r_size), nn.ELU())
        self.outcome_head = nn.Sequential(
            nn.Linear(
                self.r_size + dim_static_features // 2, self.r_size // 2 + dim_static_features // 4
            ),
            nn.ELU(),
            nn.Linear(self.r_size // 2 + dim_static_features // 4, self.dim_outcome),
        )

        self.vitals_head = nn.Sequential(
            nn.Linear(self.r_size, self.fc_hidden_units),
            nn.ELU(),
            nn.Linear(self.fc_hidden_units, self.dim_vitals),
        )

    def build_r(self, seq_output):
        r = self.elu1(self.linear1(seq_output))
        return r

    def build_outcome_vitals(self, r, static_features=None):
        vitals_pred = self.vitals_head(r)

        if static_features is not None:
            static_features = self.static_features_transform(static_features)
            r = self.rep_transform(r)
            r = torch.cat((r, static_features.unsqueeze(1).expand(-1, r.size(1), -1)), dim=-1)
        outcome_pred = self.outcome_head(r)

        vitals_outcome_pred = torch.cat((outcome_pred, vitals_pred), dim=-1)  # convention order

        return vitals_outcome_pred


class AlphaRise(Callback):
    """
    Exponential alpha rise
    """

    def __init__(self, rate="exp"):
        self.rate = rate

    def on_epoch_end(self, trainer, pl_module) -> None:
        if pl_module.hparams.exp.update_alpha:
            assert hasattr(pl_module, "br_treatment_outcome_head")
            p = float(pl_module.current_epoch + 1) / float(trainer.max_epochs)
            if self.rate == "lin":
                pl_module.br_treatment_outcome_head.alpha = (
                    p * pl_module.br_treatment_outcome_head.alpha_max
                )
            elif self.rate == "exp":
                pl_module.br_treatment_outcome_head.alpha = (
                    2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
                ) * pl_module.br_treatment_outcome_head.alpha_max
            else:
                raise NotImplementedError()


def clip_normalize_stabilized_weights(stabilized_weights, active_entries, multiple_horizons=False):
    """
    Used by RMSNs
    """
    active_entries = active_entries.astype(bool)
    stabilized_weights[~np.squeeze(active_entries)] = np.nan
    sw_tilde = np.clip(
        stabilized_weights,
        np.nanquantile(stabilized_weights, 0.01),
        np.nanquantile(stabilized_weights, 0.99),
    )
    if multiple_horizons:
        sw_tilde = sw_tilde / np.nanmean(sw_tilde, axis=0, keepdims=True)
    else:
        sw_tilde = sw_tilde / np.nanmean(sw_tilde)

    sw_tilde[~np.squeeze(active_entries)] = 0.0
    return sw_tilde
