from typing import Dict, Optional

import torch
import torch.nn as nn


class Inference_Net(nn.Module):
    """Shared temporal encoder with disentangled posterior heads.

    Heads and semantics:
    - b_*: episode-level background latent posterior
    - h0_*: time-varying healthy/shared-base latent posterior
    - s_*: time-varying fault/intervention latent posterior

    Compatibility mapping:
    - re_* aliases to b_* so existing CDVAE code path can keep working.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        z_latent_dim: int,
        num_layer: int,
        dropout_rate: float,
        activation: str = "elu",
        b_latent_dim: Optional[int] = None,
        h0_latent_dim: Optional[int] = None,
        s_latent_dim: Optional[int] = None,
        num_regimes: Optional[int] = None,
        regime_embed_dim: int = 0,
    ):
        super().__init__()
        self.activation = activation

        self.b_latent_dim = b_latent_dim or z_latent_dim
        self.h0_latent_dim = h0_latent_dim or z_latent_dim
        self.s_latent_dim = s_latent_dim or z_latent_dim

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.fc_b_mu = nn.Linear(hidden_size, self.b_latent_dim)
        self.fc_b_var = nn.Linear(hidden_size, self.b_latent_dim)

        self.fc_h0_mu = nn.Linear(hidden_size, self.h0_latent_dim)
        self.fc_h0_var = nn.Linear(hidden_size, self.h0_latent_dim)

        self.regime_embed = None
        self.regime_embed_dim = 0
        if num_regimes is not None and regime_embed_dim > 0:
            self.regime_embed = nn.Embedding(num_regimes, regime_embed_dim)
            self.regime_embed_dim = regime_embed_dim

        s_head_input_dim = hidden_size + self.regime_embed_dim
        self.fc_s_mu = nn.Linear(s_head_input_dim, self.s_latent_dim)
        self.fc_s_var = nn.Linear(s_head_input_dim, self.s_latent_dim)

        # Backward-compatible aliases used elsewhere in cdvae.py
        self.fc_mu = self.fc_b_mu
        self.fc_var = self.fc_b_var

    @staticmethod
    def _reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(
        self,
        input: torch.Tensor,
        regime_id: Optional[torch.Tensor] = None,
        return_legacy: bool = False,
    ):
        enc_feat, _ = self.gru(input)

        last_feat = enc_feat[:, -1, :]
        b_loc = self.fc_b_mu(last_feat)
        b_logvar = self.fc_b_var(last_feat)

        h0_loc = self.fc_h0_mu(enc_feat)
        h0_logvar = self.fc_h0_var(enc_feat)

        if self.regime_embed is not None and regime_id is not None:
            if regime_id.ndim > 1:
                regime_id = regime_id.squeeze(-1)
            regime_feat = self.regime_embed(regime_id.long())
            regime_feat = regime_feat.unsqueeze(1).expand(-1, enc_feat.size(1), -1)
            s_input = torch.cat([enc_feat, regime_feat], dim=-1)
        else:
            s_input = enc_feat

        s_loc = self.fc_s_mu(s_input)
        s_logvar = self.fc_s_var(s_input)

        b = self._reparameterize(b_loc, b_logvar)
        h0 = self._reparameterize(h0_loc, h0_logvar)
        s = self._reparameterize(s_loc, s_logvar)

        output: Dict[str, torch.Tensor] = {
            "q_b": {"loc": b_loc, "logvar": b_logvar},
            "q_h0": {"loc": h0_loc, "logvar": h0_logvar},
            "q_s": {"loc": s_loc, "logvar": s_logvar},
            "b": b,
            "h0": h0,
            "s": s,
            "enc_feat": enc_feat,
            # compatibility mapping for current CDVAE path
            "re_loc": b_loc,
            "re_logvar": b_logvar,
            "re_sample": b,
        }

        if return_legacy:
            return output["re_loc"], output["re_logvar"], output["enc_feat"]

        return output

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
