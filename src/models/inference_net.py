import torch
import torch.nn as nn
import torch.distributions as D


class Inference_Net(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        z_latent_dim: int,
        num_layer: int,
        dropout_rate: float,
        activation: int = "elu",
        background_latent_dim: int = None,
        health_latent_dim: int = None,
        fault_latent_dim: int = None,
        num_regimes: int = None,
        regime_embed_dim: int = 0,
    ):
        super().__init__()
        self.activation = activation
        self.z_latent_dim = z_latent_dim
        self.background_latent_dim = background_latent_dim or z_latent_dim
        self.health_latent_dim = health_latent_dim or hidden_size
        self.fault_latent_dim = fault_latent_dim or hidden_size
        self.num_regimes = num_regimes
        self.regime_embed_dim = regime_embed_dim or 0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,
            dropout=dropout_rate,
        )

        # Legacy single-latent outputs used by existing CDVAE training.
        self.fc_mu = nn.Linear(hidden_size, z_latent_dim)
        self.fc_var = nn.Linear(hidden_size, z_latent_dim)

        # Multi-head latents for industrial native-fault adaptation.
        self.background_mu = nn.Linear(hidden_size, self.background_latent_dim)
        self.background_logvar = nn.Linear(hidden_size, self.background_latent_dim)

        self.health_mu = nn.Linear(hidden_size, self.health_latent_dim)
        self.health_logvar = nn.Linear(hidden_size, self.health_latent_dim)

        self.fault_mu = nn.Linear(hidden_size, self.fault_latent_dim)
        self.fault_logvar = nn.Linear(hidden_size, self.fault_latent_dim)
        self.regime_embedding = None
        self.regime_projection = None
        if self.num_regimes is not None and self.regime_embed_dim > 0:
            self.regime_embedding = nn.Embedding(self.num_regimes, self.regime_embed_dim)
            self.regime_projection = nn.Linear(self.regime_embed_dim, hidden_size)

    def _encode(self, input: torch.Tensor):
        output, _ = self.gru(input)
        return output

    @staticmethod
    def _reparameterize(loc: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return loc + torch.randn_like(std) * std

    def posterior_dict(
        self,
        input: torch.Tensor,
        regime_context: torch.Tensor = None,
        sample: bool = True,
    ):
        """Shared encoder + disentangled heads with CDVAE-compatible fields."""
        enc_feat = self._encode(input)
        tail_feat = enc_feat[:, -1, :]
        s_feat = enc_feat
        if regime_context is not None and self.regime_embedding is not None:
            if regime_context.dtype in (torch.int32, torch.int64, torch.long):
                regime_emb = self.regime_embedding(regime_context)
            else:
                regime_emb = regime_context

            if regime_emb.dim() == 2:
                regime_emb = regime_emb.unsqueeze(1).expand(-1, enc_feat.shape[1], -1)
            regime_hidden = self.regime_projection(regime_emb)
            s_feat = enc_feat + regime_hidden

        b_loc = self.background_mu(tail_feat)
        b_logvar = self.background_logvar(tail_feat)
        # h0 is sequence-level healthy initial state (not time-varying).
        h0_loc = self.health_mu(tail_feat)
        h0_logvar = self.health_logvar(tail_feat)
        s_loc = self.fault_mu(s_feat)
        s_logvar = self.fault_logvar(s_feat)

        out = {
            "posterior_params": {
                "b_loc": b_loc,
                "b_logvar": b_logvar,
                "h0_loc": h0_loc,
                "h0_logvar": h0_logvar,
                "s_loc": s_loc,
                "s_logvar": s_logvar,
            },
            "q_b": D.Independent(D.Normal(b_loc, torch.exp(0.5 * b_logvar)), 1),
            "q_h0": D.Independent(D.Normal(h0_loc, torch.exp(0.5 * h0_logvar)), 1),
            "q_s": D.Independent(D.Normal(s_loc, torch.exp(0.5 * s_logvar)), 1),
            "enc_feat": enc_feat,
            "legacy": {
                "loc": self.fc_mu(tail_feat),
                "logvar": self.fc_var(tail_feat),
            },
            "compat": {
                "re_loc": self.fc_mu(tail_feat),
                "re_logvar": self.fc_var(tail_feat),
            },
        }
        out["compat"]["re_sample"] = self._reparameterize(
            out["compat"]["re_loc"], out["compat"]["re_logvar"]
        )
        if sample:
            out["samples"] = {
                "b": self._reparameterize(b_loc, b_logvar),
                "h0": self._reparameterize(h0_loc, h0_logvar),
                "s": self._reparameterize(s_loc, s_logvar),
            }
        return out

    def forward(
        self,
        input: torch.Tensor,
        return_dict: bool = False,
        regime_context: torch.Tensor = None,
    ) -> torch.Tensor:
        posterior = self.posterior_dict(input, regime_context=regime_context, sample=True)
        if return_dict:
            return posterior
        return posterior["legacy"]["loc"], posterior["legacy"]["logvar"], posterior["enc_feat"]

    def forward_multi_head(self, input: torch.Tensor, regime_context: torch.Tensor = None):
        """Backward-compatible alias for the three-head posterior params."""
        posterior = self.posterior_dict(input, regime_context=regime_context, sample=False)
        params = posterior["posterior_params"]
        return {
            "b_loc": params["b_loc"],
            "b_logvar": params["b_logvar"],
            "h0_loc": params["h0_loc"],
            "h0_logvar": params["h0_logvar"],
            "s_loc": params["s_loc"],
            "s_logvar": params["s_logvar"],
            "enc_feat": posterior["enc_feat"],
        }

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
