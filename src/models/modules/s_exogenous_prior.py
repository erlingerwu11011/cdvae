from __future__ import annotations

import torch
import torch.nn as nn


class SExogenousPrior(nn.Module):
    """Modular exogenous prior for fault latent s.

    Supported:
    - gaussian: regime-conditioned diagonal Gaussian + unknown fallback
    - gmm: lightweight regime-conditioned mixture (returns moment-matched mu/logvar)
    """

    def __init__(
        self,
        prior_type: str = "gaussian",
        num_regimes: int = None,
        latent_dim: int = None,
        n_components: int = 4,
    ):
        super().__init__()
        if latent_dim is None:
            raise ValueError("`latent_dim` must be provided for SExogenousPrior.")
        self.prior_type = prior_type
        self.num_regimes = num_regimes
        self.latent_dim = latent_dim
        self.n_components = n_components

        if self.prior_type == "gaussian":
            self.mu_embedding = nn.Embedding(num_regimes, latent_dim)
            self.logvar_embedding = nn.Embedding(num_regimes, latent_dim)
            self.mu_unknown = nn.Parameter(torch.zeros(latent_dim))
            self.logvar_unknown = nn.Parameter(torch.zeros(latent_dim))
            nn.init.zeros_(self.mu_embedding.weight)
            nn.init.zeros_(self.logvar_embedding.weight)
        elif self.prior_type == "gmm":
            self.logits_embedding = nn.Embedding(num_regimes, n_components)
            self.mu_embedding = nn.Embedding(num_regimes, n_components * latent_dim)
            self.logvar_embedding = nn.Embedding(num_regimes, n_components * latent_dim)
            self.logits_unknown = nn.Parameter(torch.zeros(n_components))
            self.mu_unknown = nn.Parameter(torch.zeros(n_components, latent_dim))
            self.logvar_unknown = nn.Parameter(torch.zeros(n_components, latent_dim))
            nn.init.zeros_(self.logits_embedding.weight)
            nn.init.zeros_(self.mu_embedding.weight)
            nn.init.zeros_(self.logvar_embedding.weight)
        else:
            raise ValueError(f"Unsupported s prior type: {self.prior_type}")

    def forward(
        self,
        regime_id: torch.Tensor = None,
        regime_seen_mask: torch.Tensor = None,
        seq_len: int = None,
    ):
        if regime_id is None:
            batch_size = 1 if regime_seen_mask is None else regime_seen_mask.shape[0]
            regime_id = torch.zeros(batch_size, dtype=torch.long, device=self.mu_unknown.device)
        if regime_seen_mask is None:
            regime_seen_mask = torch.ones_like(regime_id, dtype=torch.bool)

        if self.prior_type == "gaussian":
            regime_id_safe = regime_id.clamp(min=0, max=self.num_regimes - 1)
            mu_seen = self.mu_embedding(regime_id_safe)
            logvar_seen = self.logvar_embedding(regime_id_safe)
            mu_unknown = self.mu_unknown.unsqueeze(0).expand_as(mu_seen)
            logvar_unknown = self.logvar_unknown.unsqueeze(0).expand_as(logvar_seen)
            mask = regime_seen_mask.unsqueeze(-1).bool()
            mu = torch.where(mask, mu_seen, mu_unknown)
            logvar = torch.where(mask, logvar_seen, logvar_unknown)
        else:  # gmm
            regime_id_safe = regime_id.clamp(min=0, max=self.num_regimes - 1)
            logits_seen = self.logits_embedding(regime_id_safe)
            mu_seen = self.mu_embedding(regime_id_safe).view(-1, self.n_components, self.latent_dim)
            logvar_seen = self.logvar_embedding(regime_id_safe).view(
                -1, self.n_components, self.latent_dim
            )

            logits_unknown = self.logits_unknown.unsqueeze(0).expand_as(logits_seen)
            mu_unknown = self.mu_unknown.unsqueeze(0).expand_as(mu_seen)
            logvar_unknown = self.logvar_unknown.unsqueeze(0).expand_as(logvar_seen)

            comp_mask = regime_seen_mask[:, None, None].bool()
            logits = torch.where(regime_seen_mask[:, None].bool(), logits_seen, logits_unknown)
            mu_components = torch.where(comp_mask, mu_seen, mu_unknown)
            logvar_components = torch.where(comp_mask, logvar_seen, logvar_unknown)

            weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
            mu = (weights * mu_components).sum(dim=1)
            second_moment = (weights * (torch.exp(logvar_components) + mu_components**2)).sum(dim=1)
            var = (second_moment - mu**2).clamp(min=1e-6)
            logvar = torch.log(var)

        if seq_len is not None:
            mu = mu.unsqueeze(1).expand(-1, seq_len, -1)
            logvar = logvar.unsqueeze(1).expand(-1, seq_len, -1)
        return mu, logvar
