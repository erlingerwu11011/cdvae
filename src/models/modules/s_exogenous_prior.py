from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseSExogenousPrior(nn.Module):
    """Unified interface for regime-aware exogenous priors over fault latent s."""

    def forward(
        self,
        regime_id: torch.Tensor = None,
        regime_seen_mask: torch.Tensor = None,
        context: Optional[dict] = None,
        seq_len: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def kl(
        self,
        q_loc: torch.Tensor,
        q_logvar: torch.Tensor,
        regime_id: torch.Tensor = None,
        regime_seen_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        p_loc, p_logvar = self(
            regime_id=regime_id,
            regime_seen_mask=regime_seen_mask,
            seq_len=q_loc.shape[1] if q_loc.dim() == 3 else None,
        )
        var_ratio = torch.exp(q_logvar - p_logvar)
        kl = 0.5 * (
            p_logvar
            - q_logvar
            + var_ratio
            + ((q_loc - p_loc) ** 2) / torch.exp(p_logvar)
            - 1.0
        )
        return kl.sum(dim=-1)


class GaussianSExogenousPrior(BaseSExogenousPrior):
    """Gaussian prior with normal / regime-embedded / unknown fallback modes."""

    def __init__(
        self,
        num_regimes: int,
        latent_dim: int,
        mode: str = "regime_embedded",
    ):
        super().__init__()
        if latent_dim is None:
            raise ValueError("`latent_dim` must be provided.")
        if mode not in {"normal", "regime_embedded"}:
            raise ValueError(f"Unsupported GaussianSExogenousPrior mode: {mode}")

        self.num_regimes = num_regimes
        self.latent_dim = latent_dim
        self.mode = mode

        if mode == "regime_embedded":
            self.mu_embedding = nn.Embedding(num_regimes, latent_dim)
            self.logvar_embedding = nn.Embedding(num_regimes, latent_dim)
            nn.init.zeros_(self.mu_embedding.weight)
            nn.init.zeros_(self.logvar_embedding.weight)
        else:
            self.register_parameter("mu_embedding", None)
            self.register_parameter("logvar_embedding", None)

        self.mu_unknown = nn.Parameter(torch.zeros(latent_dim))
        self.logvar_unknown = nn.Parameter(torch.zeros(latent_dim))

    def _resolve_regime_inputs(
        self, regime_id: torch.Tensor = None, regime_seen_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if regime_id is None:
            batch_size = 1 if regime_seen_mask is None else regime_seen_mask.shape[0]
            device = self.mu_unknown.device
            regime_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        if regime_seen_mask is None:
            regime_seen_mask = torch.ones_like(regime_id, dtype=torch.bool)
        else:
            regime_seen_mask = regime_seen_mask.to(regime_id.device).bool()
        return regime_id, regime_seen_mask

    def forward(
        self,
        regime_id: torch.Tensor = None,
        regime_seen_mask: torch.Tensor = None,
        context: Optional[dict] = None,
        seq_len: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = context
        regime_id, regime_seen_mask = self._resolve_regime_inputs(regime_id, regime_seen_mask)

        if self.mode == "normal":
            mu_seen = torch.zeros(
                regime_id.shape[0], self.latent_dim, device=regime_id.device, dtype=self.mu_unknown.dtype
            )
            logvar_seen = torch.zeros_like(mu_seen)
        else:
            regime_id_safe = regime_id.clamp(min=0, max=self.num_regimes - 1)
            mu_seen = self.mu_embedding(regime_id_safe)
            logvar_seen = self.logvar_embedding(regime_id_safe)

        mu_unknown = self.mu_unknown.unsqueeze(0).expand_as(mu_seen)
        logvar_unknown = self.logvar_unknown.unsqueeze(0).expand_as(logvar_seen)
        mask = regime_seen_mask.unsqueeze(-1)
        mu = torch.where(mask, mu_seen, mu_unknown)
        logvar = torch.where(mask, logvar_seen, logvar_unknown)

        if seq_len is not None:
            mu = mu.unsqueeze(1).expand(-1, seq_len, -1)
            logvar = logvar.unsqueeze(1).expand(-1, seq_len, -1)
        return mu, logvar


class GMMSExogenousPrior(BaseSExogenousPrior):
    """Lightweight GMM prior returning mixture params + moment-matched Gaussian stats."""

    def __init__(self, num_regimes: int, latent_dim: int, num_components: int = 4):
        super().__init__()
        self.num_regimes = num_regimes
        self.latent_dim = latent_dim
        self.num_components = num_components

        self.logits_embedding = nn.Embedding(num_regimes, num_components)
        self.mu_embedding = nn.Embedding(num_regimes, num_components * latent_dim)
        self.logvar_embedding = nn.Embedding(num_regimes, num_components * latent_dim)

        self.logits_unknown = nn.Parameter(torch.zeros(num_components))
        self.mu_unknown = nn.Parameter(torch.zeros(num_components, latent_dim))
        self.logvar_unknown = nn.Parameter(torch.zeros(num_components, latent_dim))

        nn.init.zeros_(self.logits_embedding.weight)
        nn.init.zeros_(self.mu_embedding.weight)
        nn.init.zeros_(self.logvar_embedding.weight)

    def mixture_parameters(
        self,
        regime_id: torch.Tensor,
        regime_seen_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if regime_seen_mask is None:
            regime_seen_mask = torch.ones_like(regime_id, dtype=torch.bool)
        else:
            regime_seen_mask = regime_seen_mask.to(regime_id.device).bool()

        regime_id_safe = regime_id.clamp(min=0, max=self.num_regimes - 1)
        logits_seen = self.logits_embedding(regime_id_safe)
        mu_seen = self.mu_embedding(regime_id_safe).view(-1, self.num_components, self.latent_dim)
        logvar_seen = self.logvar_embedding(regime_id_safe).view(
            -1, self.num_components, self.latent_dim
        )

        logits_unknown = self.logits_unknown.unsqueeze(0).expand_as(logits_seen)
        mu_unknown = self.mu_unknown.unsqueeze(0).expand_as(mu_seen)
        logvar_unknown = self.logvar_unknown.unsqueeze(0).expand_as(logvar_seen)

        logits = torch.where(regime_seen_mask[:, None], logits_seen, logits_unknown)
        comp_mask = regime_seen_mask[:, None, None]
        mu_components = torch.where(comp_mask, mu_seen, mu_unknown)
        logvar_components = torch.where(comp_mask, logvar_seen, logvar_unknown)
        return logits, mu_components, logvar_components

    def forward(
        self,
        regime_id: torch.Tensor = None,
        regime_seen_mask: torch.Tensor = None,
        context: Optional[dict] = None,
        seq_len: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = context
        if regime_id is None:
            batch_size = 1 if regime_seen_mask is None else regime_seen_mask.shape[0]
            regime_id = torch.zeros(batch_size, dtype=torch.long, device=self.logits_unknown.device)
        logits, mu_components, logvar_components = self.mixture_parameters(
            regime_id=regime_id,
            regime_seen_mask=regime_seen_mask,
        )
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        mu = (weights * mu_components).sum(dim=1)
        second_moment = (weights * (torch.exp(logvar_components) + mu_components**2)).sum(dim=1)
        var = (second_moment - mu**2).clamp(min=1e-6)
        logvar = torch.log(var)

        if seq_len is not None:
            mu = mu.unsqueeze(1).expand(-1, seq_len, -1)
            logvar = logvar.unsqueeze(1).expand(-1, seq_len, -1)
        return mu, logvar


class SExogenousPrior(BaseSExogenousPrior):
    """Factory wrapper mirroring exogenous_distribution's unified entrypoint."""

    def __init__(
        self,
        prior_type: str = "gaussian",
        num_regimes: int = None,
        latent_dim: int = None,
        n_components: int = 4,
        gaussian_mode: str = "regime_embedded",
    ):
        super().__init__()
        if prior_type == "gaussian":
            self.impl: BaseSExogenousPrior = GaussianSExogenousPrior(
                num_regimes=num_regimes,
                latent_dim=latent_dim,
                mode=gaussian_mode,
            )
        elif prior_type == "gmm":
            self.impl = GMMSExogenousPrior(
                num_regimes=num_regimes,
                latent_dim=latent_dim,
                num_components=n_components,
            )
        else:
            raise ValueError(f"Unsupported s prior type: {prior_type}")
        self.prior_type = prior_type

    def forward(
        self,
        regime_id: torch.Tensor = None,
        regime_seen_mask: torch.Tensor = None,
        context: Optional[dict] = None,
        seq_len: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.impl(
            regime_id=regime_id,
            regime_seen_mask=regime_seen_mask,
            context=context,
            seq_len=seq_len,
        )

    def kl(
        self,
        q_loc: torch.Tensor,
        q_logvar: torch.Tensor,
        regime_id: torch.Tensor = None,
        regime_seen_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.impl.kl(
            q_loc=q_loc,
            q_logvar=q_logvar,
            regime_id=regime_id,
            regime_seen_mask=regime_seen_mask,
        )


def build_s_exogenous_prior(
    prior_type: str,
    num_regimes: int,
    latent_dim: int,
    n_components: int = 4,
    gaussian_mode: str = "regime_embedded",
) -> BaseSExogenousPrior:
    if prior_type == "gaussian":
        return GaussianSExogenousPrior(
            num_regimes=num_regimes,
            latent_dim=latent_dim,
            mode=gaussian_mode,
        )
    if prior_type == "gmm":
        return GMMSExogenousPrior(
            num_regimes=num_regimes,
            latent_dim=latent_dim,
            num_components=n_components,
        )
    raise ValueError(f"Unsupported s prior type: {prior_type}")
