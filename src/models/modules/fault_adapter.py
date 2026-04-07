from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BaseFaultAdapter(nn.Module):
    """Abstract interface for fault-to-effect mappings."""

    def forward(self, s: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError


class MLPFaultAdapter(BaseFaultAdapter):
    """Unconstrained MLP adapter."""

    def __init__(self, s_dim: int, context_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.s_dim = s_dim
        self.context_dim = context_dim
        self.net = nn.Sequential(
            nn.Linear(s_dim + context_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, s: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        if context is None:
            context = torch.zeros(
                *s.shape[:-1], self.context_dim, device=s.device, dtype=s.dtype
            )
        x = torch.cat([s, context], dim=-1)
        return self.net(x)


class ResidualTriangularFaultAdapter(BaseFaultAdapter):
    """Structured adapter: lower-triangular linear map + residual MLP.

    fault_effect = s @ tril(W)^T + residual_mlp([s, context])
    """

    def __init__(self, s_dim: int, context_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.s_dim = s_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.weight_raw = nn.Parameter(torch.randn(output_dim, output_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.s_to_out = nn.Identity() if s_dim == output_dim else nn.Linear(s_dim, output_dim)
        self.residual_mlp = nn.Sequential(
            nn.Linear(s_dim + context_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _triangular_weight(self) -> torch.Tensor:
        lower = torch.tril(self.weight_raw, diagonal=-1)
        diag = F.softplus(torch.diagonal(self.weight_raw, 0)) + 1e-3
        return lower + torch.diag_embed(diag)

    def forward(self, s: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        if context is None:
            context = torch.zeros(
                *s.shape[:-1], self.context_dim, device=s.device, dtype=s.dtype
            )
        s_base = self.s_to_out(s)
        w = self._triangular_weight()
        linear = torch.matmul(s_base, w.transpose(-1, -2)) + self.bias
        residual = self.residual_mlp(torch.cat([s, context], dim=-1))
        return linear + residual


class TriangularLinearFaultAdapter(ResidualTriangularFaultAdapter):
    """Backward-compatible alias."""


def build_fault_adapter(
    fault_adapter_type: str,
    s_dim: int,
    context_dim: int,
    output_dim: int,
    hidden_dim: int,
) -> BaseFaultAdapter:
    if fault_adapter_type == "mlp":
        return MLPFaultAdapter(
            s_dim=s_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
    if fault_adapter_type in {"tri_linear", "residual_triangular"}:
        return ResidualTriangularFaultAdapter(
            s_dim=s_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
    raise ValueError(f"Unsupported fault_adapter_type: {fault_adapter_type}")
