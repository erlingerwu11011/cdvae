from __future__ import annotations

import torch
import torch.nn as nn


class MLPFaultAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        _ = context
        return self.net(x)


class TriangularLinearFaultAdapter(nn.Module):
    """Lightweight lower-triangular mapping: y = s @ tril(W)^T + b."""

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.weight_raw = nn.Parameter(torch.zeros(output_dim, latent_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, s: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        _ = context
        # Keep a structured lower-triangular map when dims align; otherwise use rectangular lower mask.
        lower_mask = torch.tril(torch.ones_like(self.weight_raw))
        weight = self.weight_raw * lower_mask
        return s @ weight.t() + self.bias
