from __future__ import annotations

import torch
from torch import nn

from models.layers import CastedLinear, rms_norm

from .base import HistoryAggregator


class ParameterMatchedNoHistory(HistoryAggregator):
    """Canonical capacity control (protocol v1).

    No temporal history. Low-rank side path with the same added parameter
    count as P1: 4*D*r + 1.

        RMSNorm(z) -> D→2r -> 2r→D -> gated residual -> RMSNorm
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        norm_eps: float = 1e-5,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or rank <= 0:
            raise ValueError("hidden_size and rank must be positive")
        inner = 2 * rank
        self.down = CastedLinear(hidden_size, inner, bias=False)
        self.up = CastedLinear(inner, hidden_size, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.norm_eps = norm_eps

    def forward(
        self,
        current_z: torch.Tensor,
        history_z: torch.Tensor,
        history_lengths: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        del history_z, history_lengths  # capacity control; history unused
        normalized = rms_norm(current_z, variance_epsilon=self.norm_eps)
        memory = self.up(self.down(normalized))
        gate = torch.sigmoid(self.gate_logit).to(dtype=current_z.dtype)
        output = rms_norm(
            current_z + gate * memory, variance_epsilon=self.norm_eps
        )
        if return_diagnostics:
            return output, {"gate": gate.detach()}
        return output
