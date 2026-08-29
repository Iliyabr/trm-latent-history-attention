from __future__ import annotations

import torch
from torch import nn

from models.layers import rms_norm


class LcycleGatedHistory(nn.Module):
    """Scalar-gated uniform history control within one H-cycle."""

    def __init__(
        self,
        rms_norm_eps: float,
        gate_init: float = -2.0,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()

        self.rms_norm_eps = rms_norm_eps
        self.pre_norm = pre_norm

        self.gate_logit = nn.Parameter(
            torch.tensor(float(gate_init), dtype=torch.float32)
        )

    def forward(
        self,
        current_z: torch.Tensor,
        history_z: torch.Tensor,
    ) -> torch.Tensor:

        if current_z.ndim != 3:
            raise ValueError("current_z must have shape [B, L, D]")

        if history_z.ndim != 4:
            raise ValueError("history_z must have shape [B, K, L, D]")

        if history_z.shape[0] != current_z.shape[0]:
            raise ValueError("batch sizes must match")

        if history_z.shape[2:] != current_z.shape[1:]:
            raise ValueError("sequence/hidden dimensions must match")

        if history_z.shape[1] == 0:
            raise ValueError("history_z must contain at least one state")

        history_input = history_z

        if self.pre_norm:
            history_input = rms_norm(
                history_input,
                variance_epsilon=self.rms_norm_eps,
            )

        context = history_input.mean(dim=1)

        gate = torch.sigmoid(self.gate_logit).to(current_z.dtype)

        return rms_norm(
            current_z + gate * context.to(current_z.dtype),
            variance_epsilon=self.rms_norm_eps,
        )
