from __future__ import annotations

import torch
from torch import nn

from models.layers import CastedLinear, rms_norm


class LcycleParameterMatchedNoHistory(nn.Module):
    """Low-rank current-state control with no access to latent history."""

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int,
        rms_norm_eps: float,
        gate_init: float = -2.0,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if bottleneck_size <= 0:
            raise ValueError("bottleneck_size must be positive")

        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.rms_norm_eps = rms_norm_eps
        self.pre_norm = pre_norm

        self.down_proj = CastedLinear(
            hidden_size,
            bottleneck_size,
            bias=False,
        )

        self.up_proj = CastedLinear(
            bottleneck_size,
            hidden_size,
            bias=False,
        )

        self.gate_logit = nn.Parameter(
            torch.tensor(float(gate_init), dtype=torch.float32)
        )

    def forward(
        self,
        current_z: torch.Tensor,
    ) -> torch.Tensor:

        if current_z.ndim != 3:
            raise ValueError("current_z must have shape [B, L, D]")

        if current_z.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"got {current_z.shape[-1]}"
            )

        x = current_z

        if self.pre_norm:
            x = rms_norm(
                x,
                variance_epsilon=self.rms_norm_eps,
            )

        message = self.up_proj(
            self.down_proj(x)
        )

        gate = torch.sigmoid(self.gate_logit).to(current_z.dtype)

        return rms_norm(
            current_z + gate * message,
            variance_epsilon=self.rms_norm_eps,
        )
