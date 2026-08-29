from __future__ import annotations

import torch
from torch import nn

from models.layers import rms_norm

from .base import HistoryAggregator


class GatedHistory(HistoryAggregator):
    """Canonical gated uniform history (protocol v1).

    context = mean(RMSNorm(history states))
    z_read  = RMSNorm(z + sigmoid(gate_logit) * context)
    gate_logit_init = -2
    """

    def __init__(
        self,
        norm_eps: float = 1e-5,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.norm_eps = norm_eps
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        current_z: torch.Tensor,
        history_z: torch.Tensor,
        history_lengths: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        batch_size, max_steps = history_z.shape[:2]
        if history_lengths is None:
            history_lengths = torch.full(
                (batch_size,), max_steps, dtype=torch.long, device=history_z.device
            )
        valid = (
            torch.arange(max_steps, device=history_z.device)[None, :]
            < history_lengths[:, None]
        )
        normalized = rms_norm(history_z, variance_epsilon=self.norm_eps)
        masked = torch.where(
            valid[:, :, None, None], normalized, torch.zeros_like(normalized)
        )
        counts = history_lengths.clamp(min=1).to(current_z.dtype).view(batch_size, 1, 1)
        context = masked.sum(dim=1) / counts
        gate = torch.sigmoid(self.gate_logit).to(dtype=current_z.dtype)
        output = rms_norm(
            current_z + gate * context, variance_epsilon=self.norm_eps
        )
        has_history = (history_lengths > 0).view(batch_size, 1, 1)
        output = torch.where(has_history, output, current_z)
        if return_diagnostics:
            return output, {"gate": gate.detach()}
        return output
