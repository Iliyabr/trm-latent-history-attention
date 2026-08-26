from __future__ import annotations

import torch

from models.layers import rms_norm

from .base import HistoryAggregator


class UniformMeanHistory(HistoryAggregator):
    """B2: normalized uniform memory over preceding within-cycle states."""

    def __init__(self, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.norm_eps = norm_eps

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
                (batch_size,), max_steps, dtype=torch.long,
                device=history_z.device
            )
        positions = torch.arange(max_steps, device=history_z.device)
        valid = positions[None, :] < history_lengths[:, None]
        history_sum = torch.where(
            valid[:, :, None, None], history_z, torch.zeros_like(history_z)
        ).sum(dim=1)
        denominator = history_lengths.clamp(min=1).to(
            current_z.dtype
        ).view(batch_size, 1, 1)
        memory = history_sum / denominator
        output = rms_norm(
            current_z + memory, variance_epsilon=self.norm_eps
        )
        return (output, {}) if return_diagnostics else output
