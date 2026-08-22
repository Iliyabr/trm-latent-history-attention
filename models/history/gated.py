from __future__ import annotations

import torch
from torch import nn

from .base import HistoryAggregator


class GatedHistory(HistoryAggregator):
    """Learned scalar gate between current state and mean latent history."""

    def __init__(self) -> None:
        super().__init__()

        # alpha = 0 -> sigmoid(alpha) = 0.5
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        current_z: torch.Tensor,
        history_z: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_steps = history_z.shape[:2]

        valid_mask = (
            torch.arange(max_steps, device=history_z.device)
            .unsqueeze(0)
            < history_lengths.unsqueeze(1)
        )

        masked_history = torch.where(
            valid_mask[:, :, None, None],
            history_z,
            torch.zeros_like(history_z),
        )

        counts = history_lengths.clamp(min=1).to(
            dtype=current_z.dtype
        ).view(batch_size, 1, 1)

        mean_history = masked_history.sum(dim=1) / counts

        gate = torch.sigmoid(self.gate_logit).to(
            dtype=current_z.dtype
        )

        mixed = gate * current_z + (1.0 - gate) * mean_history

        # No valid history -> exact identity.
        has_history = (
            history_lengths > 0
        ).view(batch_size, 1, 1)

        return torch.where(
            has_history,
            mixed,
            current_z,
        )
