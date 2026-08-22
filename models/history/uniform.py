from __future__ import annotations

import torch

from .base import HistoryAggregator


class UniformMeanHistory(HistoryAggregator):
    """Parameter-free uniform averaging over current and valid previous states."""

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
            .expand(batch_size, -1)
            < history_lengths.unsqueeze(1)
        )

        masked_history = torch.where(
            valid_mask[:, :, None, None],
            history_z,
            torch.zeros_like(history_z),
        )

        history_sum = masked_history.sum(dim=1)

        denominator = (
            history_lengths.to(dtype=current_z.dtype)
            .add(1)
            .view(batch_size, 1, 1)
        )

        return (current_z + history_sum) / denominator
