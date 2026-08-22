from __future__ import annotations

import torch

from .base import HistoryAggregator


class RecencyWeightedHistory(HistoryAggregator):
    """Parameter-free exponentially recency-weighted latent history.

    The current state has weight 1.0. Previous states receive weights
    0.5, 0.25, 0.125, ... according to their temporal distance from
    the current outer reasoning step.

    Only valid history slots are included in the normalized weighted mean.
    """

    def forward(
        self,
        current_z: torch.Tensor,
        history_z: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_steps = history_z.shape[:2]

        positions = torch.arange(
            max_steps,
            device=history_z.device,
        ).unsqueeze(0)

        valid_mask = positions < history_lengths.unsqueeze(1)

        # For a sample with history length n:
        # newest valid history state has lag 1,
        # previous state lag 2, etc.
        lags = history_lengths.unsqueeze(1) - positions

        history_weights = torch.pow(
            torch.tensor(
                0.5,
                dtype=current_z.dtype,
                device=current_z.device,
            ),
            lags.to(dtype=current_z.dtype),
        )

        history_weights = torch.where(
            valid_mask,
            history_weights,
            torch.zeros_like(history_weights),
        )

        weighted_history = (
            history_z
            * history_weights[:, :, None, None]
        ).sum(dim=1)

        denominator = (
            1.0
            + history_weights.sum(dim=1)
        ).view(batch_size, 1, 1)

        return (current_z + weighted_history) / denominator
