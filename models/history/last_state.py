from __future__ import annotations

import torch
from models.layers import rms_norm

from .base import HistoryAggregator


class LastStateHistory(HistoryAggregator):
    """Parameter-free latest-state history ablation.

    Uses only the most recent valid historical state.

    The current state has weight 1.0 and the latest previous state
    has weight 0.5, matching the newest-history weight used by
    RecencyWeightedHistory.

    This isolates whether older historical states provide additional
    value beyond the immediately preceding outer reasoning state.
    """

    def forward(
        self,
        current_z: torch.Tensor,
        history_z: torch.Tensor,
        history_lengths: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        batch_size = current_z.shape[0]
        if history_lengths is None:
            history_lengths = torch.full(
                (batch_size,), history_z.shape[1], dtype=torch.long,
                device=history_z.device
            )

        has_history = history_lengths > 0

        # Clamp only for safe indexing. Samples with zero valid history
        # are replaced by current_z below.
        latest_index = history_lengths.clamp(min=1) - 1

        batch_index = torch.arange(
            batch_size,
            device=history_z.device,
        )

        latest_z = history_z[
            batch_index,
            latest_index.to(dtype=torch.long),
        ]

        mixed = rms_norm(current_z + latest_z, variance_epsilon=1e-5)

        output = torch.where(
            has_history[:, None, None],
            mixed,
            current_z,
        )
        return (output, {}) if return_diagnostics else output
