from __future__ import annotations

import torch

from .base import HistoryAggregator


class NoHistoryAggregator(HistoryAggregator):
    """Identity control: completely ignores latent history."""

    def forward(
        self,
        current_z: torch.Tensor,
        history_z: torch.Tensor,
        history_lengths: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        # Return the original tensor directly.
        # No clone, projection, parameter, or arithmetic is introduced.
        return (current_z, {}) if return_diagnostics else current_z
