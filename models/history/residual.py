from __future__ import annotations

import torch

from models.layers import rms_norm

from .base import HistoryAggregator


class ResidualHistory(HistoryAggregator):
    """B1 control: normalized residual around each recursive update."""

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
        # The model applies this mode after computing the backbone update.
        output = rms_norm(
            history_z[:, -1] + current_z, variance_epsilon=self.norm_eps
        )
        return (output, {}) if return_diagnostics else output
