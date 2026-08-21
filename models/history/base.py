from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class HistoryAggregator(nn.Module, ABC):
    """Common interface for all latent-history aggregation methods.

    Inputs
    ------
    current_z:
        Current latent state with shape [B, L, D].

    history_z:
        Detached latent-history buffer with shape [B, K, L, D].

    history_lengths:
        Number of valid historical states per batch element, shape [B].

    Returns
    -------
    torch.Tensor
        Updated latent representation with the same shape as current_z:
        [B, L, D].

    Contract
    --------
    - Aggregators must not modify the input tensors in-place.
    - Only the first history_lengths[b] entries are valid for sample b.
    - The history buffer contains only previously completed outer states.
    - Output shape and dtype must match current_z.
    """

    @abstractmethod
    def forward(
        self,
        current_z: torch.Tensor,
        history_z: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
