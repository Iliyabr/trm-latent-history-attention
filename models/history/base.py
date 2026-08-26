from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class HistoryAggregator(nn.Module, ABC):
    """Common interface for within-H-cycle latent-history readers.

    Inputs
    ------
    current_z:
        Current latent state with shape [B, L, D].

    history_z:
        States from the current H cycle with shape [B, K, L, D].  It includes
        the initial z_L and every state preceding ``current_z``.

    history_lengths:
        Optional number of valid states per batch element.  The recursive
        model normally supplies a dense, equally-sized history.

    Returns
    -------
    torch.Tensor
        Updated latent representation with the same shape as current_z:
        [B, L, D].

    Contract
    --------
    - Aggregators must not modify the input tensors in-place.
    - Only the first history_lengths[b] entries are valid for sample b.
    - History never crosses an H-cycle or ACT supervision boundary.
    - Output shape and dtype must match current_z.
    """

    @abstractmethod
    def forward(
        self,
        current_z: torch.Tensor,
        history_z: torch.Tensor,
        history_lengths: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError
