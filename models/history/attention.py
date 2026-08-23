from __future__ import annotations

import math

import torch
from torch import nn

from .base import HistoryAggregator


class HistoryAttention(HistoryAggregator):
    """Token-aligned attention over strictly previous outer latent states.

    For each sequence position independently, the current latent vector is used
    as the query and the corresponding latent vectors from valid previous outer
    states are used as keys and values.

    The attention mechanism is intentionally projection-free. This isolates
    content-dependent history selection from additional representational
    capacity.

    A single learned scalar gate mixes the current state with the retrieved
    history context:

        output = g * current + (1 - g) * context

    where g = sigmoid(gate_logit).

    Extra trainable parameters: 1.
    """

    def __init__(self) -> None:
        super().__init__()

        # gate_logit = 0 -> current/history mixture starts at 0.5 / 0.5.
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        current_z: torch.Tensor,
        history_z: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_steps, _, hidden_size = history_z.shape

        # Defensive support for an empty history dimension.
        if max_steps == 0:
            return current_z

        positions = torch.arange(
            max_steps,
            device=history_z.device,
        ).unsqueeze(0)

        valid_mask = positions < history_lengths.unsqueeze(1)
        has_history = history_lengths > 0

        # Compute attention logits in float32 for numerical stability.
        #
        # current_z:
        #   [B, L, D]
        #
        # history_z:
        #   [B, K, L, D]
        #
        # scores:
        #   [B, K, L]
        current_fp32 = current_z.to(dtype=torch.float32)
        history_fp32 = history_z.to(dtype=torch.float32)

        scores = (
            history_fp32
            * current_fp32[:, None, :, :]
        ).sum(dim=-1)

        scores = scores / math.sqrt(hidden_size)

        # Invalid history slots must never contribute.
        scores = scores.masked_fill(
            ~valid_mask[:, :, None],
            torch.finfo(scores.dtype).min,
        )

        # Avoid an all-masked softmax for samples with zero valid history.
        # Their final output is replaced by exact current_z below.
        scores = torch.where(
            has_history[:, None, None],
            scores,
            torch.zeros_like(scores),
        )

        attention_weights = torch.softmax(
            scores,
            dim=1,
        )

        # Explicitly zero invalid positions after softmax as well.
        attention_weights = (
            attention_weights
            * valid_mask[:, :, None].to(
                dtype=attention_weights.dtype
            )
        )

        context = (
            history_fp32
            * attention_weights[:, :, :, None]
        ).sum(dim=1)

        context = context.to(dtype=current_z.dtype)

        gate = torch.sigmoid(
            self.gate_logit
        ).to(dtype=current_z.dtype)

        mixed = (
            gate * current_z
            + (1.0 - gate) * context
        )

        # First outer step must remain exact identity.
        return torch.where(
            has_history[:, None, None],
            mixed,
            current_z,
        )
