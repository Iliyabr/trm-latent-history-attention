from __future__ import annotations

import math

import torch
from torch import nn

from models.layers import CastedLinear, rms_norm

from .base import HistoryAggregator


class HistoryAttention(HistoryAggregator):
    """Low-rank, token-aligned temporal attention within one H cycle."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        num_heads: int,
        window: int = 0,
        norm_eps: float = 1e-5,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if rank <= 0 or num_heads <= 0 or rank % num_heads:
            raise ValueError("history rank must be positive and divisible by heads")
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = rank // num_heads
        self.window = window
        self.norm_eps = norm_eps

        self.q_proj = CastedLinear(hidden_size, rank, bias=False)
        self.k_proj = CastedLinear(hidden_size, rank, bias=False)
        self.v_proj = CastedLinear(hidden_size, rank, bias=False)
        self.o_proj = CastedLinear(rank, hidden_size, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def _lengths(
        self, history_z: torch.Tensor, history_lengths: torch.Tensor | None
    ) -> torch.Tensor:
        if history_lengths is not None:
            return history_lengths.to(device=history_z.device, dtype=torch.long)
        return torch.full(
            (history_z.shape[0],),
            history_z.shape[1],
            device=history_z.device,
            dtype=torch.long,
        )

    def project_kv(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project one state for an ephemeral within-cycle K/V cache."""
        shape = (
            state.shape[0], 1, state.shape[1],
            self.num_heads, self.head_dim
        )
        k = self.k_proj(state).view(shape)
        v = self.v_proj(state).view(shape)
        return (
            rms_norm(k, variance_epsilon=self.norm_eps),
            rms_norm(v, variance_epsilon=self.norm_eps),
        )

    def append_kv(
        self,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.project_kv(state)
        if cache is None:
            return projected
        return (
            torch.cat((cache[0], projected[0]), dim=1),
            torch.cat((cache[1], projected[1]), dim=1),
        )

    def forward(
        self,
        current_z: torch.Tensor,
        history_z: torch.Tensor,
        history_lengths: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        delete_state: str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        max_steps = (
            kv_cache[0].shape[1] if kv_cache is not None
            else history_z.shape[1]
        )
        if max_steps == 0:
            diagnostics = {"attention_weights": current_z.new_empty(
                current_z.shape[0], self.num_heads, current_z.shape[1], 0
            )}
            return (current_z, diagnostics) if return_diagnostics else current_z

        lengths = self._lengths(history_z, history_lengths)
        positions = torch.arange(max_steps, device=history_z.device)
        valid = positions[None, :] < lengths[:, None]
        if self.window > 0:
            valid &= positions[None, :] >= (lengths - self.window)[:, None]

        # Project to low rank, split heads, then RMS-normalize Q/K/V per head.
        # CastedLinear preserves the activation dtype for mixed precision.
        q = self.q_proj(current_z).view(
            current_z.shape[0], current_z.shape[1], self.num_heads, self.head_dim
        )
        if kv_cache is None:
            k = self.k_proj(history_z).view(
                current_z.shape[0], max_steps, current_z.shape[1],
                self.num_heads, self.head_dim
            )
            v = self.v_proj(history_z).view_as(k)
            k = rms_norm(k, variance_epsilon=self.norm_eps)
            v = rms_norm(v, variance_epsilon=self.norm_eps)
        else:
            k, v = kv_cache
        q = rms_norm(q, variance_epsilon=self.norm_eps)

        # [B, heads, token, time], independently at every token position.
        scores = torch.einsum("blhd,btlhd->bhlt", q.float(), k.float())
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~valid[:, None, None, :], torch.finfo(scores.dtype).min
        )
        has_history = valid.any(dim=1)
        scores = torch.where(
            has_history[:, None, None, None], scores, torch.zeros_like(scores)
        )
        weights = torch.softmax(scores, dim=-1)
        weights = weights * valid[:, None, None, :].to(weights.dtype)
        deleted_index = None
        if return_diagnostics or delete_state is not None:
            deleted_index = torch.full(
                (current_z.shape[0],), -1, dtype=torch.long,
                device=current_z.device
            )
        if delete_state is not None:
            if delete_state not in {"most", "least"}:
                raise ValueError("delete_state must be 'most' or 'least'")
            state_scores = weights.mean(dim=(1, 2))
            if delete_state == "most":
                state_scores = state_scores.masked_fill(~valid, -torch.inf)
                candidate = state_scores.argmax(dim=-1)
            else:
                state_scores = state_scores.masked_fill(~valid, torch.inf)
                candidate = state_scores.argmin(dim=-1)
            can_delete = valid.sum(dim=-1) > 1
            deleted_index = torch.where(
                can_delete, candidate, deleted_index
            )
            deletion_mask = torch.zeros_like(valid)
            deletion_mask.scatter_(1, candidate[:, None], can_delete[:, None])
            retained = valid & ~deletion_mask
            scores = scores.masked_fill(
                ~retained[:, None, None, :], torch.finfo(scores.dtype).min
            )
            weights = torch.softmax(scores, dim=-1)
            weights = weights * retained[:, None, None, :].to(weights.dtype)
        memory = torch.einsum("bhlt,btlhd->blhd", weights, v.float())
        memory = self.o_proj(memory.reshape(*current_z.shape[:2], self.rank).to(current_z.dtype))

        gate = torch.sigmoid(self.gate_logit).to(current_z.dtype)
        output = rms_norm(
            current_z + gate * memory, variance_epsilon=self.norm_eps
        )
        output = torch.where(has_history[:, None, None], output, current_z)

        if not return_diagnostics:
            return output
        # Diagnostics are created only for an explicit request and detached.
        diagnostics = {
            "attention_weights": weights.detach(),
            "attention_entropy": (
                -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log())
                .sum(dim=-1).mean()
            ).detach(),
            "gate": gate.detach(),
            "deleted_state_index": deleted_index.detach(),
        }
        return output, diagnostics
