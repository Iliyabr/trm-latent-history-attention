from __future__ import annotations

import math

import torch
from torch import nn

from models.layers import CastedLinear, rms_norm


class LcycleLowRankHistoryAttention(nn.Module):
    """Low-rank multi-head attention over z_L states within one H-cycle.

    The caller owns the history lifetime. In the intended TRM integration,
    history is reset at the start of every H-cycle and contains the initial
    z_L plus states produced by earlier L-steps of the same H-cycle.

    Shapes
    ------
    current_z:
        [B, L, D]

    history_z:
        [B, K, L, D]

    output:
        [B, L, D]
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        num_heads: int,
        rms_norm_eps: float,
        gate_init: float = 0.0,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if rank % num_heads != 0:
            raise ValueError(
                f"rank ({rank}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_size = hidden_size
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = rank // num_heads
        self.rms_norm_eps = rms_norm_eps
        self.pre_norm = pre_norm

        self.q_proj = CastedLinear(hidden_size, rank, bias=False)
        self.k_proj = CastedLinear(hidden_size, rank, bias=False)
        self.v_proj = CastedLinear(hidden_size, rank, bias=False)
        self.o_proj = CastedLinear(rank, hidden_size, bias=False)

        # Configurable so the original pilot remains reproducible while
        # proposal-faithful runs can use gate_init=-2.0.
        self.gate_logit = nn.Parameter(
            torch.tensor(float(gate_init), dtype=torch.float32)
        )

    def forward(
        self,
        current_z: torch.Tensor,
        history_z: torch.Tensor,
    ) -> torch.Tensor:
        if current_z.ndim != 3:
            raise ValueError(
                f"current_z must have shape [B, L, D], got {tuple(current_z.shape)}"
            )

        if history_z.ndim != 4:
            raise ValueError(
                f"history_z must have shape [B, K, L, D], got {tuple(history_z.shape)}"
            )

        batch_size, seq_len, hidden_size = current_z.shape

        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden_size}"
            )

        if history_z.shape[0] != batch_size:
            raise ValueError("current_z and history_z batch sizes must match")

        if history_z.shape[2] != seq_len:
            raise ValueError("current_z and history_z sequence lengths must match")

        if history_z.shape[3] != hidden_size:
            raise ValueError("current_z and history_z hidden sizes must match")

        history_len = history_z.shape[1]
        if history_len == 0:
            raise ValueError("history_z must contain at least one state")

        # Optional proposal-faithful pre-normalization before Q/K/V.
        if self.pre_norm:
            q_input = rms_norm(
                current_z,
                variance_epsilon=self.rms_norm_eps,
            )
            kv_input = rms_norm(
                history_z,
                variance_epsilon=self.rms_norm_eps,
            )
        else:
            q_input = current_z
            kv_input = history_z

        # Low-rank projections.
        q = self.q_proj(q_input)         # [B, L, R]
        k = self.k_proj(kv_input)        # [B, K, L, R]
        v = self.v_proj(kv_input)        # [B, K, L, R]

        # Multi-head temporal attention.
        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        k = k.view(
            batch_size,
            history_len,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        v = v.view(
            batch_size,
            history_len,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        # Use float32 for attention scores/softmax for numerical stability.
        q_f = q.to(torch.float32)
        k_f = k.to(torch.float32)
        v_f = v.to(torch.float32)

        # [B, L, H, K]
        scores = torch.einsum(
            "blhd,bklhd->blhk",
            q_f,
            k_f,
        )
        scores = scores / math.sqrt(self.head_dim)

        weights = torch.softmax(scores, dim=-1)

        # [B, L, H, Hd]
        context = torch.einsum(
            "blhk,bklhd->blhd",
            weights,
            v_f,
        )

        context = context.reshape(
            batch_size,
            seq_len,
            self.rank,
        ).to(current_z.dtype)

        projected_history = self.o_proj(context)

        gate = torch.sigmoid(self.gate_logit).to(current_z.dtype)

        return rms_norm(
            current_z + gate * projected_history,
            variance_epsilon=self.rms_norm_eps,
        )
