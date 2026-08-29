from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.history import build_history_aggregator
from models.history.lcycle_lowrank_attention import LcycleLowRankHistoryAttention
from models.history.lcycle_gated import LcycleGatedHistory
from models.history.lcycle_param_matched import LcycleParameterMatchedNoHistory

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor

    # Phase 2: optional detached outer-step latent history.
    # This is recording-only for now; it must not affect vanilla computation.
    history_z_H: Optional[torch.Tensor] = None
    history_lengths: Optional[torch.Tensor] = None


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    # Phase 2: record detached z_H states across outer reasoning steps.
    # Recording is inert: no history state is consumed by the model yet.
    history_enabled: bool = False
    history_aggregator: str = "none"

    # Proposal-track history over z_L states within each H-cycle.
    # Kept separate from the validated outer-step z_H history mechanism.
    lcycle_history_enabled: bool = False
    lcycle_history_method: str = "attention"
    lcycle_history_rank: int = 32
    lcycle_history_heads: int = 4
    lcycle_history_gate_init: float = 0.0
    lcycle_history_pre_norm: bool = False

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Pluggable latent-history aggregation.
        # The default "none" implementation is parameter-free and identity.
        self.history_aggregator = build_history_aggregator(
            self.config.history_aggregator
        )

        # Proposal-track low-rank attention over z_L history.
        # This module is instantiated only when explicitly enabled so that
        # vanilla and existing z_H-history configurations gain no parameters.
        self.lcycle_history_attention = None
        self.lcycle_gated_history = None
        self.lcycle_param_matched = None

        if self.config.lcycle_history_enabled:
            method = self.config.lcycle_history_method.strip().lower()

            if method == "attention":
                self.lcycle_history_attention = LcycleLowRankHistoryAttention(
                    hidden_size=self.config.hidden_size,
                    rank=self.config.lcycle_history_rank,
                    num_heads=self.config.lcycle_history_heads,
                    rms_norm_eps=self.config.rms_norm_eps,
                    gate_init=self.config.lcycle_history_gate_init,
                    pre_norm=self.config.lcycle_history_pre_norm,
                )

            elif method == "gated":
                self.lcycle_gated_history = LcycleGatedHistory(
                    rms_norm_eps=self.config.rms_norm_eps,
                    gate_init=self.config.lcycle_history_gate_init,
                    pre_norm=self.config.lcycle_history_pre_norm,
                )

            elif method in {"parameter_matched", "param_matched"}:
                self.lcycle_param_matched = LcycleParameterMatchedNoHistory(
                    hidden_size=self.config.hidden_size,
                    bottleneck_size=2 * self.config.lcycle_history_rank,
                    rms_norm_eps=self.config.rms_norm_eps,
                    gate_init=self.config.lcycle_history_gate_init,
                    pre_norm=self.config.lcycle_history_pre_norm,
                )

            else:
                raise ValueError(
                    "Unknown lcycle_history_method: "
                    f"{self.config.lcycle_history_method!r}"
                )

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _run_L_cycle(
        self,
        z_L: torch.Tensor,
        z_H: torch.Tensor,
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run one complete L-cycle.

        When proposal-track history is enabled, history is local to this
        H-cycle. It starts with the pre-update z_L state and grows after
        every L-step.
        """

        if not self.config.lcycle_history_enabled:
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(
                    z_L,
                    z_H + input_embeddings,
                    **seq_info,
                )
            return z_L

        method = self.config.lcycle_history_method.strip().lower()

        # Parameter-matched control deliberately has no access to history.
        if method in {"parameter_matched", "param_matched"}:
            if self.lcycle_param_matched is None:
                raise RuntimeError(
                    "Parameter-matched L-cycle module is missing"
                )

            for _L_step in range(self.config.L_cycles):
                read_z = self.lcycle_param_matched(
                    current_z=z_L,
                )

                z_L = self.L_level(
                    read_z,
                    z_H + input_embeddings,
                    **seq_info,
                )

            return z_L

        # Attention and Gated use exactly the same within-H-cycle history.
        history = [z_L]

        for _L_step in range(self.config.L_cycles):
            history_z = torch.stack(history, dim=1)

            if method == "attention":
                if self.lcycle_history_attention is None:
                    raise RuntimeError(
                        "L-cycle attention module is missing"
                    )

                read_z = self.lcycle_history_attention(
                    current_z=z_L,
                    history_z=history_z,
                )

            elif method == "gated":
                if self.lcycle_gated_history is None:
                    raise RuntimeError(
                        "L-cycle gated-history module is missing"
                    )

                read_z = self.lcycle_gated_history(
                    current_z=z_L,
                    history_z=history_z,
                )

            else:
                raise RuntimeError(
                    f"Unsupported L-cycle method: {method!r}"
                )

            z_L = self.L_level(
                read_z,
                z_H + input_embeddings,
                **seq_info,
            )

            history.append(z_L)

        return z_L

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        seq_len = self.config.seq_len + self.puzzle_emb_len
        device = self.H_init.device

        history_z_H = None
        history_lengths = None

        if self.config.history_enabled:
            history_z_H = torch.zeros(
                batch_size,
                self.config.halt_max_steps,
                seq_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            )
            history_lengths = torch.zeros(
                batch_size,
                dtype=torch.int32,
                device=device,
            )

        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                seq_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                seq_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            history_z_H=history_z_H,
            history_lengths=history_lengths,
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        history_z_H = carry.history_z_H
        history_lengths = carry.history_lengths

        if self.config.history_enabled:
            assert history_z_H is not None
            assert history_lengths is not None

            # A reused batch slot must never inherit history from a previous puzzle.
            history_z_H = torch.where(
                reset_flag.view(-1, 1, 1, 1),
                torch.zeros_like(history_z_H),
                history_z_H,
            )
            history_lengths = torch.where(
                reset_flag,
                torch.zeros_like(history_lengths),
                history_lengths,
            )

        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
            history_z_H=history_z_H,
            history_lengths=history_lengths,
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                z_L = self._run_L_cycle(
                    z_L=z_L,
                    z_H=z_H,
                    input_embeddings=input_embeddings,
                    seq_info=seq_info,
                )
                z_H = self.L_level(z_H, z_L, **seq_info)

        # Final H-cycle with grad
        z_L = self._run_L_cycle(
            z_L=z_L,
            z_H=z_H,
            input_embeddings=input_embeddings,
            seq_info=seq_info,
        )
        z_H = self.L_level(z_H, z_L, **seq_info)

        # Optional history aggregation.
        #
        # IMPORTANT: carry.history_z_H contains only states from STRICTLY
        # PREVIOUS outer reasoning steps. The current state is appended only
        # after aggregation below, preserving causal history semantics.
        if self.config.history_enabled:
            assert carry.history_z_H is not None
            assert carry.history_lengths is not None

            z_H = self.history_aggregator(
                current_z=z_H,
                history_z=carry.history_z_H,
                history_lengths=carry.history_lengths,
            )

        # LM Outputs
        z_H_detached = z_H.detach()
        z_L_detached = z_L.detach()

        history_z_H = carry.history_z_H
        history_lengths = carry.history_lengths

        if self.config.history_enabled:
            assert history_z_H is not None
            assert history_lengths is not None

            # Record only AFTER the current state has been computed.
            # A future HistoryAttention module can therefore consume only
            # states from strictly earlier outer reasoning steps.
            history_z_H = history_z_H.clone()
            history_lengths = history_lengths.clone()

            batch_indices = torch.arange(
                z_H.shape[0],
                device=z_H.device,
            )
            history_slots = history_lengths.clamp(
                max=self.config.halt_max_steps - 1
            ).to(torch.long)

            history_z_H[batch_indices, history_slots] = z_H_detached

            history_lengths = torch.clamp(
                history_lengths + 1,
                max=self.config.halt_max_steps,
            )

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=z_H_detached,
            z_L=z_L_detached,
            history_z_H=history_z_H,
            history_lengths=history_lengths,
        )
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
