from typing import Any, Tuple, List, Dict, Optional
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
from models.history import build_history_aggregator, normalize_history_mode

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


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

    # Within-H-cycle z_L history. The legacy fields remain accepted so old
    # experiment configs continue to parse.
    history_enabled: bool = False
    history_aggregator: str = "none"
    history_mode: Optional[str] = None
    history_rank: int = 16
    history_heads: int = 4
    history_window: int = 0
    history_gate_init: float = -2.0

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(
        self,
        config: TinyRecursiveReasoningModel_ACTV1Config,
        extra_ffn_inter: int = 0,
    ) -> None:
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
            extra_inter=extra_ffn_inter,
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

        configured_mode = (
            self.config.history_mode
            if self.config.history_mode is not None
            else self.config.history_aggregator
        )
        if self.config.history_mode is None and not self.config.history_enabled:
            configured_mode = "none"
        self.history_mode = normalize_history_mode(configured_mode)

        # B3 spends approximately P1's 4*D*rank+1 parameters on widening every
        # shared backbone FFN. SwiGLU adds 3*D parameters per intermediate unit.
        extra_ffn_inter = 0
        if self.history_mode == "parameter_matched":
            target = 4 * self.config.hidden_size * self.config.history_rank + 1
            per_unit = (
                3 * self.config.hidden_size * max(self.config.L_layers, 1)
            )
            extra_ffn_inter = max(1, round(target / per_unit))

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[
                TinyRecursiveReasoningModel_ACTV1Block(
                    self.config, extra_ffn_inter=extra_ffn_inter
                )
                for _i in range(self.config.L_layers)
            ]
        )

        # The module is called inside every z_L recursion. It owns no carry.
        self.history_aggregator = build_history_aggregator(
            self.history_mode,
            hidden_size=self.config.hidden_size,
            rank=self.config.history_rank,
            num_heads=self.config.history_heads,
            window=self.config.history_window,
            norm_eps=self.config.rms_norm_eps,
            gate_init=self.config.history_gate_init,
        )

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

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
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def _run_L_cycle(
        self,
        z_L: torch.Tensor,
        input_injection: torch.Tensor,
        seq_info: Dict[str, Optional[torch.Tensor]],
        h_step: int,
        analysis: Optional[Dict[str, Any]] = None,
    ) -> Any:
        # Local Python references retain autograd only for this H cycle and are
        # discarded on return. No diagnostics or history enters ACT carry.
        history = [z_L]
        records: Dict[str, List[torch.Tensor]] = {}
        kv_cache = None
        if self.history_mode == "attention":
            kv_cache = self.history_aggregator.append_kv(None, z_L)
        for _L_step in range(self.config.L_cycles):
            # P1 reads projected cache directly; other ablations consume raw
            # states and therefore materialize the short temporal stack.
            stacked = (
                z_L.unsqueeze(1)
                if self.history_mode == "attention"
                else torch.stack(history, dim=1)
            )
            if self.history_mode == "residual":
                update = self.L_level(
                    z_L, input_injection, **seq_info
                )
                z_L = self.history_aggregator(update, stacked)
            else:
                diagnostics_requested = (
                    self.history_mode == "attention"
                    and analysis is not None and (
                    analysis.get("attention_weights", False)
                    or analysis.get("attention_stats", False)
                    or analysis.get("delete_state") is not None
                    )
                )
                delete_state = None
                deletion = analysis.get("delete_state") if analysis else None
                if deletion is not None:
                    if isinstance(deletion, str):
                        delete_state = deletion
                    elif (
                        deletion.get("h_step", h_step) == h_step
                        and deletion.get("l_step", _L_step) == _L_step
                    ):
                        delete_state = deletion["kind"]
                if self.history_mode == "attention":
                    lengths = torch.full(
                        (z_L.shape[0],), kv_cache[0].shape[1],
                        dtype=torch.long, device=z_L.device
                    )
                    read_result = self.history_aggregator(
                        z_L, stacked, history_lengths=lengths,
                        return_diagnostics=diagnostics_requested,
                        kv_cache=kv_cache, delete_state=delete_state,
                    )
                else:
                    read_result = self.history_aggregator(z_L, stacked)
                if diagnostics_requested:
                    read_z, step_diagnostics = read_result
                    for key, value in step_diagnostics.items():
                        if (
                            key == "attention_weights"
                            and not analysis.get("attention_weights", False)
                        ):
                            continue
                        records.setdefault(key, []).append(value.detach())
                else:
                    read_z = read_result
                z_L = self.L_level(
                    read_z, input_injection, **seq_info
                )

            corruption = analysis.get("corruption") if analysis else None
            if (
                corruption is not None
                and corruption.get("h_step", 0) == h_step
                and corruption.get("l_step", 0) == _L_step
            ):
                supplied_noise = corruption.get("noise")
                if supplied_noise is None:
                    generator = torch.Generator(device=z_L.device)
                    generator.manual_seed(int(corruption.get("seed", 0)))
                    noise = torch.randn(
                        z_L.shape, generator=generator,
                        device=z_L.device, dtype=z_L.dtype
                    )
                else:
                    noise = supplied_noise.to(
                        device=z_L.device, dtype=z_L.dtype
                    )
                    if noise.shape != z_L.shape:
                        raise ValueError(
                            "analysis corruption noise must match z_L shape"
                        )
                sample_rms = z_L.float().square().mean(
                    dim=(1, 2), keepdim=True
                ).sqrt().to(z_L.dtype)
                perturbation = (
                    float(corruption["sigma"]) * sample_rms * noise
                )
                z_L = z_L + perturbation
                records.setdefault("corruption_noise", []).append(
                    perturbation.detach()
                )

            if analysis is not None and analysis.get(
                "intermediate_logits", False
            ):
                with torch.no_grad():
                    intermediate = self.lm_head(
                        z_L
                    )[:, self.puzzle_emb_len:].detach()
                records.setdefault("intermediate_logits", []).append(
                    intermediate
                )
            if self.history_mode == "attention":
                kv_cache = self.history_aggregator.append_kv(kv_cache, z_L)
            else:
                history.append(z_L)
        return (z_L, records) if analysis is not None else z_L

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
        analysis_request: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Run one ACT step.

        ``analysis_request`` is an evaluation-only, one-call API. Supported
        flags are ``attention_weights``, ``attention_stats``,
        ``intermediate_logits``, and ``cycle_logits``. ``delete_state`` accepts
        ``{"kind": "most"|"least", "h_step": int, "l_step": int}``;
        ``corruption`` accepts selected H/L indices, ``sigma``, and either
        ``seed`` or a full-sized ``noise`` tensor. Returned analysis tensors are
        detached and never stored on the module or carry.
        """
        if analysis_request is not None and self.training:
            raise RuntimeError("analysis_request is evaluation-only")
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        analysis_cycles: List[Dict[str, List[torch.Tensor]]] = []
        cycle_logits: List[torch.Tensor] = []
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                cycle_result = self._run_L_cycle(
                    z_L, z_H + input_embeddings, seq_info,
                    h_step=_H_step, analysis=analysis_request,
                )
                if analysis_request is None:
                    z_L = cycle_result
                else:
                    z_L, cycle_records = cycle_result
                    analysis_cycles.append(cycle_records)
                z_H = self.L_level(z_H, z_L, **seq_info)
                if analysis_request is not None and analysis_request.get(
                    "cycle_logits", False
                ):
                    with torch.no_grad():
                        cycle_logits.append(
                            self.lm_head(
                                z_H
                            )[:, self.puzzle_emb_len:].detach()
                        )
        # 1 with grad
        final_h_step = self.config.H_cycles - 1
        cycle_result = self._run_L_cycle(
            z_L, z_H + input_embeddings, seq_info,
            h_step=final_h_step, analysis=analysis_request,
        )
        if analysis_request is None:
            z_L = cycle_result
        else:
            z_L, cycle_records = cycle_result
            analysis_cycles.append(cycle_records)
        z_H = self.L_level(z_H, z_L, **seq_info)
        if analysis_request is not None and analysis_request.get(
            "cycle_logits", False
        ):
            with torch.no_grad():
                cycle_logits.append(
                    self.lm_head(z_H)[:, self.puzzle_emb_len:].detach()
                )

        # LM Outputs
        z_H_detached = z_H.detach()
        z_L_detached = z_L.detach()

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=z_H_detached,
            z_L=z_L_detached,
        )
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        standard = (
            new_carry, output, (q_logits[..., 0], q_logits[..., 1])
        )
        if analysis_request is None:
            return standard

        analysis_outputs: Dict[str, torch.Tensor] = {}
        if cycle_logits:
            analysis_outputs["history_cycle_logits"] = torch.stack(
                cycle_logits, dim=0
            )
        intermediate = [
            torch.stack(cycle["intermediate_logits"], dim=0)
            for cycle in analysis_cycles
            if cycle.get("intermediate_logits")
        ]
        if intermediate:
            analysis_outputs["history_intermediate_logits"] = torch.stack(
                intermediate, dim=0
            )
        weights_by_cycle = [
            cycle.get("attention_weights", []) for cycle in analysis_cycles
        ]
        if any(weights_by_cycle):
            padded_cycles = []
            for cycle_weights in weights_by_cycle:
                padded_steps = []
                for weights in cycle_weights:
                    padding = self.config.L_cycles - weights.shape[-1]
                    padded_steps.append(F.pad(weights, (0, padding)))
                padded_cycles.append(torch.stack(padded_steps, dim=0))
            analysis_outputs["history_attention_weights"] = torch.stack(
                padded_cycles, dim=0
            )
        for source, output_key in (
            ("attention_entropy", "history_attention_entropy"),
            ("gate", "history_attention_gate"),
            ("deleted_state_index", "history_deleted_state_index"),
        ):
            values = [
                torch.stack(cycle[source], dim=0)
                for cycle in analysis_cycles if cycle.get(source)
            ]
            if values:
                analysis_outputs[output_key] = torch.stack(values, dim=0)
        corruption_noise = [
            value for cycle in analysis_cycles
            for value in cycle.get("corruption_noise", [])
        ]
        if corruption_noise:
            analysis_outputs["history_corruption_noise"] = torch.stack(
                corruption_noise, dim=0
            )
        return (*standard, analysis_outputs)


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
        
    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
        analysis_request: Optional[Dict[str, Any]] = None,
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        if analysis_request is not None and self.training:
            raise RuntimeError("analysis_request is evaluation-only")

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        inner_result = self.inner(
            new_inner_carry, new_current_data,
            analysis_request=analysis_request,
        )
        new_inner_carry, logits, (
            q_halt_logits, q_continue_logits
        ) = inner_result[:3]
        analysis_outputs = inner_result[3] if analysis_request is not None else {}

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        outputs.update(analysis_outputs)

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
                    _, _, (
                        next_q_halt_logits, next_q_continue_logits
                    ) = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
