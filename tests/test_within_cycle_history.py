from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from models.history import HistoryAttention, build_history_aggregator
from models.losses import ACTLossHead
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1_Inner,
)


def tiny_config(mode: str = "none", **overrides):
    values = dict(
        batch_size=2, seq_len=5, puzzle_emb_ndim=0,
        num_puzzle_identifiers=1, vocab_size=11,
        H_cycles=1, L_cycles=2, H_layers=0, L_layers=1,
        hidden_size=16, expansion=2, num_heads=4,
        pos_encodings="rope", halt_max_steps=2,
        halt_exploration_prob=0.0, forward_dtype="float32",
        puzzle_emb_len=0, history_mode=mode, history_rank=8,
        history_heads=2, history_window=0,
    )
    values.update(overrides)
    return TinyRecursiveReasoningModel_ACTV1Config(**values)


def run_inner(model, analysis_request=None, batch=None):
    carry = model.empty_carry(2)
    carry = model.reset_carry(torch.ones(2, dtype=torch.bool), carry)
    if batch is None:
        batch = {
            "inputs": torch.randint(0, 11, (2, 5)),
            "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
        }
    return model(carry, batch, analysis_request=analysis_request)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_shape_dtype_normalization_gradients_and_gate(dtype):
    module = HistoryAttention(16, rank=8, num_heads=2, window=2).to(dtype)
    current = torch.randn(2, 5, 16, dtype=dtype, requires_grad=True)
    history = torch.randn(2, 4, 5, 16, dtype=dtype, requires_grad=True)
    output, diagnostics = module(
        current, history, return_diagnostics=True
    )

    assert output.shape == current.shape
    assert output.dtype == dtype
    assert torch.allclose(
        output.float().square().mean(-1),
        torch.ones(2, 5),
        atol=2e-2 if dtype == torch.bfloat16 else 2e-5,
    )
    assert torch.allclose(
        torch.sigmoid(module.gate_logit.detach().float()),
        torch.tensor(0.1192029),
        atol=1e-6,
    )
    weights = diagnostics["attention_weights"]
    assert weights.shape == (2, 2, 5, 4)
    assert torch.count_nonzero(weights[..., :2]) == 0
    assert not hasattr(module, "last_attention_weights")

    output.float().sum().backward()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_attention_lengths_are_causal_and_mask_invalid_states():
    torch.manual_seed(1)
    module = HistoryAttention(8, rank=4, num_heads=2, window=0)
    current = torch.randn(2, 3, 8)
    history = torch.randn(2, 4, 3, 8)
    lengths = torch.tensor([1, 3])
    changed = history.clone()
    changed[0, 1:] = 1e6
    changed[1, 3:] = -1e6
    first = module(current, history, lengths)
    second = module(current, changed, lengths)
    assert torch.equal(first, second)


class RecordingIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.lengths = []

    def forward(self, current_z, history_z, *args, **kwargs):
        self.lengths.append(history_z.shape[1])
        return current_z


def test_history_resets_each_h_cycle_and_l_cycles_one_works():
    model = TinyRecursiveReasoningModel_ACTV1_Inner(
        tiny_config("uniform", H_cycles=2, L_cycles=3)
    )
    recorder = RecordingIdentity()
    model.history_aggregator = recorder
    run_inner(model)
    assert recorder.lengths == [1, 2, 3, 1, 2, 3]

    one = TinyRecursiveReasoningModel_ACTV1_Inner(
        tiny_config("attention", L_cycles=1)
    )
    carry, logits, _ = run_inner(one)
    assert carry.z_L.shape == (2, 5, 16)
    assert logits.shape == (2, 5, 11)
    assert not hasattr(carry, "history_z_H")


@pytest.mark.parametrize(
    "mode", ["none", "residual", "uniform", "attention", "parameter_matched"]
)
def test_all_variants_forward_and_backward(mode):
    model = TinyRecursiveReasoningModel_ACTV1_Inner(tiny_config(mode))
    _, logits, _ = run_inner(model)
    assert torch.isfinite(logits).all()
    logits.float().sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_b0_is_exact_vanilla_and_old_config_alias_loads():
    vanilla = TinyRecursiveReasoningModel_ACTV1_Inner(
        tiny_config("none", history_enabled=False)
    )
    legacy = TinyRecursiveReasoningModel_ACTV1_Inner(
        tiny_config(
            history_mode=None, history_enabled=True,
            history_aggregator="none"
        )
    )
    legacy.load_state_dict(copy.deepcopy(vanilla.state_dict()), strict=True)
    torch.manual_seed(7)
    first = run_inner(vanilla)
    torch.manual_seed(7)
    second = run_inner(legacy)
    assert torch.equal(first[0].z_H, second[0].z_H)
    assert torch.equal(first[0].z_L, second[0].z_L)
    assert torch.equal(first[1], second[1])


def test_parameter_matched_b3_is_close_to_p1_added_parameters():
    base = TinyRecursiveReasoningModel_ACTV1_Inner(tiny_config("none"))
    attention = TinyRecursiveReasoningModel_ACTV1_Inner(tiny_config("attention"))
    matched = TinyRecursiveReasoningModel_ACTV1_Inner(
        tiny_config("parameter_matched")
    )
    count = lambda module: sum(p.numel() for p in module.parameters())
    p1_added = count(attention) - count(base)
    b3_added = count(matched) - count(base)
    assert p1_added == 4 * 16 * 8 + 1
    # Nearest whole SwiGLU channel adds 3*D parameters per shared layer.
    assert abs(b3_added - p1_added) <= 3 * 16 // 2
    assert sum(
        p.numel() for p in matched.history_aggregator.parameters()
    ) == 0


def test_factory_names_and_invalid_rank():
    assert build_history_aggregator("B0").__class__.__name__ == "NoHistoryAggregator"
    assert build_history_aggregator("B1").__class__.__name__ == "ResidualHistory"
    assert build_history_aggregator("B2").__class__.__name__ == "UniformMeanHistory"
    assert build_history_aggregator("B3").__class__.__name__ == "NoHistoryAggregator"
    with pytest.raises(ValueError):
        build_history_aggregator(
            "P1", hidden_size=16, rank=7, num_heads=2
        )


def test_incremental_kv_cache_matches_full_projection():
    torch.manual_seed(11)
    module = HistoryAttention(8, rank=4, num_heads=2)
    current = torch.randn(2, 3, 8)
    history = torch.randn(2, 4, 3, 8)
    expected, expected_diag = module(
        current, history, return_diagnostics=True
    )
    cache = None
    for step in range(history.shape[1]):
        cache = module.append_kv(cache, history[:, step])
    actual, actual_diag = module(
        current, history[:, -1:],
        history_lengths=torch.full((2,), 4),
        return_diagnostics=True, kv_cache=cache,
    )
    assert torch.allclose(expected, actual, rtol=1e-6, atol=1e-6)
    assert torch.allclose(
        expected_diag["attention_weights"],
        actual_diag["attention_weights"],
        rtol=1e-6, atol=1e-6,
    )


@pytest.mark.parametrize("kind", ["most", "least"])
def test_attention_state_deletion_masks_and_renormalizes(kind):
    torch.manual_seed(12)
    module = HistoryAttention(8, rank=4, num_heads=2)
    current = torch.randn(2, 3, 8)
    history = torch.randn(2, 4, 3, 8)
    _, baseline = module(current, history, return_diagnostics=True)
    _, deleted = module(
        current, history, return_diagnostics=True, delete_state=kind
    )
    state_score = baseline["attention_weights"].mean(dim=(1, 2))
    expected = (
        state_score.argmax(dim=-1) if kind == "most"
        else state_score.argmin(dim=-1)
    )
    assert torch.equal(deleted["deleted_state_index"], expected)
    for batch_index, state_index in enumerate(expected):
        assert torch.count_nonzero(
            deleted["attention_weights"][batch_index, ..., state_index]
        ) == 0
    assert torch.allclose(
        deleted["attention_weights"].sum(dim=-1),
        torch.ones(2, 2, 3),
    )


def test_model_analysis_outputs_are_transient_and_shaped():
    model = TinyRecursiveReasoningModel_ACTV1_Inner(
        tiny_config("attention", H_cycles=2, L_cycles=3)
    )
    model.eval()
    request = {
        "attention_weights": True,
        "attention_stats": True,
        "intermediate_logits": True,
        "cycle_logits": True,
        "delete_state": {"kind": "most", "h_step": 1, "l_step": 2},
    }
    result = run_inner(model, analysis_request=request)
    outputs = result[3]
    assert outputs["history_attention_weights"].shape == (
        2, 3, 2, 2, 5, 3
    )
    assert outputs["history_attention_entropy"].shape == (2, 3)
    assert outputs["history_deleted_state_index"].shape == (2, 3, 2)
    assert outputs["history_intermediate_logits"].shape == (
        2, 3, 2, 5, 11
    )
    assert outputs["history_cycle_logits"].shape == (2, 2, 5, 11)
    assert all(not value.requires_grad for value in outputs.values())
    assert not hasattr(model.history_aggregator, "last_attention_weights")

    normal = run_inner(model)
    assert len(normal) == 3


def test_seeded_and_supplied_corruption_are_deterministic():
    model = TinyRecursiveReasoningModel_ACTV1_Inner(
        tiny_config("none", H_cycles=2, L_cycles=2)
    )
    model.eval()
    batch = {
        "inputs": torch.arange(10).view(2, 5) % 11,
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }
    request = {
        "corruption": {
            "h_step": 1, "l_step": 0, "sigma": 0.25, "seed": 123
        }
    }
    first = run_inner(model, request, batch)
    second = run_inner(model, request, batch)
    assert torch.equal(first[1], second[1])
    assert torch.equal(
        first[3]["history_corruption_noise"],
        second[3]["history_corruption_noise"],
    )

    supplied = torch.ones(2, 5, 16)
    supplied_request = {
        "corruption": {
            "h_step": 0, "l_step": 1, "sigma": 0.1,
            "noise": supplied,
        }
    }
    supplied_result = run_inner(model, supplied_request, batch)
    assert supplied_result[3]["history_corruption_noise"].shape == (
        1, 2, 5, 16
    )


def test_act_loss_return_keys_retrieves_analysis_outputs():
    config = tiny_config("attention", H_cycles=1, L_cycles=2)
    core = TinyRecursiveReasoningModel_ACTV1(config.model_dump())
    core.eval()
    loss_model = ACTLossHead(core, "softmax_cross_entropy")
    loss_model.eval()
    batch = {
        "inputs": torch.randint(0, 11, (2, 5)),
        "labels": torch.randint(0, 11, (2, 5)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }
    carry = loss_model.initial_carry(batch)
    _, _, _, outputs, _ = loss_model(
        carry=carry,
        batch=batch,
        analysis_request={"attention_stats": True},
        return_keys=["history_attention_entropy"],
    )
    assert outputs["history_attention_entropy"].shape == (1, 2)


def test_analysis_request_is_rejected_during_training():
    model = TinyRecursiveReasoningModel_ACTV1_Inner(tiny_config("attention"))
    model.train()
    with pytest.raises(RuntimeError, match="evaluation-only"):
        run_inner(model, {"attention_stats": True})


def test_init_buffers_are_registered_and_reset_carry_follows_carry_device():
    inner = TinyRecursiveReasoningModel_ACTV1_Inner(tiny_config("none"))
    buffers = dict(inner.named_buffers())
    assert "H_init" in buffers
    assert "L_init" in buffers

    carry = inner.empty_carry(2)
    reset_flag = torch.ones(2, dtype=torch.bool)
    reset = inner.reset_carry(reset_flag, carry)
    assert reset.z_H.device == carry.z_H.device
    assert torch.equal(reset.z_H[0, 0], inner.H_init)

    model = TinyRecursiveReasoningModel_ACTV1(tiny_config("none").model_dump())
    batch = {
        "inputs": torch.randint(0, 11, (2, 5)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }
    wrapped = model.initial_carry(batch)
    assert wrapped.halted.device == batch["inputs"].device
    assert wrapped.steps.device == batch["inputs"].device
    assert wrapped.inner_carry.z_H.device == batch["inputs"].device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reset_carry_accepts_cuda_carry_when_init_buffers_are_on_cpu():
    inner = TinyRecursiveReasoningModel_ACTV1_Inner(tiny_config("none"))
    assert inner.H_init.device.type == "cpu"
    cpu_carry = inner.empty_carry(2)
    device = torch.device("cuda")
    carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
        z_H=torch.empty_like(cpu_carry.z_H, device=device),
        z_L=torch.empty_like(cpu_carry.z_L, device=device),
    )
    reset_flag = torch.ones(2, dtype=torch.bool, device=device)
    reset = inner.reset_carry(reset_flag, carry)
    assert reset.z_H.device.type == "cuda"
    assert torch.allclose(
        reset.z_H[0, 0].float().cpu(), inner.H_init.float().cpu()
    )

    model = TinyRecursiveReasoningModel_ACTV1(
        tiny_config("none").model_dump()
    ).to(device)
    batch = {
        "inputs": torch.randint(0, 11, (2, 5), device=device),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long, device=device),
    }
    model.eval()
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        assert carry.halted.device.type == "cuda"
        assert model.inner.H_init.device.type == "cuda"
        model(carry, batch)
