import torch

from models.history.lcycle_lowrank_attention import (
    LcycleLowRankHistoryAttention,
)


def test_lcycle_lowrank_history_shape_dtype_and_finite():
    module = LcycleLowRankHistoryAttention(
        hidden_size=64,
        rank=32,
        num_heads=4,
        rms_norm_eps=1e-5,
    )

    current = torch.randn(2, 81, 64)
    history = torch.randn(2, 4, 81, 64)

    output = module(
        current_z=current,
        history_z=history,
    )

    assert output.shape == current.shape
    assert output.dtype == current.dtype
    assert torch.isfinite(output).all()


def test_lcycle_lowrank_history_single_state():
    module = LcycleLowRankHistoryAttention(
        hidden_size=64,
        rank=32,
        num_heads=4,
        rms_norm_eps=1e-5,
    )

    current = torch.randn(2, 81, 64)
    history = current.unsqueeze(1)

    output = module(
        current_z=current,
        history_z=history,
    )

    assert output.shape == current.shape
    assert torch.isfinite(output).all()


def test_lcycle_lowrank_history_parameter_count():
    module = LcycleLowRankHistoryAttention(
        hidden_size=64,
        rank=32,
        num_heads=4,
        rms_norm_eps=1e-5,
    )

    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )

    # Q/K/V: 3 * (64 * 32)
    # O:     32 * 64
    # gate:  1
    assert trainable == 8193


def test_lcycle_lowrank_history_requires_divisible_rank():
    try:
        LcycleLowRankHistoryAttention(
            hidden_size=64,
            rank=30,
            num_heads=4,
            rms_norm_eps=1e-5,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError when rank is not divisible by num_heads"
    )
