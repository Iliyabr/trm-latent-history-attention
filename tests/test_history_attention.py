from __future__ import annotations

import math

import torch

from models.history.attention import HistoryAttention
from models.history.factory import build_history_aggregator
from pretrain import create_model
from tests.test_history_interface import first_dev_batch, load_config


def test_history_attention_contract_cpu():
    device = torch.device("cpu")

    aggregator = HistoryAttention().to(device)
    aggregator.eval()

    assert sum(
        p.numel() for p in aggregator.parameters()
    ) == 1

    built = build_history_aggregator("attention")
    assert isinstance(built, HistoryAttention)

    current_z = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ],
        device=device,
    )

    history_z = torch.tensor(
        [
            [
                [[1000.0, 1000.0]],
                [[2000.0, 2000.0]],
                [[3000.0, 3000.0]],
            ],
            [
                [[2.0, 0.0]],
                [[5000.0, 5000.0]],
                [[6000.0, 6000.0]],
            ],
            [
                [[1.0, 0.0]],
                [[-1.0, 0.0]],
                [[7000.0, 7000.0]],
            ],
        ],
        device=device,
    )

    history_lengths = torch.tensor(
        [0, 1, 2],
        dtype=torch.int32,
        device=device,
    )

    with torch.inference_mode():
        output = aggregator(
            current_z=current_z,
            history_z=history_z,
            history_lengths=history_lengths,
        )

    assert output.shape == current_z.shape
    assert output.dtype == current_z.dtype
    assert torch.isfinite(output).all()

    # Zero-history sample must be exact identity even though invalid history
    # slots intentionally contain extreme values.
    assert torch.equal(
        output[0],
        current_z[0],
    )

    # With one valid historical state, attention context must equal that state.
    # Initial gate is 0.5.
    expected_one_history = (
        current_z[1] + history_z[1, 0]
    ) / 2.0

    assert torch.allclose(
        output[1],
        expected_one_history,
        rtol=1e-6,
        atol=1e-6,
    )

    # Two-history sample: compute the expected scaled dot-product attention
    # manually. The invalid third slot must have no effect.
    scores = torch.tensor(
        [
            1.0 / math.sqrt(2.0),
            -1.0 / math.sqrt(2.0),
        ],
        device=device,
    )

    weights = torch.softmax(scores, dim=0)

    expected_context = (
        weights[0] * history_z[2, 0]
        + weights[1] * history_z[2, 1]
    )

    expected_two_history = (
        current_z[2] + expected_context
    ) / 2.0

    assert torch.allclose(
        output[2],
        expected_two_history,
        rtol=1e-6,
        atol=1e-6,
    )

    # Changing an invalid history slot must not change the result.
    history_z_changed = history_z.clone()
    history_z_changed[2, 2] = torch.tensor(
        [[-999999.0, 999999.0]],
        device=device,
    )

    with torch.inference_mode():
        changed_output = aggregator(
            current_z=current_z,
            history_z=history_z_changed,
            history_lengths=history_lengths,
        )

    assert torch.equal(
        output[2],
        changed_output[2],
    )

    print("HISTORY ATTENTION CONTRACT PASS")


def test_history_attention_integration_cpu():
    device = torch.device("cpu")

    none_cfg = load_config(
        history_enabled=True,
        history_aggregator="none",
    )
    attention_cfg = load_config(
        history_enabled=True,
        history_aggregator="attention",
    )

    # Attention introduces one new gate parameter, so an old vanilla
    # checkpoint cannot be loaded strictly into it.
    # Build fresh models and synchronize all shared parameters explicitly.
    none_cfg.load_checkpoint = None
    attention_cfg.load_checkpoint = None

    metadata, batch = first_dev_batch(none_cfg)
    batch = {
        k: v.to(device)
        for k, v in batch.items()
    }

    none_model, _, _ = create_model(
        none_cfg,
        metadata,
        rank=0,
        world_size=1,
        device=device,
    )

    attention_model, _, _ = create_model(
        attention_cfg,
        metadata,
        rank=0,
        world_size=1,
        device=device,
    )

    missing, unexpected = attention_model.load_state_dict(
        none_model.state_dict(),
        strict=False,
    )

    assert len(unexpected) == 0
    assert len(missing) == 1
    assert missing[0].endswith(
        "history_aggregator.gate_logit"
    )

    none_model.eval()
    attention_model.eval()

    none_params = sum(
        p.numel()
        for p in none_model.parameters()
    )
    attention_params = sum(
        p.numel()
        for p in attention_model.parameters()
    )

    assert attention_params == none_params + 1

    aggregator = (
        attention_model
        .model
        .inner
        .history_aggregator
    )

    assert isinstance(
        aggregator,
        HistoryAttention,
    )

    assert torch.allclose(
        torch.sigmoid(aggregator.gate_logit),
        torch.tensor(
            0.5,
            device=device,
        ),
    )

    none_carry = none_model.initial_carry(batch)
    attention_carry = attention_model.initial_carry(batch)

    with torch.inference_mode():
        # Step 1: zero history -> exact identity.
        none_carry, _, _, none_out_1, _ = none_model(
            carry=none_carry,
            batch=batch,
            return_keys=["logits"],
        )

        attention_carry, _, _, attention_out_1, _ = attention_model(
            carry=attention_carry,
            batch=batch,
            return_keys=["logits"],
        )

        assert torch.equal(
            none_out_1["logits"],
            attention_out_1["logits"],
        )

        assert torch.equal(
            none_carry.inner_carry.z_H,
            attention_carry.inner_carry.z_H,
        )

        previous_z = (
            attention_carry
            .inner_carry
            .history_z_H[:, 0]
            .clone()
        )

        # Step 2 has exactly one previous state.
        # Attention context therefore equals that state exactly.
        none_carry, _, _, _, _ = none_model(
            carry=none_carry,
            batch=batch,
            return_keys=["logits"],
        )

        attention_carry, _, _, attention_out_2, _ = attention_model(
            carry=attention_carry,
            batch=batch,
            return_keys=["logits"],
        )

        expected_attention_z = (
            none_carry.inner_carry.z_H
            + previous_z
        ) / 2.0

        assert torch.allclose(
            attention_carry.inner_carry.z_H,
            expected_attention_z,
            rtol=1e-6,
            atol=1e-6,
        )

        assert torch.isfinite(
            attention_out_2["logits"]
        ).all()

        assert torch.equal(
            attention_carry.inner_carry.history_lengths,
            torch.full(
                (4,),
                2,
                dtype=torch.int32,
                device=device,
            ),
        )

    print("HISTORY ATTENTION TRM INTEGRATION PASS")


if __name__ == "__main__":
    test_history_attention_contract_cpu()
    test_history_attention_integration_cpu()
