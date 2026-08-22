from __future__ import annotations

import torch

from pretrain import create_model
from tests.test_history_interface import first_dev_batch, load_config


def test_uniform_history_integration_cpu():
    device = torch.device("cpu")

    none_cfg = load_config(
        history_enabled=True,
        history_aggregator="none",
    )
    uniform_cfg = load_config(
        history_enabled=True,
        history_aggregator="uniform",
    )

    metadata, batch = first_dev_batch(none_cfg)
    batch = {k: v.to(device) for k, v in batch.items()}

    none_model, _, _ = create_model(
        none_cfg,
        metadata,
        rank=0,
        world_size=1,
        device=device,
    )
    uniform_model, _, _ = create_model(
        uniform_cfg,
        metadata,
        rank=0,
        world_size=1,
        device=device,
    )

    none_model.eval()
    uniform_model.eval()

    # UniformMeanHistory is parameter-free, so both full models must
    # have exactly the same parameter count.
    none_params = sum(p.numel() for p in none_model.parameters())
    uniform_params = sum(p.numel() for p in uniform_model.parameters())
    assert none_params == uniform_params

    none_carry = none_model.initial_carry(batch)
    uniform_carry = uniform_model.initial_carry(batch)

    with torch.inference_mode():
        # Step 1: history is empty, so uniform aggregation must be identity.
        none_carry, _, _, none_out_1, _ = none_model(
            carry=none_carry,
            batch=batch,
            return_keys=["logits"],
        )
        uniform_carry, _, _, uniform_out_1, _ = uniform_model(
            carry=uniform_carry,
            batch=batch,
            return_keys=["logits"],
        )

        assert torch.equal(
            none_out_1["logits"],
            uniform_out_1["logits"],
        )
        assert torch.equal(
            none_carry.inner_carry.z_H,
            uniform_carry.inner_carry.z_H,
        )

        # Save the strictly previous state before step 2.
        previous_z = uniform_carry.inner_carry.history_z_H[:, 0].clone()

        # Step 2: both models begin from the same recurrent state.
        # Therefore the no-history model exposes the raw current z_H,
        # while the uniform model should return mean(current_z, previous_z).
        none_carry, _, _, _, _ = none_model(
            carry=none_carry,
            batch=batch,
            return_keys=["logits"],
        )
        uniform_carry, _, _, uniform_out_2, _ = uniform_model(
            carry=uniform_carry,
            batch=batch,
            return_keys=["logits"],
        )

        expected_uniform_z = (
            none_carry.inner_carry.z_H + previous_z
        ) / 2

        assert torch.allclose(
            uniform_carry.inner_carry.z_H,
            expected_uniform_z,
            rtol=1e-6,
            atol=1e-6,
        )

        assert torch.isfinite(uniform_out_2["logits"]).all()

        assert torch.equal(
            uniform_carry.inner_carry.history_lengths,
            torch.full(
                (4,),
                2,
                dtype=torch.int32,
                device=device,
            ),
        )

    print("UNIFORM HISTORY TRM INTEGRATION PASS")


if __name__ == "__main__":
    test_uniform_history_integration_cpu()

def test_recency_history_integration_cpu():
    device = torch.device("cpu")

    none_cfg = load_config(
        history_enabled=True,
        history_aggregator="none",
    )
    recency_cfg = load_config(
        history_enabled=True,
        history_aggregator="recency",
    )

    metadata, batch = first_dev_batch(none_cfg)
    batch = {k: v.to(device) for k, v in batch.items()}

    none_model, _, _ = create_model(
        none_cfg,
        metadata,
        rank=0,
        world_size=1,
        device=device,
    )
    recency_model, _, _ = create_model(
        recency_cfg,
        metadata,
        rank=0,
        world_size=1,
        device=device,
    )

    none_model.eval()
    recency_model.eval()

    none_params = sum(p.numel() for p in none_model.parameters())
    recency_params = sum(p.numel() for p in recency_model.parameters())
    assert none_params == recency_params

    none_carry = none_model.initial_carry(batch)
    recency_carry = recency_model.initial_carry(batch)

    with torch.inference_mode():
        # Step 1: no previous history, so recency aggregation is identity.
        none_carry, _, _, none_out_1, _ = none_model(
            carry=none_carry,
            batch=batch,
            return_keys=["logits"],
        )
        recency_carry, _, _, recency_out_1, _ = recency_model(
            carry=recency_carry,
            batch=batch,
            return_keys=["logits"],
        )

        assert torch.equal(
            none_out_1["logits"],
            recency_out_1["logits"],
        )
        assert torch.equal(
            none_carry.inner_carry.z_H,
            recency_carry.inner_carry.z_H,
        )

        previous_z = recency_carry.inner_carry.history_z_H[:, 0].clone()

        # Step 2: newest historical state has weight 0.5,
        # current state has weight 1.0.
        none_carry, _, _, _, _ = none_model(
            carry=none_carry,
            batch=batch,
            return_keys=["logits"],
        )
        recency_carry, _, _, recency_out_2, _ = recency_model(
            carry=recency_carry,
            batch=batch,
            return_keys=["logits"],
        )

        expected_recency_z = (
            none_carry.inner_carry.z_H + 0.5 * previous_z
        ) / 1.5

        assert torch.allclose(
            recency_carry.inner_carry.z_H,
            expected_recency_z,
            rtol=1e-6,
            atol=1e-6,
        )

        assert torch.isfinite(recency_out_2["logits"]).all()

        assert torch.equal(
            recency_carry.inner_carry.history_lengths,
            torch.full(
                (4,),
                2,
                dtype=torch.int32,
                device=device,
            ),
        )

    print("RECENCY HISTORY TRM INTEGRATION PASS")

def test_gated_history_integration_cpu():
    device = torch.device("cpu")

    none_cfg = load_config(
        history_enabled=True,
        history_aggregator="none",
    )
    gated_cfg = load_config(
        history_enabled=True,
        history_aggregator="gated",
    )

    # Old vanilla checkpoints do not contain the new gate parameter.
    # For this integration test, create both models fresh and synchronize
    # all shared parameters explicitly.
    none_cfg.load_checkpoint = None
    gated_cfg.load_checkpoint = None

    metadata, batch = first_dev_batch(none_cfg)
    batch = {k: v.to(device) for k, v in batch.items()}

    none_model, _, _ = create_model(
        none_cfg,
        metadata,
        rank=0,
        world_size=1,
        device=device,
    )
    gated_model, _, _ = create_model(
        gated_cfg,
        metadata,
        rank=0,
        world_size=1,
        device=device,
    )

    missing, unexpected = gated_model.load_state_dict(
        none_model.state_dict(),
        strict=False,
    )

    assert len(unexpected) == 0
    assert len(missing) == 1
    assert missing[0].endswith("history_aggregator.gate_logit")

    none_model.eval()
    gated_model.eval()

    none_params = sum(p.numel() for p in none_model.parameters())
    gated_params = sum(p.numel() for p in gated_model.parameters())

    assert gated_params == none_params + 1

    aggregator = gated_model.model.inner.history_aggregator
    assert aggregator.gate_logit.numel() == 1
    assert torch.allclose(
        torch.sigmoid(aggregator.gate_logit),
        torch.tensor(0.5, device=device),
    )

    none_carry = none_model.initial_carry(batch)
    gated_carry = gated_model.initial_carry(batch)

    with torch.inference_mode():
        # Step 1: empty history means exact identity.
        none_carry, _, _, none_out_1, _ = none_model(
            carry=none_carry,
            batch=batch,
            return_keys=["logits"],
        )
        gated_carry, _, _, gated_out_1, _ = gated_model(
            carry=gated_carry,
            batch=batch,
            return_keys=["logits"],
        )

        assert torch.equal(
            none_out_1["logits"],
            gated_out_1["logits"],
        )
        assert torch.equal(
            none_carry.inner_carry.z_H,
            gated_carry.inner_carry.z_H,
        )

        previous_z = gated_carry.inner_carry.history_z_H[:, 0].clone()

        # Step 2: initial gate is 0.5 and mean history contains z_1 only.
        none_carry, _, _, _, _ = none_model(
            carry=none_carry,
            batch=batch,
            return_keys=["logits"],
        )
        gated_carry, _, _, gated_out_2, _ = gated_model(
            carry=gated_carry,
            batch=batch,
            return_keys=["logits"],
        )

        expected_gated_z = (
            none_carry.inner_carry.z_H + previous_z
        ) / 2.0

        assert torch.allclose(
            gated_carry.inner_carry.z_H,
            expected_gated_z,
            rtol=1e-6,
            atol=1e-6,
        )

        assert torch.isfinite(gated_out_2["logits"]).all()

    print("GATED HISTORY TRM INTEGRATION PASS")

def test_last_state_history_integration_cpu():
    device = torch.device("cpu")

    none_cfg = load_config(
        history_enabled=True,
        history_aggregator="none",
    )
    last_cfg = load_config(
        history_enabled=True,
        history_aggregator="last_state",
    )

    metadata, batch = first_dev_batch(none_cfg)
    batch = {k: v.to(device) for k, v in batch.items()}

    none_model, _, _ = create_model(
        none_cfg,
        metadata,
        rank=0,
        world_size=1,
        device=device,
    )
    last_model, _, _ = create_model(
        last_cfg,
        metadata,
        rank=0,
        world_size=1,
        device=device,
    )

    none_model.eval()
    last_model.eval()

    none_params = sum(p.numel() for p in none_model.parameters())
    last_params = sum(p.numel() for p in last_model.parameters())
    assert none_params == last_params

    none_carry = none_model.initial_carry(batch)
    last_carry = last_model.initial_carry(batch)

    with torch.inference_mode():
        # Step 1: no valid history, so behavior must be identity.
        none_carry, _, _, none_out_1, _ = none_model(
            carry=none_carry,
            batch=batch,
            return_keys=["logits"],
        )
        last_carry, _, _, last_out_1, _ = last_model(
            carry=last_carry,
            batch=batch,
            return_keys=["logits"],
        )

        assert torch.equal(
            none_out_1["logits"],
            last_out_1["logits"],
        )
        assert torch.equal(
            none_carry.inner_carry.z_H,
            last_carry.inner_carry.z_H,
        )

        previous_z = last_carry.inner_carry.history_z_H[:, 0].clone()

        # Step 2: use only the most recent previous state with weight 0.5.
        none_carry, _, _, _, _ = none_model(
            carry=none_carry,
            batch=batch,
            return_keys=["logits"],
        )
        last_carry, _, _, last_out_2, _ = last_model(
            carry=last_carry,
            batch=batch,
            return_keys=["logits"],
        )

        expected_last_z = (
            none_carry.inner_carry.z_H + 0.5 * previous_z
        ) / 1.5

        assert torch.allclose(
            last_carry.inner_carry.z_H,
            expected_last_z,
            rtol=1e-6,
            atol=1e-6,
        )

        assert torch.isfinite(last_out_2["logits"]).all()

        assert torch.equal(
            last_carry.inner_carry.history_lengths,
            torch.full(
                (4,),
                2,
                dtype=torch.int32,
                device=device,
            ),
        )

    print("LAST-STATE HISTORY TRM INTEGRATION PASS")
