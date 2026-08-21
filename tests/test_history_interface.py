from pathlib import Path
import yaml
import torch

from pretrain import PretrainConfig, create_model
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


ROOT = Path(__file__).resolve().parents[1]


def load_config(history_enabled: bool) -> PretrainConfig:
    cfg = yaml.safe_load(
        (ROOT / "config" / "cfg_baseline_v2.yaml").read_text(encoding="utf-8")
    )
    arch = yaml.safe_load(
        (ROOT / "config" / "arch" / "trm_cpu.yaml").read_text(encoding="utf-8")
    )

    cfg.pop("defaults", None)
    cfg.pop("hydra", None)

    arch["halt_max_steps"] = 4
    arch["history_enabled"] = history_enabled

    cfg["arch"] = arch
    cfg["load_checkpoint"] = str(
        ROOT / "checkpoints" / "baseline-v2-4step-40ep" / "step_10000"
    )
    cfg["compile_model"] = False

    return PretrainConfig(**cfg)


def first_dev_batch(config: PretrainConfig):
    ds_cfg = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=[str(ROOT / "data" / "sudoku-baseline-v2")],
        global_batch_size=config.global_batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    dataset = PuzzleDataset(ds_cfg, split="dev")
    _, batch, _ = next(iter(dataset))
    return dataset.metadata, batch


def test_history_interface_cpu():
    device = torch.device("cpu")

    base_cfg = load_config(history_enabled=False)
    hist_cfg = load_config(history_enabled=True)

    metadata, batch = first_dev_batch(base_cfg)
    batch = {k: v.to(device) for k, v in batch.items()}

    base_model, _, _ = create_model(
        base_cfg, metadata, rank=0, world_size=1, device=device
    )
    hist_model, _, _ = create_model(
        hist_cfg, metadata, rank=0, world_size=1, device=device
    )

    base_model.eval()
    hist_model.eval()

    for p_base, p_hist in zip(
        base_model.parameters(), hist_model.parameters()
    ):
        assert torch.equal(p_base, p_hist)

    base_carry = base_model.initial_carry(batch)
    hist_carry = hist_model.initial_carry(batch)

    with torch.inference_mode():
        for step in range(1, 5):
            base_carry, _, _, base_out, _ = base_model(
                carry=base_carry,
                batch=batch,
                return_keys=["logits"],
            )

            hist_carry, _, _, hist_out, _ = hist_model(
                carry=hist_carry,
                batch=batch,
                return_keys=["logits"],
            )

            assert torch.equal(
                base_out["logits"],
                hist_out["logits"],
            )

            inner = hist_carry.inner_carry

            assert inner.history_z_H is not None
            assert inner.history_lengths is not None

            assert inner.history_z_H.shape == (
                4,
                4,
                81,
                64,
            )

            assert torch.equal(
                inner.history_lengths,
                torch.full(
                    (4,),
                    step,
                    dtype=torch.int32,
                    device=device,
                ),
            )

            assert not inner.history_z_H.requires_grad

            for batch_i in range(4):
                slot = int(inner.history_lengths[batch_i].item()) - 1
                assert torch.equal(
                    inner.history_z_H[batch_i, slot],
                    inner.z_H[batch_i],
                )

    print("P2-A HISTORY INTERFACE PASS")


if __name__ == "__main__":
    test_history_interface_cpu()

def test_history_reset_isolation_cpu():
    device = torch.device("cpu")
    config = load_config(history_enabled=True)

    ds_cfg = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=[str(ROOT / "data" / "sudoku-baseline-v2")],
        global_batch_size=config.global_batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    dataset = PuzzleDataset(ds_cfg, split="dev")
    iterator = iter(dataset)

    _, batch1, _ = next(iterator)
    _, batch2, _ = next(iterator)

    batch1 = {k: v.to(device) for k, v in batch1.items()}
    batch2 = {k: v.to(device) for k, v in batch2.items()}

    model, _, _ = create_model(
        config,
        dataset.metadata,
        rank=0,
        world_size=1,
        device=device,
    )
    model.eval()

    carry = model.initial_carry(batch1)

    with torch.inference_mode():
        # Build two steps of history for all four slots.
        for _ in range(2):
            carry, _, _, _, _ = model(
                carry=carry,
                batch=batch1,
                return_keys=[],
            )

        assert torch.equal(
            carry.inner_carry.history_lengths,
            torch.full((4,), 2, dtype=torch.int32, device=device),
        )

        # Simulate batch-slot reuse:
        # slots 0 and 2 start new puzzles, slots 1 and 3 continue.
        carry.halted = torch.tensor(
            [True, False, True, False],
            dtype=torch.bool,
            device=device,
        )

        carry, _, _, _, _ = model(
            carry=carry,
            batch=batch2,
            return_keys=[],
        )

        inner = carry.inner_carry

        assert torch.equal(
            inner.history_lengths,
            torch.tensor(
                [1, 3, 1, 3],
                dtype=torch.int32,
                device=device,
            ),
        )

        # Reset slots must contain only their newly computed state.
        for batch_i in (0, 2):
            assert torch.equal(
                inner.history_z_H[batch_i, 0],
                inner.z_H[batch_i],
            )
            assert torch.count_nonzero(
                inner.history_z_H[batch_i, 1:]
            ).item() == 0

        # Continuing slots retain three-step histories.
        for batch_i in (1, 3):
            assert torch.equal(
                inner.history_z_H[batch_i, 2],
                inner.z_H[batch_i],
            )

    print("P2-A HISTORY RESET ISOLATION PASS")

def test_no_history_aggregator_contract_cpu():
    device = torch.device("cpu")

    base_cfg = load_config(history_enabled=False)
    hist_cfg = load_config(history_enabled=True)

    metadata, batch = first_dev_batch(base_cfg)
    batch = {k: v.to(device) for k, v in batch.items()}

    base_model, _, _ = create_model(
        base_cfg, metadata, rank=0, world_size=1, device=device
    )
    hist_model, _, _ = create_model(
        hist_cfg, metadata, rank=0, world_size=1, device=device
    )

    # The identity history path must add zero trainable parameters.
    base_params = sum(p.numel() for p in base_model.parameters())
    hist_params = sum(p.numel() for p in hist_model.parameters())

    assert base_params == hist_params

    aggregator = hist_model.model.inner.history_aggregator
    assert aggregator.__class__.__name__ == "NoHistoryAggregator"
    assert sum(p.numel() for p in aggregator.parameters()) == 0

    print("P2-B NO-HISTORY CONTRACT PASS")
