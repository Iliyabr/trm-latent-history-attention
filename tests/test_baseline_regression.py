from pathlib import Path
import pytest
import yaml
import torch

from pretrain import PretrainConfig, create_model
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    cfg = yaml.safe_load(
        (ROOT / "config" / "cfg_baseline_v2.yaml").read_text(encoding="utf-8")
    )
    arch = yaml.safe_load(
        (ROOT / "config" / "arch" / "trm_cpu.yaml").read_text(encoding="utf-8")
    )

    cfg.pop("defaults", None)
    cfg.pop("hydra", None)
    cfg["arch"] = arch

    cfg["load_checkpoint"] = str(
        ROOT / "checkpoints" / "baseline-v2-smoke" / "step_250"
    )
    cfg["compile_model"] = False

    return PretrainConfig(**cfg)


def test_baseline_checkpoint_and_forward_cpu():
    config = load_config()
    device = torch.device("cpu")
    dataset_root = ROOT / "data" / "sudoku-baseline-v2"
    checkpoint = ROOT / "checkpoints" / "baseline-v2-smoke" / "step_250"
    if not (dataset_root / "dev" / "dataset.json").exists() or not checkpoint.exists():
        pytest.skip("optional Phase-1 baseline artifacts are not present")

    ds_cfg = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=[str(dataset_root)],
        global_batch_size=config.global_batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    dataset = PuzzleDataset(ds_cfg, split="dev")
    set_name, batch, effective_size = next(iter(dataset))

    assert set_name == "all"
    assert effective_size == 4
    assert batch["inputs"].shape == (4, 81)
    assert batch["labels"].shape == (4, 81)

    model, _, _ = create_model(
        config,
        dataset.metadata,
        rank=0,
        world_size=1,
        device=device,
    )

    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        carry = model.initial_carry(batch)
        carry, loss, metrics, outputs, all_finish = model(
            carry=carry,
            batch=batch,
            return_keys=["preds"],
        )

    assert torch.isfinite(loss)
    assert outputs["preds"].shape == (4, 81)

    for value in metrics.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all()

    print("P1-D BASELINE REGRESSION PASS")
