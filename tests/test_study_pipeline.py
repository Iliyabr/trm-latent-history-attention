from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dataset.build_sudoku_baseline_v2 import (
    assert_no_leakage,
    canonical_hash,
    identity_records,
    write_split,
)
from experiments.run_study import VARIANTS, command_for, main as run_study_main
from pretrain import PretrainConfig
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


ROOT = Path(__file__).resolve().parents[1]


def _latin_board(offset: int) -> np.ndarray:
    return np.array(
        [
            [((row * 3 + row // 3 + column + offset) % 9) + 1 for column in range(9)]
            for row in range(9)
        ],
        dtype=np.uint8,
    )


def test_identity_hashes_and_leakage_detection():
    unique = [
        _latin_board(0),
        np.ascontiguousarray(_latin_board(0).T),
        np.ascontiguousarray(np.flipud(_latin_board(0))),
    ]
    records = {
        "train": identity_records(unique, unique, [0]),
        "dev": identity_records(unique, unique, [1]),
        "test": identity_records(unique, unique, [2]),
    }
    assert_no_leakage(records)
    assert canonical_hash(unique[0], unique[0]) != canonical_hash(unique[1], unique[1])

    leaked = {
        "train": identity_records(unique, unique, [0]),
        "dev": identity_records(unique, unique, [0]),
        "test": identity_records(unique, unique, [2]),
    }
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(leaked)

    # Cyclic digit shifts are the same geometry; canonical hashes must catch them.
    digit_shift = {
        "train": identity_records([_latin_board(0)], [_latin_board(0)], [0]),
        "dev": identity_records([_latin_board(1)], [_latin_board(1)], [0]),
        "test": identity_records([_latin_board(2)], [_latin_board(2)], [0]),
    }
    with pytest.raises(AssertionError, match="canonical_sha256 leakage"):
        assert_no_leakage(digit_shift)


def test_write_split_is_readable_by_puzzle_dataset(tmp_path: Path):
    boards = [_latin_board(0), _latin_board(4)]
    write_split(tmp_path, "train", boards, boards, [0, 1], num_aug=0, augmentation_seed=0)
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=0,
            dataset_paths=[str(tmp_path)],
            global_batch_size=2,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="train",
    )
    set_name, batch, size = next(iter(dataset))
    assert set_name == "all"
    assert size == 2
    assert batch["inputs"].shape == (2, 81)
    assert int(dataset.metadata.vocab_size) == 11


def test_run_study_dry_run_emits_all_fifteen_jobs():
    buffer = StringIO()
    with patch("sys.stdout", buffer):
        assert run_study_main(["suite", "--preset", "colab", "--dry-run"]) == 0
    text = buffer.getvalue()
    for variant in ("B0", "B1", "B2", "B3", "P1"):
        for seed in (0, 1, 2):
            assert f"run_name={variant}-seed{seed}" in text
    command, run_dir = command_for(
        "P1", 0, "colab", Path("outputs/study"), None, []
    )
    assert command[1] == "-u"
    assert command[2] == "pretrain.py"
    assert "arch.history_mode=P1" in command
    assert run_dir.name == "P1-seed0"


def test_run_study_canonical_suite_is_four_models():
    buffer = StringIO()
    with patch("sys.stdout", buffer):
        assert run_study_main(["suite", "--preset", "canonical", "--dry-run"]) == 0
    text = buffer.getvalue()
    for variant in ("B0", "Gated", "P1", "B3"):
        assert f"run_name={variant}-seed0" in text
    assert "run_name=B1-seed0" not in text
    assert "experiment/sudoku_study_canonical" in text


def test_hydra_study_presets_compose():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        colab = compose(
            config_name="experiment/sudoku_study_colab",
            overrides=["arch.history_mode=P1", "arch.history_enabled=true"],
        )
        publication = compose(config_name="experiment/sudoku_study_publication")
        heavy = compose(config_name="experiment/sudoku_study_colab_heavy")
        canonical = compose(config_name="experiment/sudoku_study_canonical")
    colab_cfg = PretrainConfig(**OmegaConf.to_container(colab, resolve=True))
    publication_cfg = PretrainConfig(**OmegaConf.to_container(publication, resolve=True))
    heavy_cfg = PretrainConfig(**OmegaConf.to_container(heavy, resolve=True))
    canonical_cfg = PretrainConfig(**OmegaConf.to_container(canonical, resolve=True))
    assert colab_cfg.arch.hidden_size == 256
    assert colab_cfg.arch.history_rank == 64
    assert colab_cfg.arch.H_cycles == 2
    assert colab_cfg.arch.L_cycles == 4
    assert colab_cfg.max_runtime_minutes == 55
    assert colab_cfg.optimizer == "adamw"
    assert publication_cfg.arch.hidden_size == 512
    assert publication_cfg.arch.history_rank == 128
    assert publication_cfg.max_runtime_minutes is None
    assert heavy_cfg.arch.hidden_size == 256
    assert heavy_cfg.epochs == 1536
    assert heavy_cfg.epochs % heavy_cfg.eval_interval == 0
    assert heavy_cfg.max_runtime_minutes == 120
    assert heavy_cfg.compile_model is False
    assert canonical_cfg.arch.hidden_size == 256
    assert canonical_cfg.arch.H_cycles == 3
    assert canonical_cfg.arch.L_cycles == 6
    assert canonical_cfg.arch.L_layers == 2
    assert canonical_cfg.arch.halt_max_steps == 6
    assert canonical_cfg.arch.history_rank == 64
    assert canonical_cfg.arch.history_window == 0
    assert canonical_cfg.arch.forward_dtype == "float32"
    assert canonical_cfg.global_batch_size == 32
    assert canonical_cfg.optimizer == "adamw"
    assert canonical_cfg.compile_model is False
    assert canonical_cfg.max_runtime_minutes is None
