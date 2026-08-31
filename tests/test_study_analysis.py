import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from experiments.analyze_results import (
    aggregate,
    exact_mcnemar,
    load_artifacts,
    main as analyze_main,
    paired_bootstrap_ci,
    sample_mean_std,
)
from experiments.evaluate_study import (
    _arch_name_from_defaults,
    _load_config,
    _resolve_arch_path,
    attention_statistics,
    example_metrics,
    history_intervention,
    intervention_specs,
    load_checkpoint_robust,
    sudoku_violations,
)


def _artifact(root: Path, variant: str, seed: int, exact: list[bool], cells: list[float]) -> None:
    directory = root / variant / f"seed_{seed}"
    directory.mkdir(parents=True)
    examples = [
        {
            "example_id": f"test:{index}",
            "exact": value,
            "cell_accuracy": cells[index],
            "incorrect_cells": int(not value),
            "sudoku_violations": {"row": 0, "column": 0, "box": 0},
        }
        for index, value in enumerate(exact)
    ]
    metadata = {
        "variant": variant,
        "seed": seed,
        "metrics": {
            "examples": len(examples),
            "exact_accuracy": sum(exact) / len(exact),
            "cell_accuracy": sum(cells) / len(cells),
            "incorrect_cells": sum(not value for value in exact) / len(exact),
            "row_violations": 0,
            "column_violations": 0,
            "box_violations": 0,
        },
        "parameters": {"total": 10, "trainable": 10},
        "wall_time_seconds": 1.0 + seed,
        "peak_vram_bytes": 0,
        "throughput_examples_per_second": 4.0,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (directory / "examples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in examples), encoding="utf-8"
    )


def test_statistics_are_exact_and_deterministic():
    assert sample_mean_std([1, 3]) == pytest.approx((2, 2**0.5))
    result = exact_mcnemar(
        [True, True, True, False, False],
        [False, False, False, True, False],
    )
    assert result["left_only_correct"] == 3
    assert result["right_only_correct"] == 1
    assert result["p_value"] == pytest.approx(0.625)

    first = paired_bootstrap_ci([0, 0, 1, 1], [0, 1, 1, 1], samples=500, seed=7)
    second = paired_bootstrap_ci([0, 0, 1, 1], [0, 1, 1, 1], samples=500, seed=7)
    assert first == second
    assert first[0] == pytest.approx(0.25)
    assert first[1] <= first[0] <= first[2]


def test_sudoku_and_attention_statistics():
    solved = [
        ((row * 3 + row // 3 + column) % 9) + 1
        for row in range(9)
        for column in range(9)
    ]
    assert sudoku_violations(solved) == {"row": 0, "column": 0, "box": 0}
    encoded = [digit + 1 for digit in solved]
    assert sudoku_violations(encoded) == {"row": 0, "column": 0, "box": 0}
    broken = solved.copy()
    broken[0] = broken[1]
    violations = sudoku_violations(broken)
    assert violations["row"] == 1
    assert violations["column"] >= 1
    assert violations["box"] == 1
    encoded_broken = [digit + 1 for digit in broken]
    encoded_metrics = example_metrics(
        torch.tensor(encoded_broken).numpy(), torch.tensor(encoded).numpy()
    )
    assert encoded_metrics["sudoku_violations"] == violations

    # H=4 deliberately exceeds T-valid=3: implementations must not mistake
    # the head axis for the temporal axis.
    weights = torch.tensor([[[[0.0, 0.25, 0.75]]], [[[1.0, 0.0, 0.0]]]])
    weights = weights.expand(-1, 4, -1, -1)
    stats = attention_statistics(weights, torch.tensor([3, 2]))
    assert stats["expected_lookback"] == pytest.approx(1.625)
    assert stats["non_adjacent_mass"] == pytest.approx(0.625)
    assert stats["entropy"] > 0
    stacked = weights.unsqueeze(0).unsqueeze(0).expand(2, 3, -1, -1, -1, -1)
    assert attention_statistics(stacked, torch.tensor([3, 2])) == pytest.approx(stats)


def test_checkpoint_compatibility_and_clear_intervention_error(tmp_path: Path):
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {"state_dict": {f"module.{key}": value for key, value in source.state_dict().items()}},
        checkpoint,
    )
    info = load_checkpoint_robust(target, checkpoint, torch.device("cpu"))
    assert info["matched_keys"] == 2
    assert torch.equal(source.weight, target.weight)

    with pytest.raises(RuntimeError, match="no supported hook"):
        with history_intervention(target, {"kind": "gaussian"}, seed=0):
            pass


def test_intervention_discovery_scope_and_variant_matrix():
    class SupportedInner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.payload = None

        @contextlib.contextmanager
        def history_intervention(self, payload, seed):
            self.payload = (payload, seed)
            yield
            self.payload = None

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = SupportedInner()
            self.history_aggregator = torch.nn.Identity()

    model = Wrapper()
    with history_intervention(model, {"kind": "gaussian"}, seed=17):
        assert model.inner.payload == ({"kind": "gaussian", "seed": 17}, 17)
    assert model.inner.payload is None

    for variant in ("B0", "B1", "B2", "B3", "P1", "P1ns"):
        specs = intervention_specs(variant)
        assert [item["sigma_rms"] for item in specs if item["kind"] == "gaussian"] == [
            0.05,
            0.10,
            0.20,
        ]
        deletions = [item for item in specs if item["kind"] == "delete"]
        assert len(deletions) == (2 if variant in {"P1", "P1ns"} else 0)


def test_nested_experiment_config_resolves_ancestor_arch(tmp_path: Path):
    arch = tmp_path / "config" / "arch" / "trm_test.yaml"
    arch.parent.mkdir(parents=True)
    arch.write_text("name: test", encoding="utf-8")
    nested_config = tmp_path / "config" / "experiment" / "study.yaml"
    nested_config.parent.mkdir()
    nested_config.write_text("", encoding="utf-8")
    assert _resolve_arch_path(nested_config, "trm_test") == arch
    assert _arch_name_from_defaults([{"/arch@arch": "trm_history_colab"}]) == "trm_history_colab"
    assert _arch_name_from_defaults([{"/arch": "trm_cpu"}]) == "trm_cpu"


def test_study_yaml_loads_through_evaluate_config_loader():
    root = Path(__file__).resolve().parents[1]
    colab = _load_config(root / "config" / "experiment" / "sudoku_study_colab.yaml")
    publication = _load_config(
        root / "config" / "experiment" / "sudoku_study_publication.yaml"
    )
    assert colab.arch.hidden_size == 256
    assert colab.arch.history_rank == 64
    assert colab.arch.puzzle_emb_ndim == 256
    assert colab.optimizer == "adamw"
    assert publication.arch.hidden_size == 512
    assert publication.max_runtime_minutes is None


def test_synthetic_artifacts_aggregate_and_write_outputs(tmp_path: Path):
    root = tmp_path / "artifacts"
    for seed in (0, 1):
        _artifact(root, "B0", seed, [True, False, False, True], [1.0, 0.8, 0.7, 1.0])
        _artifact(root, "B3", seed, [True, False, True, True], [1.0, 0.8, 1.0, 1.0])
        _artifact(root, "P1", seed, [True, True, False, True], [1.0, 1.0, 0.8, 1.0])

    artifacts = load_artifacts(root)
    report = aggregate(artifacts, bootstrap_samples=500)
    assert len(report["seed_rows"]) == 6
    p1 = next(row for row in report["summaries"] if row["variant"] == "P1")
    assert p1["exact_accuracy_mean"] == pytest.approx(0.75)
    comparison = next(
        item for item in report["comparisons"] if item["left"] == "B0" and item["right"] == "P1"
    )
    assert comparison["exact_delta"] == pytest.approx(0.25)
    assert comparison["paired_examples"] == 8
    assert any(
        item["left"] == "B3" and item["right"] == "P1"
        for item in report["comparisons"]
    )

    output = tmp_path / "analysis"
    assert analyze_main(
        [
            "--input",
            str(root),
            "--output",
            str(output),
            "--bootstrap-samples",
            "100",
        ]
    ) == 0
    assert (output / "seed_results.csv").exists()
    assert (output / "aggregate_results.csv").exists()
    assert json.loads((output / "seed_results.json").read_text())
    assert json.loads((output / "aggregate_results.json").read_text())
    assert json.loads((output / "analysis.json").read_text())["comparisons"]
    for name in ("accuracy", "compute", "corruption", "attention", "learning_curves"):
        assert (output / f"{name}.pdf").stat().st_size > 0


def test_evaluate_study_imports_pretrain_without_repo_pythonpath():
    repo = Path(__file__).resolve().parents[1]
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() != "PYTHONPATH"
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import evaluate_study, pretrain; print(pretrain.__file__)",
        ],
        cwd=str(repo / "experiments"),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "pretrain.py" in result.stdout.replace("\\", "/")
