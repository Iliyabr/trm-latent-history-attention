"""Aggregate study artifacts and create publication-ready statistics/figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


VARIANT_ORDER = ("B0", "Gated", "B1", "B2", "B3", "P1", "P1ns", "P1nsMLP")
METRICS = (
    "exact_accuracy",
    "cell_accuracy",
    "incorrect_cells",
    "row_violations",
    "column_violations",
    "box_violations",
)


def sample_mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return math.nan, math.nan
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def paired_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired percentile bootstrap for mean(right-left)."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired bootstrap inputs must be non-empty with equal shape")
    differences = b - a
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    # Chunking avoids allocating samples x observations for large test sets.
    for start in range(0, samples, 1_000):
        count = min(1_000, samples - start)
        indices = rng.integers(0, differences.size, size=(count, differences.size))
        estimates[start : start + count] = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, (alpha, 1.0 - alpha))
    return float(differences.mean()), float(low), float(high)


def exact_mcnemar(left_correct: Sequence[bool], right_correct: Sequence[bool]) -> dict[str, Any]:
    """Two-sided exact McNemar test using the conditional Binomial(n, .5)."""
    left = np.asarray(left_correct, dtype=bool)
    right = np.asarray(right_correct, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("McNemar inputs must have equal shape")
    left_only = int(np.count_nonzero(left & ~right))
    right_only = int(np.count_nonzero(~left & right))
    discordant = left_only + right_only
    if discordant == 0:
        p = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
        p = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant": discordant,
        "p_value": float(p),
    }


def load_artifacts(root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for metadata_path in sorted(root.glob("**/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        examples_path = metadata_path.with_name("examples.jsonl")
        if not examples_path.exists():
            continue
        examples = [
            json.loads(line)
            for line in examples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        artifacts.append(
            {
                "path": str(metadata_path.parent),
                "metadata": metadata,
                "examples": examples,
                "intervention": metadata.get("intervention", {}).get("name", "clean"),
            }
        )
    return artifacts


def _seed_rows(artifacts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for artifact in artifacts:
        metadata = artifact["metadata"]
        metrics = metadata["metrics"]
        row = {
            "variant": metadata["variant"],
            "seed": int(metadata["seed"]),
            "intervention": artifact["intervention"],
            **{key: metrics.get(key, math.nan) for key in METRICS},
            "parameters": metadata.get("parameters", {}).get("total", math.nan),
            "wall_time_seconds": metadata.get("wall_time_seconds", math.nan),
            "peak_vram_bytes": metadata.get("peak_vram_bytes", math.nan),
            "throughput_examples_per_second": metadata.get(
                "throughput_examples_per_second", math.nan
            ),
        }
        for key, value in metadata.get("attention", {}).items():
            row[f"attention_{key}"] = value
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(artifacts: Sequence[Mapping[str, Any]], bootstrap_samples: int = 10_000) -> dict[str, Any]:
    seed_rows = _seed_rows(artifacts)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["variant"]), str(row["intervention"]))].append(row)

    summaries = []
    for (variant, intervention), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "variant": variant,
            "intervention": intervention,
            "seed_count": len(rows),
        }
        numeric = set().union(*(row.keys() for row in rows)) - {
            "variant",
            "seed",
            "intervention",
        }
        for metric in sorted(numeric):
            values = [float(row[metric]) for row in rows if metric in row and math.isfinite(float(row[metric]))]
            mean, std = sample_mean_std(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summaries.append(summary)

    clean = [a for a in artifacts if a["intervention"] == "clean"]
    by_variant_seed = {
        (a["metadata"]["variant"], int(a["metadata"]["seed"])): a for a in clean
    }
    comparisons = []
    comparison_pairs = [("B0", variant) for variant in VARIANT_ORDER[1:]]
    comparison_pairs.append(("B3", "P1"))
    for left_name, right_name in comparison_pairs:
        common_seeds = sorted(
            seed
            for (name, seed) in by_variant_seed
            if name == left_name and (right_name, seed) in by_variant_seed
        )
        if not common_seeds:
            continue
        left_exact: list[bool] = []
        right_exact: list[bool] = []
        left_cell: list[float] = []
        right_cell: list[float] = []
        for seed in common_seeds:
            left_examples = {
                row["example_id"]: row
                for row in by_variant_seed[(left_name, seed)]["examples"]
            }
            right_examples = {
                row["example_id"]: row
                for row in by_variant_seed[(right_name, seed)]["examples"]
            }
            for example_id in sorted(set(left_examples) & set(right_examples)):
                left_row, right_row = left_examples[example_id], right_examples[example_id]
                left_exact.append(bool(left_row["exact"]))
                right_exact.append(bool(right_row["exact"]))
                left_cell.append(float(left_row["cell_accuracy"]))
                right_cell.append(float(right_row["cell_accuracy"]))
        if not left_exact:
            continue
        exact_delta, exact_low, exact_high = paired_bootstrap_ci(
            left_exact, right_exact, samples=bootstrap_samples
        )
        cell_delta, cell_low, cell_high = paired_bootstrap_ci(
            left_cell, right_cell, samples=bootstrap_samples
        )
        comparisons.append(
            {
                "left": left_name,
                "right": right_name,
                "seeds": common_seeds,
                "paired_examples": len(left_exact),
                "exact_delta": exact_delta,
                "exact_ci95": [exact_low, exact_high],
                "cell_delta": cell_delta,
                "cell_ci95": [cell_low, cell_high],
                "mcnemar": exact_mcnemar(left_exact, right_exact),
            }
        )
    return {"seed_rows": seed_rows, "summaries": summaries, "comparisons": comparisons}


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def make_figures(artifacts: Sequence[Mapping[str, Any]], report: Mapping[str, Any], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create publication figures") from exc
    _style(plt)
    clean = [row for row in report["summaries"] if row["intervention"] == "clean"]
    clean.sort(key=lambda row: VARIANT_ORDER.index(row["variant"]) if row["variant"] in VARIANT_ORDER else 99)

    def bar_figure(filename: str, metric: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        names = [row["variant"] for row in clean]
        means = [row.get(f"{metric}_mean", math.nan) for row in clean]
        errors = [row.get(f"{metric}_std", 0.0) for row in clean]
        ax.bar(names, means, yerr=errors, capsize=3, color="#4472C4")
        ax.set_ylabel(ylabel)
        fig.savefig(output / filename)
        plt.close(fig)

    bar_figure("accuracy.pdf", "exact_accuracy", "Exact accuracy")
    bar_figure("compute.pdf", "wall_time_seconds", "Wall time (s)")

    interventions = [row for row in report["summaries"] if row["intervention"] != "clean"]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    if interventions:
        names = [row["intervention"] for row in interventions]
        ax.bar(names, [row["exact_accuracy_mean"] for row in interventions], color="#ED7D31")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel("Exact accuracy")
    else:
        ax.text(0.5, 0.5, "No intervention artifacts", ha="center", va="center")
        ax.set_axis_off()
    fig.savefig(output / "corruption.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    attention_keys = ("expected_lookback", "non_adjacent_mass", "entropy")
    p1 = next((row for row in clean if row["variant"] == "P1"), None)
    if p1 and any(f"attention_{key}_mean" in p1 for key in attention_keys):
        ax.bar(
            attention_keys,
            [p1.get(f"attention_{key}_mean", math.nan) for key in attention_keys],
            color="#70AD47",
        )
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.text(0.5, 0.5, "Attention weights not exposed", ha="center", va="center")
        ax.set_axis_off()
    fig.savefig(output / "attention.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    found_curve = False
    for artifact in artifacts:
        curve = artifact["metadata"].get("learning_curve")
        if not curve:
            continue
        found_curve = True
        ax.plot(curve["step"], curve["exact_accuracy"], label=artifact["metadata"]["variant"])
    if found_curve:
        ax.set_xlabel("Training step")
        ax.set_ylabel("Exact accuracy")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No learning-curve data in artifacts", ha="center", va="center")
        ax.set_axis_off()
    fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/study"))
    parser.add_argument("--output", type=Path, default=Path("results/study/analysis"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    artifacts = load_artifacts(args.input)
    if not artifacts:
        parser.error(f"no evaluation artifacts found under {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    report = aggregate(artifacts, bootstrap_samples=args.bootstrap_samples)
    _write_csv(args.output / "seed_results.csv", report["seed_rows"])
    _write_csv(args.output / "aggregate_results.csv", report["summaries"])
    (args.output / "seed_results.json").write_text(
        json.dumps(report["seed_rows"], indent=2), encoding="utf-8"
    )
    (args.output / "aggregate_results.json").write_text(
        json.dumps(report["summaries"], indent=2), encoding="utf-8"
    )
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.no_figures:
        make_figures(artifacts, report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
