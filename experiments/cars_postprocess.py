"""Training-free CARS / Confidence / Oracle trajectory selection for frozen Vanilla TRM."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.analyze_results import exact_mcnemar, paired_bootstrap_ci, sample_mean_std

# Reuse canonical Sudoku metric helpers (no training code paths).
from experiments.evaluate_study import example_metrics

ACT_STEPS = 6
TOKEN_EMPTY = 1  # encoded blank clue (raw digit 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens_to_digits(tokens: Sequence[int]) -> np.ndarray:
    """Model tokens (1=empty, 2..10 = digits 1..9) -> digit grid 0..9."""
    values = np.asarray(tokens, dtype=np.int64)
    if values.size != 81:
        raise ValueError("expected 81 Sudoku cells")
    return values - 1


def clue_mismatch_count(inputs: Sequence[int], predictions: Sequence[int]) -> int:
    """Givens are encoded inputs with token > TOKEN_EMPTY."""
    inp = np.asarray(inputs, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    given = inp > TOKEN_EMPTY
    return int(np.count_nonzero(given & (pred != inp)))


def _duplicate_excess_units(grid: np.ndarray, axis: int) -> int:
    total = 0
    if axis == 0:
        units = [grid[r, :] for r in range(9)]
    elif axis == 1:
        units = [grid[:, c] for c in range(9)]
    else:
        units = [
            grid[r : r + 3, c : c + 3].reshape(-1)
            for r in range(0, 9, 3)
            for c in range(0, 9, 3)
        ]
    for unit in units:
        for digit in range(1, 10):
            count = int(np.count_nonzero(unit == digit))
            total += max(count - 1, 0)
    return total


def structural_violations(predictions: Sequence[int]) -> dict[str, int]:
    grid = tokens_to_digits(predictions).reshape(9, 9)
    row = _duplicate_excess_units(grid, 0)
    column = _duplicate_excess_units(grid, 1)
    box = _duplicate_excess_units(grid, 2)
    return {
        "row_duplicate_excess": row,
        "column_duplicate_excess": column,
        "box_duplicate_excess": box,
        "structural_violations": row + column + box,
    }


def mean_token_confidence(logits: np.ndarray) -> float:
    """Mean over positions of max softmax probability."""
    if logits.ndim != 2:
        raise ValueError("logits must be [seq, vocab]")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    return float(probs.max(axis=-1).mean())


def act_step_metrics(
    inputs: Sequence[int],
    labels: Sequence[int],
    predictions: Sequence[int],
    logits: np.ndarray | None,
) -> dict[str, Any]:
    label_arr = np.asarray(labels, dtype=np.int64)
    pred_arr = np.asarray(predictions, dtype=np.int64)
    metrics = example_metrics(pred_arr, label_arr)
    struct = structural_violations(predictions)
    return {
        "predictions": pred_arr.tolist(),
        "exact": bool(metrics["exact"]),
        "cell_accuracy": float(metrics["cell_accuracy"]),
        "incorrect_cells": int(metrics["incorrect_cells"]),
        "clue_mismatch_count": clue_mismatch_count(inputs, predictions),
        **struct,
        "mean_confidence": mean_token_confidence(logits) if logits is not None else math.nan,
    }


def _later_tiebreak(current: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    """Return True if candidate should replace current under 'pick later step' rule."""
    return int(candidate["act_step"]) > int(current["act_step"])


def select_confidence(candidates: Sequence[Mapping[str, Any]]) -> int:
    best_idx = 0
    for idx in range(1, len(candidates)):
        cur = candidates[best_idx]
        cand = candidates[idx]
        if cand["mean_confidence"] > cur["mean_confidence"]:
            best_idx = idx
        elif cand["mean_confidence"] == cur["mean_confidence"] and _later_tiebreak(cur, cand):
            best_idx = idx
    return best_idx


def select_cars(candidates: Sequence[Mapping[str, Any]]) -> int:
    best_idx = 0
    for idx in range(1, len(candidates)):
        cur = candidates[best_idx]
        cand = candidates[idx]
        if cand["clue_mismatch_count"] < cur["clue_mismatch_count"]:
            best_idx = idx
            continue
        if cand["clue_mismatch_count"] > cur["clue_mismatch_count"]:
            continue
        if cand["structural_violations"] < cur["structural_violations"]:
            best_idx = idx
            continue
        if cand["structural_violations"] > cur["structural_violations"]:
            continue
        if cand["mean_confidence"] > cur["mean_confidence"]:
            best_idx = idx
            continue
        if cand["mean_confidence"] == cur["mean_confidence"] and _later_tiebreak(cur, cand):
            best_idx = idx
    return best_idx


def select_oracle(candidates: Sequence[Mapping[str, Any]]) -> int:
    """Diagnostic only — uses per-step exact/cell computed with labels."""
    best_idx = 0
    for idx in range(1, len(candidates)):
        cur = candidates[best_idx]
        cand = candidates[idx]
        if cand["exact"] and not cur["exact"]:
            best_idx = idx
            continue
        if cand["exact"] != cur["exact"]:
            continue
        if cand["cell_accuracy"] > cur["cell_accuracy"]:
            best_idx = idx
            continue
        if cand["cell_accuracy"] == cur["cell_accuracy"] and _later_tiebreak(cur, cand):
            best_idx = idx
    return best_idx


def summarize_method(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    count = max(len(rows), 1)
    return {
        "exact_accuracy": sum(bool(r[f"{prefix}_exact"]) for r in rows) / count,
        "cell_accuracy": sum(float(r[f"{prefix}_cell_accuracy"]) for r in rows) / count,
        "incorrect_cells": sum(int(r[f"{prefix}_incorrect_cells"]) for r in rows) / count,
        "exact_count": sum(bool(r[f"{prefix}_exact"]) for r in rows),
    }


def paired_cell_stats(
    final_cells: Sequence[float],
    selected_cells: Sequence[float],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    deltas = np.asarray(selected_cells, dtype=float) - np.asarray(final_cells, dtype=float)
    positive = int(np.count_nonzero(deltas > 0))
    negative = int(np.count_nonzero(deltas < 0))
    zero = int(np.count_nonzero(deltas == 0))
    result: dict[str, Any] = {
        "mean_delta_pp": float(deltas.mean() * 100.0),
        "median_delta_pp": float(np.median(deltas) * 100.0),
        "sd_delta_pp": float(deltas.std(ddof=1) * 100.0) if deltas.size > 1 else 0.0,
        "fraction_positive": positive / max(len(deltas), 1),
        "fraction_negative": negative / max(len(deltas), 1),
        "fraction_zero": zero / max(len(deltas), 1),
    }
    if deltas.size > 0:
        mean, low, high = paired_bootstrap_ci(
            final_cells, selected_cells, samples=bootstrap_samples, seed=seed
        )
        result["bootstrap_mean_delta"] = mean
        result["bootstrap_ci95_pp"] = [low * 100.0, high * 100.0]
    else:
        result["bootstrap_ci95_pp"] = "MISSING"
    return result


def selected_step_histogram(selected_steps: Sequence[int]) -> dict[str, float]:
    counts = np.zeros(ACT_STEPS, dtype=float)
    for step in selected_steps:
        idx = int(step) - 1
        if 0 <= idx < ACT_STEPS:
            counts[idx] += 1
    total = max(len(selected_steps), 1)
    return {str(i + 1): float(counts[i] / total) for i in range(ACT_STEPS)}


def analyze_seed_rows(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    final = summarize_method(rows, "final")
    conf = summarize_method(rows, "confidence")
    cars = summarize_method(rows, "cars")
    oracle = summarize_method(rows, "oracle")

    final_exact = [bool(r["final_exact"]) for r in rows]
    cars_exact = [bool(r["cars_exact"]) for r in rows]
    conf_exact = [bool(r["confidence_exact"]) for r in rows]

    earlier_exact_lost = [
        r
        for r in rows
        if any(r["act_steps"][i]["exact"] for i in range(ACT_STEPS - 1))
        and not r["final_exact"]
    ]
    cars_recovered = [
        r for r in earlier_exact_lost if r["cars_exact"]
    ]
    cars_damage = [
        r for r in rows if r["final_exact"] and not r["cars_exact"]
    ]

    oracle_any_exact = sum(
        any(step["exact"] for step in r["act_steps"]) for r in rows
    )

    return {
        "seed": seed,
        "examples": len(rows),
        "final": final,
        "confidence": {
            **conf,
            "delta_exact_vs_final": conf["exact_accuracy"] - final["exact_accuracy"],
            "delta_cell_pp_vs_final": (conf["cell_accuracy"] - final["cell_accuracy"]) * 100.0,
        },
        "cars": {
            **cars,
            "delta_exact_vs_final": cars["exact_accuracy"] - final["exact_accuracy"],
            "delta_cell_pp_vs_final": (cars["cell_accuracy"] - final["cell_accuracy"]) * 100.0,
        },
        "oracle_diagnostic": {
            **oracle,
            "fraction_any_step_exact": oracle_any_exact / max(len(rows), 1),
            "label": "ORACLE DIAGNOSTIC - NOT AN INFERENCE METHOD",
        },
        "exact_transitions": {
            "final_wrong_cars_correct": int(np.count_nonzero(~np.asarray(final_exact) & np.asarray(cars_exact))),
            "final_correct_cars_wrong": int(np.count_nonzero(np.asarray(final_exact) & ~np.asarray(cars_exact))),
            "mcnemar_cars_vs_final": exact_mcnemar(final_exact, cars_exact),
            "final_wrong_confidence_correct": int(
                np.count_nonzero(~np.asarray(final_exact) & np.asarray(conf_exact))
            ),
            "final_correct_confidence_wrong": int(
                np.count_nonzero(np.asarray(final_exact) & ~np.asarray(conf_exact))
            ),
            "mcnemar_confidence_vs_final": exact_mcnemar(final_exact, conf_exact),
        },
        "cell_deltas": {
            "cars_vs_final": paired_cell_stats(
                [r["final_cell_accuracy"] for r in rows],
                [r["cars_cell_accuracy"] for r in rows],
            ),
            "confidence_vs_final": paired_cell_stats(
                [r["final_cell_accuracy"] for r in rows],
                [r["confidence_cell_accuracy"] for r in rows],
            ),
        },
        "failure_modes": {
            "earlier_exact_final_wrong_count": len(earlier_exact_lost),
            "earlier_exact_final_wrong_fraction": len(earlier_exact_lost) / max(len(rows), 1),
            "cars_recovery_count": len(cars_recovered),
            "cars_recovery_rate": len(cars_recovered) / max(len(earlier_exact_lost), 1),
            "cars_damage_count": len(cars_damage),
            "cars_damage_rate": len(cars_damage) / max(len(rows), 1),
        },
        "selected_step_distribution": {
            "cars": selected_step_histogram([int(r["cars_act_step"]) for r in rows]),
            "confidence": selected_step_histogram([int(r["confidence_act_step"]) for r in rows]),
        },
        "mean_selected_act_step": {
            "cars": float(np.mean([r["cars_act_step"] for r in rows])) if rows else math.nan,
            "confidence": float(np.mean([r["confidence_act_step"] for r in rows])) if rows else math.nan,
        },
    }


def aggregate_seeds(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def agg(metric: str, field: str) -> dict[str, float]:
        values = [float(s[metric][field]) for s in per_seed]
        mean, std = sample_mean_std(values)
        return {"mean": mean, "std": std, "values": values}

    return {
        "seeds": [int(s["seed"]) for s in per_seed],
        "final_exact": agg("final", "exact_accuracy"),
        "confidence_exact": agg("confidence", "exact_accuracy"),
        "cars_exact": agg("cars", "exact_accuracy"),
        "final_cell": agg("final", "cell_accuracy"),
        "confidence_cell": agg("confidence", "cell_accuracy"),
        "cars_cell": agg("cars", "cell_accuracy"),
        "delta_exact_confidence_vs_final": {
            "values": [s["confidence"]["delta_exact_vs_final"] for s in per_seed],
            **dict(
                zip(
                    ("mean", "std"),
                    sample_mean_std([s["confidence"]["delta_exact_vs_final"] for s in per_seed]),
                )
            ),
        },
        "delta_exact_cars_vs_final": {
            "values": [s["cars"]["delta_exact_vs_final"] for s in per_seed],
            **dict(
                zip(
                    ("mean", "std"),
                    sample_mean_std([s["cars"]["delta_exact_vs_final"] for s in per_seed]),
                )
            ),
        },
        "delta_cell_pp_cars_vs_final": {
            "values": [s["cars"]["delta_cell_pp_vs_final"] for s in per_seed],
            **dict(
                zip(
                    ("mean", "std"),
                    sample_mean_std([s["cars"]["delta_cell_pp_vs_final"] for s in per_seed]),
                )
            ),
        },
    }


def make_cars_figures(report: Mapping[str, Any], figure_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for CARS figures") from exc

    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    per_seed = report["per_seed_summary"]
    seeds = [int(s["seed"]) for s in per_seed]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )

    # Figure A
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    cars_delta = [s["cars"]["delta_cell_pp_vs_final"] for s in per_seed]
    conf_delta = [s["confidence"]["delta_cell_pp_vs_final"] for s in per_seed]
    x = np.arange(len(seeds))
    width = 0.35
    axes[0].bar(x - width / 2, conf_delta, width, label="Confidence", color="#A5A5A5")
    axes[0].bar(x + width / 2, cars_delta, width, label="CARS", color="#4472C4")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, [str(s) for s in seeds])
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Δ cell accuracy vs Final (pp)")
    axes[0].legend(frameon=False)

    methods = ("final", "confidence", "cars")
    labels = ("Final", "Confidence", "CARS")
    colors = ("#595959", "#A5A5A5", "#4472C4")
    for i, seed_summary in enumerate(per_seed):
        counts = [
            seed_summary["final"]["exact_count"],
            seed_summary["confidence"]["exact_count"],
            seed_summary["cars"]["exact_count"],
        ]
        base = i * (len(methods) + 1)
        axes[1].bar(
            np.arange(base, base + len(methods)),
            counts,
            color=colors,
            width=0.8,
        )
        axes[1].text(base + 1, max(counts) + 5, f"s{seed_summary['seed']}", ha="center", fontsize=8)
    axes[1].set_ylabel("Exact puzzles / 1000")
    axes[1].set_xticks([])
    fig.savefig(figure_dir / "cars_act6_main_results.png")
    plt.close(fig)
    written.append(str(figure_dir / "cars_act6_main_results.png"))

    # Figure B
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    steps = np.arange(1, ACT_STEPS + 1)
    for seed_summary in per_seed:
        dist = seed_summary["selected_step_distribution"]["cars"]
        ys = [dist[str(step)] for step in steps]
        ax.plot(steps, ys, marker="o", label=f"seed {seed_summary['seed']}")
    ax.set_xlabel("CARS selected ACT step")
    ax.set_ylabel("Fraction of test puzzles")
    ax.set_xticks(steps)
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "cars_selected_step_distribution.png")
    plt.close(fig)
    written.append(str(figure_dir / "cars_selected_step_distribution.png"))

    # Figure C (optional)
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for seed_summary in per_seed:
        oracle_exact = int(round(seed_summary["oracle_diagnostic"]["exact_count"]))
        final_exact = seed_summary["final"]["exact_count"]
        cars_exact = seed_summary["cars"]["exact_count"]
        y = [final_exact, cars_exact, oracle_exact]
        x = np.arange(3)
        offset = (seed_summary["seed"] - 1) * 0.25
        ax.bar(x + offset, y, width=0.22, label=f"seed {seed_summary['seed']}")
    ax.set_xticks(np.arange(3), ["Final", "CARS", "Oracle"])
    ax.set_ylabel("Exact puzzles / 1000")
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "cars_recoverable_headroom.png")
    plt.close(fig)
    written.append(str(figure_dir / "cars_recoverable_headroom.png"))
    return written


def build_paper_table(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    agg = report["three_seed_aggregate"]
    rows = [
        {
            "selector": "Final Vanilla",
            "exact_acc_pct_mean_std": _fmt_mean_std(agg["final_exact"]),
            "cell_acc_pct_mean_std": _fmt_cell(agg["final_cell"]),
            "delta_exact_vs_final": 0.0,
            "delta_cell_pp_vs_final": 0.0,
            "selected_step_mean": ACT_STEPS,
            "new_params": 0,
            "retraining": "no",
        },
        {
            "selector": "Confidence Selection",
            "exact_acc_pct_mean_std": _fmt_mean_std(agg["confidence_exact"]),
            "cell_acc_pct_mean_std": _fmt_cell(agg["confidence_cell"]),
            "delta_exact_vs_final": agg["delta_exact_confidence_vs_final"]["mean"],
            "delta_cell_pp_vs_final": _fmt_delta_cell_conf(report),
            "selected_step_mean": _mean_selected(report, "confidence"),
            "new_params": 0,
            "retraining": "no",
        },
        {
            "selector": "CARS",
            "exact_acc_pct_mean_std": _fmt_mean_std(agg["cars_exact"]),
            "cell_acc_pct_mean_std": _fmt_cell(agg["cars_cell"]),
            "delta_exact_vs_final": agg["delta_exact_cars_vs_final"]["mean"],
            "delta_cell_pp_vs_final": agg["delta_cell_pp_cars_vs_final"]["mean"],
            "selected_step_mean": _mean_selected(report, "cars"),
            "new_params": 0,
            "retraining": "no",
        },
    ]
    return rows


def _fmt_mean_std(block: Mapping[str, Any]) -> str:
    return f"{block['mean'] * 100:.3f} ± {block['std'] * 100:.3f}"


def _fmt_cell(block: Mapping[str, Any]) -> str:
    return f"{block['mean'] * 100:.3f} ± {block['std'] * 100:.3f}"


def _fmt_delta_cell_conf(report: Mapping[str, Any]) -> float:
    values = [
        s["confidence"]["delta_cell_pp_vs_final"] for s in report["per_seed_summary"]
    ]
    return float(np.mean(values))


def _mean_selected(report: Mapping[str, Any], key: str) -> float:
    values = [s["mean_selected_act_step"][key] for s in report["per_seed_summary"]]
    return float(np.mean(values))


def postprocess_example(
    *,
    example_id: str,
    inputs: Sequence[int],
    labels: Sequence[int],
    act_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply frozen selectors to one puzzle's ACT-step trajectory."""
    if len(act_steps) != ACT_STEPS:
        raise ValueError(f"expected {ACT_STEPS} ACT steps, got {len(act_steps)}")

    final_idx = ACT_STEPS - 1
    conf_idx = select_confidence(act_steps)
    cars_idx = select_cars(act_steps)
    oracle_idx = select_oracle(act_steps)

    row = {
        "example_id": example_id,
        "inputs": list(inputs),
        "labels": list(labels),
        "act_steps": act_steps,
        "final_act_step": act_steps[final_idx]["act_step"],
        "confidence_act_step": act_steps[conf_idx]["act_step"],
        "cars_act_step": act_steps[cars_idx]["act_step"],
        "oracle_act_step": act_steps[oracle_idx]["act_step"],
        "selected_differs_from_final": cars_idx != final_idx,
        "cars_improved_cell": act_steps[cars_idx]["cell_accuracy"]
        > act_steps[final_idx]["cell_accuracy"],
        "cars_reduced_cell": act_steps[cars_idx]["cell_accuracy"]
        < act_steps[final_idx]["cell_accuracy"],
        "cars_unchanged_cell": act_steps[cars_idx]["cell_accuracy"]
        == act_steps[final_idx]["cell_accuracy"],
        "final_wrong_cars_exact": (not act_steps[final_idx]["exact"])
        and act_steps[cars_idx]["exact"],
        "final_exact_cars_wrong": act_steps[final_idx]["exact"]
        and (not act_steps[cars_idx]["exact"]),
        "any_earlier_exact_final_wrong": any(
            act_steps[i]["exact"] for i in range(ACT_STEPS - 1)
        )
        and not act_steps[final_idx]["exact"],
        "oracle_any_exact": any(step["exact"] for step in act_steps),
        "oracle_best_cell_accuracy": max(step["cell_accuracy"] for step in act_steps),
    }
    for prefix, idx in (
        ("final", final_idx),
        ("confidence", conf_idx),
        ("cars", cars_idx),
        ("oracle", oracle_idx),
    ):
        row.update(
            {
                f"{prefix}_act_step": int(act_steps[idx]["act_step"]),
                f"{prefix}_exact": bool(act_steps[idx]["exact"]),
                f"{prefix}_cell_accuracy": float(act_steps[idx]["cell_accuracy"]),
                f"{prefix}_incorrect_cells": int(act_steps[idx]["incorrect_cells"]),
            }
        )
    return row
