#!/usr/bin/env python
"""Reproduce the canonical CPU learning-curve statistics and paper figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


LABELS = {
    "vanilla": "Vanilla",
    "gated": "Gated",
    "attention": "HistoryAttention",
    "parammatched": "Parameter-Matched",
}


def load_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "Method": r["Method"],
                    "Seed": int(r["Seed"]),
                    "Step": int(r["Step"]),
                    "Accuracy": float(r["Accuracy"]),
                    "LMLoss": float(r["LMLoss"]),
                    "Exact": float(r["Exact"]),
                }
            )
    return rows


def paired_stats(lookup, seeds, method, step):
    d = np.array([
        100.0 * (
            lookup[(method, s, step)]["Accuracy"]
            - lookup[("vanilla", s, step)]["Accuracy"]
        )
        for s in seeds
    ])
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(len(d))
    tcrit = stats.t.ppf(0.975, len(d) - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)
    p = float(2 * stats.t.sf(abs(mean / se), len(d) - 1)) if se else 0.0
    return d, mean, sd, ci, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = load_rows(args.input)
    methods = ["vanilla", "gated", "attention", "parammatched"]
    seeds = sorted({r["Seed"] for r in rows})
    steps = sorted({r["Step"] for r in rows})
    lookup = {(r["Method"], r["Seed"], r["Step"]): r for r in rows}

    expected = len(methods) * len(seeds) * len(steps)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, found {len(rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: raw learning curves.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m in methods:
        means, sds = [], []
        for step in steps:
            vals = np.array([100 * lookup[(m, s, step)]["Accuracy"] for s in seeds])
            means.append(vals.mean())
            sds.append(vals.std(ddof=1))
        ax.errorbar(steps, means, yerr=sds, marker="o", capsize=2.5, label=LABELS[m])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Token accuracy (%)")
    ax.set_title("CPU learning curves (mean ± SD across 5 seeds)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "cpu_learning_curves_accuracy.png", dpi=350)
    plt.close(fig)

    # Figure 2: paired deltas vs Vanilla.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m in ["gated", "attention", "parammatched"]:
        means, lower, upper = [], [], []
        for step in steps:
            _, mean, _, ci, _ = paired_stats(lookup, seeds, m, step)
            means.append(mean)
            lower.append(mean - ci[0])
            upper.append(ci[1] - mean)
        ax.errorbar(
            steps, means, yerr=np.vstack([lower, upper]),
            marker="o", capsize=2.5, label=LABELS[m]
        )
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy delta vs Vanilla (pp)")
    ax.set_title("Paired CPU effect over training (mean ± 95% t CI)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "cpu_paired_delta_vs_vanilla.png", dpi=350)
    plt.close(fig)

    # Figure 3: final paired seed plot.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xpos = np.arange(len(methods), dtype=float)
    for s in seeds:
        vals = [100 * lookup[(m, s, 10000)]["Accuracy"] for m in methods]
        ax.plot(xpos, vals, marker="o", linewidth=1, alpha=0.65, label=f"seed {s}")
    means, sds = [], []
    for m in methods:
        vals = np.array([100 * lookup[(m, s, 10000)]["Accuracy"] for s in seeds])
        means.append(vals.mean())
        sds.append(vals.std(ddof=1))
    ax.errorbar(xpos, means, yerr=sds, marker="s", linewidth=2, capsize=4,
                label="mean ± SD")
    ax.set_xticks(xpos, [LABELS[m] for m in methods])
    ax.set_ylabel("Final token accuracy (%)")
    ax.set_title("Final CPU accuracy by paired seed")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "cpu_final_seed_accuracy.png", dpi=350)
    plt.close(fig)

    print(f"Loaded {len(rows)} rows: {len(methods)} models × {len(seeds)} seeds × {len(steps)} steps")
    print("\nFinal paired comparisons vs Vanilla:")
    for m in ["gated", "attention", "parammatched"]:
        d, mean, sd, ci, p = paired_stats(lookup, seeds, m, 10000)
        print(
            f"{LABELS[m]:18s}: mean={mean:+.3f} pp, SD={sd:.3f}, "
            f"95% CI=[{ci[0]:+.3f},{ci[1]:+.3f}], p={p:.3f}, "
            f"positive={(d > 0).sum()}/{len(d)}"
        )


if __name__ == "__main__":
    main()
