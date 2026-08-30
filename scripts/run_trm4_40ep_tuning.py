from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SWEEP_ROOT = ROOT / "results" / "baseline" / "trm4-40ep-sweep"
RUNS_ROOT = SWEEP_ROOT / "runs"
CHECKPOINTS_ROOT = ROOT / "checkpoints" / "trm4-40ep-sweep"
SUMMARY_PATH = SWEEP_ROOT / "tuning_results.json"


# ---------------------------------------------------------------------
# Scientific anchor
# ---------------------------------------------------------------------
# Every experiment is:
#
#   TRM
#   halt_max_steps = 4
#   epochs = 40
#   seed = 0
#
# Only ONE experimental factor changes relative to the reference.
#
# Reference:
#   lr          = 1e-3
#   hidden_size = 64
#   L_cycles    = 1
# ---------------------------------------------------------------------

FIXED_CONFIG = {
    "halt_max_steps": 4,
    "epochs": 40,
    "eval_interval": 5,
    "seed": 0,
    "global_batch_size": 4,
    "optimizer": "adamw",
}


EXPERIMENTS = [
    {
        "name": "reference",
        "factor": "reference",
        "value": "lr=1e-3, hidden_size=64, L_cycles=1",
        "overrides": [],
    },

    # -------------------------------------------------------------
    # Learning-rate control
    # -------------------------------------------------------------
    {
        "name": "lr-3e-4",
        "factor": "learning_rate",
        "value": 3e-4,
        "overrides": [
            "lr=3e-4",
        ],
    },
    {
        "name": "lr-3e-3",
        "factor": "learning_rate",
        "value": 3e-3,
        "overrides": [
            "lr=3e-3",
        ],
    },

    # -------------------------------------------------------------
    # Model-capacity control
    # -------------------------------------------------------------
    {
        "name": "hidden-96",
        "factor": "hidden_size",
        "value": 96,
        "overrides": [
            "arch.hidden_size=96",
        ],
    },
    {
        "name": "hidden-128",
        "factor": "hidden_size",
        "value": 128,
        "overrides": [
            "arch.hidden_size=128",
        ],
    },

    # -------------------------------------------------------------
    # Internal-computation control
    # -------------------------------------------------------------
    {
        "name": "Lcycles-2",
        "factor": "L_cycles",
        "value": 2,
        "overrides": [
            "arch.L_cycles=2",
        ],
    },
]


def latest_metrics_file(run_dir: Path) -> Path | None:
    files = list(run_dir.glob("metrics_step_*.json"))

    if not files:
        return None

    def step_number(path: Path) -> int:
        return int(path.stem.split("_")[-1])

    return max(files, key=step_number)


def save_summary(results: list[dict]) -> None:
    payload = {
        "study": "TRM-4 / 40-epoch controlled baseline tuning",
        "method": (
            "One-factor-at-a-time sweep. "
            "All non-tested hyperparameters are held fixed."
        ),
        "fixed_config": FIXED_CONFIG,
        "reference_config": {
            "lr": 1e-3,
            "hidden_size": 64,
            "L_cycles": 1,
        },
        "experiments": results,
    }

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_final_metrics(run_dir: Path) -> tuple[int, dict]:
    path = latest_metrics_file(run_dir)

    if path is None:
        raise RuntimeError(
            f"No metrics_step_*.json found in {run_dir}"
        )

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    step = int(payload["step"])
    metrics = payload["metrics"]["all"]

    return step, metrics


def print_result(result: dict) -> None:
    print()
    print("-" * 78)
    print("RESULT")
    print("-" * 78)

    if result["status"] != "completed":
        print(f"Status : {result['status']}")
        return

    print(f"Run            : {result['name']}")
    print(f"Factor         : {result['factor']}")
    print(f"Value          : {result['value']}")
    print(f"Final step     : {result['step']}")
    print(f"Accuracy       : {result['accuracy']:.6f}")
    print(f"Accuracy (%)   : {result['accuracy'] * 100:.3f}%")
    print(f"Exact accuracy : {result['exact_accuracy']:.6f}")
    print(f"LM loss        : {result['lm_loss']:.6f}")
    print(f"Reasoning steps: {result['steps']:.1f}")
    print(f"Runtime        : {result['runtime_seconds']:.1f} s")


def print_final_table(results: list[dict]) -> None:
    print()
    print("=" * 100)
    print("FINAL TRM-4 / 40-EPOCH TUNING SUMMARY")
    print("=" * 100)

    header = (
        f"{'Experiment':<18}"
        f"{'Factor':<18}"
        f"{'Value':<30}"
        f"{'Accuracy':>12}"
        f"{'LM loss':>12}"
    )

    print(header)
    print("-" * 100)

    for r in results:
        if r["status"] != "completed":
            print(
                f"{r['name']:<18}"
                f"{r['factor']:<18}"
                f"{str(r['value']):<30}"
                f"{'FAILED':>12}"
                f"{'-':>12}"
            )
            continue

        print(
            f"{r['name']:<18}"
            f"{r['factor']:<18}"
            f"{str(r['value']):<30}"
            f"{r['accuracy'] * 100:>11.3f}%"
            f"{r['lm_loss']:>12.6f}"
        )

    print("=" * 100)
    print()
    print("Consolidated result file:")
    print(SUMMARY_PATH.relative_to(ROOT))


def main() -> None:
    print()
    print("=" * 78)
    print("TRM-4 / 40-EPOCH CONTROLLED BASELINE TUNING")
    print("=" * 78)
    print()
    print("Fixed:")
    print("  halt_max_steps = 4")
    print("  epochs         = 40")
    print("  seed           = 0")
    print("  batch size     = 4")
    print("  optimizer      = AdamW")
    print()
    print(
        "Each experiment changes exactly ONE factor "
        "relative to the reference."
    )

    # Clean ONLY this automated sweep.
    # Previous manually-run baseline experiments are untouched.
    if SWEEP_ROOT.exists():
        print()
        print(f"Removing previous sweep results: {SWEEP_ROOT}")
        shutil.rmtree(SWEEP_ROOT)

    if CHECKPOINTS_ROOT.exists():
        print(f"Removing previous sweep checkpoints: {CHECKPOINTS_ROOT}")
        shutil.rmtree(CHECKPOINTS_ROOT)

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    total = len(EXPERIMENTS)

    for index, exp in enumerate(EXPERIMENTS, start=1):
        run_name = f"trm4-40ep-{exp['name']}"

        run_metrics_dir = RUNS_ROOT / exp["name"]
        checkpoint_dir = CHECKPOINTS_ROOT / exp["name"]

        print()
        print()
        print("#" * 78)
        print(
            f"EXPERIMENT {index}/{total}"
        )
        print("#" * 78)
        print(f"Name   : {exp['name']}")
        print(f"Factor : {exp['factor']}")
        print(f"Value  : {exp['value']}")
        print()
        print("Fixed anchor:")
        print("  TRM-4")
        print("  40 epochs")
        print("  seed = 0")
        print("#" * 78)
        print()

        cmd = [
            sys.executable,
            "pretrain.py",
            "--config-name",
            "experiment/sudoku_study_canonical",

            # Fixed scientific anchor
            "arch.halt_max_steps=4",
            "epochs=40",
            "eval_interval=5",
            "seed=0",

            # Experiment outputs
            f"run_name={run_name}",
            (
                "checkpoint_path="
                f"checkpoints/trm4-40ep-sweep/{exp['name']}"
            ),
            (
                "metrics_dir="
                f"results/baseline/trm4-40ep-sweep/runs/{exp['name']}"
            ),

            # One-factor experimental override
            *exp["overrides"],
        ]

        print("Command:")
        print(" ".join(cmd))
        print()
        print("Training output:")
        print("-" * 78)

        start = time.perf_counter()

        try:
            # No stdout capture:
            # pretrain.py output remains LIVE in the terminal.
            subprocess.run(
                cmd,
                cwd=ROOT,
                check=True,
            )

            runtime_seconds = time.perf_counter() - start

            step, metrics = read_final_metrics(run_metrics_dir)

            result = {
                "name": exp["name"],
                "factor": exp["factor"],
                "value": exp["value"],
                "status": "completed",
                "step": step,
                "accuracy": float(metrics["accuracy"]),
                "exact_accuracy": float(metrics["exact_accuracy"]),
                "lm_loss": float(metrics["lm_loss"]),
                "q_halt_accuracy": float(
                    metrics["q_halt_accuracy"]
                ),
                "q_halt_loss": float(
                    metrics["q_halt_loss"]
                ),
                "steps": float(metrics["steps"]),
                "runtime_seconds": runtime_seconds,
            }

        except Exception as exc:
            runtime_seconds = time.perf_counter() - start

            result = {
                "name": exp["name"],
                "factor": exp["factor"],
                "value": exp["value"],
                "status": "failed",
                "error": str(exc),
                "runtime_seconds": runtime_seconds,
            }

        results.append(result)

        # Update the SAME consolidated file after every run,
        # so partial results survive if a later run fails.
        save_summary(results)
        print_result(result)

        if result["status"] != "completed":
            print()
            print("Sweep stopped because this experiment failed.")
            print(
                "Completed results have already been saved to:"
            )
            print(SUMMARY_PATH.relative_to(ROOT))
            raise SystemExit(1)

    print_final_table(results)

    print()
    print("BASELINE TUNING COMPLETE")


if __name__ == "__main__":
    main()
