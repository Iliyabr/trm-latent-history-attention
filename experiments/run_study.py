"""Launch the deterministic B0/B1/B2/B3/P1 multi-seed study."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


VARIANTS = ("B0", "B1", "B2", "B3", "P1")
SEEDS = (0, 1, 2)
VARIANT_OVERRIDES = {
    "B0": ("arch.history_enabled=false", "arch.history_mode=B0"),
    "B1": ("arch.history_enabled=true", "arch.history_mode=B1"),
    "B2": ("arch.history_enabled=true", "arch.history_mode=B2"),
    "B3": ("arch.history_enabled=true", "arch.history_mode=B3"),
    "P1": ("arch.history_enabled=true", "arch.history_mode=P1"),
}


def latest_checkpoint(run_dir: Path) -> Path:
    runtime_cap = run_dir / "runtime_cap.pt"
    if runtime_cap.exists():
        return runtime_cap
    candidates = sorted(
        run_dir.glob("step_*.pt"),
        key=lambda path: int(path.stem.removeprefix("step_")),
    )
    if not candidates:
        raise FileNotFoundError(f"No resumable checkpoint found under {run_dir}")
    return candidates[-1]


def command_for(
    variant: str,
    seed: int,
    preset: str,
    output_root: Path,
    resume: Path | None,
    extras: Sequence[str],
) -> tuple[list[str], Path]:
    run_name = f"{variant}-seed{seed}"
    run_dir = output_root / preset / run_name
    command = [
        sys.executable,
        "-u",
        "pretrain.py",
        f"--config-name=experiment/sudoku_study_{preset}",
        f"seed={seed}",
        f"run_name={run_name}",
        f"checkpoint_path={run_dir.as_posix()}",
        f"metrics_dir={(run_dir / 'metrics').as_posix()}",
        f"metrics_jsonl={(run_dir / 'metrics.jsonl').as_posix()}",
        *VARIANT_OVERRIDES[variant],
    ]
    if resume is not None:
        command.append(f"resume_checkpoint={resume.as_posix()}")
    command.extend(extras)
    return command, run_dir


def execute(
    variant: str,
    seed: int,
    args: argparse.Namespace,
    resume: Path | None = None,
) -> None:
    command, run_dir = command_for(
        variant,
        seed,
        args.preset,
        args.output_root,
        resume,
        args.override,
    )
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "orchestrator.json"
    started = datetime.now(timezone.utc).isoformat()
    status_path.write_text(
        json.dumps(
            {
                "variant": variant,
                "seed": seed,
                "preset": args.preset,
                "command": command,
                "started_utc": started,
                "resume": str(resume) if resume else None,
                "status": "running",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(command, check=False, env=env)
    status_path.write_text(
        json.dumps(
            {
                "variant": variant,
                "seed": seed,
                "preset": args.preset,
                "command": command,
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "resume": str(resume) if resume else None,
                "status": "completed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        choices=("colab", "colab_heavy", "publication"),
        default="colab",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/study")
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override; repeat as needed.",
    )
    parser.add_argument("--dry-run", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    single = subparsers.add_parser("single", help="Run one variant/seed")
    single.add_argument("--variant", choices=VARIANTS, required=True)
    single.add_argument("--seed", type=int, choices=SEEDS, required=True)
    add_common(single)

    suite = subparsers.add_parser("suite", help="Run all 15 jobs serially")
    add_common(suite)

    resume = subparsers.add_parser("resume", help="Resume one variant/seed")
    resume.add_argument("--variant", choices=VARIANTS, required=True)
    resume.add_argument("--seed", type=int, choices=SEEDS, required=True)
    resume.add_argument(
        "--checkpoint",
        type=Path,
        help="Complete checkpoint; defaults to runtime_cap.pt/latest step.",
    )
    add_common(resume)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "single":
        execute(args.variant, args.seed, args)
    elif args.action == "resume":
        run_dir = args.output_root / args.preset / f"{args.variant}-seed{args.seed}"
        checkpoint = args.checkpoint
        if checkpoint is None and not args.dry_run:
            checkpoint = latest_checkpoint(run_dir)
        elif checkpoint is None:
            checkpoint = run_dir / "runtime_cap.pt"
        execute(args.variant, args.seed, args, checkpoint)
    else:
        for variant in VARIANTS:
            for seed in SEEDS:
                execute(variant, seed, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
