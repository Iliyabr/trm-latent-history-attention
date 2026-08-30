#!/usr/bin/env python3
"""Training-free CARS evaluation on frozen canonical Vanilla TRM (ACT6).

Runs inference-only forward passes, captures per-ACT-step predictions/logits,
applies Final / Confidence / CARS / Oracle selectors, and writes analysis artifacts.

No training, no backward(), no checkpoint modification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.cars_postprocess import (  # noqa: E402
    ACT_STEPS,
    act_step_metrics,
    aggregate_seeds,
    analyze_seed_rows,
    build_paper_table,
    make_cars_figures,
    postprocess_example,
    sha256_file,
)
from experiments.evaluate_study import (  # noqa: E402
    _analysis_request,
    _load_config,
    load_checkpoint_robust,
)

DEFAULT_OUTPUT = Path("results/inference-postprocess/cars-act6")
DEFAULT_DATA = Path("data/sudoku-study-v1")
DEFAULT_CONFIG = Path("config/experiment/sudoku_study_canonical.yaml")
DEFAULT_CHECKPOINTS = {
    0: Path("outputs/study/canonical/B0-seed0/step_28800.pt"),
    1: Path("outputs/study-4090/canonical/B0-seed1/step_28800.pt"),
    2: Path("outputs/study-4090/canonical/B0-seed2/step_28800.pt"),
}
METADATA_FALLBACK = {
    0: Path("results/canonical-gpu/B0/seed_0/metadata.json"),
    1: Path("results/canonical-gpu-4090/B0/seed_1/metadata.json"),
    2: Path("results/canonical-gpu-4090/B0/seed_2/metadata.json"),
}


def _example_id(set_name: str, puzzle_id: int, inputs: np.ndarray, index: int) -> str:
    digest = hashlib.sha1(inputs.astype(np.int32).tobytes()).hexdigest()[:12]
    return f"{set_name}:{puzzle_id}:{digest}:{index}"


def run_cars_inference(
    model: torch.nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One normal ACT6 forward pass; capture preds/logits at each ACT step."""
    model.eval()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    selection_started = time.perf_counter()
    postprocess_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for batch_index, (set_name, batch, effective_size) in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.device(device):
                carry = model.initial_carry(batch)  # type: ignore[attr-defined]

            act_predictions: list[np.ndarray] = []
            act_logits: list[np.ndarray] = []
            act_step = 0
            while True:
                analysis_request = _analysis_request(
                    model, "B0", seed, batch_index, act_step, None
                )
                # ACT-step level capture only; do not expand H-cycle micro-steps.
                analysis_request["cycle_logits"] = False
                carry, _, _, output, finished = model(
                    carry=carry,
                    batch=batch,
                    analysis_request=analysis_request,
                    return_keys=("preds", "logits", "q_halt_logits"),
                )
                predictions = output.get("preds")
                logits = output.get("logits")
                if predictions is None:
                    predictions = logits.argmax(-1)
                act_predictions.append(predictions.detach().cpu().numpy())
                act_logits.append(logits.detach().float().cpu().numpy())
                act_step += 1
                if bool(finished):
                    break

            if len(act_predictions) != ACT_STEPS:
                raise RuntimeError(
                    f"expected halt_max_steps={ACT_STEPS}, captured {len(act_predictions)} ACT steps"
                )

            labels = batch["labels"].cpu().numpy()
            inputs = batch["inputs"].cpu().numpy()
            identifiers = batch["puzzle_identifiers"].cpu().numpy()
            for index in range(min(int(effective_size), labels.shape[0])):
                steps: list[dict[str, Any]] = []
                for step_idx, (pred, logit) in enumerate(
                    zip(act_predictions, act_logits), start=1
                ):
                    metrics = act_step_metrics(
                        inputs[index],
                        labels[index],
                        pred[index],
                        logit[index],
                    )
                    metrics["act_step"] = step_idx
                    steps.append(metrics)

                example_id = _example_id(
                    str(set_name), int(identifiers[index]), inputs[index], len(rows)
                )
                tick = time.perf_counter()
                rows.append(
                    postprocess_example(
                        example_id=example_id,
                        inputs=inputs[index].tolist(),
                        labels=labels[index].tolist(),
                        act_steps=steps,
                    )
                )
                postprocess_seconds += time.perf_counter() - tick

    wall = time.perf_counter() - started
    metadata = {
        "variant": "B0",
        "seed": seed,
        "examples": len(rows),
        "wall_time_seconds": wall,
        "postprocess_seconds": postprocess_seconds,
        "additional_forward_passes": 0,
        "act_steps_captured": ACT_STEPS,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
    }
    return rows, metadata


def _repo_relative_checkpoint(raw: str, repo_root: Path) -> Path:
    """Map committed eval metadata paths to this clone (strip foreign home dirs)."""
    path = Path(raw)
    if not path.is_absolute():
        return path
    repo_name = repo_root.name
    for index, part in enumerate(path.parts):
        if part == repo_name and index + 1 < len(path.parts):
            return Path(*path.parts[index + 1 :])
    for index, part in enumerate(path.parts):
        if part == "outputs":
            return Path(*path.parts[index:])
    return path


def _checkpoint_search_paths(relative: Path, repo_root: Path) -> list[Path]:
    """Ordered candidates: metadata path, legacy outputs/canonical layout, defaults."""
    rel_posix = relative.as_posix()
    variants = [relative]
    if rel_posix.startswith("outputs/study/"):
        variants.append(Path(rel_posix.replace("outputs/study/", "outputs/", 1)))
    elif rel_posix.startswith("outputs/"):
        variants.append(Path("outputs/study/" + rel_posix.removeprefix("outputs/")))
    seen: set[str] = set()
    ordered: list[Path] = []
    for variant in variants:
        candidate = variant if variant.is_absolute() else repo_root / variant
        key = candidate.as_posix()
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)
    return ordered


def resolve_checkpoint(seed: int, explicit: Path | None, repo_root: Path) -> Path:
    if explicit is not None:
        candidates = _checkpoint_search_paths(explicit, repo_root)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    candidates: list[Path] = []
    meta_path = repo_root / METADATA_FALLBACK[seed]
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        relative = _repo_relative_checkpoint(str(meta["checkpoint"]), repo_root)
        candidates.extend(_checkpoint_search_paths(relative, repo_root))
    candidates.extend(_checkpoint_search_paths(DEFAULT_CHECKPOINTS[seed], repo_root))

    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return candidates[0]


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["rows"]
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    return rows


def write_seed_artifacts(output_root: Path, seed: int, rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> Path:
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    with (seed_dir / "cars_puzzles.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (seed_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return seed_dir


def build_report(
    *,
    configuration: Mapping[str, Any],
    checkpoint_provenance: Sequence[Mapping[str, Any]],
    per_seed_summary: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    missing_fields: Sequence[str],
    act16: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "configuration": configuration,
        "checkpoint_provenance": list(checkpoint_provenance),
        "method_definition": {
            "final_step": "Use last ACT-step prediction (existing evaluator baseline).",
            "confidence_selection": (
                "argmax ACT step by mean token max-softmax confidence; tie -> later step."
            ),
            "cars": {
                "lexicographic": [
                    "minimize clue_mismatch_count",
                    "minimize structural_violations (row+column+box duplicate excess)",
                    "maximize mean token confidence",
                    "tie -> later ACT step",
                ],
                "uses_ground_truth": False,
            },
            "oracle_diagnostic": {
                "label": "ORACLE DIAGNOSTIC - NOT AN INFERENCE METHOD",
                "uses_ground_truth": True,
            },
        },
        "per_seed_summary": list(per_seed_summary),
        "three_seed_aggregate": aggregate_seeds(per_seed_summary),
        "per_seed_exact_transitions": [s["exact_transitions"] for s in per_seed_summary],
        "per_seed_cell_deltas": [s["cell_deltas"] for s in per_seed_summary],
        "selected_step_distribution": [s["selected_step_distribution"] for s in per_seed_summary],
        "earlier_exact_lost_by_final": [s["failure_modes"] for s in per_seed_summary],
        "cars_recovery": [s["failure_modes"] for s in per_seed_summary],
        "cars_damage": [s["failure_modes"] for s in per_seed_summary],
        "oracle_headroom": [s["oracle_diagnostic"] for s in per_seed_summary],
        "runtime": runtime,
        "memory": {
            "new_trainable_parameters": 0,
            "checkpoint_changes": "none",
            "extra_trajectory_storage": "cars_puzzles.jsonl per seed (6 ACT steps/puzzle)",
        },
        "statistical_tests": {
            "mcnemar_per_seed": [s["exact_transitions"] for s in per_seed_summary],
            "note": "Do not pool McNemar p-values across seeds.",
        },
        "paper_table": build_paper_table(
            {"per_seed_summary": per_seed_summary, "three_seed_aggregate": aggregate_seeds(per_seed_summary)}
        ),
        "missing_fields": list(missing_fields),
    }
    if act16 is not None:
        report["optional_act16_seed0"] = act16
    return report


def render_markdown_report(report: Mapping[str, Any], status: str) -> str:
    lines = [
        "# CARS ACT6 Training-Free Inference Report",
        "",
        "## 1. Protocol and checkpoint audit",
        "",
        f"Status: **{status}**",
        "",
        "Frozen Vanilla TRM (B0), ACT6 (`halt_max_steps=6`), test split `sudoku-study-v1`.",
        "No training, no backward(), no checkpoint modification.",
        "",
        "### Checkpoint provenance",
        "",
        "| Seed | Checkpoint | SHA256 | Exists |",
        "|------|------------|--------|--------|",
    ]
    for item in report["checkpoint_provenance"]:
        lines.append(
            f"| {item['seed']} | `{item['checkpoint']}` | {item.get('sha256', 'MISSING')} | {item.get('exists', False)} |"
        )

    lines.extend(
        [
            "",
            "## 2. Implementation audit",
            "",
            "- Inference path: `scripts/eval_cars_postprocess.py` → `run_cars_inference()`",
            "- One forward pass per puzzle batch; **6 ACT-step** predictions/logits captured",
            "  (`cycle_logits=False`; final preds at each ACT step, not 18 H-cycle micro-steps).",
            "- Existing `examples.jsonl` stores 18 micro-step metrics but **not** per-step full grids;",
            "  CARS requires this dedicated capture pass.",
            "",
            "## 3. Frozen selection rule",
            "",
            "See `method_definition` in JSON. Lexicographic CARS rule is fixed pre-evaluation.",
            "",
            "## 4. Per-seed final results",
            "",
        ]
    )
    for seed_summary in report["per_seed_summary"]:
        s = seed_summary["seed"]
        lines.extend(
            [
                f"### Seed {s}",
                "",
                f"- Final: exact {seed_summary['final']['exact_accuracy']:.4f}, "
                f"cell {seed_summary['final']['cell_accuracy']:.6f}",
                f"- Confidence: exact {seed_summary['confidence']['exact_accuracy']:.4f}, "
                f"cell {seed_summary['confidence']['cell_accuracy']:.6f}, "
                f"Δcell {seed_summary['confidence']['delta_cell_pp_vs_final']:+.3f} pp",
                f"- CARS: exact {seed_summary['cars']['exact_accuracy']:.4f}, "
                f"cell {seed_summary['cars']['cell_accuracy']:.6f}, "
                f"Δcell {seed_summary['cars']['delta_cell_pp_vs_final']:+.3f} pp",
                "",
            ]
        )

    agg = report["three_seed_aggregate"]
    lines.extend(
        [
            "## 5. Three-seed aggregate",
            "",
            f"- Final exact: {agg['final_exact']['mean']:.4f} ± {agg['final_exact']['std']:.4f}",
            f"- CARS exact: {agg['cars_exact']['mean']:.4f} ± {agg['cars_exact']['std']:.4f}",
            f"- Confidence exact: {agg['confidence_exact']['mean']:.4f} ± {agg['confidence_exact']['std']:.4f}",
            f"- CARS Δ cell (pp): {agg['delta_cell_pp_cars_vs_final']['mean']:.3f} ± "
            f"{agg['delta_cell_pp_cars_vs_final']['std']:.3f}",
            "",
            "## 6. Paired exact-solve transitions",
            "",
            "See `per_seed_exact_transitions` in JSON (McNemar per seed).",
            "",
            "## 7. Paired cell-accuracy analysis",
            "",
            "See `per_seed_cell_deltas` in JSON (bootstrap CI per seed).",
            "",
            "## 8. Recursive failure-mode analysis",
            "",
            "See `earlier_exact_lost_by_final`, `cars_recovery`, `cars_damage` in JSON.",
            "",
            "## 9. Oracle trajectory headroom",
            "",
            "**ORACLE DIAGNOSTIC - NOT AN INFERENCE METHOD**",
            "",
            "## 10. Selected-step distribution",
            "",
            "See `selected_step_distribution` and figure `cars_selected_step_distribution.png`.",
            "",
            "## 11. Inference-time overhead",
            "",
            f"Additional forward passes: {report['runtime'].get('additional_forward_passes', 0)}",
            "",
            "## 12. Optional ACT16 seed-0 sensitivity",
            "",
            str(report.get("optional_act16_seed0", "NOT RUN")),
            "",
            "## 13. Paper-ready table",
            "",
            "See `paper_table` in JSON.",
            "",
            "## 14. Paper-ready figures",
            "",
            "- `docs/figures/cars_act6_main_results.png`",
            "- `docs/figures/cars_selected_step_distribution.png`",
            "- `docs/figures/cars_recoverable_headroom.png` (optional)",
            "",
            "## 15. Scientific interpretation",
            "",
            "Fill after all three seeds complete. Do not overclaim from n=3 training seeds.",
            "",
            "## 16. Proposed paper revision",
            "",
            "Draft after observed results.",
            "",
            "## 17. Missing or unresolved evidence",
            "",
        ]
    )
    for field in report["missing_fields"]:
        lines.append(f"- {field}")
    lines.extend(["", f"**CARS_ACT6_STATUS: {status}**", ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--checkpoint", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)

    repo_root = _REPO_ROOT
    explicit_ckpts: dict[int, Path] = {}
    for item in args.checkpoint:
        seed_text, _, path_text = item.partition("=")
        explicit_ckpts[int(seed_text)] = Path(path_text)

    from pretrain import create_dataloader, create_model, resolve_device

    missing: list[str] = []
    checkpoint_provenance: list[dict[str, Any]] = []
    per_seed_summary: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        seed_dir = args.output / f"seed_{seed}"
        cache_path = seed_dir / "cars_puzzles.jsonl"
        ckpt_path = resolve_checkpoint(seed, explicit_ckpts.get(seed), repo_root)
        ckpt_exists = ckpt_path.exists()
        provenance = {
            "seed": seed,
            "checkpoint": str(ckpt_path),
            "exists": ckpt_exists,
            "sha256": sha256_file(ckpt_path) if ckpt_exists else "MISSING",
        }
        checkpoint_provenance.append(provenance)
        rows: list[dict[str, Any]] = []

        if args.analyze_only:
            if not cache_path.exists():
                missing.append(f"seed {seed}: missing cached {cache_path}")
                continue
            rows = load_rows(cache_path)
            runtime_rows.append(
                json.loads((seed_dir / "metadata.json").read_text(encoding="utf-8"))
                if (seed_dir / "metadata.json").exists()
                else {"seed": seed, "wall_time_seconds": "MISSING"}
            )
        else:
            if not ckpt_exists:
                missing.append(f"seed {seed}: checkpoint not found at {ckpt_path}")
                continue
            if not args.data.exists():
                missing.append(f"dataset missing at {args.data}")
                return 2

            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = _load_config(args.config, raw)
            config.data_paths_test = [str(args.data)]
            config.seed = seed
            device = resolve_device(args.device)
            torch.manual_seed(seed)
            np.random.seed(seed)
            loader, metadata = create_dataloader(
                config,
                "test",
                test_set_mode=True,
                epochs_per_iter=1,
                global_batch_size=config.global_batch_size,
                rank=0,
                world_size=1,
                device=device,
            )
            model, _, _ = create_model(config, metadata, 0, 1, device)
            load_info = load_checkpoint_robust(model, ckpt_path, device)
            rows, meta = run_cars_inference(model, loader, device, seed=seed)
            meta["checkpoint"] = str(ckpt_path)
            meta["checkpoint_sha256"] = provenance["sha256"]
            meta["checkpoint_load"] = load_info
            write_seed_artifacts(args.output, seed, rows, meta)
            runtime_rows.append(meta)

        if rows:
            per_seed_summary.append(analyze_seed_rows(rows, seed))

    configuration = {
        "variant": "B0",
        "halt_max_steps": ACT_STEPS,
        "dataset": str(args.data),
        "split": "test",
        "seeds_requested": args.seeds,
        "seeds_completed": [s["seed"] for s in per_seed_summary],
    }
    runtime = {
        "per_seed": runtime_rows,
        "additional_forward_passes": 0,
        "postprocess_only_overhead_seconds": [r.get("postprocess_seconds") for r in runtime_rows],
    }

    status = "COMPLETE" if len(per_seed_summary) == len(args.seeds) and not missing else "INCOMPLETE"
    report = build_report(
        configuration=configuration,
        checkpoint_provenance=checkpoint_provenance,
        per_seed_summary=per_seed_summary,
        runtime=runtime,
        missing_fields=missing,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = repo_root / "docs/data/CARS_ACT6_FINAL_v1.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = repo_root / "docs/CARS_ACT6_FINAL_REPORT.md"
    md_path.write_text(render_markdown_report(report, status), encoding="utf-8")

    if status == "COMPLETE" and not args.skip_figures:
        fig_dir = repo_root / "docs/figures"
        written = make_cars_figures(report, fig_dir)
        report["figures"] = written
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"status": status, "missing": missing, "json": str(json_path)}, indent=2))
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
