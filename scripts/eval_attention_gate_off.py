from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from pretrain import (
    PretrainConfig,
    TrainState,
    create_dataloader,
    create_model,
    evaluate,
    resolve_device,
)

CHECKPOINT_ROOT = Path("checkpoints/lcycle-history")
METRICS_ROOT = Path("results/lcycle-history")
OUTPUT_PATH = Path(
    "results/lcycle-history-mechanistic/cpu_attention_gate_off_v1.json"
)

FINAL_STEP = 10000
SEEDS = range(5)
GATE_OFF_LOGIT = -100.0
TOL = 1e-6


def to_builtin(x):
    if isinstance(x, dict):
        return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x


def flatten_numeric(obj, prefix=""):
    out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            out.update(flatten_numeric(v, key))

    elif isinstance(obj, (int, float, np.number)):
        out[prefix] = float(obj)

    return out


def audit_config(raw, seed):
    arch = raw["arch"]

    expected = {
        "hidden_size": 64,
        "halt_max_steps": 4,
        "H_cycles": 3,
        "L_cycles": 6,
        "L_layers": 2,
        "num_heads": 4,
        "history_enabled": False,
        "history_aggregator": "none",
        "lcycle_history_enabled": True,
        "lcycle_history_rank": 16,
        "lcycle_history_heads": 4,
        "lcycle_history_gate_init": -2.0,
        "lcycle_history_pre_norm": True,
    }

    for key, value in expected.items():
        actual = arch.get(key)

        if actual != value:
            raise RuntimeError(
                f"seed{seed}: config mismatch for {key}: "
                f"expected={value!r}, actual={actual!r}"
            )

    method = arch.get("lcycle_history_method", "attention")

    if method not in (None, "attention"):
        raise RuntimeError(
            f"seed{seed}: unexpected lcycle_history_method={method!r}"
        )

    # Older canonical Attention configs may predate explicit method dispatch.
    arch["lcycle_history_method"] = "attention"


def make_eval_loader(config, device):
    return create_dataloader(
        config,
        config.eval_split,
        rank=0,
        world_size=1,
        device=device,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )


def evaluate_model(config, state, device):
    eval_loader, eval_metadata = make_eval_loader(config, device)

    state.model.eval()

    return evaluate(
        config,
        state,
        eval_loader,
        eval_metadata,
        evaluators=[],
        rank=0,
        world_size=1,
        cpu_group=None,
        device=device,
    )


all_results = []

for seed in SEEDS:
    print()
    print("=" * 72)
    print(f"SEED {seed}")
    print("=" * 72)

    run_name = f"proposal-h3l6-l2-attention-40ep-seed{seed}"

    run_dir = CHECKPOINT_ROOT / run_name
    checkpoint = run_dir / f"step_{FINAL_STEP}"
    config_file = run_dir / "all_config.yaml"

    stored_metrics_file = (
        METRICS_ROOT / run_name / f"metrics_step_{FINAL_STEP}.json"
    )

    for required in (checkpoint, config_file, stored_metrics_file):
        if not required.exists():
            raise FileNotFoundError(required)

    raw = yaml.safe_load(
        config_file.read_text(encoding="utf-8")
    )

    audit_config(raw, seed)

    # Runtime-only changes. No scientific setting is changed.
    raw["seed"] = seed
    raw["device"] = "cpu"
    raw["compile_model"] = False
    raw["dataloader_num_workers"] = 0
    raw["wandb_mode"] = "disabled"

    # Load the frozen final checkpoint.
    raw["load_checkpoint"] = str(checkpoint)

    # Critical: prevent evaluate() from writing predictions/evaluator files.
    raw["checkpoint_path"] = None
    raw["metrics_dir"] = None
    raw["eval_save_outputs"] = []

    config = PretrainConfig(**raw)

    device = resolve_device(config.device)

    if config.cpu_threads is not None:
        torch.set_num_threads(config.cpu_threads)

    torch.manual_seed(seed)

    # We only need training metadata to instantiate the model with the
    # exact vocabulary/shape used during training. The loader is never iterated.
    _, train_metadata = create_dataloader(
        config,
        "train",
        rank=0,
        world_size=1,
        device=device,
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )

    model, _, _ = create_model(
        config,
        train_metadata,
        rank=0,
        world_size=1,
        device=device,
    )

    state = TrainState(
        model=model,
        optimizers=(),
        optimizer_lrs=(),
        carry=None,
        step=FINAL_STEP,
        total_steps=FINAL_STEP,
    )

    # -------------------------------------------------------------
    # 1) NORMAL frozen-checkpoint evaluation
    # -------------------------------------------------------------
    print("\n--- NORMAL EVALUATION ---")
    normal_metrics = to_builtin(
        evaluate_model(config, state, device)
    )

    # Compare against the already frozen metric file.
    stored_record = json.loads(
        stored_metrics_file.read_text(encoding="utf-8")
    )
    stored_metrics = stored_record["metrics"]

    normal_flat = flatten_numeric(normal_metrics)
    stored_flat = flatten_numeric(stored_metrics)

    common_keys = sorted(
        set(normal_flat).intersection(stored_flat)
    )

    if not common_keys:
        raise RuntimeError(
            f"seed{seed}: no common metric keys between "
            "re-evaluation and stored metrics"
        )

    differences = {
        key: abs(normal_flat[key] - stored_flat[key])
        for key in common_keys
    }

    max_diff = max(differences.values())

    print(f"Stored-metric max abs diff: {max_diff:.10g}")

    if max_diff > TOL:
        print("MISMATCHED METRICS:")
        for key in common_keys:
            if differences[key] > TOL:
                print(
                    key,
                    stored_flat[key],
                    normal_flat[key],
                    differences[key],
                )
        raise RuntimeError(
            f"seed{seed}: normal re-evaluation does not reproduce "
            f"frozen metrics within tolerance {TOL}"
        )

    print("NORMAL RE-EVALUATION MATCH: PASS")

    # -------------------------------------------------------------
    # Locate learned Attention gate
    # -------------------------------------------------------------
    matches = [
        (name, parameter)
        for name, parameter in state.model.named_parameters()
        if name.endswith(
            "lcycle_history_attention.gate_logit"
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"seed{seed}: expected exactly one Attention gate; "
            f"found {[name for name, _ in matches]}"
        )

    gate_name, gate_param = matches[0]

    original_logit = float(gate_param.detach().cpu())
    original_gate = float(
        torch.sigmoid(gate_param.detach()).cpu()
    )

    print(
        f"Learned gate: {gate_name} "
        f"logit={original_logit:+.6f} "
        f"sigmoid={original_gate:.6f}"
    )

    # -------------------------------------------------------------
    # 2) GATE-OFF evaluation
    # -------------------------------------------------------------
    with torch.no_grad():
        gate_param.fill_(GATE_OFF_LOGIT)

    actual_off_gate = float(
        torch.sigmoid(gate_param.detach()).cpu()
    )

    print(
        f"\n--- GATE-OFF EVALUATION "
        f"(logit={GATE_OFF_LOGIT}, gate={actual_off_gate:.3e}) ---"
    )

    gate_off_metrics = to_builtin(
        evaluate_model(config, state, device)
    )

    # Restore original learned value in memory.
    with torch.no_grad():
        gate_param.fill_(original_logit)

    restored_logit = float(gate_param.detach().cpu())

    if abs(restored_logit - original_logit) > 1e-7:
        raise RuntimeError(
            f"seed{seed}: failed to restore gate"
        )

    off_flat = flatten_numeric(gate_off_metrics)

    deltas = {}

    for key in sorted(set(normal_flat).intersection(off_flat)):
        deltas[key] = off_flat[key] - normal_flat[key]

    print("\nMetric deltas: gate-off minus normal")

    for key, delta in deltas.items():
        if "accuracy" in key.lower():
            print(f"{key}: {delta:+.8f} ({100*delta:+.4f} pp)")
        else:
            print(f"{key}: {delta:+.8f}")

    all_results.append(
        {
            "seed": seed,
            "run_name": run_name,
            "checkpoint": str(checkpoint),
            "stored_metrics_file": str(stored_metrics_file),
            "normal_reproduction_max_abs_diff": max_diff,
            "gate_name": gate_name,
            "learned_gate_logit": original_logit,
            "learned_gate": original_gate,
            "gate_off_logit": GATE_OFF_LOGIT,
            "gate_off_gate": actual_off_gate,
            "normal_metrics": normal_metrics,
            "gate_off_metrics": gate_off_metrics,
            "gate_off_minus_normal": deltas,
        }
    )


# -----------------------------------------------------------------
# Aggregate paired deltas across seeds
# -----------------------------------------------------------------
metric_keys = sorted(
    set.intersection(
        *[
            set(r["gate_off_minus_normal"].keys())
            for r in all_results
        ]
    )
)

aggregate = {}

print()
print("=" * 72)
print("FIVE-SEED GATE-OFF SUMMARY")
print("=" * 72)

for key in metric_keys:
    values = np.array(
        [
            r["gate_off_minus_normal"][key]
            for r in all_results
        ],
        dtype=float,
    )

    aggregate[key] = {
        "mean_delta": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "per_seed_delta": values.tolist(),
    }

    if "accuracy" in key.lower():
        print(
            f"{key}: "
            f"mean={100*values.mean():+.4f} pp, "
            f"SD={100*values.std(ddof=1):.4f} pp, "
            f"negative={(values < 0).sum()}/5"
        )
    else:
        print(
            f"{key}: "
            f"mean={values.mean():+.6f}, "
            f"SD={values.std(ddof=1):.6f}"
        )


output = {
    "analysis": "POST-HOC INFERENCE-ONLY MECHANISTIC ABLATION",
    "scientific_status": "EXPLORATORY",
    "intervention": (
        "Set trained HistoryAttention scalar gate_logit to -100 "
        "in memory during evaluation only."
    ),
    "training_performed": False,
    "checkpoint_modified_on_disk": False,
    "seeds": list(SEEDS),
    "results": all_results,
    "aggregate_gate_off_minus_normal": aggregate,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH.write_text(
    json.dumps(output, indent=2),
    encoding="utf-8",
)

print()
print("SAVED:", OUTPUT_PATH)
print("DONE — NO TRAINING WAS PERFORMED")
