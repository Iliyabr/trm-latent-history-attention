"""Paper-ready evaluation for B0/B1/B2/B3/P1 checkpoints."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import yaml

# `python experiments/evaluate_study.py` puts experiments/ on sys.path, not the
# repo root. Insert the root so `from pretrain import ...` works on Colab.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

VARIANTS = ("B0", "B1", "B2", "B3", "P1", "Gated")
CORRUPTION_SIGMAS = (0.05, 0.10, 0.20)
IGNORE_LABEL_ID = -100


def sudoku_violations(
    prediction: Sequence[int], *, encoded_tokens: bool | None = None
) -> dict[str, int]:
    """Count invalid units for raw digits or model tokens (digit + 1)."""
    if len(prediction) != 81:
        return {"row": 0, "column": 0, "box": 0}
    values = np.asarray(prediction, dtype=np.int64)
    if encoded_tokens is None:
        encoded_tokens = bool(np.any(values == 10))
    if encoded_tokens:
        values = values - 1
    grid = values.reshape(9, 9)

    def bad(unit: np.ndarray) -> bool:
        return bool(np.any((unit < 1) | (unit > 9)) or np.unique(unit).size != 9)

    return {
        "row": int(sum(bad(grid[i, :]) for i in range(9))),
        "column": int(sum(bad(grid[:, i]) for i in range(9))),
        "box": int(
            sum(
                bad(grid[r : r + 3, c : c + 3].reshape(-1))
                for r in range(0, 9, 3)
                for c in range(0, 9, 3)
            )
        ),
    }


def example_metrics(prediction: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    valid = label != IGNORE_LABEL_ID
    incorrect = int(np.count_nonzero(prediction[valid] != label[valid]))
    cells = int(valid.sum())
    return {
        "exact": bool(cells and incorrect == 0),
        "cell_accuracy": float((cells - incorrect) / cells) if cells else 0.0,
        "incorrect_cells": incorrect,
        # PuzzleDataset Sudoku labels/predictions encode digits 1..9 as 2..10.
        "sudoku_violations": sudoku_violations(
            prediction[valid].tolist(), encoded_tokens=True
        ),
    }


def attention_statistics(
    weights: torch.Tensor, lengths: torch.Tensor | None = None
) -> dict[str, float]:
    """Summarize attention with temporal history on the final axis.

    Supports [B,H,L,T] and stacked [Hcycle,Lstep,B,H,L,T] diagnostics.
    """
    w = weights.detach().float().cpu()
    if w.ndim < 2:
        raise ValueError("attention weights must include batch and history dimensions")
    if w.ndim == 3:
        # Legacy/synthetic [B,T,...] layout.
        w = w.movedim(1, -1)
    w = w.clamp_min(0)
    k = w.shape[-1]
    if lengths is None:
        if w.ndim >= 6:
            # Model diagnostics are [H-cycle,L-step,B,heads,tokens,T].
            shape = [1] * w.ndim
            shape[1] = w.shape[1]
            expanded_lengths = torch.arange(
                1, w.shape[1] + 1, dtype=torch.long
            ).view(shape)
        else:
            expanded_lengths = torch.full(
                (1,) * (w.ndim - 1) + (1,), k, dtype=torch.long
            )
    else:
        length_view = lengths.detach().cpu().long().clamp(0, k)
        candidates = [
            axis for axis, size in enumerate(w.shape[:-1]) if size == length_view.numel()
        ]
        if not candidates:
            raise ValueError(
                f"cannot align {length_view.numel()} history lengths with weights {tuple(w.shape)}"
            )
        # The specified diagnostics place B at axis 0 or, when cycle/step
        # dimensions are stacked, axis 2.
        batch_axis = 2 if w.ndim >= 6 and 2 in candidates else candidates[0]
        shape = [1] * w.ndim
        shape[batch_axis] = length_view.numel()
        expanded_lengths = length_view.view(shape)
    slots = torch.arange(k).view((1,) * (w.ndim - 1) + (k,))
    valid = (slots < expanded_lengths).expand_as(w)
    w = torch.where(valid, w, 0)
    w = w / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).eps)
    lookback = (expanded_lengths - slots).clamp_min(0)
    expected = (w * lookback).sum(-1)
    adjacent_slot = (slots == (expanded_lengths - 1)).expand_as(w)
    non_adjacent = torch.where(~adjacent_slot, w, 0).sum(-1)
    entropy = -(w * w.clamp_min(torch.finfo(w.dtype).eps).log()).sum(-1)
    usable = valid.any(-1)
    return {
        "expected_lookback": float(expected[usable].mean()) if usable.any() else 0.0,
        "non_adjacent_mass": float(non_adjacent[usable].mean()) if usable.any() else 0.0,
        "entropy": float(entropy[usable].mean()) if usable.any() else 0.0,
    }


def _checkpoint_state(raw: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(raw, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            value = raw.get(key)
            if isinstance(value, Mapping) and value and all(torch.is_tensor(v) for v in value.values()):
                return value
        if raw and all(torch.is_tensor(v) for v in raw.values()):
            return raw
    raise ValueError("checkpoint contains no recognizable model state_dict")


def load_checkpoint_robust(
    model: torch.nn.Module, checkpoint: Path, device: torch.device
) -> dict[str, Any]:
    state = dict(_checkpoint_state(torch.load(checkpoint, map_location=device, weights_only=False)))
    expected = set(model.state_dict())
    candidates = [state]
    for prefix in ("module.", "_orig_mod."):
        candidates += [
            {k.removeprefix(prefix): v for k, v in state.items()},
            {prefix + k: v for k, v in state.items()},
        ]
    best = max(candidates, key=lambda candidate: len(expected & set(candidate)))
    target_state = model.state_dict()
    for key in expected & set(best):
        if best[key].shape == target_state[key].shape:
            continue
        if "puzzle_emb" in key and best[key].ndim == target_state[key].ndim:
            mean = best[key].mean(dim=0, keepdim=True)
            best[key] = mean.expand(target_state[key].shape).contiguous()
        else:
            raise RuntimeError(
                f"{checkpoint}: tensor shape mismatch for {key}: "
                f"{tuple(best[key].shape)} versus {tuple(target_state[key].shape)}"
            )
    result = model.load_state_dict(best, strict=False)
    model.to(device)
    matched = len(expected & set(best))
    if not matched:
        raise RuntimeError(f"{checkpoint}: no checkpoint keys match this model")
    return {
        "matched_keys": matched,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def _model_chain(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    pending = [model]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for name in ("_orig_mod", "model", "inner", "history_aggregator"):
            child = getattr(current, name, None)
            if isinstance(child, torch.nn.Module):
                pending.append(child)


def _history_module(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in _model_chain(model):
        if "history" in type(module).__name__.lower() and "aggregator" not in type(module).__name__.lower():
            return module
        child = getattr(module, "history_aggregator", None)
        if isinstance(child, torch.nn.Module):
            return child
    return None


@contextlib.contextmanager
def history_intervention(model: torch.nn.Module, specification: Mapping[str, Any], seed: int):
    """Activate only an explicit intervention API; never fake an ablation."""
    payload = {**dict(specification), "seed": seed}
    for module in _model_chain(model):
        for factory_name in ("history_intervention", "intervention"):
            factory = getattr(module, factory_name, None)
            if callable(factory):
                kwargs = {"seed": seed} if "seed" in inspect.signature(factory).parameters else {}
                with factory(payload, **kwargs):
                    yield
                return
        for setter_name, clearer_name in (
            ("set_history_intervention", "clear_history_intervention"),
            ("set_intervention", "clear_intervention"),
        ):
            setter = getattr(module, setter_name, None)
            clearer = getattr(module, clearer_name, None)
            if callable(setter) and callable(clearer):
                kwargs = {"seed": seed} if "seed" in inspect.signature(setter).parameters else {}
                setter(payload, **kwargs)
                try:
                    yield
                finally:
                    clearer()
                return
    # Current TRM exposes interventions through an explicit, per-forward
    # analysis_request. Yield the immutable specification to the evaluator
    # instead of storing mutable state on the model.
    for module in _model_chain(model):
        if "analysis_request" in inspect.signature(module.forward).parameters:
            yield payload
            return
    raise RuntimeError(
        "history intervention requested, but no supported hook was found on the "
        "model, inner model, or history aggregator; expected history_intervention "
        "or paired set/clear methods. "
        "Deletion must use attention ranking and Gaussian noise must modify history_z."
    )


def intervention_specs(variant: str) -> list[dict[str, Any]]:
    result = [
        {
            "name": f"gaussian_{sigma:.2f}",
            "kind": "gaussian",
            "sigma_rms": sigma,
            "matched_draw_key": f"gaussian_{sigma:.2f}",
        }
        for sigma in CORRUPTION_SIGMAS
    ]
    if variant == "P1":
        result = [
            {"name": "delete_most_attended", "kind": "delete", "rank": "most"},
            {"name": "delete_least_attended", "kind": "delete", "rank": "least"},
            *result,
        ]
    return result


def _analysis_request(
    model: torch.nn.Module,
    variant: str,
    seed: int,
    batch_index: int,
    act_step: int,
    intervention: Mapping[str, Any] | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {"cycle_logits": True}
    if variant == "P1":
        request.update(attention_weights=True, attention_stats=True)
    if intervention is None:
        return request

    kind = intervention["kind"]
    if kind == "delete":
        request["delete_state"] = {"kind": intervention["rank"]}
        return request
    if kind != "gaussian":
        raise ValueError(f"unknown intervention kind: {kind}")

    h_cycles = l_cycles = 1
    for module in _model_chain(model):
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "H_cycles"):
            h_cycles = int(config.H_cycles)
            l_cycles = int(config.L_cycles)
            break
    # Select a reproducible inner state independently for every batch/ACT
    # call. Identical variant seeds receive identical steps and Gaussian draws.
    draw_seed = int(seed + 1_000_003 * batch_index + 10_007 * act_step)
    rng = random.Random(draw_seed)
    request["corruption"] = {
        "sigma": float(intervention["sigma_rms"]),
        "seed": draw_seed,
        "h_step": rng.randrange(h_cycles),
        "l_step": rng.randrange(l_cycles),
    }
    return request


def _extract_attention(model: torch.nn.Module, outputs: Mapping[str, Any], carry: Any):
    candidates = [outputs.get("attention_weights"), outputs.get("history_attention_weights")]
    for module in _model_chain(model):
        candidates += [
            getattr(module, "last_attention_weights", None),
            getattr(module, "attention_weights", None),
        ]
    lengths = getattr(getattr(carry, "inner_carry", carry), "history_lengths", None)
    for candidate in candidates:
        if torch.is_tensor(candidate):
            return attention_statistics(candidate, lengths)
    return None


def _arch_name_from_defaults(defaults: Sequence[Any]) -> str | None:
    for item in defaults:
        if not isinstance(item, Mapping):
            continue
        for key, value in item.items():
            name = str(key)
            if name in {"arch", "/arch"} or name.startswith("/arch@"):
                return str(value)
    return None


def _resolve_arch_path(config_path: Path, arch_name: str) -> Path:
    for directory in (config_path.parent, *config_path.parents):
        candidate = directory / "arch" / f"{arch_name}.yaml"
        if candidate.exists():
            return candidate
    raise ValueError(
        f"{config_path}: cannot find arch/{arch_name}.yaml in any ancestor"
    )


def _load_config(path: Path, checkpoint_raw: Any | None = None):
    from omegaconf import OmegaConf
    from pretrain import PretrainConfig

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = raw.pop("defaults", [])
    raw.pop("hydra", None)
    if "arch" not in raw:
        arch_name = _arch_name_from_defaults(defaults)
        if not arch_name:
            raise ValueError(f"{path}: architecture config cannot be resolved")
        raw["arch"] = yaml.safe_load(
            _resolve_arch_path(path, arch_name).read_text(encoding="utf-8")
        )
    if isinstance(checkpoint_raw, Mapping):
        embedded = checkpoint_raw.get("config") or checkpoint_raw.get("cfg")
        if isinstance(embedded, Mapping) and isinstance(embedded.get("arch"), Mapping):
            raw["arch"] = dict(embedded["arch"])
    raw.update(compile_model=False, load_checkpoint=None, dataloader_num_workers=0)
    resolved = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
    return PretrainConfig(**resolved)


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    count = max(len(rows), 1)
    return {
        "examples": len(rows),
        "exact_accuracy": sum(bool(row["exact"]) for row in rows) / count,
        "cell_accuracy": sum(float(row["cell_accuracy"]) for row in rows) / count,
        "incorrect_cells": sum(int(row["incorrect_cells"]) for row in rows) / count,
        **{
            f"{unit}_violations": sum(int(row["sudoku_violations"][unit]) for row in rows) / count
            for unit in ("row", "column", "box")
        },
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    variant: str,
    seed: int,
    intervention: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    attention: list[dict[str, float]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch_index, (set_name, batch, effective_size) in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.device(device):
                carry = model.initial_carry(batch)  # type: ignore[attr-defined]
            trajectory: list[np.ndarray] = []
            q_trajectory: list[np.ndarray] = []
            tick = time.perf_counter()
            act_step = 0
            while True:
                analysis_request = _analysis_request(
                    model, variant, seed, batch_index, act_step, intervention
                )
                carry, _, _, output, finished = model(
                    carry=carry,
                    batch=batch,
                    analysis_request=analysis_request,
                    return_keys=(
                        "preds",
                        "logits",
                        "q_halt_logits",
                        "history_attention_weights",
                        "history_cycle_logits",
                    ),
                )
                cycle_logits = output.get("history_cycle_logits")
                if cycle_logits is not None:
                    trajectory.extend(
                        cycle_logits.argmax(-1).cpu().numpy()
                    )
                else:
                    predictions = output.get("preds")
                    if predictions is None:
                        predictions = output["logits"].argmax(-1)
                    trajectory.append(predictions.cpu().numpy())
                if "q_halt_logits" in output:
                    q_trajectory.append(output["q_halt_logits"].float().cpu().numpy())
                stats = _extract_attention(model, output, carry)
                if stats:
                    attention.append(stats)
                act_step += 1
                if bool(finished):
                    break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - tick)
            labels = batch["labels"].cpu().numpy()
            inputs = batch["inputs"].cpu().numpy()
            identifiers = batch["puzzle_identifiers"].cpu().numpy()
            for index in range(min(int(effective_size), labels.shape[0])):
                step_metrics = [example_metrics(step[index], labels[index]) for step in trajectory]
                valid = labels[index] != IGNORE_LABEL_ID
                corrections = regressions = 0
                for before, after in zip(trajectory, trajectory[1:]):
                    before_ok = (before[index] == labels[index]) & valid
                    after_ok = (after[index] == labels[index]) & valid
                    corrections += int(np.count_nonzero(~before_ok & after_ok & valid))
                    regressions += int(np.count_nonzero(before_ok & ~after_ok & valid))
                digest = hashlib.sha1(inputs[index].astype(np.int32).tobytes()).hexdigest()[:12]
                rows.append(
                    {
                        "example_id": (
                            f"{set_name}:{int(identifiers[index])}:{digest}:{len(rows)}"
                        ),
                        "set": str(set_name),
                        "puzzle_identifier": int(identifiers[index]),
                        "inputs": inputs[index].tolist(),
                        "labels": labels[index].tolist(),
                        "predictions": trajectory[-1][index].tolist(),
                        **step_metrics[-1],
                        "trajectory": {
                            "steps": len(trajectory),
                            "exact": [metric["exact"] for metric in step_metrics],
                            "cell_accuracy": [metric["cell_accuracy"] for metric in step_metrics],
                            "q_halt_logits": [float(q[index]) for q in q_trajectory],
                            "corrections": corrections,
                            "regressions": regressions,
                        },
                    }
                )
    wall = time.perf_counter() - started
    metadata: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "metrics": _summary(rows),
        "wall_time_seconds": wall,
        "inference_latency_seconds": {
            "mean_batch": float(np.mean(latencies)) if latencies else 0.0,
            "p50_batch": float(np.median(latencies)) if latencies else 0.0,
        },
        "throughput_examples_per_second": len(rows) / sum(latencies) if latencies else 0.0,
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    if attention:
        metadata["attention"] = {
            key: float(np.mean([item[key] for item in attention])) for key in attention[0]
        }
    return rows, metadata


def _write_artifact(path: Path, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]):
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (path / "examples.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="A common PATH or repeatable VARIANT=PATH",
    )
    parser.add_argument("--checkpoint", action="append", required=True, metavar="VARIANT=PATH")
    parser.add_argument("--data", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("results/study"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--interventions", action="store_true")
    args = parser.parse_args(argv)

    from pretrain import create_dataloader, create_model, resolve_device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = resolve_device(args.device)
    common_config: Path | None = None
    variant_configs: dict[str, Path] = {}
    for assignment in args.config:
        name, separator, value = assignment.partition("=")
        if separator:
            if name not in VARIANTS:
                parser.error(f"unknown config variant {name!r}")
            variant_configs[name] = Path(value)
        elif common_config is None:
            common_config = Path(assignment)
        else:
            parser.error("only one common --config PATH may be supplied")
    for assignment in args.checkpoint:
        variant, separator, value = assignment.partition("=")
        if not separator or variant not in VARIANTS:
            parser.error(f"--checkpoint must be one of {VARIANTS} followed by =PATH")
        checkpoint = Path(value)
        config_path = variant_configs.get(variant, common_config)
        if config_path is None:
            parser.error(f"no --config supplied for {variant}")
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = _load_config(config_path, raw)
        if args.data:
            config.data_paths_test = [str(path) for path in args.data]
        if args.batch_size:
            config.global_batch_size = args.batch_size
        config.seed = args.seed
        loader, metadata = create_dataloader(
            config,
            split=args.split,
            rank=0,
            world_size=1,
            device=device,
            global_batch_size=config.global_batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
        )
        model, _, _ = create_model(config, metadata, 0, 1, device)
        load_info = load_checkpoint_robust(model, checkpoint, device)
        rows, result = evaluate_model(model, loader, device, variant, args.seed)
        result.update(checkpoint=str(checkpoint), checkpoint_load=load_info)
        destination = args.output / variant / f"seed_{args.seed}"
        _write_artifact(destination, rows, result)
        if args.interventions:
            for intervention in intervention_specs(variant):
                with history_intervention(
                    model, intervention, args.seed
                ) as intervention_request:
                    altered_rows, altered = evaluate_model(
                        model,
                        loader,
                        device,
                        variant,
                        args.seed,
                        intervention=intervention_request,
                    )
                altered["intervention"] = intervention
                _write_artifact(destination / intervention["name"], altered_rows, altered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
