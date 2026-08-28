from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import json
import math
import random
import sys
import time
import hashlib
import yaml
import shutil
import copy

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
try:
    import wandb
except ImportError:  # Local JSONL logging and checkpoints do not require W&B.
    wandb = None  # type: ignore[assignment]
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
try:
    from adam_atan2 import AdamATan2
except ImportError:  # Optional for the CPU-first configuration.
    AdamATan2 = None  # type: ignore[assignment,misc]

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    eval_split: str = "test"
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    metrics_dir: Optional[str] = None
    metrics_jsonl: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

    # Runtime
    device: str = "auto"
    compile_model: bool = True
    dataloader_num_workers: int = 1
    cpu_threads: Optional[int] = None
    optimizer: str = "adam_atan2"
    wandb_mode: str = "online"
    max_runtime_minutes: Optional[float] = None
    deterministic: bool = True
    best_dev_metric: str = "exact_accuracy"
    best_dev_mode: str = "max"
    plateau_patience_evals: int = 3

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return resolved


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, device: torch.device, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=None,
        num_workers=config.dataloader_num_workers,
        pin_memory=device.type == "cuda",
    )
    if config.dataloader_num_workers > 0:
        loader_kwargs.update(prefetch_factor=8, persistent_workers=True)
    dataloader = DataLoader(**loader_kwargs)
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int, device: torch.device):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        model = model.to(device)
        if config.compile_model and "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config, device)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    def dense_optimizer():
        if config.optimizer == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            )
        if config.optimizer != "adam_atan2":
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        if AdamATan2 is None:
            raise RuntimeError(
                "optimizer=adam_atan2 requires the optional adam-atan2 package. "
                "Use optimizer=adamw for the CPU configuration."
            )
        return AdamATan2(
            model.parameters(),
            lr=0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )

    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            dense_optimizer()
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            dense_optimizer()
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int, device: torch.device):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size, device=device)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_train_state(
    config: PretrainConfig,
    train_state: TrainState,
    ema_helper: Optional[EMAHelper] = None,
    model_for_evaluation: Optional[nn.Module] = None,
    filename: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Atomically save all state required for an eval-boundary resume.

    Carry is deliberately omitted unless it is ``None``. A live ACT carry is
    batch-specific and can retain an autograd graph, so replaying it with the
    next loader batch is unsafe. Study checkpoints are written at evaluation
    boundaries where training restarts with a fresh carry.
    """
    if config.checkpoint_path is None:
        return None

    os.makedirs(config.checkpoint_path, exist_ok=True)
    output = os.path.join(
        config.checkpoint_path, filename or f"step_{train_state.step}.pt"
    )
    temporary = output + ".tmp"
    payload = {
        "schema_version": 2,
        # "model" remains compatible with model-only evaluation/loading. When
        # EMA is used it can hold the exact weights that produced dev metrics.
        "model": (
            model_for_evaluation.state_dict()
            if model_for_evaluation is not None
            else train_state.model.state_dict()
        ),
        # Resume always pairs optimizer moments with the actual training model.
        "training_model": train_state.model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in train_state.optimizers],
        "optimizer_lrs": list(train_state.optimizer_lrs),
        "step": train_state.step,
        "total_steps": train_state.total_steps,
        "scheduler": {
            "kind": "cosine_with_warmup",
            "step": train_state.step,
            "warmup_steps": config.lr_warmup_steps,
            "min_ratio": config.lr_min_ratio,
        },
        "carry": None,
        "carry_resume_safe": train_state.carry is None,
        "ema": ema_helper.state_dict() if ema_helper is not None else None,
        "config": config.model_dump(mode="json"),
        "rng_state": _rng_state(),
        "extra": extra or {},
    }
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return output


def restore_train_state(
    config: PretrainConfig,
    train_state: TrainState,
    ema_helper: Optional[EMAHelper],
    device: torch.device,
) -> dict[str, Any]:
    if config.resume_checkpoint is None:
        return {}
    print(f"Resuming complete checkpoint {config.resume_checkpoint}")
    payload = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "optimizers" not in payload:
        raise ValueError(
            "resume_checkpoint must be a complete v2 checkpoint; use "
            "load_checkpoint for model-only weights"
        )
    train_state.model.load_state_dict(payload.get("training_model", payload["model"]))
    if len(payload["optimizers"]) != len(train_state.optimizers):
        raise ValueError("Checkpoint optimizer count does not match this configuration")
    for optimizer, state in zip(train_state.optimizers, payload["optimizers"]):
        optimizer.load_state_dict(state)
    train_state.step = int(payload["step"])
    # Preserve the current planned duration while restoring scheduler position.
    if train_state.step > train_state.total_steps:
        raise ValueError("Checkpoint step is beyond the configured training duration")
    train_state.carry = payload.get("carry") if payload.get("carry_resume_safe") else None
    if ema_helper is not None and payload.get("ema") is not None:
        ema_helper.load_state_dict(payload["ema"])
    if payload.get("rng_state"):
        _restore_rng_state(payload["rng_state"])
    return dict(payload.get("extra") or {})


def load_checkpoint(model: nn.Module, config: PretrainConfig, device: torch.device):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        loaded = torch.load(config.load_checkpoint, map_location=device, weights_only=False)
        state_dict = loaded.get("model", loaded) if isinstance(loaded, dict) else loaded

        # Resize and reset puzzle embedding only when this architecture uses one.
        puzzle_emb_ndim = int(
            (config.arch.__pydantic_extra__ or {}).get("puzzle_emb_ndim", 0)
        )

        if puzzle_emb_ndim > 0:
            puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
            expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore

            if puzzle_emb_name in state_dict:
                puzzle_emb = state_dict[puzzle_emb_name]
                if puzzle_emb.shape != expected_shape:
                    print(
                        f"Resetting puzzle embedding as shape is different. "
                        f"Found {puzzle_emb.shape}, Expected {expected_shape}"
                    )
                    state_dict[puzzle_emb_name] = (
                        torch.mean(puzzle_emb, dim=0, keepdim=True)
                        .expand(expected_shape)
                        .contiguous()
                    )
        target_state = model.state_dict()
        exact = (
            set(state_dict) == set(target_state)
            and all(
                state_dict[key].shape == target_state[key].shape
                for key in state_dict
            )
        )
        if exact:
            model.load_state_dict(state_dict, assign=True)
        else:
            # Model-only checkpoints may warm-start a history ablation. Keep
            # every shape-compatible backbone tensor while leaving new P1
            # projections (or widened B3 FFN slices) freshly initialized.
            compatible = {}
            resized = 0
            for key, value in state_dict.items():
                if key not in target_state:
                    continue
                target = target_state[key]
                if value.shape == target.shape:
                    compatible[key] = value
                elif value.ndim == target.ndim:
                    # B3 widens existing FFN matrices. Preserve the old block
                    # in the overlapping prefix and initialize only new rows /
                    # columns from the current model.
                    merged = target.clone()
                    overlap = tuple(
                        slice(0, min(old, new))
                        for old, new in zip(value.shape, target.shape)
                    )
                    merged[overlap] = value[overlap]
                    compatible[key] = merged
                    resized += 1
            incompatible = model.load_state_dict(
                compatible, strict=False, assign=True
            )
            print(
                "Partially loaded model checkpoint: "
                f"{len(compatible)}/{len(target_state)} tensors matched; "
                f"{resized} resized; "
                f"{len(incompatible.missing_keys)} initialized from config."
            )


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int, device: torch.device):
    if train_state.step >= train_state.total_steps:  # At most total_steps updates.
        return
    train_state.step += 1

    # To device
    batch = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device(device):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
    device: torch.device,
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in batch.items()}
            with torch.device(device):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device=device
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code when W&B is active; local provenance is always written.
    if wandb is not None and wandb.run is not None:
        wandb.run.log_code(config.checkpoint_path)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def append_metrics(config: PretrainConfig, record: dict[str, Any]) -> None:
    path = config.metrics_jsonl
    if path is None and config.metrics_dir is not None:
        path = os.path.join(config.metrics_dir, "metrics.jsonl")
    if path is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_value(record), sort_keys=True) + "\n")


def metric_by_suffix(metrics: dict[str, Any], suffix: str) -> Optional[float]:
    matches: list[float] = []

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}/{key}" if path else str(key))
        elif path == suffix or path.endswith("/" + suffix):
            try:
                matches.append(float(value))
            except (TypeError, ValueError):
                pass

    visit(metrics)
    return matches[0] if matches else None


def write_run_metadata(
    config: PretrainConfig, device: torch.device, train_state: TrainState
) -> None:
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    config_json = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    metadata = {
        "schema_version": 1,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "seed": config.seed,
        "deterministic": config.deterministic,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device_type": device.type,
        "cuda_version": torch.version.cuda,
        "total_steps": train_state.total_steps,
    }
    with open(
        os.path.join(config.checkpoint_path, "run_metadata.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    config = PretrainConfig(**hydra_config)  # Resolve runtime settings before distributed setup.
    device = resolve_device(config.device)
    if device.type == "cpu" and config.cpu_threads is not None:
        torch.set_num_threads(config.cpu_threads)

    print(f"Runtime device: {device}; PyTorch: {torch.__version__}; threads: {torch.get_num_threads()}")

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        if device.type != "cuda":
            raise RuntimeError("Distributed CPU training is not supported by this entry point.")
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    device = resolve_device(config.device)
    if config.best_dev_mode not in {"min", "max"}:
        raise ValueError("best_dev_mode must be 'min' or 'max'")
    if config.max_runtime_minutes is not None and config.max_runtime_minutes <= 0:
        raise ValueError("max_runtime_minutes must be positive when set")
    if config.plateau_patience_evals < 1:
        raise ValueError("plateau_patience_evals must be at least 1")

    # Seed all RNGs to ensure consistency.
    process_seed = config.seed + RANK
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.random.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    if config.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE, device=device)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, config.eval_split, test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE, device=device)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE, device=device)

    # Progress bar, checkpoint state, and logger
    progress_bar = None
    ema_helper = None
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)
    resume_extra = restore_train_state(config, train_state, ema_helper, device)
    best_metric = resume_extra.get("best_metric")
    no_improvement_evals = int(resume_extra.get("no_improvement_evals", 0))
    if RANK == 0:
        progress_bar = tqdm.tqdm(
            total=train_state.total_steps,
            initial=train_state.step,
            file=sys.stdout,
            mininterval=2.0,
            dynamic_ncols=True,
        )
        if wandb is not None:
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config.model_dump(),
                mode=config.wandb_mode,
                settings=wandb.Settings(_disable_stats=True),
            )
            wandb.log(
                {"num_params": sum(x.numel() for x in train_state.model.parameters())},
                step=train_state.step,
            )
        save_code_and_config(config)
        write_run_metadata(config, device, train_state)
        append_metrics(
            config,
            {
                "event": "run_start",
                "run_name": config.run_name,
                "seed": config.seed,
                "step": train_state.step,
                "total_steps": train_state.total_steps,
                "resumed_from": config.resume_checkpoint,
                "num_params": sum(x.numel() for x in train_state.model.parameters()),
            },
        )

    # Training Loop
    steps_per_iter = max(1, train_state.total_steps // max(1, total_iters))
    start_iter = min(
        total_iters,
        int(resume_extra.get("completed_iters", train_state.step // steps_per_iter)),
    )
    # PuzzleDataset seeds each iterator from this counter.
    if hasattr(train_loader.dataset, "_iters"):
        train_loader.dataset._iters = start_iter
    run_started = time.perf_counter()
    examples_seen = 0
    runtime_cap_reached = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _iter_id in range(start_iter, total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE, device=device)
            if metrics is None:
                break
            examples_seen += global_batch_size

            if RANK == 0 and metrics is not None:
                elapsed = time.perf_counter() - run_started
                metrics["runtime/wall_seconds"] = elapsed
                metrics["runtime/examples_per_second"] = (
                    examples_seen / elapsed if elapsed else 0.0
                )
                metrics["runtime/peak_vram_bytes"] = (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0
                )
                if wandb is not None and wandb.run is not None:
                    wandb.log(metrics, step=train_state.step)
                append_metrics(
                    config,
                    {
                        "event": "train",
                        "run_name": config.run_name,
                        "seed": config.seed,
                        "step": train_state.step,
                        "metrics": metrics,
                    },
                )
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)
            if (
                config.max_runtime_minutes is not None
                and time.perf_counter() - run_started
                >= config.max_runtime_minutes * 60
            ):
                runtime_cap_reached = True

        if WORLD_SIZE > 1:
            cap_flag = torch.tensor(
                int(runtime_cap_reached), device=device, dtype=torch.int32
            )
            dist.all_reduce(cap_flag, op=dist.ReduceOp.MAX)
            runtime_cap_reached = bool(cap_flag.item())

        if runtime_cap_reached:
            if RANK == 0:
                save_train_state(
                    config,
                    train_state,
                    ema_helper,
                    filename="runtime_cap.pt",
                    extra={
                        "best_metric": best_metric,
                        "no_improvement_evals": no_improvement_evals,
                        "completed_iters": _iter_id + 1,
                    },
                )
                append_metrics(
                    config,
                    {
                        "event": "runtime_cap",
                        "step": train_state.step,
                        "runtime_seconds": time.perf_counter() - run_started,
                    },
                )
            break

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.copy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state.model)
                train_state_eval.carry = None
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config, 
                train_state_eval, 
                eval_loader, 
                eval_metadata, 
                evaluators,
                rank=RANK, 
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP,
                device=device)

            if RANK == 0 and metrics is not None:
                if wandb is not None and wandb.run is not None:
                    wandb.log(metrics, step=train_state.step)

                if config.metrics_dir is not None:
                    os.makedirs(config.metrics_dir, exist_ok=True)
                    metrics_file = os.path.join(
                        config.metrics_dir,
                        f"metrics_step_{train_state.step}.json",
                    )
                    metrics_record = {
                        "run_name": config.run_name,
                        "step": train_state.step,
                        "eval_split": config.eval_split,
                        "metrics": metrics,
                    }
                    with open(metrics_file, "w", encoding="utf-8") as f:
                        json.dump(
                            metrics_record,
                            f,
                            indent=2,
                            default=lambda x: x.item() if hasattr(x, "item") else str(x),
                        )
                    print(f"SAVED METRICS: {metrics_file}")

                candidate = metric_by_suffix(metrics, config.best_dev_metric)
                improved = False
                if candidate is not None:
                    improved = best_metric is None or (
                        candidate > best_metric
                        if config.best_dev_mode == "max"
                        else candidate < best_metric
                    )
                    if improved:
                        best_metric = candidate
                        no_improvement_evals = 0
                        save_train_state(
                            config,
                            train_state,
                            ema_helper,
                            model_for_evaluation=train_state_eval.model,
                            filename="best_dev.pt",
                            extra={
                                "best_metric": best_metric,
                                "best_metric_name": config.best_dev_metric,
                                "no_improvement_evals": 0,
                                "completed_iters": _iter_id + 1,
                            },
                        )
                    else:
                        no_improvement_evals += 1
                append_metrics(
                    config,
                    {
                        "event": "eval",
                        "run_name": config.run_name,
                        "seed": config.seed,
                        "step": train_state.step,
                        "eval_split": config.eval_split,
                        "metrics": metrics,
                        "best_metric_name": config.best_dev_metric,
                        "best_metric": best_metric,
                        "improved": improved,
                        "no_improvement_evals": no_improvement_evals,
                        "plateau": no_improvement_evals
                        >= config.plateau_patience_evals,
                        "runtime_seconds": time.perf_counter() - run_started,
                    },
                )
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(
                    config,
                    train_state,
                    ema_helper,
                    model_for_evaluation=train_state_eval.model,
                    extra={
                        "best_metric": best_metric,
                        "no_improvement_evals": no_improvement_evals,
                        "completed_iters": _iter_id + 1,
                    },
                )

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    if RANK == 0:
        elapsed = time.perf_counter() - run_started
        append_metrics(
            config,
            {
                "event": "run_end",
                "step": train_state.step,
                "runtime_seconds": elapsed,
                "examples_seen": examples_seen,
                "examples_per_second": examples_seen / elapsed if elapsed else 0.0,
                "peak_vram_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0
                ),
                "runtime_cap_reached": runtime_cap_reached,
                "best_metric": best_metric,
                "no_improvement_evals": no_improvement_evals,
                "plateau": no_improvement_evals >= config.plateau_patience_evals,
            },
        )
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    launch()
