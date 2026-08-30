# RUNBOOK — TRM Latent-History Project

This document records the supported execution workflow for the project.

The canonical CPU training campaign is already complete. The training commands
below are retained for **reproduction**, not for adding new post-hoc training
variants to the frozen CPU evidence.

## 1. Local CPU environment

```powershell
conda create -n trm-cpu -c defaults --override-channels python=3.12 -y
conda activate trm-cpu

python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements-analysis.txt
```

Verified development environment:

```text
Python 3.12.x
PyTorch 2.7.0+cpu
device = cpu
compile_model = false
dataloader_num_workers = 0
```

## 2. Smoke pipeline

Generate the deterministic toy Sudoku data:

```powershell
python scripts/create_tiny_sudoku_dataset.py
```

Run the CPU smoke configuration:

```powershell
python pretrain.py --config-name cfg_cpu_smoke
```

This validates the pipeline only. The toy dataset is **not** scientific
evidence.

## 3. Unit tests

Canonical L-cycle modules:

```powershell
python -m pytest tests/test_lcycle_lowrank_history.py -q
```

Frozen observed result:

```text
4 passed
```

## 4. Canonical CPU configuration

The base configuration is `config/cfg_baseline_v2.yaml` with matched overrides:

```text
hidden_size = 64
halt_max_steps = 4
H_cycles = 3
L_cycles = 6
L_layers = 2
spatial_heads = 4

epochs = 40
global_batch_size = 4
lr = 1e-3
weight_decay = 0.01
EMA = false
device = cpu
```

HistoryAttention:

```text
rank = 16
temporal_heads = 4
gate_logit_init = -2
pre-QKV RMSNorm = true
```

## 5. Reproduce one canonical CPU run

### Vanilla

```powershell
$seed = 0
$run = "proposal-h3l6-l2-vanilla-40ep-seed$seed"

python pretrain.py --config-name cfg_baseline_v2 `
  arch.halt_max_steps=4 `
  arch.H_cycles=3 `
  arch.L_cycles=6 `
  arch.L_layers=2 `
  +arch.history_enabled=false `
  +arch.history_aggregator=none `
  +arch.lcycle_history_enabled=false `
  epochs=40 `
  eval_interval=5 `
  seed=$seed `
  run_name=$run `
  checkpoint_path="checkpoints/lcycle-history/$run" `
  metrics_dir="results/lcycle-history/$run"
```

### HistoryAttention

```powershell
$seed = 0
$run = "proposal-h3l6-l2-attention-40ep-seed$seed"

python pretrain.py --config-name cfg_baseline_v2 `
  arch.halt_max_steps=4 `
  arch.H_cycles=3 `
  arch.L_cycles=6 `
  arch.L_layers=2 `
  +arch.history_enabled=false `
  +arch.history_aggregator=none `
  +arch.lcycle_history_enabled=true `
  +arch.lcycle_history_method=attention `
  +arch.lcycle_history_rank=16 `
  +arch.lcycle_history_heads=4 `
  +arch.lcycle_history_gate_init=-2.0 `
  +arch.lcycle_history_pre_norm=true `
  epochs=40 `
  eval_interval=5 `
  seed=$seed `
  run_name=$run `
  checkpoint_path="checkpoints/lcycle-history/$run" `
  metrics_dir="results/lcycle-history/$run"
```

### Gated Uniform History

Use the same command as Attention with:

```text
lcycle_history_method=gated
run name: proposal-h3l6-l2-gated-40ep-seedN
```

### Parameter-Matched No-History

Use the same command as Attention with:

```text
lcycle_history_method=parameter_matched
run name: proposal-h3l6-l2-parammatched-40ep-seedN
```

The Parameter-Matched branch must receive **no history tensor**.

## 6. Canonical run matrix

CPU:

```text
Vanilla:        seeds 0,1,2,3,4
Attention:      seeds 0,1,2,3,4
Gated:          seeds 0,1,2,3,4
ParameterMatch: seeds 0,1,2,3,4
```

Final step:

```text
10,000
```

Evaluation checkpoints:

```text
1250
2500
3750
5000
6250
7500
8750
10000
```

Do not add new training variants to the canonical CPU matrix under Protocol v1.

## 7. Reproduce CPU statistics and figures

Input:

```text
docs/data/CPU_LCYCLE_ALL_METRICS_v1.csv
```

Run:

```powershell
python scripts/analyze_cpu_lcycle.py `
  --input docs/data/CPU_LCYCLE_ALL_METRICS_v1.csv `
  --output-dir docs/figures
```

Expected paper figures:

```text
docs/figures/cpu_paired_delta_vs_vanilla.png
docs/figures/cpu_learning_curves_accuracy.png
docs/figures/cpu_final_seed_accuracy.png
```

## 8. Extract final learned gates

```powershell
python scripts/extract_lcycle_gates.py `
  --checkpoint-root checkpoints/lcycle-history
```

The script reads checkpoints only; it does not modify them.

Reference initialization:

```text
gate_logit = -2
sigmoid(gate_logit) = 0.119203
```

## 9. Artifact policy

Tracked / paper-facing:

```text
docs/data/CPU_LCYCLE_ALL_METRICS_v1.csv
docs/figures/*
docs/*FINAL*.md
scripts/*
tests/*
models/history/*
```

Local-only:

```text
checkpoints/
logs/
results/lcycle-history/
results/history-depth/ raw run directories
results/history-multiseed/ raw run directories
```

Tracked summary JSONs under `results/` remain tracked.

No non-empty local artifact should be deleted during project cleanup. Historical
material is moved to `docs/archive/` or `.local_archive/` instead.

## 10. GPU canonical workflow

The GPU campaign must follow:

```text
docs/TRM_HISTORY_CANONICAL_PROTOCOL_v1.md
docs/TRM_HISTORY_DEPLOYMENT_MANIFEST_v1.md
```

Before long runs:

```text
1. Verify canonical pre-QKV HistoryAttention implementation.
2. Verify Gated and Parameter-Matched controls.
3. Run unit/smoke tests.
4. Profile Vanilla ACT6 and ACT16.
5. Verify one common effective batch.
6. Record actual GPU, dtype, PyTorch/CUDA, VRAM, and sec/step.
7. Select ACT using feasibility only.
8. Select one fixed optimization-step budget N using Vanilla dev + compute only.
9. Lock the deployment manifest.
10. Run the matched seed matrix.
```

Canonical GPU scientific backbone:

```text
D = 256
H_cycles = 3
L_cycles = 6
L_layers = 2
spatial_heads = 4

Attention rank = 64
temporal_heads = 4
gate_logit_init = -2
pre-QKV RMSNorm = true

optimizer = AdamW
lr = 1e-4
beta = (0.9, 0.95)
weight_decay = 0.1
EMA = 0.999

dataset = sudoku-study-v1
effective batch target = 32
minimum seeds = 0,1,2
```

The final numerical ACT, physical batch, gradient accumulation, dtype, GPU, and
fixed update budget belong in the deployment manifest and must not be guessed
from this RUNBOOK.

## 11. Final paper update

When GPU results are frozen:

```text
1. Fill the deployment manifest.
2. Record per-seed GPU metrics and paired deltas.
3. Create the final GPU result table.
4. Update docs/FINAL_PROJECT_STATUS.md.
5. Replace GPU placeholders in the paper.
6. Run a final provenance audit.
7. Freeze the final paper PDF.
```
