#!/usr/bin/env bash
# One-shot canonical GPU campaign (protocol v1) — RTX 4090 server.
#
# Hardware: RTX 4090 24 GB, bfloat16, CPU cap 16 threads / 4 loader workers.
# Seed 1 only. The other live GPU box uses scripts/run_canonical_server.sh
# (seed 0, float32). Do not average the two machines as one matched family
# (protocol §22: one GPU/dtype regime per paired comparison).
#
# Four models: B0, P1, Gated, B3. Last checkpoint only.
# Label results SCREENING (single seed on this card).
#
# Usage (from repo root, inside the venv):
#   bash scripts/run_canonical_4090.sh setup
#   bash scripts/run_canonical_4090.sh data
#   bash scripts/run_canonical_4090.sh train
#   bash scripts/run_canonical_4090.sh eval
#   bash scripts/run_canonical_4090.sh all
#
# Resume a crashed variant:
#   bash scripts/run_canonical_4090.sh resume B0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRESET=canonical
SEED=1
VARIANTS=(B0 P1 Gated B3)
OUTPUT_ROOT=outputs/study-4090
EVAL_OUT=results/canonical-gpu-4090
CPU_THREADS=16
DATALOADER_WORKERS=4

# Host cap: 16 cores (user limit 15–20). OMP=2 so four loader workers cannot
# explode BLAS threads. taskset is applied at launch when it exists.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

launch() {
  if command -v taskset >/dev/null 2>&1; then
    taskset -c 0-15 "$@"
  else
    "$@"
  fi
}

TRAIN_OVERRIDES=(
  --override checkpoint_every_eval=false
  --override compile_model=false
  --override max_runtime_minutes=null
  --override arch.forward_dtype=bfloat16
  --override dataloader_num_workers="${DATALOADER_WORKERS}"
)

run_dir() {
  local variant="$1"
  echo "${OUTPUT_ROOT}/${PRESET}/${variant}-seed${SEED}"
}

last_ckpt() {
  local variant="$1"
  python - "$ROOT/$(run_dir "$variant")" <<'PY'
from pathlib import Path
import sys
run = Path(sys.argv[1])
caps = run / "runtime_cap.pt"
if caps.exists():
    print(caps)
    raise SystemExit(0)
steps = sorted(run.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
if not steps:
    raise SystemExit(f"no last checkpoint under {run}")
print(steps[-1])
PY
}

cmd_setup() {
  python - "$CPU_THREADS" <<'PY'
import os
import sys
import torch

cpu_cap = int(sys.argv[1])
torch.set_num_threads(cpu_cap)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("omp_num_threads", os.environ.get("OMP_NUM_THREADS"))
print("torch_num_threads", torch.get_num_threads())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the 4090 campaign")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability()
print("gpu", name)
print("capability", cap)
print("vram_bytes", torch.cuda.get_device_properties(0).total_memory)
bf = torch.cuda.is_bf16_supported()
print("bf16_supported", bf)
x = torch.ones(1, device="cuda", dtype=torch.bfloat16)
print("bf16_runtime_ok", float((x * 2).float()))
if cap < (8, 0):
    raise SystemExit(f"expected Ampere+ for bfloat16, got {name} {cap}")
if "4090" not in name and "4080" not in name:
    print("WARNING: this script is tuned for RTX 4090; found", name)
PY
  mkdir -p "$OUTPUT_ROOT/$PRESET" "$EVAL_OUT" artifacts
  {
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "script=run_canonical_4090.sh"
    echo "seed=${SEED}"
    echo "dtype=bfloat16"
    echo "cpu_threads=${CPU_THREADS}"
    echo "dataloader_workers=${DATALOADER_WORKERS}"
    echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status=$(git status --porcelain | tr '\n' ';')"
    nproc --all 2>/dev/null || true
    nvidia-smi -L || true
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
  } | tee "$OUTPUT_ROOT/$PRESET/provenance.txt"
}

cmd_data() {
  if [[ -d data/sudoku-study-v1/train && -d data/sudoku-study-v1/dev && -d data/sudoku-study-v1/test ]]; then
    echo "dataset already present: data/sudoku-study-v1"
    return
  fi
  python dataset/build_sudoku_baseline_v2.py
}

cmd_train() {
  for variant in "${VARIANTS[@]}"; do
    echo "===== TRAIN ${variant} seed ${SEED} (4090 / bfloat16) ====="
    launch python experiments/run_study.py single \
      --preset "$PRESET" \
      --variant "$variant" \
      --seed "$SEED" \
      --output-root "$OUTPUT_ROOT" \
      "${TRAIN_OVERRIDES[@]}"
  done
}

cmd_resume() {
  local variant="${1:?usage: bash scripts/run_canonical_4090.sh resume VARIANT}"
  local dir
  dir="$(run_dir "$variant")"
  local ckpt=""
  if [[ -f "$dir/runtime_cap.pt" ]]; then
    ckpt="$dir/runtime_cap.pt"
  elif compgen -G "$dir/step_*.pt" > /dev/null; then
    ckpt="$(last_ckpt "$variant")"
  elif [[ -f "$dir/best_dev.pt" ]]; then
    echo "WARNING: resuming from best_dev.pt (no last-step checkpoint yet)"
    ckpt="$dir/best_dev.pt"
  else
    echo "No checkpoint in $dir; start train from scratch instead."
    exit 1
  fi
  launch python experiments/run_study.py resume \
    --preset "$PRESET" \
    --variant "$variant" \
    --seed "$SEED" \
    --output-root "$OUTPUT_ROOT" \
    --checkpoint "$ckpt" \
    "${TRAIN_OVERRIDES[@]}"
}

cmd_eval() {
  local args=(
    --config config/experiment/sudoku_study_canonical.yaml
    --data data/sudoku-study-v1
    --split test
    --seed "$SEED"
    --device cuda
    --interventions
    --output "$EVAL_OUT"
  )
  for variant in "${VARIANTS[@]}"; do
    local ckpt
    if ! ckpt="$(last_ckpt "$variant")"; then
      echo "SKIP eval ${variant}: no last checkpoint"
      continue
    fi
    echo "EVAL ${variant} <- $ckpt"
    args+=(--checkpoint "${variant}=${ckpt}")
  done
  launch python experiments/evaluate_study.py "${args[@]}"
  launch python experiments/analyze_results.py \
    --input "$EVAL_OUT" \
    --output "$EVAL_OUT/analysis"
}

cmd_all() {
  cmd_setup
  cmd_data
  cmd_train
  cmd_eval
}

case "${1:-all}" in
  setup) cmd_setup ;;
  data) cmd_data ;;
  train) cmd_train ;;
  resume) cmd_resume "${2:-}" ;;
  eval) cmd_eval ;;
  all) cmd_all ;;
  *)
    echo "usage: bash scripts/run_canonical_4090.sh {setup|data|train|resume VARIANT|eval|all}"
    exit 2
    ;;
esac
