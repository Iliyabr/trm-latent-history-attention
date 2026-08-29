#!/usr/bin/env bash
# One-shot canonical GPU campaign (protocol v1).
#
# Four models, seed 0 only. Last checkpoint only (no step_*.pt every eval).
# Label results SCREENING, not confirmatory multi-seed.
#
# Usage (from repo root, inside the venv):
#   bash scripts/run_canonical_server.sh setup
#   bash scripts/run_canonical_server.sh data
#   bash scripts/run_canonical_server.sh train
#   bash scripts/run_canonical_server.sh eval
#   bash scripts/run_canonical_server.sh all
#
# Resume a crashed variant:
#   bash scripts/run_canonical_server.sh resume B0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRESET=canonical
SEED=0
# Highest-value order if the job is killed late: Vanilla, Attention, Gated, B3.
VARIANTS=(B0 P1 Gated B3)
OUTPUT_ROOT=outputs/study
EVAL_OUT=results/canonical-gpu
# 1080 Ti: keep float32 (canonical default). T4 only: uncomment the next line
# after a runtime dtype check.
# DTYPE_OVERRIDE=(--override arch.forward_dtype=bfloat16)
DTYPE_OVERRIDE=()

TRAIN_OVERRIDES=(
  --override checkpoint_every_eval=false
  --override compile_model=false
  --override max_runtime_minutes=null
  "${DTYPE_OVERRIDE[@]}"
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
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability())
    print("dtype_default float32")
else:
    raise SystemExit("CUDA is required for the server campaign")
PY
  mkdir -p "$OUTPUT_ROOT/$PRESET" "$EVAL_OUT" artifacts
  {
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status=$(git status --porcelain | tr '\n' ';')"
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
    echo "===== TRAIN ${variant} seed ${SEED} ====="
    python experiments/run_study.py single \
      --preset "$PRESET" \
      --variant "$variant" \
      --seed "$SEED" \
      --output-root "$OUTPUT_ROOT" \
      "${TRAIN_OVERRIDES[@]}"
  done
}

cmd_resume() {
  local variant="${1:?usage: bash scripts/run_canonical_server.sh resume VARIANT}"
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
  python experiments/run_study.py resume \
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
  python experiments/evaluate_study.py "${args[@]}"
  python experiments/analyze_results.py \
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
    echo "usage: bash scripts/run_canonical_server.sh {setup|data|train|resume VARIANT|eval|all}"
    exit 2
    ;;
esac
