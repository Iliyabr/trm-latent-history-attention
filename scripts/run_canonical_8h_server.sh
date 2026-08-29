#!/usr/bin/env bash
# Longer canonical run — GTX 1080 Ti, ~8 h wall-clock budget.
#
# Differences from scripts/run_canonical_server.sh (1024-epoch screening):
#   - Transformer L-stack (mlp_t=false, RoPE) — NOT the paper MLP variant
#   - ACT halt_max_steps=16 (paper default) instead of 6
#   - max_runtime_minutes wall-clock cap (default 480 min = 8 h)
#   - Separate output dirs so screening artifacts are not overwritten
#
# Timing calibration (seed 0, ACT6 screening run on this card):
#   B0 ~1.1 h, P1 ~1.3 h, Gated ~1.2 h, B3 ~1.1 h for 28 800 steps each.
# ACT16 is slower; the runtime cap stops training safely before 8 h.
#
# Usage (repo root, venv active):
#   bash scripts/run_canonical_8h_server.sh setup
#   bash scripts/run_canonical_8h_server.sh data
#
#   # Recommended: one variant, full 8 h (best chance at higher exact accuracy)
#   bash scripts/run_canonical_8h_server.sh train B0
#
#   # Or all four variants, ~2 h each (~8 h total)
#   bash scripts/run_canonical_8h_server.sh train-all
#
#   bash scripts/run_canonical_8h_server.sh resume B0
#   bash scripts/run_canonical_8h_server.sh eval
#
# Environment overrides:
#   MAX_RUNTIME_MINUTES=480   per-job cap (default 480; train-all uses 120)
#   SEED=0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRESET=canonical_8h
SEED="${SEED:-0}"
VARIANTS=(B0 P1 Gated B3)
OUTPUT_ROOT=outputs/study-8h
EVAL_OUT=results/canonical-gpu-8h

# Full 8 h for a single variant; ~2 h each when running train-all.
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-480}"
TRAIN_ALL_RUNTIME_MINUTES="${TRAIN_ALL_RUNTIME_MINUTES:-120}"

TRAIN_OVERRIDES=(
  --override arch.mlp_t=false
  --override arch.pos_encodings=rope
  --override arch.halt_max_steps=16
  --override checkpoint_every_eval=false
  --override compile_model=false
  --override dataloader_num_workers=1
)

run_dir() {
  local variant="$1"
  echo "${OUTPUT_ROOT}/${PRESET}/${variant}-seed${SEED}"
}

# Prefer best_dev.pt for eval (accuracy chase); fall back to last step / cap.
eval_ckpt() {
  local variant="$1"
  python - "$ROOT/$(run_dir "$variant")" <<'PY'
from pathlib import Path
import sys

run = Path(sys.argv[1])
for name in ("best_dev.pt", "runtime_cap.pt"):
    path = run / name
    if path.exists():
        print(path)
        raise SystemExit(0)
steps = sorted(run.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
if not steps:
    raise SystemExit(f"no checkpoint under {run}")
print(steps[-1])
PY
}

last_resume_ckpt() {
  local variant="$1"
  python - "$ROOT/$(run_dir "$variant")" <<'PY'
from pathlib import Path
import sys

run = Path(sys.argv[1])
cap = run / "runtime_cap.pt"
if cap.exists():
    print(cap)
    raise SystemExit(0)
steps = sorted(run.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
if steps:
    print(steps[-1])
    raise SystemExit(0)
best = run / "best_dev.pt"
if best.exists():
    print(best)
    raise SystemExit(0)
raise SystemExit(f"no resumable checkpoint under {run}")
PY
}

cmd_setup() {
  python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print("gpu", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability())
print("preset", "canonical_8h")
print("arch", "transformer L-stack (mlp_t=false, rope)")
print("act", "halt_max_steps=16")
PY
  mkdir -p "$OUTPUT_ROOT/$PRESET" "$EVAL_OUT" artifacts
  {
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "script=run_canonical_8h_server.sh"
    echo "seed=${SEED}"
    echo "max_runtime_minutes_default=${MAX_RUNTIME_MINUTES}"
    echo "train_all_runtime_minutes=${TRAIN_ALL_RUNTIME_MINUTES}"
    echo "arch=mlp_t=false,pos_encodings=rope,halt_max_steps=16"
    echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    nvidia-smi -L || true
  } | tee "$OUTPUT_ROOT/$PRESET/provenance.txt"
}

cmd_data() {
  if [[ -d data/sudoku-study-v1/train && -d data/sudoku-study-v1/dev && -d data/sudoku-study-v1/test ]]; then
    echo "dataset already present: data/sudoku-study-v1"
    return
  fi
  python dataset/build_sudoku_baseline_v2.py
}

cmd_train_one() {
  local variant="${1:?usage: bash scripts/run_canonical_8h_server.sh train VARIANT}"
  local cap="${2:-$MAX_RUNTIME_MINUTES}"
  echo "===== TRAIN ${variant} seed ${SEED} (cap ${cap} min, ACT16) ====="
  python experiments/run_study.py single \
    --preset "$PRESET" \
    --variant "$variant" \
    --seed "$SEED" \
    --output-root "$OUTPUT_ROOT" \
    --override "max_runtime_minutes=${cap}" \
    "${TRAIN_OVERRIDES[@]}"
}

cmd_train() {
  cmd_train_one "${1:-B0}"
}

cmd_train_all() {
  for variant in "${VARIANTS[@]}"; do
    cmd_train_one "$variant" "$TRAIN_ALL_RUNTIME_MINUTES"
  done
}

cmd_resume() {
  local variant="${1:?usage: bash scripts/run_canonical_8h_server.sh resume VARIANT}"
  local dir
  dir="$(run_dir "$variant")"
  local ckpt
  ckpt="$(last_resume_ckpt "$variant")"
  echo "RESUME ${variant} <- ${ckpt}"
  python experiments/run_study.py resume \
    --preset "$PRESET" \
    --variant "$variant" \
    --seed "$SEED" \
    --output-root "$OUTPUT_ROOT" \
    --checkpoint "$ckpt" \
    --override "max_runtime_minutes=${MAX_RUNTIME_MINUTES}" \
    "${TRAIN_OVERRIDES[@]}"
}

cmd_eval() {
  local args=(
    --config "config/experiment/sudoku_study_${PRESET}.yaml"
    --data data/sudoku-study-v1
    --split test
    --seed "$SEED"
    --device cuda
    --interventions
    --output "$EVAL_OUT"
  )
  for variant in "${VARIANTS[@]}"; do
    local ckpt
    if ! ckpt="$(eval_ckpt "$variant")"; then
      echo "SKIP eval ${variant}: no checkpoint"
      continue
    fi
    echo "EVAL ${variant} <- $ckpt"
    args+=(--checkpoint "${variant}=${ckpt}")
  done
  if [[ ${#args[@]} -le 8 ]]; then
    echo "No checkpoints found under ${OUTPUT_ROOT}/${PRESET}/"
    exit 1
  fi
  python experiments/evaluate_study.py "${args[@]}"
  python experiments/analyze_results.py \
    --input "$EVAL_OUT" \
    --output "$EVAL_OUT/analysis"
}

cmd_all() {
  cmd_setup
  cmd_data
  cmd_train B0
  cmd_eval
}

case "${1:-}" in
  setup) cmd_setup ;;
  data) cmd_data ;;
  train) cmd_train "${2:-B0}" ;;
  train-all) cmd_train_all ;;
  resume) cmd_resume "${2:-}" ;;
  eval) cmd_eval ;;
  all) cmd_all ;;
  *)
    cat <<'EOF'
usage: bash scripts/run_canonical_8h_server.sh COMMAND [ARG]

  setup              GPU + provenance check
  data               build sudoku-study-v1 if missing
  train [VARIANT]    one model, ~8 h cap (default B0)
  train-all          B0,P1,Gated,B3 ~2 h each (~8 h total)
  resume VARIANT     continue after crash or cap
  eval               test eval -> results/canonical-gpu-8h/
  all                setup + data + train B0 + eval

Examples:
  tmux new -s trm8h
  bash scripts/run_canonical_8h_server.sh train B0
  bash scripts/run_canonical_8h_server.sh train Gated
  MAX_RUNTIME_MINUTES=600 bash scripts/run_canonical_8h_server.sh train B0
EOF
    exit 2
    ;;
esac
