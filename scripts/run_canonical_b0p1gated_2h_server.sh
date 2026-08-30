#!/usr/bin/env bash
# B0 + P1 + Gated canonical run — GTX 1080 Ti, 2 h per model (~6 h total).
#
# Same training recipe as run_canonical_8h_server.sh (Transformer + RoPE, ACT16).
# Skips B3. Uses separate output dirs from screening and full 8h runs.
#
# Usage (repo root, venv active):
#   bash scripts/run_canonical_b0p1gated_2h_server.sh setup
#   bash scripts/run_canonical_b0p1gated_2h_server.sh data
#   bash scripts/run_canonical_b0p1gated_2h_server.sh train-all
#   bash scripts/run_canonical_b0p1gated_2h_server.sh eval
#
#   bash scripts/run_canonical_b0p1gated_2h_server.sh train B0   # one model, 2 h
#   bash scripts/run_canonical_b0p1gated_2h_server.sh resume P1
#
# Environment overrides:
#   RUNTIME_MINUTES=120   per-model cap (default 120 = 2 h)
#   SEED=0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRESET=canonical_8h
SEED="${SEED:-0}"
VARIANTS=(B0 P1 Gated)
OUTPUT_ROOT=outputs/study-b0p1gated-2h
EVAL_OUT=results/canonical-gpu-b0p1gated-2h
RUNTIME_MINUTES="${RUNTIME_MINUTES:-120}"

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
print("variants", "B0 P1 Gated")
print("runtime_minutes_per_model", 120)
print("preset", "canonical_8h (ACT16, transformer)")
PY
  mkdir -p "$OUTPUT_ROOT/$PRESET" "$EVAL_OUT" artifacts
  {
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "script=run_canonical_b0p1gated_2h_server.sh"
    echo "seed=${SEED}"
    echo "runtime_minutes=${RUNTIME_MINUTES}"
    echo "variants=B0,P1,Gated"
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
  local variant="${1:?usage: train VARIANT}"
  echo "===== TRAIN ${variant} seed ${SEED} (cap ${RUNTIME_MINUTES} min) ====="
  python experiments/run_study.py single \
    --preset "$PRESET" \
    --variant "$variant" \
    --seed "$SEED" \
    --output-root "$OUTPUT_ROOT" \
    --override "max_runtime_minutes=${RUNTIME_MINUTES}" \
    "${TRAIN_OVERRIDES[@]}"
}

cmd_train() {
  cmd_train_one "${1:?usage: bash scripts/run_canonical_b0p1gated_2h_server.sh train VARIANT}"
}

cmd_train_all() {
  for variant in "${VARIANTS[@]}"; do
    cmd_train_one "$variant"
  done
}

cmd_resume() {
  local variant="${1:?usage: bash scripts/run_canonical_b0p1gated_2h_server.sh resume VARIANT}"
  local ckpt
  ckpt="$(last_resume_ckpt "$variant")"
  echo "RESUME ${variant} <- ${ckpt}"
  python experiments/run_study.py resume \
    --preset "$PRESET" \
    --variant "$variant" \
    --seed "$SEED" \
    --output-root "$OUTPUT_ROOT" \
    --checkpoint "$ckpt" \
    --override "max_runtime_minutes=${RUNTIME_MINUTES}" \
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
  local found=0
  for variant in "${VARIANTS[@]}"; do
    local ckpt
    if ! ckpt="$(eval_ckpt "$variant")"; then
      echo "SKIP eval ${variant}: no checkpoint"
      continue
    fi
    found=1
    echo "EVAL ${variant} <- $ckpt"
    args+=(--checkpoint "${variant}=${ckpt}")
  done
  if [[ "$found" -eq 0 ]]; then
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
  cmd_train_all
  cmd_eval
}

case "${1:-all}" in
  setup) cmd_setup ;;
  data) cmd_data ;;
  train) cmd_train "${2:-}" ;;
  train-all) cmd_train_all ;;
  resume) cmd_resume "${2:-}" ;;
  eval) cmd_eval ;;
  all) cmd_all ;;
  *)
    cat <<'EOF'
usage: bash scripts/run_canonical_b0p1gated_2h_server.sh COMMAND [ARG]

  setup        GPU check + provenance
  data         build sudoku-study-v1 if missing
  train-all    B0, P1, Gated — 2 h each (~6 h total)
  train VAR    one variant, 2 h (B0 | P1 | Gated)
  resume VAR   continue after crash or cap
  eval         test eval -> results/canonical-gpu-b0p1gated-2h/
  all          setup + data + train-all + eval

Examples:
  tmux new -s trm3
  bash scripts/run_canonical_b0p1gated_2h_server.sh all
  RUNTIME_MINUTES=150 bash scripts/run_canonical_b0p1gated_2h_server.sh train P1
EOF
    exit 2
    ;;
esac
