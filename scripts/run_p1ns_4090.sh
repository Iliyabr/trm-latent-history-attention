#!/usr/bin/env bash
# P1ns (no-skip attention) — RTX 4090, ACT6, L_cycles ∈ {6, 10}, seeds 1–2.
#
# Ablation: readout = RMSNorm(memory) only (no z + gate·memory residual).
# Matches refined 1080 Ti seed-0 budget (epochs=8192, wall-stopped) but on
# bfloat16. Protocol §22: do NOT pool 4090 seeds with 1080 Ti seed 0.
#
# Usage (repo root, venv active):
#   SEED=1 bash scripts/run_p1ns_4090.sh all
#   SEED=2 bash scripts/run_p1ns_4090.sh all
#   SEED=1 bash scripts/run_p1ns_4090.sh resume L6
#
# Overnight both seeds:
#   SEED=1 bash scripts/run_p1ns_4090.sh all && SEED=2 bash scripts/run_p1ns_4090.sh all
#
# Environment overrides:
#   SEED=1|2
#   RUNTIME_L6_MINUTES=240
#   RUNTIME_L10_MINUTES=210
#   EPOCHS=8192

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRESET=canonical
VARIANT=P1ns
SEED="${SEED:-1}"
HALT_MAX_STEPS=6
EPOCHS="${EPOCHS:-8192}"
RUNTIME_L6_MINUTES="${RUNTIME_L6_MINUTES:-240}"
RUNTIME_L10_MINUTES="${RUNTIME_L10_MINUTES:-210}"
EVAL_OUT=results/p1ns-act6-4090
CPU_THREADS=16
DATALOADER_WORKERS=1

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

declare -A OUTPUT_ROOT_FOR=(
  [L6]=outputs/study-p1ns-4090-L6
  [L10]=outputs/study-p1ns-4090-L10
)
declare -A L_CYCLES_FOR=(
  [L6]=6
  [L10]=10
)
declare -A RUNTIME_FOR=(
  [L6]="${RUNTIME_L6_MINUTES}"
  [L10]="${RUNTIME_L10_MINUTES}"
)
JOBS=(L6 L10)

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
  --override arch.forward_dtype=bfloat16
  --override "dataloader_num_workers=${DATALOADER_WORKERS}"
)

run_dir() {
  local job="$1"
  echo "${OUTPUT_ROOT_FOR[$job]}/${PRESET}/${VARIANT}-seed${SEED}"
}

eval_ckpt() {
  local job="$1"
  python - "$ROOT/$(run_dir "$job")" <<'PY'
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
  local job="$1"
  python - "$ROOT/$(run_dir "$job")" <<'PY'
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
  launch python - "$CPU_THREADS" <<PY
import os
import sys
import torch

cpu_cap = int(sys.argv[1])
torch.set_num_threads(cpu_cap)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print("gpu", torch.cuda.get_device_name(0))
print("variant", "${VARIANT}")
print("seed", ${SEED})
print("dtype", "bfloat16")
print("halt_max_steps", ${HALT_MAX_STEPS})
print("epochs", ${EPOCHS})
print("jobs", "L6=${L_CYCLES_FOR[L6]} (${RUNTIME_L6_MINUTES} min), L10=${L_CYCLES_FOR[L10]} (${RUNTIME_L10_MINUTES} min)")
PY
  for job in "${JOBS[@]}"; do
    mkdir -p "${OUTPUT_ROOT_FOR[$job]}/$PRESET"
  done
  mkdir -p "$EVAL_OUT" artifacts
  {
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "script=run_p1ns_4090.sh"
    echo "seed=${SEED}"
    echo "variant=${VARIANT}"
    echo "dtype=bfloat16"
    echo "history_mode=P1ns (attention_no_skip)"
    echo "halt_max_steps=${HALT_MAX_STEPS}"
    echo "epochs=${EPOCHS}"
    echo "L6_cycles=${L_CYCLES_FOR[L6]} runtime_min=${RUNTIME_L6_MINUTES}"
    echo "L10_cycles=${L_CYCLES_FOR[L10]} runtime_min=${RUNTIME_L10_MINUTES}"
    echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    nvidia-smi -L || true
  } | tee "artifacts/p1ns_act6_4090_seed${SEED}_provenance.txt"
}

cmd_data() {
  if [[ -d data/sudoku-study-v1/train && -d data/sudoku-study-v1/dev && -d data/sudoku-study-v1/test ]]; then
    echo "dataset already present: data/sudoku-study-v1"
    return
  fi
  python dataset/build_sudoku_baseline_v2.py
}

cmd_train_one() {
  local job="${1:?usage: train L6|L10}"
  if [[ -z "${L_CYCLES_FOR[$job]+x}" ]]; then
    echo "unknown job ${job}; expected L6 or L10"
    exit 2
  fi
  local l="${L_CYCLES_FOR[$job]}"
  local minutes="${RUNTIME_FOR[$job]}"
  local out_root="${OUTPUT_ROOT_FOR[$job]}"
  local dir
  dir="$(run_dir "$job")"
  echo "===== TRAIN ${VARIANT} ACT${HALT_MAX_STEPS} L_cycles=${l} seed=${SEED} (4090 / bfloat16, cap ${minutes} min) ====="
  echo "output -> ${dir}"
  launch python experiments/run_study.py single \
    --preset "$PRESET" \
    --variant "$VARIANT" \
    --seed "$SEED" \
    --output-root "$out_root" \
    --override "arch.L_cycles=${l}" \
    --override "arch.halt_max_steps=${HALT_MAX_STEPS}" \
    --override "epochs=${EPOCHS}" \
    --override "max_runtime_minutes=${minutes}" \
    "${TRAIN_OVERRIDES[@]}"
}

cmd_train() {
  cmd_train_one "${1:?usage: bash scripts/run_p1ns_4090.sh train L6|L10}"
}

cmd_train_all() {
  for job in "${JOBS[@]}"; do
    cmd_train_one "$job"
  done
}

cmd_resume() {
  local job="${1:?usage: bash scripts/run_p1ns_4090.sh resume L6|L10}"
  if [[ -z "${L_CYCLES_FOR[$job]+x}" ]]; then
    echo "unknown job ${job}; expected L6 or L10"
    exit 2
  fi
  local l="${L_CYCLES_FOR[$job]}"
  local minutes="${RUNTIME_FOR[$job]}"
  local out_root="${OUTPUT_ROOT_FOR[$job]}"
  local ckpt
  ckpt="$(last_resume_ckpt "$job")"
  echo "RESUME ${VARIANT} L${l} seed=${SEED} <- ${ckpt}"
  launch python experiments/run_study.py resume \
    --preset "$PRESET" \
    --variant "$VARIANT" \
    --seed "$SEED" \
    --output-root "$out_root" \
    --checkpoint "$ckpt" \
    --override "arch.L_cycles=${l}" \
    --override "arch.halt_max_steps=${HALT_MAX_STEPS}" \
    --override "epochs=${EPOCHS}" \
    --override "max_runtime_minutes=${minutes}" \
    "${TRAIN_OVERRIDES[@]}"
}

cmd_eval() {
  local found=0
  for job in "${JOBS[@]}"; do
    local l ckpt
    l="${L_CYCLES_FOR[$job]}"
    if ! ckpt="$(eval_ckpt "$job")"; then
      echo "SKIP eval P1ns L${l} seed=${SEED}: no checkpoint"
      continue
    fi
    found=1
    echo "EVAL P1ns L${l} seed=${SEED} <- $ckpt"
    launch python experiments/evaluate_study.py \
      --config "config/experiment/sudoku_study_${PRESET}.yaml" \
      --data data/sudoku-study-v1 \
      --split test \
      --seed "$SEED" \
      --device cuda \
      --interventions \
      --output "${EVAL_OUT}/L${l}" \
      --checkpoint "${VARIANT}=${ckpt}"
    launch python experiments/analyze_results.py \
      --input "${EVAL_OUT}/L${l}" \
      --output "${EVAL_OUT}/L${l}/analysis"
  done
  if [[ "$found" -eq 0 ]]; then
    echo "No checkpoints for seed=${SEED} under outputs/study-p1ns-4090-L{6,10}/"
    exit 1
  fi
  echo ""
  echo "===== Compare (seed ${SEED}) ====="
  python - "$SEED" <<'PY'
import json
import sys
from pathlib import Path

seed = sys.argv[1]
paths = {
    f"4090 B0 ACT6": Path(f"results/canonical-gpu-4090/B0/seed_{seed}/metadata.json"),
    f"4090 P1 ACT6": Path(f"results/canonical-gpu-4090/P1/seed_{seed}/metadata.json"),
    f"P1ns L6": Path(f"results/p1ns-act6-4090/L6/P1ns/seed_{seed}/metadata.json"),
    f"P1ns L10": Path(f"results/p1ns-act6-4090/L10/P1ns/seed_{seed}/metadata.json"),
    f"1080 P1ns L6 s0": Path("results/p1ns-act6/L6/P1ns/seed_0/metadata.json"),
}
for name, path in paths.items():
    if not path.exists():
        print(f"{name:22s}  MISSING")
        continue
    m = json.loads(path.read_text())["metrics"]
    print(
        f"{name:22s}  cell={100*m['cell_accuracy']:.3f}%  "
        f"exact={100*m['exact_accuracy']:.2f}%"
    )
PY
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
usage: SEED=1|2 bash scripts/run_p1ns_4090.sh COMMAND [ARG]

  setup        GPU check + provenance
  data         build sudoku-study-v1 if missing
  train-all    P1ns L6 then L10 (~4 h + ~3.5 h per seed)
  train L6|L10 one job
  resume L6|L10
  eval         test eval + comparison table
  all          setup + data + train-all + eval

Both seeds overnight:
  SEED=1 bash scripts/run_p1ns_4090.sh all && SEED=2 bash scripts/run_p1ns_4090.sh all
EOF
    exit 2
    ;;
esac
