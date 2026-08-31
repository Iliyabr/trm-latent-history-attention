#!/usr/bin/env bash
# P1 no-skip ablation (P1ns) — ACT6, L_cycles ∈ {6, 10}, ~8 h wall budget.
#
# Ablation vs canonical P1:
#   P1:   readout = RMSNorm(z + σ(g)·memory)
#   P1ns: readout = RMSNorm(memory)   # no residual onto current z_L
#
# Same otherwise: pre-QKV RMSNorm, rank/heads/gate init, ACT halt_max_steps=6,
# float32 (1080 Ti). Train P1ns only; compare later to existing ACT6 / ACT16 P1.
#
# Wall plan (~8 h total on 1080 Ti):
#   L=6  train  ~2.5 h  (matches / exceeds ACT6 screening length)
#   L=10 train  ~3.5 h  (extra inner cycles cost more per step)
#   eval both   ~minutes
#
# Usage (repo root, venv active):
#   bash scripts/run_p1ns_act6_server.sh setup
#   bash scripts/run_p1ns_act6_server.sh data
#   bash scripts/run_p1ns_act6_server.sh train-all
#   bash scripts/run_p1ns_act6_server.sh eval
#   bash scripts/run_p1ns_act6_server.sh all
#
#   bash scripts/run_p1ns_act6_server.sh train L6
#   bash scripts/run_p1ns_act6_server.sh train L10
#   bash scripts/run_p1ns_act6_server.sh resume L6
#
# Environment overrides:
#   SEED=0
#   RUNTIME_L6_MINUTES=150
#   RUNTIME_L10_MINUTES=210

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRESET=canonical
VARIANT=P1ns
SEED="${SEED:-0}"
HALT_MAX_STEPS=6
RUNTIME_L6_MINUTES="${RUNTIME_L6_MINUTES:-150}"
RUNTIME_L10_MINUTES="${RUNTIME_L10_MINUTES:-210}"
EVAL_OUT=results/p1ns-act6

# Separate output roots so run_study's ${VARIANT}-seed${SEED} dirs do not collide.
declare -A OUTPUT_ROOT_FOR=(
  [L6]=outputs/study-p1ns-act6-L6
  [L10]=outputs/study-p1ns-act6-L10
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
  python - <<PY
import torch

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print("gpu", torch.cuda.get_device_name(0))
print("variant", "${VARIANT}")
print("halt_max_steps", ${HALT_MAX_STEPS})
print("jobs", "L6=${L_CYCLES_FOR[L6]} (${RUNTIME_L6_MINUTES} min), L10=${L_CYCLES_FOR[L10]} (${RUNTIME_L10_MINUTES} min)")
print("preset", "${PRESET}")
PY
  for job in "${JOBS[@]}"; do
    mkdir -p "${OUTPUT_ROOT_FOR[$job]}/$PRESET"
  done
  mkdir -p "$EVAL_OUT" artifacts
  {
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "script=run_p1ns_act6_server.sh"
    echo "seed=${SEED}"
    echo "variant=${VARIANT}"
    echo "history_mode=P1ns (attention_no_skip)"
    echo "halt_max_steps=${HALT_MAX_STEPS}"
    echo "L6_cycles=${L_CYCLES_FOR[L6]} runtime_min=${RUNTIME_L6_MINUTES} root=${OUTPUT_ROOT_FOR[L6]}"
    echo "L10_cycles=${L_CYCLES_FOR[L10]} runtime_min=${RUNTIME_L10_MINUTES} root=${OUTPUT_ROOT_FOR[L10]}"
    echo "ablation=RMSNorm(memory) without residual z + gate*memory"
    echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    nvidia-smi -L || true
  } | tee artifacts/p1ns_act6_provenance.txt
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
  echo "===== TRAIN ${VARIANT} ACT${HALT_MAX_STEPS} L_cycles=${l} seed=${SEED} (cap ${minutes} min) ====="
  echo "output -> ${dir}"
  python experiments/run_study.py single \
    --preset "$PRESET" \
    --variant "$VARIANT" \
    --seed "$SEED" \
    --output-root "$out_root" \
    --override "arch.L_cycles=${l}" \
    --override "arch.halt_max_steps=${HALT_MAX_STEPS}" \
    --override checkpoint_every_eval=false \
    --override compile_model=false \
    --override dataloader_num_workers=1 \
    --override "max_runtime_minutes=${minutes}"
}

cmd_train() {
  cmd_train_one "${1:?usage: bash scripts/run_p1ns_act6_server.sh train L6|L10}"
}

cmd_train_all() {
  for job in "${JOBS[@]}"; do
    cmd_train_one "$job"
  done
}

cmd_resume() {
  local job="${1:?usage: bash scripts/run_p1ns_act6_server.sh resume L6|L10}"
  if [[ -z "${L_CYCLES_FOR[$job]+x}" ]]; then
    echo "unknown job ${job}; expected L6 or L10"
    exit 2
  fi
  local l="${L_CYCLES_FOR[$job]}"
  local minutes="${RUNTIME_FOR[$job]}"
  local out_root="${OUTPUT_ROOT_FOR[$job]}"
  local ckpt
  ckpt="$(last_resume_ckpt "$job")"
  echo "RESUME ${VARIANT} L${l} <- ${ckpt}"
  python experiments/run_study.py resume \
    --preset "$PRESET" \
    --variant "$VARIANT" \
    --seed "$SEED" \
    --output-root "$out_root" \
    --checkpoint "$ckpt" \
    --override "arch.L_cycles=${l}" \
    --override "arch.halt_max_steps=${HALT_MAX_STEPS}" \
    --override checkpoint_every_eval=false \
    --override compile_model=false \
    --override dataloader_num_workers=1 \
    --override "max_runtime_minutes=${minutes}"
}

cmd_eval() {
  local found=0
  for job in "${JOBS[@]}"; do
    local l ckpt
    l="${L_CYCLES_FOR[$job]}"
    if ! ckpt="$(eval_ckpt "$job")"; then
      echo "SKIP eval P1ns L${l}: no checkpoint"
      continue
    fi
    found=1
    echo "EVAL P1ns L${l} <- $ckpt"
    # Arch (L_cycles, history_mode) is restored from the checkpoint embed.
    python experiments/evaluate_study.py \
      --config "config/experiment/sudoku_study_${PRESET}.yaml" \
      --data data/sudoku-study-v1 \
      --split test \
      --seed "$SEED" \
      --device cuda \
      --interventions \
      --output "${EVAL_OUT}/L${l}" \
      --checkpoint "${VARIANT}=${ckpt}"
    python experiments/analyze_results.py \
      --input "${EVAL_OUT}/L${l}" \
      --output "${EVAL_OUT}/L${l}/analysis"
  done
  if [[ "$found" -eq 0 ]]; then
    echo "No checkpoints found under outputs/study-p1ns-act6-L{6,10}/"
    exit 1
  fi
  echo ""
  echo "===== Compare to prior campaigns ====="
  echo "ACT6 P1 (with skip):  results/canonical-gpu/P1/seed_${SEED}/metadata.json"
  echo "ACT16 P1 (with skip): results/canonical-gpu-8h/P1/seed_${SEED}/metadata.json"
  echo "This run P1ns L6:     ${EVAL_OUT}/L6/P1ns/seed_${SEED}/metadata.json"
  echo "This run P1ns L10:    ${EVAL_OUT}/L10/P1ns/seed_${SEED}/metadata.json"
  echo ""
  echo "Quick cell/exact dump:"
  python - <<'PY'
import json
from pathlib import Path

paths = {
    "ACT6 B0": Path("results/canonical-gpu/B0/seed_0/metadata.json"),
    "ACT6 P1": Path("results/canonical-gpu/P1/seed_0/metadata.json"),
    "ACT16 B0": Path("results/canonical-gpu-8h/B0/seed_0/metadata.json"),
    "ACT16 P1": Path("results/canonical-gpu-8h/P1/seed_0/metadata.json"),
    "P1ns L6": Path("results/p1ns-act6/L6/P1ns/seed_0/metadata.json"),
    "P1ns L10": Path("results/p1ns-act6/L10/P1ns/seed_0/metadata.json"),
}
for name, path in paths.items():
    if not path.exists():
        print(f"{name:12s}  MISSING ({path})")
        continue
    m = json.loads(path.read_text())["metrics"]
    print(
        f"{name:12s}  cell={100*m['cell_accuracy']:.3f}%  "
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
usage: bash scripts/run_p1ns_act6_server.sh COMMAND [ARG]

  setup        GPU check + provenance
  data         build sudoku-study-v1 if missing
  train-all    P1ns ACT6 L=6 then L=10 (~6 h train)
  train L6|L10 one job
  resume L6|L10 continue after crash or cap
  eval         test eval + comparison dump vs ACT6/ACT16
  all          setup + data + train-all + eval

Examples:
  tmux new -s p1ns
  git pull origin feature/latent-history-attention
  source .venv/bin/activate
  bash scripts/run_p1ns_act6_server.sh all
EOF
    exit 2
    ;;
esac
