#!/usr/bin/env bash
# L-mix no-skip compare — GTX 1080 Ti (float32).
#
# Three arms (seed 0 by default), same ACT6 / H3 / L6 / D256 / history rank:
#   B0      — vanilla, transformer L-stack (mlp_t=false, RoPE), no history
#   P1ns    — history attention NO residual skip, transformer L-stack
#   P1nsMLP — same no-skip attention, paper MLP L-stack (mlp_t=true, no RoPE)
#
# History ablation (both P1ns*):
#   P1:   readout = RMSNorm(z + σ(g)·memory)
#   P1ns: readout = RMSNorm(memory)
#
# Training is wall-clock capped with epochs=8192 so the clock (not epoch
# budget) stops the run — long enough for temporal heads to train.
# Default total budget: 18 h for train-all → 360 min (~6 h) per arm.
#
# Usage (repo root, venv active):
#   bash scripts/run_lmix_noskip_server.sh setup
#   bash scripts/run_lmix_noskip_server.sh data
#   bash scripts/run_lmix_noskip_server.sh train-all
#   bash scripts/run_lmix_noskip_server.sh eval
#   bash scripts/run_lmix_noskip_server.sh analyze
#   bash scripts/run_lmix_noskip_server.sh all
#
#   bash scripts/run_lmix_noskip_server.sh train B0
#   bash scripts/run_lmix_noskip_server.sh train P1ns
#   bash scripts/run_lmix_noskip_server.sh train P1nsMLP
#   bash scripts/run_lmix_noskip_server.sh resume P1ns
#
# Environment overrides:
#   SEED=0
#   TOTAL_RUNTIME_MINUTES=1080   # train-all wall budget (default 18 h)
#   MAX_RUNTIME_MINUTES=360     # per-arm cap; default = TOTAL / 3
#   EPOCHS=8192                 # must stay >> wall budget

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRESET=lmix_noskip
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-8192}"
# 18 h total across B0 / P1ns / P1nsMLP unless overridden.
TOTAL_RUNTIME_MINUTES="${TOTAL_RUNTIME_MINUTES:-1080}"
NUM_ARMS=3
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-$((TOTAL_RUNTIME_MINUTES / NUM_ARMS))}"
HALT_MAX_STEPS=6
OUTPUT_ROOT=outputs/study-lmix-noskip
EVAL_OUT=results/lmix-noskip
VARIANTS=(B0 P1ns P1nsMLP)

COMMON_OVERRIDES=(
  --override "arch.halt_max_steps=${HALT_MAX_STEPS}"
  --override "arch.H_cycles=3"
  --override "arch.L_cycles=6"
  --override "arch.L_layers=2"
  --override "epochs=${EPOCHS}"
  --override checkpoint_every_eval=false
  --override compile_model=false
  --override dataloader_num_workers=1
  --override "max_runtime_minutes=${MAX_RUNTIME_MINUTES}"
)

backbone_overrides() {
  local variant="$1"
  case "$variant" in
    B0|P1ns)
      echo --override arch.mlp_t=false --override arch.pos_encodings=rope
      ;;
    P1nsMLP)
      echo --override arch.mlp_t=true --override arch.pos_encodings=none
      ;;
    *)
      echo "unknown variant ${variant}; expected B0|P1ns|P1nsMLP" >&2
      return 2
      ;;
  esac
}

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
  python - <<PY
import torch

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print("gpu", torch.cuda.get_device_name(0))
print("preset", "${PRESET}")
print("variants", " ".join("${VARIANTS[@]}".split()))
print("halt_max_steps", ${HALT_MAX_STEPS})
print("epochs", ${EPOCHS}, "(wall clock should stop first)")
print("total_runtime_minutes", ${TOTAL_RUNTIME_MINUTES}, "(~${TOTAL_RUNTIME_MINUTES}/60 h for train-all)")
print("max_runtime_minutes", ${MAX_RUNTIME_MINUTES}, "per arm")
print("compare", "B0 (transformer) vs P1ns (transformer, no skip) vs P1nsMLP (MLP, no skip)")
PY
  mkdir -p "$OUTPUT_ROOT/$PRESET" "$EVAL_OUT" artifacts
  {
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "script=run_lmix_noskip_server.sh"
    echo "seed=${SEED}"
    echo "preset=${PRESET}"
    echo "variants=${VARIANTS[*]}"
    echo "halt_max_steps=${HALT_MAX_STEPS}"
    echo "epochs=${EPOCHS}"
    echo "total_runtime_minutes=${TOTAL_RUNTIME_MINUTES}"
    echo "max_runtime_minutes=${MAX_RUNTIME_MINUTES}"
    echo "B0=vanilla transformer L (mlp_t=false, rope)"
    echo "P1ns=attention_no_skip + transformer L"
    echo "P1nsMLP=attention_no_skip + mlp_t=true pos_encodings=none"
    echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    nvidia-smi -L || true
  } | tee artifacts/lmix_noskip_provenance.txt
}

cmd_data() {
  if [[ -d data/sudoku-study-v1/train && -d data/sudoku-study-v1/dev && -d data/sudoku-study-v1/test ]]; then
    echo "dataset already present: data/sudoku-study-v1"
    return
  fi
  python dataset/build_sudoku_baseline_v2.py
}

cmd_train_one() {
  local variant="${1:?usage: train B0|P1ns|P1nsMLP}"
  local bb
  # shellcheck disable=SC2207
  bb=($(backbone_overrides "$variant"))
  local dir
  dir="$(run_dir "$variant")"
  echo "===== TRAIN ${variant} seed=${SEED} (cap ${MAX_RUNTIME_MINUTES} min) ====="
  echo "output -> ${dir}"
  python experiments/run_study.py single \
    --preset "$PRESET" \
    --variant "$variant" \
    --seed "$SEED" \
    --output-root "$OUTPUT_ROOT" \
    "${COMMON_OVERRIDES[@]}" \
    "${bb[@]}"
}

cmd_train() {
  cmd_train_one "${1:?usage: bash scripts/run_lmix_noskip_server.sh train B0|P1ns|P1nsMLP}"
}

cmd_train_all() {
  for variant in "${VARIANTS[@]}"; do
    cmd_train_one "$variant"
  done
}

cmd_resume() {
  local variant="${1:?usage: bash scripts/run_lmix_noskip_server.sh resume B0|P1ns|P1nsMLP}"
  local bb
  # shellcheck disable=SC2207
  bb=($(backbone_overrides "$variant"))
  local ckpt
  ckpt="$(last_resume_ckpt "$variant")"
  echo "RESUME ${variant} <- ${ckpt}"
  python experiments/run_study.py resume \
    --preset "$PRESET" \
    --variant "$variant" \
    --seed "$SEED" \
    --output-root "$OUTPUT_ROOT" \
    --checkpoint "$ckpt" \
    "${COMMON_OVERRIDES[@]}" \
    "${bb[@]}"
}

cmd_eval() {
  local found=0
  local ckpt_args=()
  for variant in "${VARIANTS[@]}"; do
    local ckpt
    if ! ckpt="$(eval_ckpt "$variant")"; then
      echo "SKIP eval ${variant}: no checkpoint"
      continue
    fi
    found=1
    ckpt_args+=(--checkpoint "${variant}=${ckpt}")
    echo "EVAL ${variant} <- $ckpt"
  done
  if [[ "$found" -eq 0 ]]; then
    echo "No checkpoints found under ${OUTPUT_ROOT}/${PRESET}/"
    exit 1
  fi
  python experiments/evaluate_study.py \
    --config "config/experiment/sudoku_study_${PRESET}.yaml" \
    --data data/sudoku-study-v1 \
    --split test \
    --seed "$SEED" \
    --device cuda \
    --interventions \
    --output "$EVAL_OUT" \
    "${ckpt_args[@]}"
}

cmd_analyze() {
  if [[ ! -d "$EVAL_OUT" ]]; then
    echo "missing ${EVAL_OUT}; run eval first"
    exit 1
  fi
  python experiments/analyze_results.py \
    --input "$EVAL_OUT" \
    --output "${EVAL_OUT}/analysis"
  echo ""
  echo "===== L-mix no-skip compare (test) ====="
  python - <<'PY'
import json
from pathlib import Path

paths = {
    "B0": Path("results/lmix-noskip/B0/seed_0/metadata.json"),
    "P1ns": Path("results/lmix-noskip/P1ns/seed_0/metadata.json"),
    "P1nsMLP": Path("results/lmix-noskip/P1nsMLP/seed_0/metadata.json"),
}
for name, path in paths.items():
    if not path.exists():
        print(f"{name:10s}  MISSING ({path})")
        continue
    m = json.loads(path.read_text())["metrics"]
    print(
        f"{name:10s}  cell={100*m['cell_accuracy']:.3f}%  "
        f"exact={100*m['exact_accuracy']:.2f}%"
    )
print("tables:", Path("results/lmix-noskip/analysis"))
PY
}

cmd_all() {
  cmd_setup
  cmd_data
  cmd_train_all
  cmd_eval
  cmd_analyze
}

case "${1:-all}" in
  setup) cmd_setup ;;
  data) cmd_data ;;
  train) cmd_train "${2:-}" ;;
  train-all) cmd_train_all ;;
  resume) cmd_resume "${2:-}" ;;
  eval) cmd_eval ;;
  analyze) cmd_analyze ;;
  all) cmd_all ;;
  *)
    cat <<'EOF'
usage: bash scripts/run_lmix_noskip_server.sh COMMAND [ARG]

  setup              GPU check + provenance
  data               build sudoku-study-v1 if missing
  train-all          B0 then P1ns then P1nsMLP (~6 h each; 18 h total by default)
  train B0|P1ns|P1nsMLP
  resume B0|P1ns|P1nsMLP
  eval               test eval (+ interventions) for all available ckpts
  analyze            tables + cell/exact dump for the three arms
  all                setup + data + train-all + eval + analyze

Examples:
  tmux new -s lmix
  git pull origin feature/latent-history-attention
  source .venv/bin/activate
  bash scripts/run_lmix_noskip_server.sh all

  # Different total budget (e.g. 12 h → 4 h / arm):
  TOTAL_RUNTIME_MINUTES=720 bash scripts/run_lmix_noskip_server.sh train-all
EOF
    exit 2
    ;;
esac
