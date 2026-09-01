# P1ns (no-skip attention) — RTX 4090 server runbook

Train and evaluate **P1ns** (`RMSNorm(memory)` only, no residual onto `z_L`) on the
4090 box for **seeds 1 and 2**, matching the refined ACT6 budget used on the 1080 Ti
(seed 0): `L_cycles ∈ {6, 10}`, `halt_max_steps=6`, `epochs=8192`, wall-stopped.

| Machine | Variant | Seeds | Dtype | Script |
|---------|---------|-------|-------|--------|
| 1080 Ti | P1ns | **0** | float32 | `scripts/run_p1ns_act6_server.sh` |
| RTX 4090 | P1ns | **1, 2** | bfloat16 | `scripts/run_p1ns_4090.sh` |

**Protocol §22:** Do not average 1080 Ti seed 0 with 4090 seeds 1–2 as one matched
family (different GPU + dtype). Report them separately; compare P1ns vs canonical
P1 **within each machine**.

---

## 1. Pull latest code

```bash
cd ~/trm-latent-history-attention
git fetch origin
git checkout feature/latent-history-attention
git pull origin feature/latent-history-attention
```

Confirm the launcher exists:

```bash
ls scripts/run_p1ns_4090.sh docs/P1NS_4090_SERVER.md
git log -1 --oneline
```

---

## 2. Environment

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

Dataset (skip if `data/sudoku-study-v1/` already present):

```bash
bash scripts/run_p1ns_4090.sh data
```

Smoke check:

```bash
python -c "from models.history import build_history_aggregator; m=build_history_aggregator('P1ns', hidden_size=8, rank=4, num_heads=2); assert m.use_skip is False; print('P1ns ok')"
```

---

## 3. Run (recommended: tmux)

### One seed (~7.5 h train + eval)

```bash
tmux new -s p1ns-s1
cd ~/trm-latent-history-attention
source .venv/bin/activate
SEED=1 bash scripts/run_p1ns_4090.sh all
```

Detach: `Ctrl+b` then `d`.

### Both seeds overnight (~15 h)

```bash
tmux new -s p1ns-4090
cd ~/trm-latent-history-attention
source .venv/bin/activate
SEED=1 bash scripts/run_p1ns_4090.sh all && SEED=2 bash scripts/run_p1ns_4090.sh all
```

### Resume after crash or cap

```bash
SEED=1 bash scripts/run_p1ns_4090.sh resume L6
# or
SEED=2 bash scripts/run_p1ns_4090.sh resume L10
```

### Eval only (checkpoints already trained)

```bash
SEED=1 bash scripts/run_p1ns_4090.sh eval
SEED=2 bash scripts/run_p1ns_4090.sh eval
```

---

## 4. Outputs

| Artifact | Path |
|----------|------|
| L6 checkpoints | `outputs/study-p1ns-4090-L6/canonical/P1ns-seed{1,2}/` |
| L10 checkpoints | `outputs/study-p1ns-4090-L10/canonical/P1ns-seed{1,2}/` |
| L6 test eval | `results/p1ns-act6-4090/L6/P1ns/seed_{1,2}/` |
| L10 test eval | `results/p1ns-act6-4090/L10/P1ns/seed_{1,2}/` |
| Provenance | `artifacts/p1ns_act6_4090_seed{N}_provenance.txt` |

Compare to existing canonical 4090 ACT6 (with-skip P1):

- `results/canonical-gpu-4090/P1/seed_1/metadata.json`
- `results/canonical-gpu-4090/P1/seed_2/metadata.json`

Compare to 1080 Ti P1ns seed 0 (separate regime):

- `results/p1ns-act6/L6/P1ns/seed_0/metadata.json`
- `results/p1ns-act6/L10/P1ns/seed_0/metadata.json`

Quick check after eval:

```bash
for s in 1 2; do
  echo "=== seed $s ==="
  jq '.metrics | {cell: .cell_accuracy, exact: .exact_accuracy}' \
    results/p1ns-act6-4090/L6/P1ns/seed_$s/metadata.json
  jq '.metrics | {cell: .cell_accuracy, exact: .exact_accuracy}' \
    results/p1ns-act6-4090/L10/P1ns/seed_$s/metadata.json
done
```

---

## 5. Push results to GitHub

Commit **metadata + analysis only** (no `.pt`, no `examples.jsonl` unless you need them).

```bash
cd ~/trm-latent-history-attention

git add results/p1ns-act6-4090/
git add artifacts/p1ns_act6_4090_seed_*.txt

# Optional: training metrics (no checkpoints)
git add -f \
  outputs/study-p1ns-4090-L6/canonical/P1ns-seed*/all_config.yaml \
  outputs/study-p1ns-4090-L6/canonical/P1ns-seed*/metrics.jsonl \
  outputs/study-p1ns-4090-L6/canonical/P1ns-seed*/orchestrator.json \
  outputs/study-p1ns-4090-L6/canonical/P1ns-seed*/run_metadata.json \
  outputs/study-p1ns-4090-L10/canonical/P1ns-seed*/all_config.yaml \
  outputs/study-p1ns-4090-L10/canonical/P1ns-seed*/metrics.jsonl \
  outputs/study-p1ns-4090-L10/canonical/P1ns-seed*/orchestrator.json \
  outputs/study-p1ns-4090-L10/canonical/P1ns-seed*/run_metadata.json

git status   # must NOT list *.pt

git commit -m "Add P1ns ACT6 L6/L10 4090 evals for seeds 1–2."
git push origin feature/latent-history-attention
```

---

## 6. Pull on local Windows

```powershell
cd "C:\Users\mahya\Documents\New folder\Term_6\DeepLearning\TRM\trm-latent-history-attention"
git pull origin feature/latent-history-attention
```

If untracked local copies block the pull, move them aside first (same as CARS).

---

## 7. Tunables

Override without editing the script:

```bash
RUNTIME_L6_MINUTES=300 RUNTIME_L10_MINUTES=240 SEED=1 bash scripts/run_p1ns_4090.sh all
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `SEED` | `1` | Training/eval seed (use `1` or `2` on 4090) |
| `RUNTIME_L6_MINUTES` | `240` | Wall cap per L6 job |
| `RUNTIME_L10_MINUTES` | `210` | Wall cap per L10 job |
| `EPOCHS` | `8192` | Must exceed wall budget so the clock stops training |
