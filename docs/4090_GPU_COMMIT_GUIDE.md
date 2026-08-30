# RTX 4090 Canonical GPU — Minimal Git Commit Guide

**Branch:** `feature/latent-history-attention`  
**Base campaign commit:** `315ab13585da2e5cc103daf73d1fb28b287f6b63`  
**Hardware:** NVIDIA GeForce RTX 4090, bfloat16, PyTorch 2.6.0+cu124  
**Regime:** canonical ACT6, 28 800 steps, four variants (B0, P1, Gated, B3), seeds **1** and **2**

This document specifies exactly what to commit after the two-seed 4090 evaluation, without staging checkpoints, large raw outputs, or unrelated local files.

---

## 1. What to commit

| Path | Commit? | Rationale |
|------|---------|-----------|
| `scripts/run_canonical_4090.sh` | **Yes** | Required runtime fix (`DATALOADER_WORKERS=1`) + stale comment cleanup |
| `artifacts/data/sudoku_study_v1_manifest.json` | **Yes** | Small dataset provenance / leakage record (default output of `build_sudoku_baseline_v2.py`) |
| `results/canonical-gpu-4090/analysis/*` | **Yes** | Aggregates: `*.json`, `*.csv`, `*.pdf` — paper-ready summaries |
| `results/canonical-gpu-4090/**/metadata.json` | **Yes** | Compact per-run metrics (clean + interventions) |
| `results/canonical-gpu-4090/**/examples.jsonl` | **No** | ~24 MB per seed (~48 MB for seeds 1+2); headline numbers live in metadata + analysis |
| `scripts/run_canonical_4090_seed2.local.sh` | **No** | Local one-line wrapper; not needed in repo |

### Size reference (1080 Ti precedent)

For `results/canonical-gpu/` (single seed, same eval layout):

| Artifact | Count | Size |
|----------|-------|------|
| `examples.jsonl` | 18 | ~24 MB |
| `metadata.json` | 18 | ~14 KB |
| Full tree | 46 files | ~24 MB |

Committing metadata + analysis for 4090 (two seeds) keeps the commit at roughly **1–2 MB** instead of **~50 MB**.

### `examples.jsonl` decision

Existing 1080 Ti campaigns (`canonical-gpu`, `canonical-gpu-8h`) did commit `examples.jsonl`. For this commit, **prefer a small history**: commit metadata + analysis only. Per-example files remain on the server; checkpoints under ignored `outputs/study-4090/` can be re-evaluated if needed.

**All seeds and variants must be included as-is.** Do not omit seed 2’s negative P1 result.

---

## 2. What to leave untracked / ignored

| Path | Action |
|------|--------|
| `outputs/study-4090/` | Ignored by `.gitignore` (`outputs/*`) — **do not force-add** |
| `data/` | Ignored — keep local |
| `.venv/`, `*.pt`, `wandb/` | Ignored |
| `scripts/run_canonical_4090_seed2.local.sh` | Leave untracked (or delete locally after commit) |
| Unrelated modified files (`experiments/run_study.py`, other docs, configs, etc.) | **Exclude** from this commit |

---

## 3. Minimal cleanup (script only)

Edit `scripts/run_canonical_4090.sh` before staging. **No architecture or hyperparameter changes.**

### Required functional fix

```diff
-DATALOADER_WORKERS=4
+DATALOADER_WORKERS=1
```

**Reason:** `puzzle_dataset.py` asserts that multithreaded data loading is not supported. `workers=4` caused the smoke run to fail before the first training batch.

### Stale comment corrections

```diff
-# Hardware: RTX 4090 24 GB, bfloat16, CPU cap 16 threads / 4 loader workers.
+# Hardware: RTX 4090 24 GB, bfloat16, CPU cap 16 threads / 1 loader worker.
```

```diff
-# Host cap: 16 cores (user limit 15–20). OMP=2 so four loader workers cannot
-# explode BLAS threads. taskset is applied at launch when it exists.
+# Host cap: 16 cores (user limit 15–20). OMP=2 limits BLAS threads.
+# Sudoku dataset rejects multithreaded loading (puzzle_dataset.py); workers must be 1.
```

Do **not** reset, restore, or delete anything else.

### Seed 2 without the local helper script

`SEED=1` is hardcoded in the launcher. Seed 2 was run via `scripts/run_canonical_4090_seed2.local.sh` (do not commit). For future reruns, a one-line generic fix would be `SEED="${SEED:-1}"` so `SEED=2 bash scripts/run_canonical_4090.sh train` works — optional, not required for this commit.

---

## 4. Campaign results (preserved in commit)

Clean test cell / exact accuracy:

| Variant | Seed 1 | Seed 2 |
|---------|--------|--------|
| B0 (Vanilla) | 63.936% / 1.2% | 67.021% / 1.6% |
| B3 | 64.909% / 1.3% | 65.978% / 1.5% |
| Gated | 65.846% / 1.4% | 66.485% / 1.8% |
| P1 (HistoryAttention) | 66.926% / 1.8% | 63.070% / 0.4% |

P1 vs B0 (cell):

- Seed 1: **+2.990 pp**
- Seed 2: **−3.951 pp**

This seed sensitivity must remain fully visible in `analysis/seed_results.json` and related aggregates.

---

## 5. Exact git commands

Run from repo root on the server. **Do not use `git add .`, `git add -A`, or broad staging.**

```bash
cd ~/trm-latent-history-attention

# Script fix (after editing comments above)
git add scripts/run_canonical_4090.sh

# Dataset provenance
git add artifacts/data/sudoku_study_v1_manifest.json

# Eval aggregates
git add results/canonical-gpu-4090/analysis/aggregate_results.csv
git add results/canonical-gpu-4090/analysis/aggregate_results.json
git add results/canonical-gpu-4090/analysis/analysis.json
git add results/canonical-gpu-4090/analysis/seed_results.csv
git add results/canonical-gpu-4090/analysis/seed_results.json
git add results/canonical-gpu-4090/analysis/accuracy.pdf
git add results/canonical-gpu-4090/analysis/attention.pdf
git add results/canonical-gpu-4090/analysis/compute.pdf
git add results/canonical-gpu-4090/analysis/corruption.pdf
git add results/canonical-gpu-4090/analysis/learning_curves.pdf

# Per-run metadata only (no examples.jsonl)
git add results/canonical-gpu-4090/B0/seed_1/metadata.json
git add results/canonical-gpu-4090/B0/seed_2/metadata.json
git add results/canonical-gpu-4090/P1/seed_1/metadata.json
git add results/canonical-gpu-4090/P1/seed_2/metadata.json
git add results/canonical-gpu-4090/Gated/seed_1/metadata.json
git add results/canonical-gpu-4090/Gated/seed_2/metadata.json
git add results/canonical-gpu-4090/B3/seed_1/metadata.json
git add results/canonical-gpu-4090/B3/seed_2/metadata.json

git add results/canonical-gpu-4090/B0/seed_1/gaussian_0.05/metadata.json
git add results/canonical-gpu-4090/B0/seed_1/gaussian_0.10/metadata.json
git add results/canonical-gpu-4090/B0/seed_1/gaussian_0.20/metadata.json
git add results/canonical-gpu-4090/B0/seed_2/gaussian_0.05/metadata.json
git add results/canonical-gpu-4090/B0/seed_2/gaussian_0.10/metadata.json
git add results/canonical-gpu-4090/B0/seed_2/gaussian_0.20/metadata.json

git add results/canonical-gpu-4090/P1/seed_1/gaussian_0.05/metadata.json
git add results/canonical-gpu-4090/P1/seed_1/gaussian_0.10/metadata.json
git add results/canonical-gpu-4090/P1/seed_1/gaussian_0.20/metadata.json
git add results/canonical-gpu-4090/P1/seed_1/delete_most_attended/metadata.json
git add results/canonical-gpu-4090/P1/seed_1/delete_least_attended/metadata.json
git add results/canonical-gpu-4090/P1/seed_2/gaussian_0.05/metadata.json
git add results/canonical-gpu-4090/P1/seed_2/gaussian_0.10/metadata.json
git add results/canonical-gpu-4090/P1/seed_2/gaussian_0.20/metadata.json
git add results/canonical-gpu-4090/P1/seed_2/delete_most_attended/metadata.json
git add results/canonical-gpu-4090/P1/seed_2/delete_least_attended/metadata.json

git add results/canonical-gpu-4090/Gated/seed_1/gaussian_0.05/metadata.json
git add results/canonical-gpu-4090/Gated/seed_1/gaussian_0.10/metadata.json
git add results/canonical-gpu-4090/Gated/seed_1/gaussian_0.20/metadata.json
git add results/canonical-gpu-4090/Gated/seed_2/gaussian_0.05/metadata.json
git add results/canonical-gpu-4090/Gated/seed_2/gaussian_0.10/metadata.json
git add results/canonical-gpu-4090/Gated/seed_2/gaussian_0.20/metadata.json

git add results/canonical-gpu-4090/B3/seed_1/gaussian_0.05/metadata.json
git add results/canonical-gpu-4090/B3/seed_1/gaussian_0.10/metadata.json
git add results/canonical-gpu-4090/B3/seed_1/gaussian_0.20/metadata.json
git add results/canonical-gpu-4090/B3/seed_2/gaussian_0.05/metadata.json
git add results/canonical-gpu-4090/B3/seed_2/gaussian_0.10/metadata.json
git add results/canonical-gpu-4090/B3/seed_2/gaussian_0.20/metadata.json
```

If any path differs on disk:

```bash
find results/canonical-gpu-4090 -name metadata.json
```

Add any missing paths explicitly. Still **no** `git add .`.

### Verify before commit

```bash
git status
git diff --cached --stat
git diff --cached scripts/run_canonical_4090.sh

# Must print "OK" — no examples.jsonl, checkpoints, outputs/, or local seed2 script
git diff --cached --name-only | grep -E 'examples\.jsonl|\.pt$|outputs/|seed2\.local' \
  && echo "STOP: unwanted files staged" || echo "OK: no heavy/unwanted files"
```

### Commit

```bash
git commit -m "$(cat <<'EOF'
Add RTX 4090 canonical GPU evals (seeds 1–2) and fix DataLoader workers.

Record two-seed test metrics and analysis for B0/P1/Gated/B3 on the 4090
bfloat16 campaign; set dataloader workers to 1 for Sudoku dataset compatibility.
EOF
)"
```

### Post-commit check

```bash
git show --stat HEAD
```

Expected: ~44 files, ~1–2 MB total, one script diff, no checkpoints.

**Do not push** unless explicitly requested.

---

## 6. Commit message (copy-paste)

```
Add RTX 4090 canonical GPU evals (seeds 1–2) and fix DataLoader workers.

Record two-seed test metrics and analysis for B0/P1/Gated/B3 on the 4090
bfloat16 campaign; set dataloader workers to 1 for Sudoku dataset compatibility.
```
