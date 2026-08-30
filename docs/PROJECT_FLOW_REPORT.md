# Latent-History TRM — Project Flow Report

Living document for the report: methods, training campaigns, evaluation, and
results from **canonical protocol v1** onward (Aug 2026).

**Branch:** `feature/latent-history-attention`  
**Repo:** [trm-latent-history-attention](https://github.com/Iliyabr/trm-latent-history-attention)

---

## 1. Research question

Does **learned low-rank attention over within-H-cycle inner latent states** (`z_L`)
improve Sudoku reasoning compared to:

- **B0** — vanilla TRM (no history),
- **Gated** — uniform history with a learned gate,
- **B3** — exact parameter-matched no-history control (`4·D·r+1` extra params)?

History is **within one outer H-cycle only** (not across ACT steps). The current
inner state queries past inner states from that same cycle before the shared TRM
update.

---

## 2. Timeline (high level)

| Phase | When | What happened |
|-------|------|----------------|
| Pre-protocol | Aug 25–28 | Colab screening (`colab` / `colab_heavy` presets, H2/L4). Pipeline and eval tooling built. **Not mixed with canonical results.** |
| Protocol alignment | Aug 29 | Code/config aligned to **TRM_HISTORY_CANONICAL_PROTOCOL_v1**: pre-QKV P1, canonical Gated, exact B3 param match, D256/H3/L6/L2. |
| 1080 Ti screening | Aug 29 | Four models × seed 0, preset `canonical`, ~28k steps (~1.1 h/model). Eval → `results/canonical-gpu/`. |
| 8h campaign | Aug 29–30 | Preset `canonical_8h`: ACT16, 2 h/model (`train-all`). Eval → `results/canonical-gpu-8h/`. |
| Extended L_cycle | Aug 30+ | `L_CYCLES=10`, B0/P1/Gated, 2 h/model. Script: `run_extended_l_cycle_server.sh`. **In progress on server** (outputs not in repo yet). |
| 4090 campaign | Aug 29+ | Separate box, seed **1**, `run_canonical_4090.sh`. Verified smoke; full four-model run planned/in progress separately. |

---

## 3. Protocol v1 — what we fixed

Before the canonical protocol, the inherited implementation attended over **ACT
supervision steps** (wrong scope). Protocol v1 locked:

### Architecture (`config/arch/trm_history_canonical.yaml`)

| Setting | Value |
|---------|--------|
| Hidden size D | 256 |
| H-cycles / L-cycles / L layers | 3 / **6** / 2 Transformer |
| Spatial heads | 4 |
| Positional encoding | RoPE |
| L-block | Transformer (`mlp_t=false`) |
| ACT (screening) | 6 halt steps |
| Temporal rank / heads | 64 / 4 |
| History window | 0 (full causal within H-cycle) |
| Gate init | −2 → σ(gate) ≈ 0.12 at start |
| Dtype (1080 Ti) | float32 |

### Variant definitions (`models/history/`)

| ID | Mechanism |
|----|-----------|
| **B0** | Identity — vanilla TRM |
| **P1** | Low-rank multi-head **temporal** attention; **pre-QKV RMSNorm** on query and history keys/values |
| **Gated** | Uniform mean of normalized history + learned gate |
| **B3** | Widened FFN side path matching P1’s **exactly 65,537** extra parameters |

Legacy Colab variants **B1/B2** remain in code but are **not** canonical controls.

### Training defaults (`config/experiment/sudoku_study_canonical.yaml`)

- Dataset: `data/sudoku-study-v1`
- Batch: 32, AdamW lr 1e-4, wd 0.1, EMA 0.999
- Dev eval every 64 epochs; `best_dev.pt` by **exact_accuracy**
- Epochs 1024 = **placeholder** (~28,800 steps); formal step budget **N not frozen** as of screening

See also: [CANONICAL_PROTOCOL_v1.md](CANONICAL_PROTOCOL_v1.md)

---

## 4. Dataset and preprocessing

**Build:** `python dataset/build_sudoku_baseline_v2.py`  
**Source:** Hugging Face `sapientinc/sudoku-extreme`

| Split | Bases | Augmentation | Role |
|-------|-------|--------------|------|
| Train | 900 | 64 `shuffle_sudoku` per base → **58,500** examples | Gradient updates |
| Dev | 100 | none | Checkpoint selection (`best_dev.pt`) |
| Test | 1,000 | none | Final eval only (`evaluate_study.py`) |

Augmentation: valid Sudoku symmetries (digit permutation, optional transpose,
3×3 band/stack shuffles). Splits chosen **before** augmentation with leakage
checks (exact + digit-canonical hashes).

**Not** a full paper-scale Sudoku-Extreme reproduction (paper uses 1k bases ×
1k aug, 50k epochs, optional MLP backbone).

---

## 5. Hardware and seed policy

| Machine | GPU | Seed | Script | Dtype |
|---------|-----|------|--------|-------|
| `development` (1080 Ti) | GTX 1080 Ti | **0** | `run_canonical_server.sh`, `run_canonical_8h_server.sh`, `run_extended_l_cycle_server.sh` | float32 |
| 4090 server | RTX 4090 | **1** | `run_canonical_4090.sh` | bfloat16 |
| Colab | T4 | 0–2 | notebook / `run_study.py` | bfloat16 (screening only) |

**Do not pool** 1080 Ti and 4090 results as one matched seed family (protocol §22).

---

## 6. Training campaigns (1080 Ti, seed 0)

### 6.1 Screening — preset `canonical`

| Item | Value |
|------|--------|
| Script | `bash scripts/run_canonical_server.sh train` |
| ACT | 6 |
| L_cycles | 6 |
| Variants | B0, P1, Gated, B3 |
| Steps | ~28,800 (~1024 epochs) |
| Time | ~1.1 h per model |
| Train output | `outputs/study/canonical/<VARIANT>-seed0/` |
| Eval checkpoint | last `step_*.pt` |
| Eval output | `results/canonical-gpu/` |
| Completed | Aug 29, 2026 |

### 6.2 Longer run — preset `canonical_8h`

| Item | Value |
|------|--------|
| Script | `bash scripts/run_canonical_8h_server.sh train-all` |
| ACT | **16** |
| L_cycles | 6 |
| Runtime cap | **120 min/model** (`train-all`; 480 min available for single-model) |
| Variants | B0, P1, Gated, B3 |
| Train output | `outputs/study-8h/canonical_8h/<VARIANT>-seed0/` |
| Eval checkpoint | **`best_dev.pt`** |
| Eval output | `results/canonical-gpu-8h/` |
| Completed | Aug 30, 2026 (eval in repo) |

### 6.3 Extended inner cycles — preset `canonical_8h` + `L_cycles=10`

| Item | Value |
|------|--------|
| Script | `L_CYCLES=10 bash scripts/run_extended_l_cycle_server.sh train-all` |
| ACT | 16 |
| L_cycles | **10** |
| Runtime cap | 120 min/model |
| Variants | B0, P1, Gated (**no B3**) |
| Train output | `outputs/study-extended-lcycle/canonical_8h/` |
| Eval output | `results/canonical-gpu-extended-lcycle/` |
| Status | **Running / pending eval** (not committed to GitHub as of Aug 30) |

---

## 7. Evaluation protocol

**Script:** `experiments/evaluate_study.py` (+ `experiments/analyze_results.py`)

- **Split:** test (1,000 puzzles), seed 0
- **Metrics:** cell accuracy, exact accuracy, Sudoku constraint violations
- **Interventions (all variants):** Gaussian noise on latents σ ∈ {0.05, 0.10, 0.20}
- **P1 only:** delete most-attended / delete least-attended history state
- **Analysis:** `results/.../analysis/` — CSV, JSON, PDF figures

Launch via server scripts (`... eval`) or manually with `--checkpoint VARIANT=path`.

---

## 8. Results — clean test (headline)

All numbers: **seed 0**, **test split**, intervention **`clean`**, 1080 Ti campaigns.

### 8.1 Screening (`results/canonical-gpu/`)

| Variant | Cell acc | Exact acc | Exact (n/1000) |
|---------|----------|-----------|----------------|
| **Gated** | **66.5%** | **1.2%** | 12 |
| B3 | 64.3% | 0.1% | 1 |
| P1 | 63.9% | 0.9% | 9 |
| B0 | 63.7% | 0.9% | 9 |

**Screening takeaway:** Gated best on cell accuracy; exact accuracy still very low
(0–1.2%). No variant near paper-scale Sudoku exact (~75–87%).

### 8.2 8h campaign (`results/canonical-gpu-8h/`)

| Variant | Cell acc | Exact acc | Exact (n/1000) |
|---------|----------|-----------|----------------|
| **B0** | **67.9%** | **2.5%** | 25 |
| P1 | 67.2% | 1.9% | 19 |
| B3 | 63.9% | 1.6% | 16 |
| Gated | 64.3% | 1.6% | 16 |

**8h takeaway:** Ranking **flipped** — vanilla B0 best after longer ACT16 training.
Gated lost its screening lead. P1 second on cell acc; attention ablations still
flat (see §9).

### 8.3 Screening → 8h delta (cell / exact)

| Variant | Δ cell | Δ exact |
|---------|--------|---------|
| B0 | +4.2 pp | +1.6 pp |
| P1 | +3.3 pp | +1.0 pp |
| B3 | −0.4 pp | +1.5 pp |
| Gated | **−2.2 pp** | +0.4 pp |

---

## 9. Attention analysis (P1, clean test)

Aggregate attention stats (L_cycles=6, both evals very similar):

| Metric | Screening | 8h |
|--------|-----------|-----|
| Expected lookback | 2.31 | 2.28 |
| Non-adjacent mass | 60.2% | 59.7% |
| Entropy | 1.01 | 1.00 |

**Interpretation:**

- Bias toward **recent** inner steps (~2–3 back), not only the immediate previous step
- Not uniform over full window (uniform L=6 would be ~3.5 lookback)
- **Ablation:** deleting most vs least attended history **does not change** test accuracy (~67.1% vs ~67.2% cell on 8h P1) → attention pattern weakly coupled to predictions at this budget

Per-cell attention heatmaps are **not** stored in eval artifacts (only pooled scalars).

---

## 10. Gaussian interventions (robustness)

Gaussian noise on latent states (σ = 0.05, 0.10, 0.20) produced **stable**
metrics across variants in both campaigns — no large corruption sensitivity in
this low-accuracy regime. See `results/*/analysis/corruption.pdf`.

---

## 11. Gaps vs original TRM paper

Our study is a **resource-adapted** Sudoku setting. Do **not** compare absolute
accuracy to published Sudoku-Extreme numbers without caveats.

| Factor | Paper (upstream README) | Our canonical study |
|--------|-------------------------|---------------------|
| Train aug | 1000 per base | 64 per base |
| Train examples | ~1M | 58,500 |
| Epochs | 50,000 | ~28k steps (screening) or time-capped |
| L-block | MLP option (~87% exact) or Transformer (~75%) | Transformer + RoPE |
| wd | 1.0 | 0.1 |
| D | 512 optional | 256 |

Low exact accuracy (1–3%) is expected under this budget; the scientific question
is **relative** variant ranking and mechanism diagnostics, not paper reproduction.

---

## 12. Repository map (for the report)

| Artifact | Path |
|----------|------|
| Screening metrics (train) | `outputs/canonical/<VARIANT>-seed0/metrics.jsonl` |
| Screening test eval | `results/canonical-gpu/` |
| 8h test eval | `results/canonical-gpu-8h/` |
| Extended L_cycle (when done) | `results/canonical-gpu-extended-lcycle/` |
| Aggregate tables | `results/*/analysis/aggregate_results.csv` |
| Protocol spec | `CANONICAL SCIENTIFIC PROTOCOL v1.txt`, `docs/CANONICAL_PROTOCOL_v1.md` |
| Server runbooks | `docs/SERVER_GPU.md`, `docs/SERVER_4090.md` |

---

## 13. Open items (update as you go)

- [ ] **Extended L_cycle=10** — train + eval on server; commit results
- [ ] **4090 seed-1** four-model canonical run — separate results tree
- [ ] **Freeze training budget N** — still placeholder at 28,800 steps
- [ ] **Multi-seed** confirmatory campaign (protocol target: seeds 0, 1, 2 per variant)
- [ ] Update this doc when extended-lcycle numbers land

---

## 14. One-paragraph methods summary (copy-paste starter)

We study within-H-cycle latent history in Tiny Recursive Reasoning Models (TRM)
on a fixed Sudoku-Extreme split (`sudoku-study-v1`: 900 train bases with 64
symmetry augmentations, 100 dev, 1000 test). Four variants share a D256
Transformer backbone (H3/L6/L2, RoPE): vanilla B0, low-rank temporal attention
P1 (pre-QKV RMSNorm), gated uniform history, and parameter-matched B3. Training
used AdamW (lr 1e-4, wd 0.1, batch 32, EMA 0.999) on a GTX 1080 Ti (float32,
seed 0). We ran a short screening campaign (ACT6, ~28k steps), a longer ACT16
campaign (2 h/model, best-dev checkpoints), and an extended inner-cycle campaign
(L_cycles=10). Test evaluation reports cell and exact accuracy plus latent
Gaussian corruptions and, for P1, history-state deletion ablations. Results
remain far below paper-scale Sudoku accuracy; comparisons are within-study only.

---

*Last updated: 2026-08-30. Edit this file when new campaigns finish.*
