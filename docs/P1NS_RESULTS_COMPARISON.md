# P1ns Ablation — Results Comparison

**Scope:** Training-free architectural ablation of HistoryAttention readout.  
**P1 (canonical):** `RMSNorm(z + σ(g)·memory)`  
**P1ns (ablation):** `RMSNorm(memory)` only — no residual onto current `z_L`

All numbers are **clean test** metrics on `sudoku-study-v1` (1000 puzzles) unless noted.

**Sources:**
- ACT6 screening (1080 Ti, float32, seed 0): `results/canonical-gpu/`
- ACT16 longer run (1080 Ti, float32, seed 0): `results/canonical-gpu-8h/`
- ACT6 4090 (bfloat16, seeds 1–2): `results/canonical-gpu-4090/`
- P1ns ACT6 (1080 Ti, float32, seed 0): `results/p1ns-act6/`

**Protocol §22:** Do not pool 1080 Ti float32 with 4090 bfloat16 as one matched seed family. Within-machine comparisons are primary.

**Status:** P1ns seeds 1–2 on 4090 are **not yet in this clone** (`results/p1ns-act6-4090/` missing). Table below uses seed-0 P1ns only.

---

## 1. Clean test results (primary table)

| Model | L_cycles | ACT | Seed / GPU | Cell acc | Exact acc | vs ACT6 P1 (cell) |
|-------|----------|-----|------------|----------|-----------|-------------------|
| B0 (Vanilla) | 6 | 6 | 0 / 1080 Ti | 63.738% | 0.90% | −0.16 pp |
| **P1** (with skip) | 6 | 6 | 0 / 1080 Ti | 63.899% | 0.90% | baseline |
| Gated | 6 | 6 | 0 / 1080 Ti | 66.546% | 1.20% | +2.65 pp |
| B3 (param-matched) | 6 | 6 | 0 / 1080 Ti | 64.341% | 0.10% | +0.44 pp |
| B0 (Vanilla) | 6 | 16 | 0 / 1080 Ti | 67.894% | 2.50% | +4.00 pp |
| P1 (with skip) | 6 | 16 | 0 / 1080 Ti | 67.158% | 1.90% | +3.26 pp |
| Gated | 6 | 16 | 0 / 1080 Ti | 64.263% | 1.60% | +0.36 pp |
| B3 | 6 | 16 | 0 / 1080 Ti | 63.864% | 1.60% | −0.03 pp |
| **P1ns L6** (no skip) | 6 | 6 | 0 / 1080 Ti | **68.380%** | **3.50%** | **+4.48 pp** |
| P1ns L10 (no skip) | 10 | 6 | 0 / 1080 Ti | 67.532% | 3.20% | +3.63 pp |

### Δ vs matched controls (seed 0)

| Comparison | Δ cell | Δ exact |
|------------|--------|---------|
| P1ns L6 − ACT6 P1 | **+4.48 pp** | **+2.60 pp** |
| P1ns L6 − ACT6 B0 | +4.64 pp | +2.60 pp |
| P1ns L6 − ACT16 P1 | **+1.22 pp** | **+1.60 pp** |
| P1ns L6 − ACT16 B0 | +0.49 pp | +1.00 pp |
| P1ns L6 − ACT6 Gated | +1.83 pp | +2.30 pp |
| P1ns L10 − P1ns L6 | −0.85 pp | −0.30 pp |

---

## 2. Context: 4090 canonical ACT6 (with-skip P1)

These are **not** P1ns; included so seed-1/2 P1ns results can be compared later.

| Model | Seed | Cell acc | Exact acc | P1 − B0 (cell) |
|-------|------|----------|-----------|----------------|
| B0 | 1 | 63.936% | 1.20% | — |
| P1 | 1 | 66.926% | 1.80% | **+2.99 pp** |
| Gated | 1 | 65.846% | 1.40% | — |
| B3 | 1 | 64.909% | 1.30% | — |
| B0 | 2 | 67.021% | 1.60% | — |
| P1 | 2 | 63.070% | 0.40% | **−3.95 pp** |
| Gated | 2 | 66.485% | 1.80% | — |
| B3 | 2 | 65.978% | 1.50% | — |

P1 is **seed-sensitive** on the 4090 (+2.99 pp seed 1, −3.95 pp seed 2). That is why multi-seed P1ns on the same card matters.

---

## 3. Attention diagnostics (seed 0)

| Model | Expected lookback | Non-adjacent mass | Entropy |
|-------|-------------------|-------------------|---------|
| ACT6 P1 | 2.314 | 0.602 | 1.012 |
| ACT16 P1 | 2.278 | 0.597 | 1.002 |
| **P1ns L6** | 2.243 | 0.585 | 1.052 |
| P1ns L10 | 3.297 | 0.708 | 1.457 |

P1ns L6 attention pattern is close to canonical P1 (lookback ~2.2–2.3, non-adjacent ~0.58–0.60). L10 uses longer history (lookback ~3.3) but does **not** improve accuracy.

---

## 4. Interventions (seed 0)

### Deletion ablations

| Model | Clean exact | Delete-most exact | Delete-least exact | Clean cell | Delete-most cell |
|-------|-------------|-------------------|--------------------|------------|------------------|
| ACT6 P1 | 0.90% | 0.90% | 0.90% | 63.899% | 63.944% |
| ACT16 P1 | 1.90% | 1.90% | 1.90% | 67.158% | 67.114% |
| **P1ns L6** | **3.50%** | **2.70%** | 3.40% | **68.380%** | 68.157% |
| P1ns L10 | 3.20% | 2.20% | 1.90% | 67.532% | 67.102% |

Delete-most hurts P1ns exact (−0.8 pp L6, −1.0 pp L10), so the attended history is still used. Canonical ACT6/ACT16 P1 show almost no deletion sensitivity in this low-exact regime.

### Gaussian latent corruption (cell %)

| Model | Clean | σ=0.05 | σ=0.10 | σ=0.20 |
|-------|-------|--------|--------|--------|
| ACT6 P1 | 63.899 | 63.911 | 63.907 | 63.909 |
| ACT16 P1 | 67.158 | 67.146 | 67.173 | 67.198 |
| P1ns L6 | 68.380 | 68.340 | 68.340 | 68.295 |
| P1ns L10 | 67.532 | 67.526 | 67.494 | 67.517 |

All variants are stable under these corruptions; P1ns L6 remains highest.

---

## 5. Training-budget caveat

| Campaign | Approx. budget | Checkpoint used in eval |
|----------|----------------|-------------------------|
| ACT6 screening P1 | ~28.8k steps (~1.1 h) | `step_28800.pt` |
| ACT16 8h P1 | ~2 h wall, ACT16 | `best_dev.pt` |
| **P1ns L6 / L10** | ~4 h / ~3.5 h wall, `epochs=8192` | `best_dev.pt` |

P1ns received **more wall-clock training** than ACT6 screening P1. The fairest same-machine comparison is vs **ACT16 P1** (longer train, still with skip): P1ns L6 still wins (+1.22 pp cell, +1.60 pp exact).

Do **not** claim the skip ablation alone explains the full +4.5 pp vs ACT6 P1 without acknowledging the budget mismatch.

---

## 6. Takeaways

1. **Best model in this table:** P1ns L6 — 68.38% cell / 3.5% exact (seed 0, 1080 Ti).
2. **Removing the residual skip looks beneficial** vs both short ACT6 P1 and longer ACT16 P1.
3. **L=10 does not help** over L=6 for P1ns (−0.85 pp cell).
4. **Attention still matters** for P1ns (delete-most drops exact); history is not inert.
5. **Multi-seed confirmation pending** on 4090 (`docs/P1NS_4090_SERVER.md`). Canonical P1 was seed-sensitive there; P1ns needs the same check.

---

## 7. Missing / next

- [ ] `results/p1ns-act6-4090/` seeds 1–2 (L6 + L10)
- [ ] Update this doc after 4090 P1ns evals land
- [ ] Optional: matched-budget retrain of P1 **with skip** at the same ~4 h wall as P1ns L6 for a cleaner ablation

**Report status:** SCREENING (single seed for P1ns). Not confirmatory multi-seed.
