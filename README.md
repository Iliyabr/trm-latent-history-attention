# Inference-Time Trajectory Selection and Latent History for Tiny Recursive Models

Deep Learning Course Project — Sharif University of Technology

**Authors:** Iliya Barari, Mahyar Mohammad Alipour  
**Department:** Electrical Engineering, Sharif University of Technology

---

## Overview

Tiny Recursive Models (TRMs) obtain large effective reasoning depth by repeatedly applying a very small shared network in latent space. This project studies a natural question that arises from recursive reasoning:

> **When, why, and under what conditions does explicit access to the recursive trajectory help Tiny Recursive Models?**

We investigate this question in two complementary directions:

1. **CARS — Constraint-Aware Recursive Selection**  
   A strictly training-free inference-time method that selects among predictions already produced by a frozen TRM trajectory.

2. **Latent-History Retrieval**  
   A mechanistic study of whether later recursive states benefit from explicit access to earlier latent states through low-rank temporal attention.

The project includes matched controls, CPU and GPU studies, statistical analysis, mechanistic interventions, compute measurements, and an exploratory ACT-depth sensitivity study.

---

## Main Contributions

### 1. CARS: Training-Free Inference-Time Selection

CARS operates on an already trained and frozen Vanilla TRM checkpoint.

During normal ACT6 inference, TRM already produces intermediate predictions. CARS keeps these predictions and selects one using a fixed deterministic rule:

1. minimize mismatch with the original Sudoku clues;
2. minimize row, column, and 3×3 box constraint violations;
3. maximize mean token confidence;
4. choose the later ACT step if a tie remains.

CARS:

- requires **no retraining**;
- introduces **0 trainable parameters**;
- performs **0 additional model forward passes**;
- does not use ground-truth solutions during selection;
- always returns a prediction actually produced by the frozen model.

Across the three frozen Vanilla ACT6 seeds:

| Selector | Exact Accuracy | Cell Accuracy |
|---|---:|---:|
| Final ACT step | 1.233 ± 0.351% | 64.895 ± 1.847% |
| Confidence Selection | 1.367 ± 0.569% | 64.921 ± 1.853% |
| **CARS** | **1.400 ± 0.529%** | **64.956 ± 1.814%** |

Across 3,000 seed–puzzle evaluations:

- Final-step Vanilla solved **37** puzzles exactly.
- Confidence Selection solved **41**.
- CARS solved **42**.
- **5 puzzles were exactly correct at an earlier ACT step but incorrect at the final step.**
- CARS recovered **all 5** of these cases.
- CARS damaged **0** final exact solutions.

The average cell-accuracy improvement is small, but the result exposes an important recursive failure mode:

> **Recursive refinement is not strictly monotonic; a later recursive step can destroy an earlier correct solution.**

Measured CARS selection cost is approximately **0.024 ms per puzzle**, excluding the model inference that is already required by Vanilla TRM.

---

### 2. Low-Rank Latent HistoryAttention

The main learned history mechanism gives the current within-cycle latent state causal access to previous low-level recursive states.

For a current latent state z and historical states h_i:

q = W_Q RMSNorm(z)

k_i = W_K RMSNorm(h_i)

v_i = W_V RMSNorm(h_i)

alpha_i = softmax_i(q^T k_i / sqrt(d_h))

m = W_O sum_i alpha_i v_i

z_read = RMSNorm(z + sigmoid(g) m)

The canonical implementation uses:

- causal within-H-cycle history;
- strict **READ → UPDATE → APPEND** ordering;
- history reset at each H-cycle;
- no history carry across ACT steps;
- low-rank Q/K/V/O projections;
- rank ratio r/D = 1/4;
- 4 temporal heads;
- pre-QKV RMSNorm;
- scalar gate initialized with g = -2.

HistoryAttention is a **learned inference-path augmentation**, not a training-free post-processing method. CARS is the strictly inference-time contribution of the project.

---

## Why Four Models?

The final study compares four variants to separate history access, temporal selectivity, and parameter capacity.

| Model | Uses History? | Selective Retrieval? | Extra Capacity? | Purpose |
|---|---:|---:|---:|---|
| Vanilla TRM | No | No | No | Baseline |
| Gated Uniform History | Yes | No | Negligible | Tests value of history access |
| Low-Rank HistoryAttention | Yes | Yes | Yes | Tests learned temporal retrieval |
| Parameter-Matched No-History | No | No | Yes | Controls for added capacity |

Important comparisons:

- **Vanilla vs Gated** → does explicit history access help?
- **Gated vs HistoryAttention** → does temporal selectivity help beyond simple aggregation?
- **Vanilla vs Parameter-Matched** → are changes explained by added parameters?
- **HistoryAttention vs Parameter-Matched** → does retrieving previous states matter beyond capacity?

---

## Track A — Supporting Outer-History Study

Track A was an earlier supporting experiment over outer recursive z_H states.

| Recursion Depth | Vanilla Cell Acc. | Attention Cell Acc. | Delta |
|---:|---:|---:|---:|
| 2 | 48.268% | 48.311% | +0.043 pp |
| 4 | 46.757% | 46.905% | +0.148 pp |
| 8 | 42.659% | 43.389% | **+0.730 pp** |
| 16 | 42.113% | 42.171% | +0.058 pp |

At depth 8:

- 5/5 seeds were positive;
- 95% CI: **[+0.367, +1.110] pp**.

This shows that explicit history can help in a specific intermediate recursion regime, but the effect is not monotonic with depth.

---

## Track B — Proposal-Faithful Within-Cycle History

Track B is the primary latent-history experiment.

History is built from the within-H-cycle z_L states.

The causal ordering is:

```text
READ previous history
        ↓
TRM UPDATE
        ↓
APPEND the new z_L state
```

The newly produced state is never visible to itself.

History:

- begins with the z_L state entering the H-cycle;
- resets at every H-cycle;
- is not carried across ACT steps;
- never contains future states.

This design prevents current-state leakage and makes temporal retrieval explicitly causal.

---

## CPU Study

### Frozen CPU Configuration

| Field | Value |
|---|---|
| Hidden size | D = 64 |
| H-cycles | 3 |
| L-cycles | 6 |
| L-layers | 2 |
| History rank | 16 |
| Temporal heads | 4 |
| ACT max steps | 4 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 0.01 |
| Batch size | 4 |
| Final step | 10,000 |
| Seeds | 5 |
| Precision | CPU FP32 |

### Final CPU Results

| Model | Cell Accuracy | LM Loss | Delta vs Vanilla |
|---|---:|---:|---:|
| **Vanilla** | **45.867 ± 0.858%** | **1.3329** | — |
| Gated | 45.747 ± 0.503% | 1.3398 | -0.120 pp |
| HistoryAttention | 45.459 ± 1.520% | 1.3443 | -0.407 pp |
| Parameter-Matched | 45.428 ± 0.655% | 1.3466 | -0.438 pp |

HistoryAttention vs Vanilla at the final checkpoint:

- mean delta: **-0.407 pp**;
- 95% CI: **[-2.447, +1.632]**;
- p = 0.609.

Therefore the final CPU experiment does **not** support a persistent selective-history advantage.

### Early Optimization Effect

At step 1,250:

- HistoryAttention − Vanilla = **+0.688 pp**;
- positive in **5/5 seeds**;
- 95% CI: **[+0.347, +1.028]**;
- Holm-adjusted p = 0.040.

At step 2,500:

- mean delta = **+3.099 pp**;
- positive in 5/5 seeds;
- the effect does not survive checkpoint-wise Holm correction.

Main interpretation:

> HistoryAttention changes optimization dynamics early, but the advantage decays near convergence.

---

## CPU Mechanistic Evidence

The gate starts at sigmoid(-2) ≈ 0.119.

Final gate values:

| Model | Final Gate |
|---|---:|
| HistoryAttention | 0.139 ± 0.011 |
| Gated History | 0.228 ± 0.027 |
| Parameter-Matched | 0.136 ± 0.006 |

A frozen-checkpoint inference-only gate-off intervention on HistoryAttention gave:

- mean change: **-0.684 ± 1.186 pp**;
- decreases in 4/5 seeds;
- 95% CI: **[-2.156, +0.788]**;
- p ≈ 0.267.

The effect is heterogeneous and statistically inconclusive, but suggests that trained models can functionally use the history branch.

---

## Project-Canonical GPU ACT6 Study

This GPU campaign is canonical for the **frozen project protocol**, not a full reproduction of the original TRM Sudoku operating regime.

### Configuration

| Field | Value |
|---|---|
| Hidden size | D = 256 |
| H-cycles | 3 |
| L-cycles | 6 |
| L-layers | 2 |
| History rank | 64 |
| Temporal heads | 4 |
| ACT max steps | 6 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Betas | (0.9, 0.95) |
| Weight decay | 0.1 |
| EMA | 0.999 |
| Effective batch | 32 |
| Training steps | 28,800 |
| Train base puzzles | 900 |
| Dev puzzles | 100 |
| Clean test puzzles | 1,000 |
| Canonical seeds | 0, 1, 2 |

Seed 0 was run on GTX 1080 Ti / FP32.  
Seeds 1 and 2 were run on RTX 4090 / BF16.

Accuracy is aggregated across the three protocol-matched seeds. Runtime and VRAM comparisons should only be interpreted within a common hardware regime.

### Three-Seed ACT6 Results

| Model | Exact Accuracy | Cell Accuracy | Delta Cell vs Vanilla |
|---|---:|---:|---:|
| Vanilla | 1.233 ± 0.351% | 64.898 ± 1.841% | — |
| **Gated** | **1.467 ± 0.306%** | **66.292 ± 0.388%** | **+1.394 pp** |
| HistoryAttention | 1.033 ± 0.709% | 64.632 ± 2.030% | -0.267 pp |
| Parameter-Matched | 0.967 ± 0.757% | 65.076 ± 0.831% | +0.177 pp |

HistoryAttention − Vanilla per seed:

- seed 0: **+0.160 pp**
- seed 1: **+2.990 pp**
- seed 2: **-3.951 pp**

Three-seed mean:

- **-0.267 pp**
- SD: **3.490 pp**
- positive in 2/3 seeds
- paired 95% CI approximately **[-8.936, +8.403]**

The GPU evidence therefore does not support robust superiority of selective HistoryAttention.

The strongest descriptive GPU direction instead favors the much simpler **Gated Uniform History** control.

---

## GPU Compute Cost

On the matched RTX 4090 runs:

| Model | Relative Train Time | Throughput | Peak Train VRAM |
|---|---:|---:|---:|
| Vanilla | 1.000× | 213.57 ex/s | 663.0 MiB |
| Gated | 1.167× | 183.11 ex/s | 754.6 MiB |
| HistoryAttention | **1.576×** | 135.49 ex/s | 786.4 MiB |
| Parameter-Matched | 1.196× | 178.59 ex/s | 722.2 MiB |

HistoryAttention adds exactly **65,537 parameters**, approximately **+3.83%** over Vanilla.

In the available matched clean RTX 4090 seed-1 evaluation:

- Vanilla: approximately **67.0 examples/s**
- HistoryAttention: approximately **31.6 examples/s**

This corresponds to roughly **2.1× inference time per processed example** for HistoryAttention in that measurement.

---

## GPU Attention Diagnostics

For canonical ACT6 HistoryAttention seed 1:

- attention entropy: **1.036**
- expected lookback: **2.306 states**
- non-adjacent attention mass: **60.3%**

Deleting either the most- or least-attended history state changes cell accuracy by less than approximately **0.06 pp**.

Interpretation:

> History is being accessed, but useful information may be redundant across several previous states rather than concentrated in one causally essential memory.

---

## Exploratory ACT16 Sensitivity

A separate seed-0 experiment increased the ACT ceiling from 6 to 16.

This experiment is **exploratory**, because ACT16 also differs in training duration and checkpoint selection.

| Model | ACT6 Cell / Exact | ACT16 Cell / Exact |
|---|---:|---:|
| Vanilla | 63.74 / 0.9 | **67.89 / 2.5** |
| HistoryAttention | 63.90 / 0.9 | 67.16 / 1.9 |
| Gated | **66.55 / 1.2** | 64.26 / 1.6 |
| Parameter-Matched | 64.34 / 0.1 | 63.86 / 1.6 |

All models used the full ACT ceiling.

Increasing ACT from 6 to 16 increased trajectory length from 18 to 48 internal steps and raised evaluation cost by roughly 2–2.7×.

HistoryAttention remained approximately **0.736 pp below Vanilla** at ACT16.

---

## Main Scientific Findings

### What worked

- Track A shows a reproducible history benefit at a specific recursion depth.
- HistoryAttention shows a reproducible early CPU optimization advantage.
- Some GPU seeds benefit from HistoryAttention.
- Gated Uniform History has the strongest descriptive GPU direction.
- CARS recovers earlier exact solutions that are later destroyed by recursion.

### What did not work

- HistoryAttention does not retain its early CPU advantage at convergence.
- HistoryAttention is highly seed-sensitive on GPU.
- Learned temporal selectivity does not outperform simpler history access consistently.
- Increasing ACT to 16 does not preferentially improve HistoryAttention.

### Overall interpretation

> **Recursive reasoning is useful, but neither deeper recursion nor more complex memory is automatically better. The value of history depends on when and how it is accessed.**

---

## Important Limitations

This project should not be interpreted as a reproduction of the full original TRM Sudoku regime.

Important limitations include:

- resource-adapted CPU and GPU settings;
- zero exact-grid accuracy in the reduced CPU study;
- low absolute exact-grid accuracy in the GPU study;
- only three canonical GPU training seeds;
- different deployment hardware for seed 0 versus seeds 1–2;
- exploratory, confounded ACT16 comparison;
- HistoryAttention requires training;
- CARS currently uses Sudoku-specific constraints.

Therefore the project establishes conclusions about **relative mechanism behavior under the evaluated protocol**, not about full-scale TRM performance.

---

## Repository Structure

A simplified view of the project layout:

```text
.
├── README.md
├── RUNBOOK.md
├── config/
├── configs/
├── docs/
│   ├── data/
│   ├── figures/
│   ├── archive/
│   ├── TRM_HISTORY_CANONICAL_PROTOCOL_v1.md
│   └── TRM_HISTORY_DEPLOYMENT_MANIFEST_v1.md
├── models/
│   └── history/
├── scripts/
├── tests/
└── paper/
```

Important history modules include:

```text
models/history/
├── attention.py
├── gated.py
├── uniform.py
├── recency.py
├── last_state.py
├── lcycle_lowrank_attention.py
├── lcycle_gated.py
└── lcycle_param_matched.py
```

Important CPU analysis utilities include:

```text
scripts/analyze_cpu_lcycle.py
scripts/extract_lcycle_gates.py
scripts/eval_attention_gate_off.py
```

The GPU/CARS branch additionally contains the frozen inference-time evaluation and CARS result artifacts.

---

## Reproducibility

Recommended starting points:

- README.md
- RUNBOOK.md
- docs/TRM_HISTORY_CANONICAL_PROTOCOL_v1.md
- docs/TRM_HISTORY_DEPLOYMENT_MANIFEST_v1.md

CPU module validation:

```bash
pytest tests/test_lcycle_lowrank_history.py -q
```

The canonical CPU history tests pass successfully.

CARS is evaluated only on frozen Vanilla checkpoints and must not create an optimizer, call backward, or modify checkpoints.

---

## Figures

Main project figures are stored under:

```text
docs/figures/
```

Important figures include:

- CPU learning curves;
- CPU paired HistoryAttention–Vanilla deltas;
- GPU three-seed paired effects;
- CARS ACT6 results;
- CARS selected-step distribution;
- CARS recoverable exact-solution headroom.

---

## Reporting Policy

The project intentionally preserves mixed and negative findings.

We avoid unsupported claims such as:

- “HistoryAttention always improves TRM.”
- “More recursion always makes history more useful.”
- “Attention weights prove causal importance.”
- “The resource-adapted GPU study reproduces original TRM performance.”

Instead, conclusions are stated conservatively using language such as:

- supports;
- suggests;
- statistically inconclusive;
- descriptive;
- regime-dependent;
- optimization-sensitive;
- exploratory.

---

## Final Status

| Component | Status |
|---|---|
| Supporting outer-history study | Complete |
| Canonical five-seed CPU Track B | Complete |
| CPU statistical analysis | Complete |
| CPU mechanistic gate analysis | Complete |
| CPU gate-off intervention | Complete |
| Project-canonical GPU ACT6 seeds 0–2 | Complete |
| GPU mechanistic diagnostics | Complete / partial by diagnostic |
| ACT16 seed-0 sensitivity | Complete, exploratory |
| CARS frozen-checkpoint ACT6 evaluation | Complete |
| Final paper | Complete / final formatting stage |
| Comprehensive project report | Complete |

---

## Reference

Base model:

**A. Jolicoeur-Martineau, “Less is More: Recursive Reasoning with Tiny Networks,” 2025.**

This repository is an academic course project built on top of the Tiny Recursive Models codebase and extends it for latent-history and inference-time trajectory-selection experiments.

---

## License

Please follow the license terms of the original Tiny Recursive Models repository and the license included in this repository.
