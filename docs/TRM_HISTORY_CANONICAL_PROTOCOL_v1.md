# TRM Latent-History Project — Canonical Scientific Protocol v1

**Status: FROZEN**
**Scope:** Shared scientific protocol for the CPU and GPU branches of the project.
**Change control:** Any change to scientific model definitions, history semantics, controls, dataset, normalization, rank policy, evaluation, seed policy, or interpretation requires explicit agreement by both project members and a new `TRM_HISTORY_CANONICAL_PROTOCOL_v2`.

Hardware-dependent values are recorded separately in `TRM_HISTORY_DEPLOYMENT_MANIFEST_v1.md`. Filling that manifest according to this protocol is not a scientific protocol change.

---

## 1. Research question

The project is **not** framed as “HistoryAttention improves TRM.”

The canonical research question is:

> **When, why, and under what conditions does explicit latent-history access help Tiny Recursive Models?**

More specifically:

> Does explicit retrieval from previous recursive latent states improve TRM, and does learned selective retrieval provide value beyond simple history access or additional model capacity?

### Rationale

Current evidence does not support a universal improvement claim. We have a positive outer-history regime, a non-positive mean effect for proposal-oriented within-cycle attention on the reduced CPU model, and substantial seed sensitivity. The paper therefore characterizes history access, selectivity, capacity, stability, and scale transfer rather than optimizing toward a predetermined positive result.

---

## 2. Two scientific tracks

### Track A — outer-step `z_H` history

Supporting mechanistic study only. It is not the final proposal mechanism.

Completed evidence:

| Depth | Vanilla | Attention | Delta |
|---:|---:|---:|---:|
| 2 | 48.268% | 48.311% | +0.043 pp |
| 4 | 46.757% | 46.905% | +0.148 pp |
| 8 | 42.659% | 43.389% | +0.730 pp |
| 16 | 42.113% | 42.171% | +0.058 pp |

At depth 8, the paired 95% CI for Attention − Vanilla was `[+0.367,+1.110]` pp and all 5/5 seeds were positive. The depth-8 vs depth-4 interaction was not conclusive, so a monotonic depth claim is not allowed. Exact-grid accuracy was 0 in this reduced regime.

**Allowed claim:** explicit latent-history retrieval can help in a particular recursive regime.
**Not allowed:** universal or monotonically increasing benefit.

Track A is complete and must not be rerun unless a genuine bug is discovered.

### Track B — within-H-cycle `z_L` history

This is the **primary contribution** and the only history track used for new canonical experiments.

---

## 3. Frozen history semantics

For each H-cycle:

```text
history = [z_L at H-cycle entry]

for each L-step:
    READ from causal history
    TRM UPDATE z_L
    APPEND new z_L
```

Frozen rules:

- History location: within-H-cycle `z_L`.
- Order: **read → update → append**.
- The newly updated state is not inserted before the current retrieval.
- History is not detached inside the final gradient-bearing H-cycle.
- History resets at every H-cycle boundary.
- History is not carried through ACT.
- No history may leak between puzzles or reset batch slots.

For `L_cycles=6`, queried history lengths are `1,2,3,4,5,6`; maximum queried length is 6. The paper description is **full causal within-H-cycle history**. If a numeric window is required, use `history_window=6`.

No primary history-window sweep is permitted under v1.

---

## 4. Canonical Low-Rank HistoryAttention

The proposal-faithful formulation is frozen as:

```text
q   = W_Q RMSNorm_D(z)
k_i = W_K RMSNorm_D(h_i)
v_i = W_V RMSNorm_D(h_i)

score_i = (q · k_i) / sqrt(d_head)
alpha   = softmax_over_recursive_time(score)
context = sum_i alpha_i v_i
memory  = W_O(context)

g      = sigmoid(gate_logit)
z_read = RMSNorm(z + g * memory)
```

with:

```text
gate_logit_init = -2
```

### Normalization rule

Normalization is **pre-QKV in D-dimensional latent space**:

```text
W_Q(RMSNorm_D(z))
```

not the previous GPU screening form:

```text
RMSNorm_head(W_Q(z))
```

The old GPU implementation and its results are preserved as **PRE-ALIGNMENT SCREENING**. All new confirmatory GPU runs must use the canonical pre-QKV form.

---

## 5. Low-rank scaling rule

Preserve:

```text
rank / hidden_size = 1/4
```

| Regime | D | Rank | Temporal heads | P1 added params |
|---|---:|---:|---:|---:|
| CPU | 64 | 16 | 4 | 4,097 |
| Canonical GPU | 256 | 64 | 4 | 65,537 |
| D512 reference | 512 | 128 | 4 | 262,145 |

P1 added parameters follow `4*D*r + 1`.

---

## 6. Canonical four-model family

Exactly four scientific variants form the primary comparison.

### A. Vanilla TRM

No explicit history. It is the recurrent reference baseline.

### B. Gated Uniform History

Uses the exact same causal `z_L` history as Attention but removes query-dependent selection.

Canonical form:

```text
h_bar  = mean(pre-normalized causal history)
z_read = RMSNorm(z + sigmoid(gate_logit) * h_bar)
gate_logit_init = -2
```

Question answered: **Does history access itself help?**

### C. Low-Rank HistoryAttention

Uses the canonical formulation in Section 4.

Question answered: **Does learned selective retrieval add value beyond uniform history access?**

### D. Parameter-Matched No-History

Receives no history tensor. It adds a low-rank current-state transformation with the same parameter budget as P1.

Preferred canonical side path:

```text
RMSNorm(z) -> D -> 2r -> D -> scalar-gated residual -> RMSNorm
```

For `D=256, r=64`:

```text
256*128 + 128*256 + 1 = 65,537
```

which exactly matches P1.

Question answered: **Can the effect be explained by extra capacity rather than history retrieval?**

Before long runs, verify programmatically that P1 and Parameter-Matched add the same parameter count and that Parameter-Matched receives no history tensor.

---

## 7. Old GPU controls are not canonical by default

- Old B1 is not the intended Gated control.
- Old B2 is ungated `RMSNorm(z + mean(history))` and is not the canonical Gated control.
- The old standalone `GatedHistory` differs in gate semantics/init/norm and is not canonical.
- Old B3 is useful as an exploratory capacity control but only approximately parameter-matched and widens the shared FFN.

Preserve old results, but do not relabel them as canonical v1 results.

---

## 8. Frozen CPU configuration

```text
D = 64
H_cycles = 3
L_cycles = 6
L_layers = 2
spatial_heads = 4
rank = 16
temporal_heads = 4
gate_init = -2
pre-QKV RMSNorm
ACT = 4
batch = 4
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 0.01
EMA = off
epochs = 40
final optimization step = 10,000
seeds = 0,1,2,3,4
```

Completed five-seed Vanilla vs Attention result:

| Seed | Vanilla | Attention | Delta |
|---:|---:|---:|---:|
| 0 | 45.525% | 46.728% | +1.204 pp |
| 1 | 47.352% | 45.241% | -2.111 pp |
| 2 | 45.654% | 45.309% | -0.346 pp |
| 3 | 45.136% | 43.117% | -2.019 pp |
| 4 | 45.667% | 46.901% | +1.235 pp |

Aggregate:

- Vanilla mean: **45.867%**
- Attention mean: **45.459%**
- Paired mean delta: **-0.407 pp**
- Positive Attention seeds: **2/5**
- Vanilla mean LM loss: **1.332880**
- Attention mean LM loss: **1.344272**
- Exact accuracy: **0** for both

Allowed interpretation:

- **Not supported:** within-cycle Attention improves the reduced CPU model on average.
- **Supported:** the effect is strongly seed/optimization sensitive.

### Remaining CPU work

Do not rerun completed Track A, Vanilla 0–4, Attention 0–4, or completed Gated seeds.

Remaining canonical CPU runs:

```text
Gated: seed 3, seed 4
Parameter-Matched: seeds 0,1,2,3,4
```

After the final five-seed four-model aggregation, the CPU track is closed. No further CPU architecture tuning is allowed under v1.

---

## 9. Existing GPU screening result

Reported pre-alignment screening:

- B0 cell accuracy: ~68.248%
- P1 cell accuracy: ~63.210%
- Cell delta: ~-5.038 pp
- B0 exact: 2.5%
- P1 exact: 1.0%
- Exact delta: -1.5 pp

Classification: **PRE-ALIGNMENT GPU SCREENING**.

Reasons:

- H2/L4 rather than H3/L6.
- Old post-projection per-head RMSNorm P1.
- Only one known training seed in the reported comparison.
- Complete provenance was not available in the audited checkout.

Preserve the result, recover provenance if possible, and never average it with canonical H3/L6 results.

---

## 10. Frozen canonical GPU scientific configuration

```text
hidden_size = 256
H_cycles = 3
L_cycles = 6
L_layers = 2
spatial_heads = 4
rank = 64
temporal_heads = 4
gate_init = -2
pre-QKV RMSNorm
full causal within-H-cycle z_L history
```

### Why D256

It is a meaningful 4× width increase over D64 while remaining suitable for matched multi-seed replication.

### Why H3/L6/L2

Preserving the recursion structure is more important to the scientific question than maximizing width. The scale-transfer experiment should change scale while keeping the proposal’s recursive structure.

### D512 policy

D512/r128 is not the primary campaign. It may receive a feasibility smoke, but replacing D256 requires explicit agreement and Protocol v2. Multi-seed replication takes priority over a single very expensive D512 seed.

---

## 11. Frozen canonical GPU dataset

Dataset: `sudoku-study-v1`

Verified design:

- 900 train base puzzles
- 64 augmentations per training base
- 58,500 stored training examples
- 100 development puzzles
- 1,000 test puzzles
- train/dev split before augmentation
- leakage checks applied

Paper label: **resource-adapted larger-scale Sudoku setting**.

Do not call it a full Sudoku-Extreme reproduction and do not compare its absolute accuracy directly with the published full-scale Sudoku-Extreme numbers.

---

## 12. Frozen GPU optimization family

```text
optimizer = AdamW
learning_rate = 1e-4
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
EMA = 0.999
compile_model = false
```

All four canonical variants use identical optimization settings.

---

## 13. Deployment parameters and selection rules

The following are hardware-dependent deployment values, not open scientific design choices:

- ACT
- physical batch size
- gradient accumulation
- actual GPU
- actual dtype
- fixed optimization-step budget `N`

Their selection rule is frozen; their final values belong in `TRM_HISTORY_DEPLOYMENT_MANIFEST_v1.md`.

### ACT

Candidates: ACT6 and ACT16.

Choose using only:

- peak VRAM
- seconds per optimization step
- expected full-campaign runtime
- hardware compatibility

Accuracy must not influence ACT selection. Once selected, use the same ACT for every model and seed.

### Batch

Target effective batch: **32**.

Allowed realizations include:

- physical 32 × accumulation 1
- physical 16 × accumulation 2
- physical 8 × accumulation 4

Choose one realization that fits both Vanilla and canonical Attention. All variants use the same effective batch.

### Hardware / dtype

Available project GPUs include Colab T4 and GTX 1080 Ti. Use one hardware/dtype regime for a matched primary comparison whenever possible. Verify actual dtype at runtime; never infer it solely from YAML.

Record GPU, VRAM, compute capability, PyTorch, CUDA, actual dtype, peak allocated/reserved VRAM, and seconds/step.

### Fixed training budget

Every canonical GPU model/seed receives the same fixed `N` optimization updates.

Choose `N` before using canonical Attention outcomes, using only:

- available compute budget
- Vanilla development learning behavior
- measured profiling runtime

No model-specific early stopping, no extra Attention training, and no extension because one model looks weak.

---

## 14. Current GPU implementation requirement

Before canonical GPU long training, align P1 from the old post-projection form to the frozen canonical pre-QKV formulation for Q/K/V. This is an implementation alignment, not a new design decision.

After alignment, run short validation only:

- forward succeeds
- backward succeeds
- finite loss
- finite Q/K/V/O gradients
- finite gate gradient
- read → update → append
- reset every H-cycle
- no ACT history carry
- max queried history length = 6
- full causal history
- checkpoint save/load
- P1 added params = 65,537
- peak VRAM and sec/step recorded

Smoke runs are engineering validation, not paper performance results.

---

## 15. Canonical GPU seed and run policy

Minimum canonical training seeds: `0,1,2`.

Preferred full experiment matrix:

| Model | Seeds |
|---|---|
| Vanilla | 0,1,2 |
| Low-Rank HistoryAttention | 0,1,2 |
| Gated Uniform History | 0,1,2 |
| Parameter-Matched No-History | 0,1,2 |

Total preferred GPU long runs: **12**.

Priority:

1. Vanilla 0–2 + Attention 0–2
2. Gated 0–2
3. Parameter-Matched 0–2

If five seeds are later affordable, add seeds 3–4 symmetrically to all primary variants. Do not add extra seeds only to a model whose result is uncertain.

---

## 16. Test-set policy

Development data may be used for engineering checks and Vanilla-only budget selection.

Test data is reserved for frozen final evaluation.

Do not use test results to select rank, normalization, gate, ACT, optimizer, history method, training budget, or architecture.

---

## 17. Metrics

Primary GPU endpoint: **exact-grid accuracy**.

Key secondary endpoints:

- cell/token accuracy
- LM loss

Additional diagnostics:

- incorrect cells
- row violations
- column violations
- box violations
- parameter count
- runtime
- seconds/step
- VRAM
- throughput

Never call cell accuracy “exact accuracy.”

CPU exact accuracy being zero means the CPU evidence is primarily token-level/mechanistic.

---

## 18. Statistical reporting

For every training seed report:

- absolute Vanilla metric
- absolute modified-model metric
- paired difference

Then report mean and SD across training seeds.

With only three GPU training seeds, primary inference is descriptive. A hierarchical/two-level bootstrap may be computed but must be labeled exploratory. McNemar may be reported for paired exact-grid predictions.

Do not treat many test puzzles as a substitute for independent training seeds.

---

## 19. Mechanistic analyses

Run only after the clean canonical comparisons are complete.

Allowed secondary analyses:

- attention entropy
- expected lookback
- non-adjacent history mass
- delete-most-attended state
- delete-least-attended state
- Gaussian latent corruption

Existing screening supports that P1 can attend to non-adjacent states. It does not establish that the most-attended state is causally necessary or that Attention improves corruption recovery.

---

## 20. Provenance requirement

No number enters a primary paper table unless it can be traced through:

```text
RUN ID
-> FULL CONFIG
-> EXACT GIT COMMIT
-> CHECKPOINT
-> RAW METRICS
-> DATASET / SPLIT
-> SEED
```

Every future run must preserve run ID, date, scientific model, variant, seed, git branch/commit/status, dataset/splits, architecture, ACT, rank/heads/window/norm/gate, optimizer settings, dtype, batch/accumulation, steps, cell/exact/loss, parameter counts, GPU/VRAM/runtime, checkpoint, metrics path, experiment status, and protocol deviations.

---

## 21. Experiment status labels

Use:

`SMOKE`, `PILOT`, `SCREENING`, `CONFIRMATORY`, `FINAL`, `INTERRUPTED`, `INVALID`.

Current classifications:

- Outer `z_H` study: **CONFIRMATORY MECHANISM STUDY**
- CPU `z_L` five-seed study: **CONFIRMATORY REDUCED-COMPUTE STUDY**
- Old H2/L4 GPU result: **PRE-ALIGNMENT SCREENING**
- Canonical D256/H3/L6 GPU study: **CONFIRMATORY SCALE-TRANSFER STUDY**

---

## 22. Paper result organization

Maintain two primary result families.

### Table A — Reduced CPU regime

Rows: Vanilla, Gated Uniform History, Low-Rank HistoryAttention, Parameter-Matched No-History.

Columns: Seeds, Cell Accuracy, Exact Accuracy, LM Loss, Delta vs Vanilla.

Use five training seeds.

### Table B — GPU scale-transfer

Rows: Vanilla, Gated Uniform History, Low-Rank HistoryAttention, Parameter-Matched No-History.

Columns: Seeds, Cell Accuracy, Exact Accuracy, LM Loss, Added Parameters, Runtime, VRAM, Delta vs Vanilla.

Use at least three training seeds.

Do not causally compare absolute CPU and GPU accuracies. Cross-scale comparison focuses on paired effect direction, stability, control relationships, and whether conclusions transfer.

---

## 23. Paper structure

1. **Introduction** — Is the current TRM latent always a sufficient summary? Study explicit latent-history access without promising improvement.
2. **Related Work** — TRM/HRM, recursive reasoning, recurrent memory, hidden-state history, temporal attention, parameter-efficient retrieval. Clarify that history attention operates over recursive time for the same token and does not replace spatial attention.
3. **Method** — Primary within-H-cycle `z_L` Low-Rank HistoryAttention; causality/reset/pre-QKV norm/low-rank QKV/gate/residual; Gated and Parameter-Matched controls; Track A briefly as prior mechanism study.
4. **Results** — Track A mechanism evidence; five-seed reduced CPU Track B; control decomposition; canonical GPU scale transfer; mechanistic diagnostics only when defensible.
5. **Conclusion / Limitations** — reduced CPU regime, CPU exact=0, resource-adapted GPU setup, GPU seed count, different CPU/GPU training regimes, pre-alignment GPU screening, and optimization instability if observed.

---

## 24. Current claim policy

Supported now:

- Explicit latent-history access can affect TRM behavior.
- Outer-step HistoryAttention improved the reduced depth-8 setting.
- Within-cycle Attention is strongly seed-sensitive in the reduced CPU regime.
- Screening-level Attention can allocate substantial mass to non-adjacent states.

Not supported now:

- Universal or monotonic history benefit.
- Mean improvement of proposal-faithful within-cycle Attention on D64 CPU.
- Superiority of the existing GPU screening P1.
- Most-attended state is causally necessary.
- Attention improves latent-corruption recovery.

Not yet tested:

- Canonical proposal-faithful GPU scale transfer.

---

## 25. Result-dependent final interpretation

- **Attention > Vanilla, Gated, and Parameter-Matched consistently:** evidence for learned selective retrieval beyond access and capacity.
- **Gated > Vanilla, Attention ≤ Gated:** history access may help while learned selectivity adds little or optimization difficulty.
- **Attention ≈ Parameter-Matched:** capacity may explain much of the effect.
- **Vanilla > all history variants:** recurrent latent may already be a sufficient summary in the tested regime.
- **Large positive/negative effects across seeds:** optimization stability is the central limitation.
- **CPU unstable/negative but GPU consistently positive across multiple seeds:** effect may be capacity/training-regime dependent.

No scale-dependence claim from one GPU seed.

---

## 26. Stop rules

### CPU

Stop after Gated seeds 3–4, Parameter-Matched seeds 0–4, and the final five-seed four-model aggregation.

### GPU

Minimum useful canonical scale-transfer evidence: 3 Vanilla + 3 Attention runs. Preferred complete matrix: 3 seeds × 4 canonical models.

After the canonical matrix and planned mechanistic analyses, stop. No Maze, ARC, full-rank sweeps, history-window sweeps, or architecture search without explicit agreement and Protocol v2.

---

## 27. Change control

Allowed without Protocol v2:

- genuine bug fixes
- filling deployment-manifest values according to this protocol
- archiving interrupted runs
- improving logging without changing model behavior

A bug fix must be documented, affected runs identified, invalid runs marked `INVALID`, and old artifacts preserved.

Requires Protocol v2:

- changing D/H/L/layers
- rank policy
- history semantics
- norm order
- gate formulation
- canonical controls
- dataset
- optimizer family
- primary metrics
- seed policy
- scientific question

---

## 28. Final team principle

We are not trying to make Attention win.

We are testing what explicit latent-history retrieval contributes.

Positive, negative, null, and seed-sensitive results are all valid outcomes.

Every future run should optimize for **comparability, reproducibility, interpretability, and scientific honesty**, not favorable performance.
