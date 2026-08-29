# CPU Within-H-Cycle Latent-History Study — Final Evidence v1

## Status

**CPU CANONICAL TRAINING TRACK: CLOSED / FROZEN**

No additional CPU training is permitted under protocol v1.

## Canonical configuration

- Device: CPU / float32
- Hidden size: 64
- H cycles: 3
- L cycles: 6
- L layers: 2
- Spatial heads: 4
- halt_max_steps: 4
- Global batch size: 4
- Epochs: 40
- Final optimization step: 10,000
- Learning rate: 1e-3
- Weight decay: 0.01
- Seeds: 0, 1, 2, 3, 4
- Evaluation checkpoints:
  1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000

HistoryAttention:
- within-H-cycle z_L history
- rank: 16
- temporal heads: 4
- gate_logit_init: -2
- pre-QKV RMSNorm: true
- causal READ -> UPDATE -> APPEND semantics
- history reset every H-cycle

## Canonical model family

1. Vanilla TRM
2. Gated Uniform History
3. Low-Rank HistoryAttention
4. Parameter-Matched No-History

Total completed canonical runs: **20 / 20**.

Total stored learning-curve evaluations: **160 / 160**.

---

## Final seed-level results

### Vanilla

| Seed | Accuracy | LM Loss | Exact |
|---:|---:|---:|---:|
| 0 | 45.5247% | 1.344894 | 0 |
| 1 | 47.3518% | 1.262387 | 0 |
| 2 | 45.6543% | 1.353546 | 0 |
| 3 | 45.1358% | 1.342637 | 0 |
| 4 | 45.6667% | 1.360933 | 0 |

### Attention

| Seed | Accuracy | LM Loss | Exact |
|---:|---:|---:|---:|
| 0 | 46.7284% | 1.323962 | 0 |
| 1 | 45.2407% | 1.353584 | 0 |
| 2 | 45.3086% | 1.362693 | 0 |
| 3 | 43.1173% | 1.382082 | 0 |
| 4 | 46.9012% | 1.299041 | 0 |

### Gated

| Seed | Accuracy | LM Loss | Exact |
|---:|---:|---:|---:|
| 0 | 46.1728% | 1.314445 | 0 |
| 1 | 45.8765% | 1.335045 | 0 |
| 2 | 45.4383% | 1.355991 | 0 |
| 3 | 46.2099% | 1.318154 | 0 |
| 4 | 45.0370% | 1.375306 | 0 |

### Parameter-Matched

| Seed | Accuracy | LM Loss | Exact |
|---:|---:|---:|---:|
| 0 | 46.5000% | 1.314588 | 0 |
| 1 | 44.9630% | 1.375883 | 0 |
| 2 | 45.1420% | 1.342083 | 0 |
| 3 | 44.9383% | 1.350950 | 0 |
| 4 | 45.5988% | 1.349736 | 0 |

---

## Final aggregate results

| Model | Accuracy mean ± SD | LM Loss mean ± SD |
|---|---:|---:|
| Vanilla | 45.867% ± 0.858 | 1.332880 ± 0.040071 |
| Gated | 45.747% ± 0.503 | 1.339788 ± 0.025772 |
| Attention | 45.459% ± 1.520 | 1.344272 ± 0.032840 |
| Parameter-Matched | 45.428% ± 0.655 | 1.346648 ± 0.021974 |

Final paired accuracy deltas relative to Vanilla:

- Gated - Vanilla: **-0.120 pp**
- Attention - Vanilla: **-0.407 pp**
- Parameter-Matched - Vanilla: **-0.438 pp**

Additional comparison:

- Gated - Attention: **+0.288 pp**
- Parameter-Matched - Attention: **-0.031 pp**

Exact-grid accuracy was zero for all 20 runs.

---

## Optimization trajectory evidence

Paired mean accuracy delta relative to Vanilla:

| Step | Attention | Gated | Parameter-Matched |
|---:|---:|---:|---:|
| 1250 | +0.688 pp | +0.536 pp | +0.532 pp |
| 2500 | +3.099 pp | +0.500 pp | +0.826 pp |
| 3750 | +0.733 pp | +0.628 pp | +0.598 pp |
| 5000 | +0.986 pp | +0.342 pp | +0.447 pp |
| 6250 | +0.080 pp | +0.480 pp | +0.109 pp |
| 7500 | -0.001 pp | -0.323 pp | -0.270 pp |
| 8750 | -0.212 pp | -0.273 pp | -0.419 pp |
| 10000 | -0.407 pp | -0.120 pp | -0.438 pp |

Attention was positive relative to Vanilla in 5/5 seeds at steps 1250 and
2500, but only 2/5 seeds at the final checkpoint.

This supports an optimization-dynamics interpretation rather than a final
accuracy improvement claim.

A transient optimization anomaly occurred at step 2500, especially for
Gated seed 4 (32.599% accuracy, LM loss 3.0287). The run recovered by the
final checkpoint and is retained; no run is excluded post hoc.

---

## Supported interpretation

The reduced CPU study does **not** support the claim that within-cycle
HistoryAttention improves final TRM accuracy on average.

The study does support the following observations:

1. Explicit history-related modifications alter early optimization dynamics.
2. HistoryAttention shows a strong early advantage, including 5/5 positive
   paired seeds at the first two evaluation checkpoints.
3. The early Attention advantage is not sustained to convergence.
4. Attention exhibits the largest between-seed variance at the final
   checkpoint.
5. Gated Uniform History is more stable at the final checkpoint than
   HistoryAttention.
6. Parameter-Matched No-History finishes almost identically to Attention in
   mean accuracy, so the reduced CPU evidence does not isolate a final benefit
   from selective history retrieval.
7. Exact Sudoku solving is not established in this reduced CPU regime.

Paper-ready summary:

"Explicit history-related modifications altered early optimization dynamics,
but none produced a consistent final improvement over Vanilla in the reduced
CPU regime. Selective HistoryAttention showed the largest early gains, yet
these gains were not sustained and became strongly seed-dependent near
convergence."

---

## Provenance audit

Repository branch:

`feature/lcycle-lowrank-history`

Git HEAD observed during final audit:

`976630fe6fa5ad0168c19b06de419d4d24d6b9b8`

Two TRM source snapshots were used during the canonical campaign.

Vanilla + Attention checkpoint snapshot:

`9C2DFD4D94B0C361F6BC6C21199572E08F5E2E843838098B2D41172DA87AA0A3`

Gated + Parameter-Matched checkpoint snapshot:

`129A0473FFC3D8847D84970CB745FC5A974AD0D1A8B3945A946611B6FDCFACA3`

The source diff was audited. The later snapshot adds explicit method dispatch,
Gated Uniform History, and Parameter-Matched No-History controls while
preserving the pre-existing Vanilla and Attention computation paths.

Interrupted Gated seed-2 directories are archival only and are not included in
the canonical 20-run result set.

The working tree was not clean during the final audit. Results are therefore
identified by saved run configurations, checkpoint source snapshots, metrics,
and recorded hashes rather than by Git commit alone.

---

## Freeze decision

CPU model training under protocol v1 is complete.

Future CPU work may analyze existing artifacts, generate tables/figures, or
verify provenance, but must not introduce new training variants into the
canonical CPU result set.

Primary remaining experimental work is the matched GPU scale-transfer study.
