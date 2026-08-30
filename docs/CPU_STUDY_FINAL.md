# CPU Study Final — TRM Latent-History Project

**Status:** FINAL / CPU TRAINING FROZEN
**Scope:** Consolidated final record for all CPU evidence used by the paper.

## 1. Research question

The project does not assume that HistoryAttention improves TRM. The CPU study asks:

> When, why, and under what conditions does explicit latent-history access help Tiny Recursive Models?

Two mechanisms were studied separately:

- **Track A:** outer-step `z_H` history, used as a supporting mechanism study.
- **Track B:** within-H-cycle `z_L` history, the proposal-faithful primary method.

These tracks must not be averaged or presented as replications of the same mechanism.

---

## 2. Track A — outer-step `z_H` history

Track A uses projection-free, token-aligned attention over strictly previous outer-step latent states. Historical states are detached and reset-isolated between puzzles.

### Final depth summary

| Outer depth | Seeds | Vanilla | Recency | HistoryAttention | Attention − Vanilla |
|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 48.268% | 48.996% | 48.311% | +0.043 pp |
| 4 | 5 | 46.757% | 47.095% | 46.905% | +0.148 pp |
| 8 | 5 | 42.659% | 42.933% | 43.389% | **+0.730 pp** |
| 16 | 3 | 42.113% | 42.329% | 42.171% | +0.058 pp |

At depth 8, the paired 95% confidence interval for Attention − Vanilla was
`[+0.367,+1.110]` pp and Attention was positive in `5/5` seeds.

The depth-8 minus depth-4 interaction was `+0.581 pp` with 95% CI
`[-0.628,+1.756]` pp, so the evidence does **not** establish a monotonic
increase in the benefit of history with recursion depth.

**Supported Track-A claim:** explicit latent-history retrieval can help in a
specific recursive regime.

---

## 3. Track B — within-H-cycle `z_L` history

Track B is the primary proposal-faithful mechanism.

For every H-cycle:

```text
history = [z_L at H-cycle entry]

for each L-step:
    READ from the complete causal history
    UPDATE z_L with the TRM block
    APPEND the new z_L
```

The history is reset at every H-cycle boundary, is not carried through ACT, and
does not leak between puzzles. The newly updated state is never available to
the retrieval performed in the same L-step.

### Canonical Low-Rank HistoryAttention

```text
q   = W_Q RMSNorm(z)
k_i = W_K RMSNorm(h_i)
v_i = W_V RMSNorm(h_i)

alpha   = softmax((q · k_i) / sqrt(d_head))
context = sum_i alpha_i v_i
memory  = W_O(context)

g      = sigmoid(gate_logit)
z_read = RMSNorm(z + g * memory)
```

CPU values:

```text
D = 64
rank = 16
temporal heads = 4
gate_logit_init = -2
pre-QKV RMSNorm = true
```

---

## 4. Four-model causal decomposition

The final Track-B comparison contains exactly four variants.

| Model | Scientific question |
|---|---|
| Vanilla TRM | Baseline without explicit history |
| Gated Uniform History | Does access to causal history help without learned selection? |
| Low-Rank HistoryAttention | Does query-dependent temporal selection add value? |
| Parameter-Matched No-History | Can the effect be explained by extra capacity alone? |

The Parameter-Matched model receives no history tensor.

---

## 5. Frozen CPU configuration

```text
hidden_size = 64
H_cycles = 3
L_cycles = 6
L_layers = 2
spatial_heads = 4
halt_max_steps = 4

rank = 16
temporal_heads = 4
gate_logit_init = -2
pre-QKV RMSNorm = true

batch = 4
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 0.01
EMA = off
epochs = 40
final optimization step = 10,000
seeds = 0,1,2,3,4
device = CPU / float32
```

All 20 canonical runs completed. Each run has evaluation metrics at steps
`1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000`, yielding 160 stored
learning-curve evaluations.

---

## 6. Final Track-B results

| Model | Final token accuracy (mean ± SD) | LM loss (mean ± SD) | Exact | Paired delta vs Vanilla |
|---|---:|---:|---:|---:|
| Vanilla | **45.867 ± 0.858%** | **1.332880 ± 0.040071** | 0 | — |
| Gated Uniform History | 45.747 ± 0.503% | 1.339788 ± 0.025772 | 0 | −0.120 pp |
| Low-Rank HistoryAttention | 45.459 ± 1.520% | 1.344272 ± 0.032840 | 0 | −0.407 pp |
| Parameter-Matched No-History | 45.428 ± 0.655% | 1.346648 ± 0.021974 | 0 | −0.438 pp |

Paired 95% t intervals at the final checkpoint:

| Comparison | Mean delta | 95% CI | Positive seeds |
|---|---:|---:|---:|
| Gated − Vanilla | −0.120 pp | [−1.380,+1.141] | 2/5 |
| Attention − Vanilla | −0.407 pp | [−2.447,+1.632] | 2/5 |
| Parameter-Matched − Vanilla | −0.438 pp | [−1.959,+1.083] | 1/5 |

None of the final intervals excludes zero. With five training seeds, these
intervals are wide and do not establish statistical equivalence.

---

## 7. Optimization trajectory

HistoryAttention showed a reproducible early advantage that was not sustained.

| Step | Attention − Vanilla | Gated − Vanilla | Parameter-Matched − Vanilla |
|---:|---:|---:|---:|
| 1250 | **+0.688 pp** | +0.536 pp | +0.532 pp |
| 2500 | **+3.099 pp** | +0.500 pp | +0.826 pp |
| 3750 | +0.733 pp | +0.628 pp | +0.598 pp |
| 5000 | +0.986 pp | +0.342 pp | +0.447 pp |
| 6250 | +0.080 pp | +0.480 pp | +0.109 pp |
| 7500 | −0.001 pp | −0.323 pp | −0.270 pp |
| 8750 | −0.212 pp | −0.273 pp | −0.419 pp |
| 10000 | −0.407 pp | −0.120 pp | −0.438 pp |

For Attention − Vanilla at step 1250:

```text
mean = +0.688 pp
95% t CI = [+0.347,+1.028] pp
raw p = 0.005
Holm-adjusted p across 8 checkpoint comparisons = 0.040
positive seeds = 5/5
```

At step 2500 the mean advantage was `+3.099 pp` and all `5/5` seeds were
positive, but the checkpoint-level result did not survive Holm correction.

The early two-checkpoint average and early-to-late contrast are post-hoc
trajectory summaries and should be labeled exploratory if reported.

---

## 8. Learned gate analysis

All three added branches moved away from the common initialization
`sigmoid(-2) = 0.1192`.

| Model | Final gate mean ± SD | Ratio to initial gate |
|---|---:|---:|
| HistoryAttention | **0.1390 ± 0.0113** | 1.17× |
| Gated Uniform History | **0.2279 ± 0.0267** | 1.91× |
| Parameter-Matched No-History | **0.1361 ± 0.0058** | 1.14× |

This rules out the simple explanation that the added branches failed only
because their learned gates remained at initialization. Gated Uniform History
opened substantially more than the Attention and capacity-control branches,
yet did not improve final token accuracy over Vanilla.

Because there are only five seeds, correlations between final gate magnitude
and final accuracy are treated as exploratory and are not primary paper claims.

---

## 9. Final CPU interpretation

The CPU evidence does **not** support the claim that proposal-faithful
within-cycle HistoryAttention improves final TRM accuracy on average.

The combined CPU evidence supports a more specific conclusion:

1. Explicit latent-history access can matter in a particular recursive regime
   (Track A, depth 8).
2. Proposal-faithful within-cycle HistoryAttention changes optimization
   dynamics and gives a reproducible early advantage.
3. That early advantage disappears by the frozen final checkpoint and becomes
   strongly seed-sensitive.
4. The Gated and Parameter-Matched controls show that neither history access,
   selectivity, nor extra capacity alone yields a robust final CPU improvement.
5. Learned gates move away from initialization, so the negative/null final
   result is not explained by a completely inactive auxiliary branch.

Paper-ready sentence:

> Explicit latent-history access can alter TRM reasoning and optimization, but
> its benefit is regime-dependent: a supporting outer-history study shows a
> reproducible gain at intermediate recursion depth, whereas proposal-faithful
> within-cycle selective retrieval yields an early optimization advantage that
> is not sustained to convergence in the reduced CPU regime.

---

## 10. Limitations

- Exact-grid accuracy is zero in the reduced CPU studies; CPU conclusions are
  token-level/mechanistic rather than evidence of improved complete Sudoku
  solving.
- Track A and Track B use different history semantics and must remain separate.
- Five seeds remain a small sample for training-run uncertainty.
- The CPU regime is deliberately resource-reduced and should not be compared
  causally to larger GPU absolute accuracy.
- Post-hoc trajectory summaries must be labeled exploratory.
- The canonical GPU scale-transfer experiment is required to test whether the
  Track-B behavior changes with scale/training regime.

---

## 11. Provenance and artifacts

Primary final evidence:

```text
docs/CPU_LCYCLE_FINAL_EVIDENCE_v1.md
docs/CPU_LCYCLE_STATISTICAL_ANALYSIS_v1.md
docs/data/CPU_LCYCLE_ALL_METRICS_v1.csv
docs/figures/cpu_paired_delta_vs_vanilla.png
docs/figures/cpu_learning_curves_accuracy.png
docs/figures/cpu_final_seed_accuracy.png
```

Track-A evidence:

```text
docs/CPU_HISTORY_FINAL_SUMMARY.md
results/history-depth/trm4-vs-trm8-depth-interaction.json
results/history-multiseed/trm4-multiseed-statistics.json
```

Canonical implementation/evidence commit:

```text
cfa323f  Add canonical L-cycle history controls and freeze CPU evidence
```

CPU analysis/figure commit:

```text
53e91f8  Add CPU statistical analysis and paper figures
```

The historical `PAPER_EVIDENCE_LOG.md` is a research log, not the final
authoritative summary.

---

## 12. Freeze decision

CPU **training** under protocol v1 is closed.

Permitted remaining CPU work is analysis of existing artifacts, figure
generation, reproducibility checks, documentation, or clearly labeled
post-hoc inference-only mechanistic ablations. No new training variant should
enter the canonical CPU result family.
