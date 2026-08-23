# TRM Latent-History Project — Paper Evidence Log

This file is the canonical running record of experimentally supported results,
paper-ready statements, methodological decisions, caveats, and open scientific
questions for the TRM latent-history project.

It should be updated whenever an important experimental result or paper-relevant
interpretation is obtained.

---

# 1. Experimental Protocol

## Primary development setting

The current controlled development anchor is:

- Architecture: reduced CPU TRM
- halt_max_steps: 4
- epochs: 40
- seed: 0
- global batch size: 4
- learning rate: 1e-3
- hidden size: 64
- L_cycles: 1
- evaluation split: dev
- official test split: not used for model development
- dataset: sudoku-baseline-v2

TRM-4 is used as the primary latent-history development baseline because it
provides non-trivial outer-step history while remaining substantially less
degraded than TRM-8 and TRM-16 under the reduced CPU optimization budget.

Paper-ready wording:

"TRM-4 was selected as the primary latent-history development setting because it
provides non-trivial recursive history while avoiding the stronger degradation
observed at larger outer-step budgets in the reduced CPU regime."

---

# 2. Baseline TRM Results

Matched 40-epoch seed-0 development results:

| Model | Dev Accuracy | LM Loss | Exact Accuracy |
|---|---:|---:|---:|
| TRM-2 | 0.482469 | 1.218538 | 0.0 |
| TRM-4 | 0.458395 | 1.357913 | 0.0 |
| TRM-8 | 0.428148 | 1.444620 | 0.0 |
| TRM-16 | 0.421728 | 1.462849 | 0.0 |

Supported interpretation:

"Under the reduced CPU regime and a fixed 40-epoch optimization budget,
increasing outer reasoning depth alone did not improve development performance."

Important caveat:

This result does not establish that recursion is intrinsically harmful.
Deeper configurations may require different optimization budgets or model scale.

---

# 3. Latent-History Interface

The latent-history mechanism records previous outer-step z_H states.

At outer step t, valid history is:

H_t = {z_1, ..., z_(t-1)}

The current state is never inserted into the history before aggregation.

Historical states are detached before storage.

Supported methodological statement:

"We preserve the original truncated-gradient semantics by storing detached
outer-step latent states and exposing only strictly previous states to the
history aggregation module."

Reset isolation is implemented per batch slot to prevent cross-puzzle latent
state leakage.

---

# 4. UniformMeanHistory

Definition:

UniformMeanHistory performs a parameter-free average over the current state and
all valid previous latent states.

Final TRM-4 / 40 epoch / seed-0 result:

- Dev accuracy: 0.456420
- LM loss: 1.308096
- Exact accuracy: 0.0
- Extra trainable parameters: 0

Vanilla reference:

- Dev accuracy: 0.458395
- LM loss: 1.357913

Difference versus vanilla:

- Accuracy: approximately -0.20 percentage points
- LM loss: approximately -0.04982

Trajectory observation:

UniformMeanHistory achieved higher intermediate development accuracy than
vanilla TRM at every recorded checkpoint before the final evaluation, but the
final development accuracy was nearly unchanged relative to vanilla.

Paper-ready wording:

"Under the matched seed-0 TRM-4 setting, UniformMeanHistory produced nearly
unchanged final development accuracy relative to vanilla TRM while achieving a
lower language-model loss."

Additional interpretation:

"Uniform latent-history averaging appeared to improve optimization behavior and
probabilistic fit, but did not translate into higher final token accuracy under
the evaluated 40-epoch setting."

Statistical caveat:

This is a single-seed development observation and does not establish superiority,
degradation, or equivalence.

---

# 5. RecencyWeightedHistory

Definition:

The current state receives weight 1.0, while previous states receive exponentially
decaying weights:

- lag 1: 0.5
- lag 2: 0.25
- lag 3: 0.125
- etc.

This method is parameter-free.

Final TRM-4 / 40 epoch / seed-0 result:

- Dev accuracy: 0.472099
- LM loss: 1.274998
- Exact accuracy: 0.0
- Extra trainable parameters: 0

Difference versus vanilla:

- Accuracy: approximately +1.37 percentage points
- LM loss: approximately -0.08291

Difference versus UniformMeanHistory:

- Accuracy: approximately +1.57 percentage points
- LM loss: approximately -0.03310

Trajectory observation:

RecencyWeightedHistory maintained a development-accuracy advantage over vanilla
TRM throughout the recorded training trajectory and ended with the highest final
accuracy among the currently tested non-attention methods.

Paper-ready wording:

"Under the matched seed-0 TRM-4 development setting, parameter-free
recency-weighted latent-history aggregation improved development accuracy by
approximately 1.37 percentage points over vanilla TRM while also reducing
language-model loss."

Paper-ready interpretation:

"The gain from RecencyWeightedHistory cannot be attributed simply to increased
trainable model capacity, since the method introduces no additional trainable
parameters."

Scientific interpretation:

"The contrast between UniformMeanHistory and RecencyWeightedHistory suggests
that temporal structure within latent history may matter: treating all previous
states equally was not beneficial for final accuracy, whereas emphasizing recent
states produced a stronger development result."

Statistical caveat:

This is still a single-seed development result and does not establish statistical
superiority.

---

# 6. GatedHistory

Definition:

GatedHistory learns a single scalar gate:

z'_t = g * z_t + (1 - g) * h_t

where h_t is the mean valid latent history.

The initial gate is:

g = 0.5

Extra trainable parameters:

1

Final TRM-4 / 40 epoch / seed-0 result:

- Dev accuracy: 0.465556
- LM loss: 1.261310
- Exact accuracy: 0.0

Final learned values:

- gate_logit: -1.228600
- current-state weight: approximately 0.2264
- history weight: approximately 0.7736

Difference versus vanilla:

- Accuracy: approximately +0.72 percentage points
- LM loss: approximately -0.09660

Difference versus RecencyWeightedHistory:

- Accuracy: approximately -0.65 percentage points
- LM loss: approximately -0.01369

Trajectory observation:

GatedHistory remained above vanilla TRM in development accuracy throughout the
recorded trajectory and achieved the lowest final LM loss among the tested
non-attention methods.

Paper-ready wording:

"Under the matched seed-0 TRM-4 setting, GatedHistory improved final development
accuracy over vanilla TRM and achieved the lowest language-model loss among the
tested non-attention history mechanisms."

Paper-ready interpretation:

"The learned scalar gate moved substantially from its initial 0.5 value toward
greater reliance on latent history, assigning approximately 77.4 percent of the
final mixture weight to mean historical state."

Important caveat:

The learned gate value is descriptive evidence from one run and should not be
interpreted as a general property of the architecture without multi-seed
confirmation.

---

# 7. Current Non-Attention Comparison

Current seed-0 TRM-4 / 40 epoch results:

| Method | Dev Accuracy | LM Loss | Extra Params |
|---|---:|---:|---:|
| Vanilla TRM | 0.458395 | 1.357913 | 0 |
| UniformMeanHistory | 0.456420 | 1.308096 | 0 |
| GatedHistory | 0.465556 | 1.261310 | 1 |
| RecencyWeightedHistory | 0.472099 | 1.274998 | 0 |

Current descriptive ranking:

- Highest final accuracy: RecencyWeightedHistory
- Lowest final LM loss: GatedHistory
- UniformMeanHistory: lower loss but nearly unchanged final accuracy
- Vanilla TRM: reference condition

Paper-ready synthesis:

"Across the current seed-0 development experiments, access to latent history
consistently improved language-model loss, but the effect on final accuracy
depended strongly on how historical states were weighted."

Additional synthesis:

"The strongest final token accuracy was obtained with a fixed recency bias,
whereas a learned scalar gate achieved the lowest language-model loss. This
suggests that history access alone is insufficient to explain performance;
the mechanism used to prioritize or combine historical states appears important."

---

# 8. LastStateHistory

Definition:

LastStateHistory uses only the most recent previous latent state:

z'_t = (z_t + 0.5 * z_(t-1)) / 1.5

The method is parameter-free.

Scientific role:

This is a direct ablation of RecencyWeightedHistory. It tests whether the
Recency gain can be explained primarily by the immediately preceding latent
state, or whether older historical states provide additional useful information.

Final TRM-4 / 40 epoch / seed-0 result:

- Dev accuracy: 0.463889
- LM loss: 1.300170
- Exact accuracy: 0.0
- Extra trainable parameters: 0

Vanilla reference:

- Dev accuracy: 0.458395
- LM loss: 1.357913

RecencyWeightedHistory reference:

- Dev accuracy: 0.472099
- LM loss: 1.274998

Difference versus vanilla:

- Accuracy: approximately +0.55 percentage points
- LM loss: approximately -0.05774

Difference versus RecencyWeightedHistory:

- Accuracy: approximately -0.82 percentage points
- LM loss: approximately +0.02517

Trajectory observation:

LastStateHistory increased steadily through step 7500, where it reached its
highest recorded accuracy of approximately 46.64 percent, and then slightly
declined toward the final checkpoint.

RecencyWeightedHistory, in contrast, continued improving through the final
checkpoint and finished at approximately 47.21 percent.

Paper-ready wording:

"Under the matched seed-0 TRM-4 setting, the LastState ablation improved
development accuracy over vanilla TRM but remained approximately 0.82
percentage points below RecencyWeightedHistory."

Paper-ready interpretation:

"The LastState ablation peaked earlier and slightly declined toward the end of
training, whereas RecencyWeightedHistory continued to improve through the final
checkpoint, suggesting that access to a broader temporally weighted latent
history may support more sustained optimization."

Scientific interpretation:

"The advantage of RecencyWeightedHistory over LastStateHistory suggests that its
development gain may not be explained solely by access to the immediately
preceding latent state. Older latent states may provide additional useful
information."

Statistical caveat:

This is a single-seed development observation. It does not yet establish that
multi-step history is statistically superior to latest-state-only history.

---

# 9. Statistical Policy

Current experiments are development-stage seed-0 runs.

Do not make final statistical claims from these runs.

Preferred final protocol:

- run important methods on multiple seeds
- preferably seeds 0, 1, 2, 3, 4
- report mean and standard deviation
- perform paired comparisons using the same dev puzzles
- use puzzle-level analysis rather than treating all individual Sudoku cells as
  fully independent samples
- consider paired bootstrap confidence intervals over puzzles
- define any practical-equivalence margin before inspecting final statistical
  results

Paper wording when evidence is inconclusive:

"The method did not provide statistically supported evidence of improvement
over the matched vanilla baseline under the evaluated setting."

Do not interpret failure to reject a null hypothesis as proof of equivalence.

---

# 10. Planned Experimental Path

Immediate:

1. Finish LastStateHistory 40-epoch result.
2. Record its trajectory and comparison with RecencyWeightedHistory.
3. Freeze LastStateHistory implementation and result.

Then:

4. Complete HistoryAttention implementation on the teammate branch.
5. Compare Attention against:
   - Vanilla
   - Uniform
   - LastState
   - Recency
   - Gated

Then:

6. Select the important methods for multi-seed evaluation.
7. Run multi-seed TRM-4 comparisons.
8. Evaluate promising methods across recursion depths:
   - TRM-2
   - TRM-4
   - TRM-8
   - TRM-16

Main future scientific question:

"Does the usefulness of explicit latent-history retrieval increase with outer
reasoning depth?"

---

# 11. Paper-Ready Core Narrative

Current tentative narrative:

"Explicit access to recursive latent history does not appear uniformly useful.
Simple uniform averaging improves probabilistic fit but does not improve final
development accuracy. In contrast, emphasizing recent latent states yields a
clearer development gain without increasing trainable parameter count, while a
learned scalar gate also improves over vanilla TRM and strongly shifts toward
using historical information. These observations motivate selective
history-retrieval mechanisms such as HistoryAttention."

This narrative is provisional and must be updated after multi-seed and
HistoryAttention results.

---

# 12. Important Writing Restrictions

Do not currently claim:

- that RecencyWeightedHistory is statistically superior
- that GatedHistory is statistically superior
- that recursion is intrinsically harmful
- that more epochs do not help
- that latent history improves exact Sudoku solving
- that Attention is necessary

Exact puzzle accuracy is currently zero for the tested reduced development
setting.

Final claims must distinguish:

- development observations
- multi-seed evidence
- statistical confidence
- exact-solve performance

# 13. HistoryAttention

## Method

HistoryAttention performs token-aligned selective retrieval over strictly
previous outer-step latent states.

For each sequence position, the current latent vector acts as the query and the
corresponding vectors from valid previous outer states act directly as keys and
values.

The first implementation is intentionally projection-free. It therefore tests
content-dependent history selection without introducing learned query, key, or
value projections.

Attention is computed across the outer-step history dimension only.

The retrieved context is fused with the current state using one learned scalar
gate:

z'_t = g * z_t + (1 - g) * c_t

where:

g = sigmoid(gate_logit)

The initial gate is 0.5.

Extra trainable parameters:

1

This matches the trainable parameter overhead of GatedHistory.

## Seed-0 TRM-4 Result

Matched configuration:

- halt_max_steps: 4
- epochs: 40
- seed: 0
- global batch size: 4
- learning rate: 1e-3
- hidden size: 64
- L_cycles: 1
- evaluation split: dev

Final result:

- Dev accuracy: 0.4768518806
- LM loss: 1.2298237085
- Exact accuracy: 0.0
- gate_logit: -1.0763931274
- current-state weight: approximately 0.2542
- history weight: approximately 0.7458

Differences versus important controls:

Versus Vanilla TRM:

- Accuracy: approximately +1.85 percentage points
- LM loss: approximately -0.12809

Versus GatedHistory:

- Accuracy: approximately +1.13 percentage points
- LM loss: approximately -0.03149

Versus RecencyWeightedHistory:

- Accuracy: approximately +0.48 percentage points
- LM loss: approximately -0.04517

Training trajectory:

| Step | Dev Accuracy | LM Loss |
|---:|---:|---:|
| 1250 | 0.429136 | 1.444027 |
| 2500 | 0.441728 | 1.396567 |
| 3750 | 0.461975 | 1.340990 |
| 5000 | 0.464012 | 1.289414 |
| 6250 | 0.472716 | 1.253708 |
| 7500 | 0.475864 | 1.240461 |
| 8750 | 0.475432 | 1.233669 |
| 10000 | 0.476852 | 1.229824 |

Paper-ready development wording:

"Under the matched seed-0 TRM-4 development setting, projection-free
HistoryAttention achieved the highest final token accuracy and lowest
language-model loss among the evaluated latent-history mechanisms."

Important controlled-comparison interpretation:

"HistoryAttention and GatedHistory each introduce only one additional trainable
scalar parameter. The observed seed-0 advantage of HistoryAttention over
GatedHistory therefore cannot be attributed to a larger trainable parameter
count."

Scientific interpretation:

"The seed-0 result provides preliminary evidence that content-dependent
selection over previous latent states may provide value beyond both fixed
recency weighting and learned but non-selective history mixing."

Statistical caveat:

This is a single-seed development result and does not establish statistical
superiority over Vanilla TRM, RecencyWeightedHistory, or GatedHistory.

Exact puzzle accuracy remains zero.

---

# 14. Multi-Seed Validation Status

The primary methods selected for multi-seed validation are:

- Vanilla TRM
- RecencyWeightedHistory
- GatedHistory
- HistoryAttention

Planned seeds:

0, 1, 2, 3, 4

Per-puzzle evaluation export was validated without modifying the canonical
training or evaluation implementation.

The exported tensors are:

- inputs: [200, 81]
- labels: [200, 81]
- puzzle_identifiers: [200]
- logits: [200, 81, 11]

Recomputed token accuracy from the exported logits matches the evaluation
pipeline metric to floating-point precision.

The current puzzle_identifiers field is not unique across the development set
and must not be used as the pairing key.

Pairing across runs is instead validated by exact equality of the stored inputs
and labels. This was verified between independent evaluation runs.

## HistoryAttention Seed 1

Final TRM-4 / 40 epoch / seed-1 result:

- Dev accuracy: 0.4703086913
- LM loss: 1.2740758657
- Exact accuracy: 0.0
- gate_logit: -1.1674356461
- current-state weight: approximately 0.2373
- history weight: approximately 0.7627

The per-puzzle exported prediction file contains all 200 development puzzles.

Recomputed accuracy:

0.4703086317

Reported accuracy:

0.4703086913

The approximately 6e-8 difference is consistent with floating-point reduction
precision.

Preliminary observation:

Both HistoryAttention seed 0 and seed 1 learned substantial reliance on latent
history, assigning approximately 74.6 percent and 76.3 percent of the scalar
mixture weight to retrieved historical context, respectively.

This observation is descriptive only and requires completion of the planned
multi-seed experiment before interpretation.
