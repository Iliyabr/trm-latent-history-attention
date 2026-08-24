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
## 15. Final CPU Latent-History Study

### Status

CPU latent-history experimentation is CLOSED.

The CPU experiments are treated as controlled mechanism validation, not as
a reproduction of the full-scale TRM Sudoku result.

### Experimental scope

Confirmatory:
- TRM-4: 5 matched seeds (0-4)
- TRM-8: 5 matched seeds (0-4)
- paired puzzle-level prediction export
- exact input/label ordering verification
- paired two-way bootstrap over seeds and puzzles

Exploratory:
- TRM-2: 3 seeds (0-2)
- TRM-16: 3 seeds (0-2)

All reduced-compute experiments used the same 200-puzzle development set.

### Reduced CPU architecture

- hidden_size = 64
- num_heads = 4
- H_cycles = 1
- L_cycles = 1
- H_layers = 0
- L_layers = 1
- puzzle embeddings disabled
- global batch size = 4
- learning rate = 1e-3
- training budget = 40 epochs
- CPU threads = 8

### HistoryAttention mechanism

HistoryAttention performs projection-free, token-aligned attention over
strictly previous outer-step latent states.

The mechanism:
- never attends to the current state as history
- masks unused history slots
- preserves exact identity when no previous state exists
- uses detached historical latent states
- maintains per-sample reset isolation
- introduces exactly one additional trainable scalar parameter

The scalar gate mixes the current latent state and retrieved historical
context.

### Final accuracy summary

| Depth | Seeds | Vanilla | Recency | HistoryAttention | Attention - Vanilla |
|---|---:|---:|---:|---:|---:|
| 2 | 3 | 48.268 +/- 0.119% | 48.996 +/- 0.586% | 48.311 +/- 0.603% | +0.043 pp |
| 4 | 5 | 46.757 +/- 1.337% | 47.095 +/- 0.903% | 46.905 +/- 0.574% | +0.148 pp |
| 8 | 5 | 42.659 +/- 0.203% | 42.933 +/- 0.513% | 43.389 +/- 0.145% | +0.730 pp |
| 16 | 3 | 42.113 +/- 0.131% | 42.329 +/- 0.566% | 42.171 +/- 0.147% | +0.058 pp |

### Confirmatory paired results

TRM-4:

- Attention vs Vanilla:
  - mean difference = +0.148 percentage points
  - 95% paired two-way bootstrap CI = [-1.043, +1.317] pp

- Attention vs Recency:
  - mean difference = -0.190 percentage points
  - 95% paired two-way bootstrap CI = [-0.869, +0.458] pp

No statistically resolved HistoryAttention advantage was observed at TRM-4.

TRM-8:

- Attention vs Vanilla:
  - mean difference = +0.730 percentage points
  - 95% paired two-way bootstrap CI = [+0.367, +1.110] pp
  - Attention outperformed Vanilla in 5/5 seeds

- Attention vs Recency:
  - mean difference = +0.456 percentage points
  - 95% paired two-way bootstrap CI = [-0.122, +0.998] pp

HistoryAttention therefore shows a reproducible improvement over Vanilla at
TRM-8 under the reduced-compute regime. Superiority over the Recency baseline
is not statistically resolved.

### Depth interaction

For the change in the Attention-vs-Vanilla effect from depth 4 to depth 8:

- interaction = +0.581 percentage points
- 95% CI = [-0.628, +1.756] pp

The interaction interval includes zero.

Therefore the current data do NOT establish a statistically resolved,
monotonic increase in HistoryAttention benefit with recursive depth.

### Exploratory depth controls

TRM-2:
- Attention - Vanilla mean = approximately +0.043 pp
- Recency showed a larger positive descriptive effect
- selective retrieval has very little history to choose between at this depth

TRM-16:
- Attention - Vanilla mean = approximately +0.058 pp across three seeds
- the TRM-8 advantage did not persist
- learning-curve inspection did not provide strong evidence that simply
  extending the 40-epoch budget would resolve the result

These depth-2 and depth-16 results are exploratory and should not be presented
as five-seed confirmatory findings.

### Learning-curve audit

Across TRM-16 runs, best validation checkpoints frequently occurred before
the final step for Vanilla, Recency, and HistoryAttention.

The observed pattern does not support the simple explanation that the
TRM-16 result failed solely because training ended too early.

A longer-budget TRM-16 campaign is therefore not currently justified.

### Exact-accuracy limitation

Exact puzzle accuracy remained zero throughout the reduced CPU experiments.

Accordingly, CPU-phase claims are limited to:
- token-level accuracy
- language-model loss
- optimization behavior
- comparative latent-history mechanism effects

The CPU experiments do NOT establish improved exact Sudoku solving.

### Scale limitation

The official TRM architecture is substantially larger than the CPU
foundation.

Reduced CPU foundation:
- hidden_size = 64
- H_cycles = 1
- L_cycles = 1
- L_layers = 1
- puzzle embeddings disabled

Official TRM configuration:
- hidden_size = 512
- H_cycles = 3
- L_cycles = 6
- L_layers = 2
- num_heads = 8
- puzzle_emb_len = 16
- puzzle_emb_ndim = hidden_size

A capacity-bridge CPU smoke run using the wider TRM architecture reached
approximately 46.7 seconds per optimization step (75/250 steps required
approximately 59 minutes).

Large-model experimentation on the laptop CPU was therefore judged
computationally impractical and stopped.

### Paper-ready interpretation

In a controlled reduced-compute TRM setting, selective retrieval over
previous recursive latent states produced a reproducible improvement over
vanilla recurrence at an intermediate recursion depth. At TRM-8,
HistoryAttention improved mean token accuracy by +0.730 percentage points
and the paired bootstrap confidence interval excluded zero. The benefit was
not observed consistently at shallower or deeper recursion depths, and the
depth interaction itself was not statistically resolved. We therefore
interpret the CPU experiments as evidence that latent-history retrieval can
be useful in a specific recursive regime rather than as evidence of a
universal or monotonically depth-dependent advantage.

Because these experiments used a deliberately reduced architecture and
exact puzzle accuracy remained zero, scale transfer is evaluated separately
under a larger GPU-trained TRM configuration.

### Next phase

The next experimental phase is GPU scale-transfer validation.

Initial GPU comparison:
1. Vanilla TRM
2. RecencyWeightedHistory
3. HistoryAttention

The first GPU runs will use matched conditions and a single screening seed.
Additional seeds will only be launched after runtime, memory, and initial
signal are verified.
