# GPU Phase Handoff - TRM Latent-History Project

## Current status

The reduced-compute CPU phase is complete.

Do not redesign or rerun the CPU campaign unless a later reviewer-driven
analysis specifically requires it.

## Research question

Does explicit selective retrieval over previous recursive latent states
improve TRM reasoning, and does the effect transfer from the reduced CPU
regime to a larger GPU-trained architecture?

## Proposed mechanism

HistoryAttention:
- projection-free
- token-aligned
- attention over strictly previous outer-step z_H states
- detached history
- masked variable-length history
- exact zero-history identity
- one learned scalar fusion gate
- exactly +1 trainable parameter

## Key CPU finding

At TRM-8, across five matched seeds:

Vanilla:
- mean token accuracy = 42.659%
- SD = 0.203%

Recency:
- mean token accuracy = 42.933%
- SD = 0.513%

HistoryAttention:
- mean token accuracy = 43.389%
- SD = 0.145%

Attention vs Vanilla:
- delta = +0.730 pp
- 95% paired two-way bootstrap CI = [+0.367, +1.110] pp
- positive in 5/5 seeds

Attention vs Recency:
- delta = +0.456 pp
- 95% CI = [-0.122, +0.998] pp

The latter comparison is not statistically resolved.

## Cross-depth CPU result

Approximate Attention-minus-Vanilla mean token-accuracy differences:

- depth 2:  +0.043 pp (3 seeds, exploratory)
- depth 4:  +0.148 pp (5 seeds)
- depth 8:  +0.730 pp (5 seeds)
- depth 16: +0.058 pp (3 seeds, exploratory)

The effect is not monotonic with depth.

Depth-4 to depth-8 interaction:
- +0.581 pp
- 95% CI = [-0.628, +1.756] pp

Therefore do not claim statistically established depth-dependent scaling.

## CPU limitation

Exact Sudoku accuracy remained zero in the reduced-compute regime.

CPU claims therefore concern token accuracy, loss, optimization, and
mechanism comparison, not exact puzzle solving.

## Official TRM Sudoku reference from repository

README Sudoku-Extreme attention configuration:

- arch = trm
- hidden_size = 512
- num_heads = 8
- H_cycles = 3
- L_cycles = 6
- L_layers = 2
- halt_max_steps = 16 from arch config
- puzzle_emb_len = 16
- puzzle_emb_ndim = hidden_size
- learning rate = 1e-4
- puzzle embedding learning rate = 1e-4 in Sudoku command
- weight_decay = 1.0
- puzzle_emb_weight_decay = 1.0
- epochs = 50000
- eval_interval = 5000
- EMA enabled
- dataset = sudoku-extreme-1k-aug-1000

Repository README reports approximately 75% exact accuracy for the
attention-based Sudoku configuration and runtime below 20 hours on one
L40S GPU.

These numbers are repository reference values, not results reproduced by
this project.

## CPU-to-full-scale difference

CPU foundation:
- hidden_size = 64
- num_heads = 4
- H_cycles = 1
- L_cycles = 1
- L_layers = 1
- puzzle embeddings disabled
- global batch size = 4
- lr = 1e-3

Official architecture:
- hidden_size = 512
- num_heads = 8
- H_cycles = 3
- L_cycles = 6
- L_layers = 2
- puzzle embeddings available
- forward_dtype = bfloat16

A larger-architecture CPU smoke run was stopped because it required roughly
46.7 seconds per optimization step.

## GPU target

Initial hardware:
- Google Colab
- NVIDIA T4

Important:
The repository's published timing target assumes an L40S, not a T4.
Therefore the official 50,000-epoch experiment must NOT be launched blindly.

## GPU phase sequence

### G0 - Environment audit
Verify:
- GPU model
- CUDA availability
- PyTorch CUDA build
- available GPU memory
- repository revision
- dataset availability
- dependency compatibility

### G1 - Full-width Vanilla smoke
Run a very short full-width TRM smoke test.

Measure:
- GPU memory
- seconds/step
- numerical stability
- dtype compatibility
- checkpoint/evaluation functionality

### G2 - History compatibility smoke
Run matched:
- Vanilla
- Recency
- HistoryAttention

Use identical architecture/training settings.

Goal:
verify that the history interface works correctly under the full-width GPU
architecture before any expensive experiment.

### G3 - Scale-transfer pilot
Run one matched screening seed for:
- Vanilla
- Recency
- HistoryAttention

Use a budget selected only after G1 runtime measurement.

Primary question:
Does the positive HistoryAttention signal observed in the reduced CPU regime
survive at larger scale?

### G4 - Multi-seed confirmation
Only if the pilot gives a useful signal:
- add seeds 1 and 2
- use paired evaluation
- preserve prediction exports when practical

## Scientific guardrails

Do not:
- claim CPU results reproduce the original TRM paper
- claim monotonic improvement with depth
- claim HistoryAttention is superior to Recency based on current CPU results
- compare unmatched training budgets
- change multiple architectural factors during a controlled comparison
- introduce Q/K/V projections or multi-head history attention before the
  scale-transfer result is known

## Key project paths

CPU final summary:
docs/CPU_HISTORY_FINAL_SUMMARY.md

Living paper evidence:
docs/PAPER_EVIDENCE_LOG.md

HistoryAttention implementation:
models/history/attention.py

History factory:
models/history/factory.py

Attention tests:
tests/test_history_attention.py

TRM-4 statistics:
results/history-multiseed/trm4-multiseed-statistics.json

TRM-4 vs TRM-8 interaction:
results/history-depth/trm4-vs-trm8-depth-interaction.json

## Git reference

Validated HistoryAttention implementation was frozen in commit:

55b60c6 Add validated latent-history attention

Raw multi-seed prediction/checkpoint directories should not be committed
wholesale.

## Immediate next action

Start a clean Google Colab/T4 session and perform G0 environment audit
before copying or running any large training command.
