# CPU L-cycle Study — Statistical Analysis v1

## Data integrity

- Rows: 160
- Design: 4 models × 5 seeds × 8 evaluation checkpoints = 160 rows
- Evaluation steps: 1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000
- Exact-grid accuracy: 0 for every stored evaluation row

## Final paired comparisons at step 10,000

| Comparison | Mean delta (pp) | SD of paired delta | 95% t CI (pp) | two-sided p | Positive seeds |
|---|---:|---:|---:|---:|---:|
| Gated − Vanilla | -0.120 | 1.015 | [-1.380, +1.141] | 0.805 | 2/5 |
| HistoryAttention − Vanilla | -0.407 | 1.643 | [-2.447, +1.632] | 0.609 | 2/5 |
| Parameter-Matched − Vanilla | -0.438 | 1.225 | [-1.959, +1.083] | 0.469 | 1/5 |

None of the final paired intervals excludes zero. With only five seeds, the intervals are wide; this does not establish statistical equivalence.

## Attention vs Vanilla over training

| Step | Mean delta (pp) | 95% t CI (pp) | raw p | Holm-adjusted p across 8 checkpoints | Positive seeds |
|---:|---:|---:|---:|---:|---:|
| 1250 | +0.688 | [+0.347, +1.028] | 0.005 | 0.040 | 5/5 |
| 2500 | +3.099 | [+0.200, +5.997] | 0.041 | 0.288 | 5/5 |
| 3750 | +0.733 | [-3.049, +4.516] | 0.619 | 1.000 | 3/5 |
| 5000 | +0.986 | [-1.132, +3.104] | 0.266 | 1.000 | 2/5 |
| 6250 | +0.080 | [-0.692, +0.853] | 0.787 | 1.000 | 3/5 |
| 7500 | -0.001 | [-1.273, +1.270] | 0.998 | 1.000 | 3/5 |
| 8750 | -0.212 | [-1.934, +1.509] | 0.749 | 1.000 | 3/5 |
| 10000 | -0.407 | [-2.447, +1.632] | 0.609 | 1.000 | 2/5 |

The step-1,250 paired Attention advantage is the strongest checkpoint-level result: +0.688 pp, 95% CI [+0.347, +1.028], raw p=0.005, Holm-adjusted p=0.040.

The step-2,500 advantage is also positive in 5/5 seeds, but does not survive Holm correction across the eight checkpoint comparisons.

## Exploratory trajectory summaries

- Mean Attention−Vanilla delta averaged over steps 1,250 and 2,500: +1.893 pp, 95% CI [+0.537, +3.249], p=0.018.
- Change from the early two-checkpoint average to the late two-checkpoint average (8,750 and 10,000): -2.203 pp, 95% CI [-4.013, -0.394], p=0.028.
- Observed-range AULC (steps 1,250–10,000), Attention−Vanilla: +0.689 pp, 95% CI [-0.991, +2.369], p=0.318; positive in 4/5 seeds.

These trajectory summaries were selected after inspecting the learning curves and should be described as exploratory/post-hoc rather than confirmatory.

## Recommended paper interpretation

The reduced CPU experiment does not show a final-accuracy improvement from HistoryAttention. However, the paired learning curves show a reproducible early optimization advantage, especially at step 1,250, which then decays and reverses slightly by the final checkpoint. The parameter-matched and gated controls indicate that some early improvement is not unique to selective history retrieval. This makes the CPU study useful as a mechanistic/optimization result rather than as evidence of improved final Sudoku solving.

## Suggested figure usage

1. Main CPU figure: `cpu_paired_delta_vs_vanilla.png` — strongest compact evidence for the early-gain / late-decay story.
2. Secondary figure or appendix: `cpu_learning_curves_accuracy.png` — shows raw accuracy trajectories and seed variability.
3. Secondary/diagnostic figure: `cpu_final_seed_accuracy.png` — makes the final seed sensitivity visible.