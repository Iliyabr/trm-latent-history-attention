# CPU Latent-History Study - Final Research Summary

## Status

CPU experimentation is closed.

The CPU experiments serve as controlled mechanism validation and not as a
reproduction of the full-scale TRM Sudoku result.

## Experimental roles

Confirmatory:
- TRM-4: 5 seeds
- TRM-8: 5 seeds
- Paired puzzle-level evaluation
- Two-way bootstrap over seeds and puzzles

Exploratory:
- TRM-2: 3 seeds
- TRM-16: 3 seeds

## Architecture

Reduced CPU TRM:
- hidden_size: 64
- num_heads: 4
- H_cycles: 1
- L_cycles: 1
- L_layers: 1
- puzzle embeddings: disabled
- global batch size: 4
- learning rate: 1e-3
- epochs: 40
- CPU threads: 8

HistoryAttention:
- projection-free
- token-aligned attention across strictly previous outer-step states
- scalar learned fusion gate
- one additional trainable parameter
- historical states detached
- causal history only
- reset-isolated between puzzles

## Final accuracy summary

| Depth | Seeds | Vanilla | Recency | HistoryAttention | Attention - Vanilla |
|---|---:|---:|---:|---:|---:|
| 2 | 3 | 48.268 +/- 0.119% | 48.996 +/- 0.586% | 48.311 +/- 0.603% | +0.043 pp |
| 4 | 5 | 46.757 +/- 1.337% | 47.095 +/- 0.903% | 46.905 +/- 0.574% | +0.148 pp |
| 8 | 5 | 42.659 +/- 0.203% | 42.933 +/- 0.513% | 43.389 +/- 0.145% | +0.730 pp |
| 16 | 3 | 42.113 +/- 0.131% | 42.329 +/- 0.566% | 42.171 +/- 0.147% | +0.058 pp |

## Key paired statistical results

TRM-4:
- Attention vs Vanilla:
  - delta: +0.148 pp
  - 95% CI: [-1.043, +1.317] pp
- Attention vs Recency:
  - delta: -0.190 pp
  - 95% CI: [-0.869, +0.458] pp

TRM-8:
- Attention vs Vanilla:
  - delta: +0.730 pp
  - 95% CI: [+0.367, +1.110] pp
  - Attention better in 5/5 seeds
- Attention vs Recency:
  - delta: +0.456 pp
  - 95% CI: [-0.122, +0.998] pp

Depth interaction:
- (Attention - Vanilla) depth8 minus depth4:
  - delta: +0.581 pp
  - 95% CI: [-0.628, +1.756] pp
- Therefore a monotonic depth-dependent advantage is NOT established.

## Interpretation

The CPU experiments establish that explicit latent-history access is not
uniformly beneficial.

HistoryAttention shows its strongest and most reproducible advantage at
TRM-8, where it improves token accuracy over vanilla in all five seeds and
the paired bootstrap confidence interval is entirely above zero.

The effect is weak at TRM-2 and TRM-4 and disappears again in the exploratory
TRM-16 screening. The evidence therefore supports an intermediate-depth
operating regime rather than a universal or monotonically depth-increasing
benefit.

Recency remains an important control. At shallow recursion depths, a simple
temporal prior can match or outperform learned selective retrieval.

## Statistical limitations

- Only TRM-4 and TRM-8 have five-seed confirmation.
- TRM-2 and TRM-16 are exploratory three-seed controls.
- 200 development puzzles were used.
- Puzzle pairing was verified exactly across runs.
- Bootstrap resampled both training seeds and puzzles while preserving pairing.
- Five training seeds remain a small sample for between-run uncertainty.
- Failure of a confidence interval to exclude zero is not evidence of equivalence.

## Exact-accuracy limitation

Exact puzzle accuracy was 0 across the reduced-compute experiments.

Therefore CPU conclusions are limited to:
- token-level accuracy
- language-model loss
- optimization behavior
- comparative history-mechanism effects

They do not establish improved exact Sudoku solving.

## Scale-transfer motivation

The official TRM architecture differs substantially from the CPU foundation:

CPU foundation:
- hidden_size = 64
- H_cycles = 1
- L_cycles = 1
- L_layers = 1
- puzzle embedding disabled

Official TRM architecture:
- hidden_size = 512
- H_cycles = 3
- L_cycles = 6
- L_layers = 2
- num_heads = 8
- puzzle embedding available

A CPU smoke test of the larger architecture reached approximately 46.7
seconds per optimization step, making meaningful large-scale training
impractical on the laptop CPU.

The next phase is therefore a GPU scale-transfer study.

## Paper-ready conclusion

In a controlled reduced-compute TRM setting, selective retrieval over
previous recursive latent states produced a reproducible improvement over
vanilla recurrence at an intermediate recursion depth, but the benefit did
not generalize monotonically across depths. These findings motivate a
separate scale-transfer evaluation under a larger, GPU-trained TRM regime.
