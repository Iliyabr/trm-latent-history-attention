# Latent History Module Interface

This directory defines the shared interface for within-H-cycle `z_L` history readers.

## Scientific purpose

History is reset at the start of every outer H cycle. During that cycle the
module stores the initial `z_L` and every subsequently produced inner state:

```
H = [z_0, z_1, ..., z_{k-1}]
```

The current state queries this history *before* the shared TRM update. After
the answer/`z_H` update, only the final `y`/`z` pair is carried to the next
outer or ACT step. History never crosses an H-cycle or ACT boundary.

## Tensor contract

Every history aggregator receives:

- `current_z`: `[B, L, D]`
- `history_z`: `[B, K, L, D]`
- optional `history_lengths`: `[B]`

and returns a tensor with the same shape and dtype as `current_z`.

## Experiment modes

| ID | Config name | Behavior |
|---|---|---|
| B0 | `none` | Identity / vanilla TRM |
| B1 | `residual` | RMSNorm residual around the backbone update |
| B2 | `uniform` | Uniform mean of the within-cycle history |
| P1 | `attention` | Low-rank multi-head temporal attention |
| B3 | `parameter_matched` | Identity reader; extra FFN width lives in the backbone |
| optional | `static_lag` | Latest-state history ablation |

Diagnostics, state deletion, and latent corruption are evaluation-only. They
are requested through `analysis_request` and are not stored on the module
during training.
