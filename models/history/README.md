# Latent History Module Interface

Within-H-cycle `z_L` history readers for **TRM_HISTORY_CANONICAL_PROTOCOL_v1**.

## Semantics

History resets at every outer H-cycle. Order: **read → TRM update → append**.
History never crosses an H-cycle or ACT boundary.

## Canonical family

| ID | Config name | Behavior |
|----|-------------|----------|
| B0 | `none` | Vanilla TRM (identity) |
| Gated | `gated` | Mean of RMSNorm'd history; `z + σ(g)·context` then RMSNorm; gate init −2 |
| P1 | `attention` | Low-rank temporal attention; **pre-QKV** `W RMSNorm_D(·)`; readout `RMSNorm(z + σ(g)·memory)`; gate init −2 |
| P1ns | `attention_no_skip` | Same attention as P1, but readout is `RMSNorm(memory)` only (no residual onto current `z`) |
| B3 | `parameter_matched` | No history; `D→2r→D` gated side path; **exactly** `4·D·r+1` params |

## Legacy (screening only; not protocol controls)

| ID | Config name | Behavior |
|----|-------------|----------|
| B1 | `residual` | RMSNorm residual around backbone update |
| B2 | `uniform` | Ungated `RMSNorm(z + mean(history))` |

## Tensor contract

- `current_z`: `[B, L, D]`
- `history_z`: `[B, K, L, D]`
- optional `history_lengths`: `[B]`
- returns same shape/dtype as `current_z`
