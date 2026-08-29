# Canonical protocol v1 — what is set in the repo

Implements **TRM_HISTORY_CANONICAL_PROTOCOL_v1** settings in code/config.
Training budget `N` is still chosen after Vanilla-only feasibility (see
`docs/PRE_FREEZE_FEASIBILITY.md`).

## Backbone (`config/arch/trm_history_canonical.yaml`)

| Item | Value |
|------|--------|
| D | 256 |
| H / L / L_layers | 3 / 6 / 2 |
| Spatial heads | 4 |
| ACT | 6 |
| Rank / temporal heads | 64 / 4 |
| Gate init | −2 |
| History window | 0 (full causal within H-cycle) |
| Default dtype | float32 (override `bfloat16` on T4 if verified) |

## Training (`config/experiment/sudoku_study_canonical.yaml`)

| Item | Value |
|------|--------|
| Dataset | `data/sudoku-study-v1` |
| Effective batch | 32 |
| Optimizer | AdamW, lr 1e-4, wd 0.1, β=(0.9, 0.95) |
| EMA | 0.999 |
| compile | false |
| Runtime cap | none |
| Epochs | 1024 placeholder until `N` is frozen from Vanilla profiling |

## Models

| Runner flag | Mechanism |
|-------------|-----------|
| `B0` | Vanilla |
| `Gated` | Canonical gated uniform history |
| `P1` | Pre-QKV low-rank HistoryAttention |
| `B3` | Exact parameter-matched no-history (`4·D·r+1`) |

## Launch

```bash
python experiments/run_study.py single --preset canonical --variant B0 --seed 0
python experiments/run_study.py suite --preset canonical --dry-run
```

1080 Ti: keep float32. T4: add `--override arch.forward_dtype=bfloat16` only after
runtime confirmation.

Old `colab` / `colab_heavy` H2/L4 + post-projection P1 runs stay **pre-alignment
screening**; do not mix them with this preset.
