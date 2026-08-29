# Pre-freeze feasibility

Profiles must run on **Colab T4** or **GTX 1080 Ti** (not the laptop).

Canonical backbone is now in-repo: see `docs/CANONICAL_PROTOCOL_v1.md` and
`--preset canonical` (D256, H3/L6/L2, ACT6, r64, pre-QKV P1, Gated, exact B3).

Still open before freeze: measure ACT6 vs ACT16 cost, confirm effective batch 32,
confirm dtype on the chosen GPU, freeze update budget `N` from Vanilla-only
dev curves.

```bash
python experiments/run_study.py single --preset canonical --variant B0 --seed 0 --dry-run
# 1080 Ti: default float32
# T4: add --override arch.forward_dtype=bfloat16 after verifying runtime dtype
```
