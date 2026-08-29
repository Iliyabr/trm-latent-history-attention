# TRM History Deployment Manifest v1

**Protocol:** `TRM_HISTORY_CANONICAL_PROTOCOL_v1.md`
**Status:** PENDING PROFILE â†’ LOCKED before canonical GPU long runs

Fill only from measured GPU runtime evidence. Accuracy must not be used to choose deployment values.

| Field | Final value |
|---|---|
| GPU | PENDING |
| VRAM | PENDING |
| Compute capability | PENDING |
| PyTorch | PENDING |
| CUDA | PENDING |
| Actual dtype | PENDING |
| ACT | PENDING (candidate: 6 or 16) |
| Physical batch | PENDING |
| Gradient accumulation | PENDING |
| Effective batch | 32 target |
| Fixed optimization steps N | PENDING |
| Vanilla peak allocated VRAM | PENDING |
| Vanilla peak reserved VRAM | PENDING |
| Vanilla sec/step | PENDING |
| Attention peak allocated VRAM | PENDING |
| Attention peak reserved VRAM | PENDING |
| Attention sec/step | PENDING |
| Canonical code commit | PENDING |
| Dataset revision/hash | PENDING |
| Profiling date | PENDING |
| Status | PENDING |

## Required pre-lock checks

- P1 aligned to pre-QKV RMSNorm.
- Vanilla ACT6 profile completed.
- Vanilla ACT16 profile completed.
- Canonical P1 smoke completed.
- Common effective batch verified for Vanilla and P1.
- Actual GPU/dtype verified at runtime.
- Fixed update budget selected from Vanilla dev behavior + compute budget only.

Once locked, deployment values must remain unchanged across the canonical GPU matched campaign.
