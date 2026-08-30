# Markdown Audit Decisions — 2026-08-30

No non-empty file should be deleted during finalization. Historical files are
preserved by moving/copying them into an archive before rewriting canonical
entry points.

| Current file | Status | Final action |
|---|---|---|
| `README.md` | **STALE** | Preserve original under `docs/archive/upstream/`, then replace with project-specific final README |
| `CPU_LOCAL.md` | KEEP / SMALL UPDATE | Keep root link; update scope text so it no longer says scientific CPU pilots are still future work |
| `docs/CPU_HISTORY_FINAL_SUMMARY.md` | AUTHORITATIVE TRACK A | Keep; merge its paper-facing conclusions into `docs/CPU_STUDY_FINAL.md` |
| `docs/CPU_LCYCLE_FINAL_EVIDENCE_v1.md` | AUTHORITATIVE TRACK B | Keep unchanged as frozen evidence |
| `docs/CPU_LCYCLE_STATISTICAL_ANALYSIS_v1.md` | KEEP / ENCODING FIX | Preserve numbers; repair mojibake/encoding and link from final CPU summary |
| `docs/GPU_PHASE_HANDOFF.md` | **STALE** | Archive; it describes the old projection-free outer-history GPU plan, not the canonical Track-B campaign |
| `docs/PAPER_EVIDENCE_LOG.md` | HISTORICAL RESEARCH LOG | Preserve/archive as a development log; do not use as final source of truth |
| `docs/TRM_HISTORY_CANONICAL_PROTOCOL_v1.md` | FROZEN SCIENTIFIC INTENT | Keep. Do not rewrite scientific design. Its old “remaining CPU work” section is historical execution state and is superseded by final evidence |
| `docs/TRM_HISTORY_DEPLOYMENT_MANIFEST_v1.md` | ACTIVE | Keep and fill/lock only from actual GPU profiling |
| `experiments/baseline/PHASE1_RESULTS.md` | HISTORICAL SUPPORT | Keep under experiments or archive after final paper; not a primary final result |
| `experiments/baseline/README.md` | LEGACY PHASE-1 GUIDE | Add a legacy notice if retained |
| `models/history/README.md` | **STALE / INCOMPLETE** | Rewrite to distinguish Track A (`z_H`) from canonical Track B (`z_L`) |
| `docs/archive/phase1/*` | ARCHIVE | Already correctly archived; leave unchanged |

## Main inconsistencies found

1. Root `README.md` still says HistoryAttention, tests, and real experiments have
   not been completed. This is false in the current project state.
2. `GPU_PHASE_HANDOFF.md` describes projection-free outer-step HistoryAttention,
   Recency, T4 screening, and single-seed promotion logic. It conflicts with the
   frozen canonical D256/H3/L6 four-model GPU campaign.
3. `models/history/README.md` documents only the outer `z_H` interface and could
   mislead a reader into thinking it is the primary method.
4. `PAPER_EVIDENCE_LOG.md` contains valuable provenance but mixes preliminary
   seed-0 observations, old Track-A logic, noncanonical pilots, and final Track-B
   results. It should not remain the canonical paper source.
5. `TRM_HISTORY_CANONICAL_PROTOCOL_v1.md` contains an old execution-status
   section saying some CPU runs remain. The scientific protocol itself should
   remain frozen; completion status should be documented separately.
6. `CPU_LCYCLE_STATISTICAL_ANALYSIS_v1.md` contains visible mojibake such as
   `â€”`, `Ã—`, and `âˆ’`. The statistical content is useful, but encoding should
   be repaired before final submission.

## Recommended final authoritative documents

```text
README.md
RUNBOOK.md

docs/FINAL_PROJECT_STATUS.md
docs/CPU_STUDY_FINAL.md
docs/CPU_LCYCLE_FINAL_EVIDENCE_v1.md
docs/CPU_LCYCLE_STATISTICAL_ANALYSIS_v1.md
docs/TRM_HISTORY_CANONICAL_PROTOCOL_v1.md
docs/TRM_HISTORY_DEPLOYMENT_MANIFEST_v1.md

docs/data/CPU_LCYCLE_ALL_METRICS_v1.csv
docs/figures/*
docs/archive/*
```

When the canonical GPU campaign finishes, add:

```text
docs/GPU_STUDY_FINAL.md
```

and update `FINAL_PROJECT_STATUS.md`, `README.md`, and the paper.
