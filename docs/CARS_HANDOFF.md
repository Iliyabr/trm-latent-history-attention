# CARS Handoff — Training-Free Inference Post-Processing

This document tracks implementation of **CARS (Constraint-Aware Recursive Selection)** per `Handoff.txt`.

## Status

| Item | Status |
|------|--------|
| Selection logic (`experiments/cars_postprocess.py`) | **Implemented** |
| Inference runner (`scripts/eval_cars_postprocess.py`) | **Implemented** |
| Unit tests (`tests/test_cars_postprocess.py`) | **Implemented** |
| Seeds 0–2 evaluation | **NOT RUN locally** (no checkpoints / dataset in clone) |
| Final report + JSON | **Scaffold** → run script on server |
| Figures | Generated when status = COMPLETE |

**CARS_ACT6_STATUS: INCOMPLETE** until all three frozen B0 ACT6 checkpoints are evaluated on the server.

---

## Implementation audit (inference path)

### Existing evaluator (`experiments/evaluate_study.py`)

- One ACT6 forward pass per batch (`halt_max_steps=6`).
- With `cycle_logits=True`, stores **18 H-cycle micro-steps** per puzzle in `trajectory.exact` / `trajectory.cell_accuracy`.
- Stores **final-step** `predictions` only — **not** full grids at each ACT step.
- **CARS cannot run from existing `examples.jsonl` alone** (needs per-ACT-step grids + logits for confidence).

### CARS runner (`scripts/eval_cars_postprocess.py`)

- `model.eval()` + `torch.inference_mode()` only — **no training, no backward**.
- Sets `cycle_logits=False`; captures **`preds` + `logits` at each of 6 ACT steps**.
- Applies frozen selectors: Final, Confidence, CARS, Oracle (diagnostic).
- Writes `results/inference-postprocess/cars-act6/seed_{N}/cars_puzzles.jsonl`.
- Aggregates to `docs/data/CARS_ACT6_FINAL_v1.json` and `docs/CARS_ACT6_FINAL_REPORT.md`.
- Figures → `docs/figures/cars_act6_*.png`.

**Additional forward passes: 0** (same single trajectory; extra CPU for selection only).

---

## Frozen selection rules

### Confidence Selection

`Confidence(t) = mean_j max_c p_t(j,c)` → argmax over ACT steps; tie → **later** step.

### CARS (lexicographic, frozen)

1. Minimize `clue_mismatch_count` (given cells only; input token > 1)
2. Minimize `structural_violations` (row + column + box duplicate excess)
3. Maximize mean token confidence
4. Tie → **later** ACT step

No ground truth in CARS or Confidence selection.

### Oracle (diagnostic only)

Best exact, then best cell, then later step. **Uses labels — not deployable.**

---

## Checkpoints (frozen Vanilla ACT6)

| Seed | Expected checkpoint | Eval metadata fallback |
|------|---------------------|------------------------|
| 0 | `outputs/study/canonical/B0-seed0/step_28800.pt` | `results/canonical-gpu/B0/seed_0/metadata.json` |
| 1 | `outputs/study-4090/canonical/B0-seed1/step_28800.pt` | `results/canonical-gpu-4090/B0/seed_1/metadata.json` |
| 2 | `outputs/study-4090/canonical/B0-seed2/step_28800.pt` | `results/canonical-gpu-4090/B0/seed_2/metadata.json` |

Verify SHA256 before/after run (recorded in output metadata).

---

## Server run commands

From repo root with venv + dataset + checkpoints present:

```bash
# Full run (all three seeds)
python scripts/eval_cars_postprocess.py \
  --seeds 0 1 2 \
  --data data/sudoku-study-v1 \
  --config config/experiment/sudoku_study_canonical.yaml

# Explicit checkpoint overrides (if paths differ)
python scripts/eval_cars_postprocess.py \
  --checkpoint 0=outputs/study/canonical/B0-seed0/step_28800.pt \
  --checkpoint 1=outputs/study-4090/canonical/B0-seed1/step_28800.pt \
  --checkpoint 2=outputs/study-4090/canonical/B0-seed2/step_28800.pt

# Re-analyze cached trajectories only (no GPU)
python scripts/eval_cars_postprocess.py --analyze-only
```

**Do not** modify `results/canonical-gpu/` or `results/canonical-gpu-4090/`.

---

## Optional ACT16 seed-0

Only after ACT6 seeds 0–2 complete. Re-run same script with `--seeds 0 --checkpoint 0=.../B0-seed0/best_dev.pt` and ACT16 halt_max_steps checkpoint — keep separate output dir (not mixed into ACT6 aggregate).

---

## Output files (required by handoff)

| File | Purpose |
|------|---------|
| `scripts/eval_cars_postprocess.py` | Inference + analysis entry point |
| `experiments/cars_postprocess.py` | Frozen selectors + statistics |
| `docs/data/CARS_ACT6_FINAL_v1.json` | Machine-readable results |
| `docs/CARS_ACT6_FINAL_REPORT.md` | Paper-facing report |
| `docs/figures/cars_act6_main_results.png` | Figure A |
| `docs/figures/cars_selected_step_distribution.png` | Figure B |
| `docs/figures/cars_recoverable_headroom.png` | Figure C (optional) |
| `results/inference-postprocess/cars-act6/` | Raw per-puzzle trajectories (gitignored-friendly) |

---

## Validation checklist (before COMPLETE)

- [ ] No optimizer / backward / training executed
- [ ] Checkpoint SHA256 unchanged after eval
- [ ] CARS + Confidence rules identical across seeds
- [ ] No ground truth in CARS/Confidence selection
- [ ] Oracle labeled diagnostic-only
- [ ] Existing canonical result dirs untouched
- [ ] All three seeds evaluated on same 1000-puzzle test split
- [ ] P1 seed sensitivity preserved in separate 4090 campaign (not part of CARS Vanilla run)
