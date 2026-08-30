# CARS ACT6 Training-Free Inference Report

## 1. Protocol and checkpoint audit

Status: **INCOMPLETE**

Frozen Vanilla TRM (B0), ACT6 (`halt_max_steps=6`), test split `sudoku-study-v1`.  
No training, no backward(), no checkpoint modification.

### Checkpoint provenance

| Seed | Checkpoint | SHA256 | Exists |
|------|------------|--------|--------|
| 0 | `outputs/study/canonical/B0-seed0/step_28800.pt` | MISSING | false |
| 1 | `outputs/study-4090/canonical/B0-seed1/step_28800.pt` | MISSING | false |
| 2 | `outputs/study-4090/canonical/B0-seed2/step_28800.pt` | MISSING | false |

Seeds 1–2 correspond to the RTX 4090 bfloat16 campaign; seed 0 to 1080 Ti float32 screening. Do not pool wall-clock across hardware.

## 2. Implementation audit

- **Inference path:** `scripts/eval_cars_postprocess.py` → `run_cars_inference()`
- One forward pass per batch; **6 ACT-step** predictions and logits captured (`cycle_logits=False`).
- Existing `results/canonical-gpu/B0/seed_0/examples.jsonl` stores 18 H-cycle micro-step metrics but **not** per-ACT-step full grids — insufficient for CARS constraint scoring.
- **Selection logic:** `experiments/cars_postprocess.py` (unit-tested in `tests/test_cars_postprocess.py`).
- **Additional forward passes:** 0 (post-processing only over captured trajectory).

See also `docs/CARS_HANDOFF.md`.

## 3. Frozen selection rule

### Confidence Selection

Mean token max-softmax confidence per ACT step; argmax; tie → later step. No Sudoku constraints, no ground truth.

### CARS

Lexicographic: (1) minimize clue mismatches on givens, (2) minimize structural duplicate excess, (3) maximize confidence, (4) later step. Frozen pre-evaluation.

### Oracle

**ORACLE DIAGNOSTIC - NOT AN INFERENCE METHOD** — uses labels to estimate trajectory headroom.

## 4. Per-seed final results

**MISSING** — run on server:

```bash
python scripts/eval_cars_postprocess.py --seeds 0 1 2 --data data/sudoku-study-v1
```

## 5. Three-seed aggregate

MISSING

## 6. Paired exact-solve transitions

MISSING (McNemar per seed after run)

## 7. Paired cell-accuracy analysis

MISSING (bootstrap CI per seed after run)

## 8. Recursive failure-mode analysis

MISSING — key metrics: earlier-exact-lost-by-final, CARS recovery rate, CARS damage rate

## 9. Oracle trajectory headroom

MISSING

## 10. Selected-step distribution

MISSING — see Figure B after run

## 11. Inference-time overhead

- New trainable parameters: **0**
- Retraining: **no**
- Additional model forward passes: **0**
- Post-processing: measured per seed in output metadata (`postprocess_seconds`)

## 12. Optional ACT16 seed-0 sensitivity

NOT RUN

## 13. Paper-ready table

MISSING — populated in `docs/data/CARS_ACT6_FINAL_v1.json` after evaluation

## 14. Paper-ready figures

Pending COMPLETE status:

- `docs/figures/cars_act6_main_results.png`
- `docs/figures/cars_selected_step_distribution.png`
- `docs/figures/cars_recoverable_headroom.png`

## 15. Scientific interpretation

Pending results. Do not overclaim from n=3 training seeds.

## 16. Proposed paper revision

Draft after observed CARS vs Final outcomes on seeds 0–2.

## 17. Missing or unresolved evidence

- All three frozen B0 checkpoints not evaluated in this clone
- Dataset `data/sudoku-study-v1` not present locally
- Figures not generated
- ACT16 optional branch not run
- Scientific interpretation and paper framing pending numbers

**CARS_ACT6_STATUS: INCOMPLETE**
