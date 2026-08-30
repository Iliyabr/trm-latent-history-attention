# CARS ACT6 Training-Free Inference Report

## 1. Protocol and checkpoint audit

Status: **COMPLETE**

Frozen Vanilla TRM (B0), ACT6 (`halt_max_steps=6`), test split `sudoku-study-v1`.
No training, no backward(), no checkpoint modification.

### Checkpoint provenance

| Seed | Checkpoint | SHA256 | Exists |
|------|------------|--------|--------|
| 0 | `/home/mahyar/trm-latent-history-attention/outputs/study/canonical/B0-seed0/step_28800.pt` | de10685c2f21d38450c6e07b7498c420997cac7607a75b346f5b99cb8e68096b | True |
| 1 | `/home/mahyar/trm-latent-history-attention/outputs/study-4090/canonical/B0-seed1/step_28800.pt` | f50e0a078ddc82d9da4deaa518e1148057f185ebe8d9e38bef306daff96042d9 | True |
| 2 | `/home/mahyar/trm-latent-history-attention/outputs/study-4090/canonical/B0-seed2/step_28800.pt` | 6a6cf25f86f02c8d33cc157ee808e0fb171c5d5ce1a73032f4c7beee28cf037b | True |

## 2. Implementation audit

- Inference path: `scripts/eval_cars_postprocess.py` → `run_cars_inference()`
- One forward pass per puzzle batch; **6 ACT-step** predictions/logits captured
  (`cycle_logits=False`; final preds at each ACT step, not 18 H-cycle micro-steps).
- Existing `examples.jsonl` stores 18 micro-step metrics but **not** per-step full grids;
  CARS requires this dedicated capture pass.

## 3. Frozen selection rule

See `method_definition` in JSON. Lexicographic CARS rule is fixed pre-evaluation.

## 4. Per-seed final results

### Seed 0

- Final: exact 0.0090, cell 0.637383
- Confidence: exact 0.0090, cell 0.638148, Δcell +0.077 pp
- CARS: exact 0.0100, cell 0.637728, Δcell +0.035 pp

### Seed 1

- Final: exact 0.0120, cell 0.639222
- Confidence: exact 0.0120, cell 0.638889, Δcell -0.033 pp
- CARS: exact 0.0120, cell 0.640519, Δcell +0.130 pp

### Seed 2

- Final: exact 0.0160, cell 0.670259
- Confidence: exact 0.0200, cell 0.670605, Δcell +0.035 pp
- CARS: exact 0.0200, cell 0.670444, Δcell +0.019 pp

## 5. Three-seed aggregate

- Final exact: 0.0123 ± 0.0035
- CARS exact: 0.0140 ± 0.0053
- Confidence exact: 0.0137 ± 0.0057
- CARS Δ cell (pp): 0.061 ± 0.060

## 6. Paired exact-solve transitions

See `per_seed_exact_transitions` in JSON (McNemar per seed).

## 7. Paired cell-accuracy analysis

See `per_seed_cell_deltas` in JSON (bootstrap CI per seed).

## 8. Recursive failure-mode analysis

See `earlier_exact_lost_by_final`, `cars_recovery`, `cars_damage` in JSON.

## 9. Oracle trajectory headroom

**ORACLE DIAGNOSTIC - NOT AN INFERENCE METHOD**

## 10. Selected-step distribution

See `selected_step_distribution` and figure `cars_selected_step_distribution.png`.

## 11. Inference-time overhead

Additional forward passes: 0

## 12. Optional ACT16 seed-0 sensitivity

NOT RUN

## 13. Paper-ready table

See `paper_table` in JSON.

## 14. Paper-ready figures

- `docs/figures/cars_act6_main_results.png`
- `docs/figures/cars_selected_step_distribution.png`
- `docs/figures/cars_recoverable_headroom.png` (optional)

## 15. Scientific interpretation

Fill after all three seeds complete. Do not overclaim from n=3 training seeds.

## 16. Proposed paper revision

Draft after observed results.

## 17. Missing or unresolved evidence


**CARS_ACT6_STATUS: COMPLETE**
