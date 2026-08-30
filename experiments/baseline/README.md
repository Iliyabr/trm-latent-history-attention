# Baseline Experiment Configuration (historical)

Phase 1 used `sudoku-small`, `sudoku-baseline-v2`, and related CPU configs.
Those datasets and configs were removed in favor of the canonical study dataset:

- build: `python dataset/build_sudoku_baseline_v2.py`
- output: `data/sudoku-study-v1/`
- manifest: `artifacts/data/sudoku_study_v1_manifest.json`

Use `--preset canonical` (or `canonical_8h`) via `experiments/run_study.py` for
all new training. See [experiments/README.md](../README.md).

`PHASE1_RESULTS.md` remains as a historical record only.
