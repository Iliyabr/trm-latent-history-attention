# Baseline Experiment Configuration

## Canonical Phase 1 configuration

The canonical scientific baseline configuration is:

    config/cfg_baseline_v2.yaml

It uses:

- deterministic `sudoku-baseline-v2`
- seed 0
- 1000 training source puzzles
- 200 disjoint development puzzles
- development evaluation via `eval_split: dev`
- 20 epochs
- batch size 4
- CPU execution
- metrics JSON output
- dedicated `baseline-v2` checkpoint path

## Legacy configurations

The following files are retained only as historical development records:

- `config/cfg_baseline_cpu.yaml`
- `config/cfg_baseline_5epoch.yaml`
- `config/cfg_baseline_20epoch.yaml`
- `experiments/baseline/configs/cfg_baseline_cpu.yaml`

They must not be used for new Phase 1 or HistoryAttention experiments.

All future controlled comparisons should derive from:

    config/cfg_baseline_v2.yaml
