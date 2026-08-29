# Latent-history Sudoku study

This study compares five history mechanisms over three fixed training seeds:
`B0`, `B1`, `B2`, `B3`, and `P1` × seeds `0`, `1`, and `2` = **15 runs**.
The Colab preset has a 55-minute runtime cap and targets roughly one hour per
run. That target is not a guarantee that a run will finish or converge.

## Prepare data

```bash
python dataset/build_sudoku_baseline_v2.py
```

Defaults are 900 train bases, 100 dev puzzles, 1,000 bounded official test
puzzles, seed 0, and 64 train augmentations per base. Splitting and leakage
checks happen before augmentation. Outputs use the `PuzzleDataset` layout:

- `data/sudoku-study-v1/{train,dev,test}/`
- `artifacts/data/sudoku_study_v1_manifest.json`

The manifest records source/generated-file hashes, exact input-solution hashes,
digit-canonical hashes, selected source indices, and zero-overlap assertions.
The official test split is isolated from development.

## Launch

Inspect commands without starting training:

```bash
python experiments/run_study.py suite --dry-run
```

Run one job, all 15 jobs serially, or resume an interrupted/capped job:

```bash
python experiments/run_study.py single --variant P1 --seed 0
python experiments/run_study.py suite
python experiments/run_study.py resume --variant P1 --seed 0
```

Use `--preset publication` for the D512/rank128/H3/L6/ACT16 configuration.
Additional Hydra settings can be repeated with `--override`, for example:

```bash
python experiments/run_study.py single --variant B0 --seed 1 \
  --override global_batch_size=16 --override max_runtime_minutes=50
```

## Presets and interpretation

`colab` uses D256, four spatial heads, two transformer layers, H2/L4, ACT6,
and temporal rank 64 with four heads. Its batch size is a conservative T4
starting point; reduce it if a variant exceeds available VRAM.

**Canonical protocol v1** (`--preset canonical`) is the primary GPU track:
D256, **H3/L6/L2**, ACT6, rank 64, pre-QKV P1, variants **B0 / Gated / P1 / B3**.
See [docs/CANONICAL_PROTOCOL_v1.md](../docs/CANONICAL_PROTOCOL_v1.md).

`colab_heavy` keeps the same D256 architecture but raises the epoch budget to
1536 (~43k optimizer steps, about 1.5–2 hours on a T4) and the wall-clock cap
to 120 minutes. Use it for a longer B0 vs P1 seed-0 comparison. Outputs go under
`outputs/study/colab_heavy/` so they do not overwrite the short Colab runs.

On a GTX 1080 Ti or other Pascal GPU, follow [docs/SERVER_GPU.md](../docs/SERVER_GPU.md):
override `arch.forward_dtype=float32` and `max_runtime_minutes=null`. Do not
install `requirements.txt`. For canonical runs, prefer `--preset canonical`
(float32 is already the default there).

`publication` uses D512, H3/L6, ACT16, and temporal rank 128. The scaled Colab
preset changes model capacity and inference depth. Its inference results are
useful for pipeline validation and directional comparisons, but must not be
reported as publication-scale inference.

## Outputs

Each run writes under `outputs/study/<preset>/<variant>-seed<seed>/`:

- `step_<n>.pt`: resumable complete checkpoints
- `runtime_cap.pt`: checkpoint written when the cap is reached
- `best_dev.pt`: best observed development checkpoint
- `metrics.jsonl`: local metrics independent of W&B
- `run_metadata.json`: deterministic config/environment fingerprint
- `orchestrator.json`: command, status, timestamps, and return code

Checkpoints contain model and optimizer states, scheduler-relevant step, EMA,
RNG states, and resolved config. Batch-specific ACT carry is only restored when
marked safe; evaluation-boundary resumes start with a fresh carry.

`metrics.jsonl` records wall time, throughput, peak VRAM, development metrics,
best-metric decisions, and consecutive non-improvement counts. A plateau flag
is evidence from the configured patience window, not proof of convergence.

## Evaluation and paper tables

After training, evaluate checkpoints on the held-out test split and aggregate
seed-level results:

```bash
python experiments/evaluate_study.py \
  --config config/experiment/sudoku_study_colab.yaml \
  --checkpoint P1=outputs/study/colab/P1-seed0/best_dev.pt \
  --checkpoint B0=outputs/study/colab/B0-seed0/best_dev.pt \
  --data data/sudoku-study-v1 --split test --interventions --seed 0

python experiments/analyze_results.py \
  --input results/study --output results/study/analysis
```

`evaluate_study.py` writes per-example predictions, exact/cell accuracy,
Sudoku constraint violations, ACT/cycle trajectories, parameters, wall time,
peak VRAM, and latency. With `--interventions` it also runs matched Gaussian
latent corruption (`0.05/0.10/0.20 × RMS`) on every variant and
most/least-attended history deletion on P1.

`analyze_results.py` reports each seed, mean ± sample standard deviation,
paired bootstrap 95% CIs, exact McNemar tests, and the pre-specified P1 vs B0
and P1 vs B3 comparisons. Figures are written as `accuracy.pdf`,
`learning_curves.pdf`, `compute.pdf`, `corruption.pdf`, and `attention.pdf`.

The companion `TRM_Latent_History_Colab.ipynb` covers setup, data preparation,
single/suite execution, resume, and output inspection. Colab should install
`requirements-colab.txt` (not `requirements.txt`): the latter includes
`adam-atan2` and `triton`, which fail to build on Colab Python 3.13. The Colab
preset uses AdamW.
