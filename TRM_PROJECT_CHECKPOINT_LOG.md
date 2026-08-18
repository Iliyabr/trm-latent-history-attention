# TRM Project Development Log

## Project

Less is More: Recursive Reasoning with Tiny Networks

## Purpose of this document

This file is a handoff checkpoint. It stores important decisions,
experiments, commands, outputs, and unresolved items so the project can
continue in another chat, with another AI assistant, or by another team
member without reconstructing the history.

------------------------------------------------------------------------

# Current Phase

## Phase 0: CPU Foundation

Status: COMPLETE

Verified items: - Repository structure created. - CPU-only environment
validated. - PyTorch CPU execution validated. - Tiny Sudoku dataset
pipeline validated. - TRM model imports validated. - Config files
validated. - CPU smoke training completed. - Checkpoint generation
validated. - Reproducible local workflow documented.

Important environment: - Conda environment: trm-cpu - Python: 3.12.13 -
PyTorch: 2.7.0+cpu - CUDA: False - CPU threads: 8

------------------------------------------------------------------------

# Git Status

Branches: - main - codex/cpu-foundation - phase1-baseline (current
development branch)

Important decision: - Phase 0 foundation was kept isolated from future
experimental changes. - Phase 1 starts from a clean development branch.

------------------------------------------------------------------------

# Phase 0 Artifacts

Important files:

-   README.md
-   CPU_LOCAL.md
-   requirements-cpu.txt
-   config/cfg_cpu_smoke.yaml
-   config/arch/trm_cpu.yaml
-   scripts/create_tiny_sudoku_dataset.py

Smoke checkpoint:

checkpoints/trm-cpu-smoke/local-smoke/

Contains: - all_config.yaml - losses.py - trm.py - step_16

------------------------------------------------------------------------

# Phase 1: Baseline Stabilization

Status: IN PROGRESS

Goal: Create a reproducible baseline before implementing
HistoryAttention.

Reason: Future architectural changes must be compared against a stable
TRM baseline.

Current branch: phase1-baseline

Created structure:

experiments/ └── baseline/ ├── configs/ │ └── cfg_baseline_cpu.yaml └──
results/

------------------------------------------------------------------------

# Important Design Decisions

## Dataset strategy

Three levels are planned:

1.  sudoku-tiny

-   Only for smoke tests.
-   Never used for scientific claims.

2.  sudoku-small

-   Phase 1 baseline dataset.
-   CPU-compatible.
-   Used for comparing original TRM and future modifications.

3.  sudoku-extreme

-   Full research-scale experiments.
-   Requires stronger hardware/cloud GPU.

------------------------------------------------------------------------

# Dataset Work

Created:

dataset/build_sudoku_baseline.py

Purpose: A wrapper around the original TRM Sudoku preprocessing
pipeline.

Current command:

python dataset/build_sudoku_baseline.py

Configuration:

Output: data/sudoku-small

Parameters: - subsample_size: 1000 - num_aug: 10

Meaning: - 1000 original puzzles - 10 augmentations per puzzle

------------------------------------------------------------------------

# Current Dataset Generation Status

Last observed log:

Running: python dataset/build_sudoku_dataset.py --output-dir
data/sudoku-small --subsample-size 1000 --num-aug 10

Download: train.csv from HuggingFace

Observed: - HuggingFace Xet warning is not an error. - Falling back to
normal HTTP download. - Dataset download was progressing normally.

No action required: hf_xet package is optional.

------------------------------------------------------------------------

# Next Steps

1.  Finish sudoku-small generation.

2.  Validate generated dataset:

    -   files exist
    -   metadata correct
    -   numpy shapes correct
    -   PuzzleDataset loader works

3.  Update baseline config:

    -   data path -\> data/sudoku-small
    -   project name -\> baseline experiment
    -   checkpoint path -\> baseline checkpoint folder

4.  Run first TRM baseline experiment.

5.  Only after baseline is stable: Implement HistoryAttention module.

------------------------------------------------------------------------

# Important Rule

Do not modify upstream TRM files unnecessarily.

Keep: - original TRM - baseline experiments - HistoryAttention
experiments

separated for clean comparisons.
