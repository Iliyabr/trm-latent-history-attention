# Local CPU setup (Windows 11)

This path is for development checks and small pilot runs on a laptop without
CUDA. It does not reproduce the full paper-scale experiments.

## 1. Create an isolated environment

```powershell
conda create -n trm-cpu -c defaults --override-channels python=3.12 -y
conda activate trm-cpu
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
```

The CPU requirements intentionally omit `triton`, CUDA wheels, and
`adam-atan2`. The smoke configuration uses PyTorch `AdamW` instead.
The verified environment uses Python 3.12.13 and PyTorch 2.7.0+cpu.

## 2. Build the deterministic toy dataset

```powershell
python scripts/create_tiny_sudoku_dataset.py
```

The generated `data/sudoku-tiny` directory is ignored by Git. It is synthetic
and exists only to verify the local pipeline.

## 3. Run the smoke test

```powershell
python pretrain.py --config-name cfg_cpu_smoke
```

Expected behavior:

- the log prints `Runtime device: cpu`;
- 16 tiny training steps and two evaluation passes complete;
- a small checkpoint appears under `checkpoints/trm-cpu-smoke/`;
- Weights & Biases stays disabled and no login is required.

## Scope warning

The upstream Sudoku experiment uses a much larger model, batch, augmentation
count, and training schedule. The smoke result is not an accuracy baseline and
must never be included as evidence in the final report. The completed canonical
CPU study is documented in docs/CPU_STUDY_FINAL.md; reproduction commands are
maintained in RUNBOOK.md.
