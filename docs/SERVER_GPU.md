# Train and test on a local GPU server (GTX 1080 Ti)

This path is for the latent-history Sudoku study on a Linux box with an NVIDIA
GPU. It is written for a **GTX 1080 Ti** (11 GB, Pascal, compute 6.1). A newer
GPU (T4, RTX, A100) can drop the `float32` override and keep `bfloat16`.

Do not use `docs/CPU_LOCAL.md` here. Do not install `requirements.txt`
(`adam-atan2` / `triton`). Use `requirements-colab.txt` and PyTorch `AdamW`.

Work on branch `feature/latent-history-attention`. `main` does not contain the
study runner.

On Debian/Ubuntu the interpreter is `python3`, not `python`. After you activate
a venv, `python` exists inside that venv.

## 1. Get the code

```bash
git clone -b feature/latent-history-attention https://github.com/Iliyabr/trm-latent-history-attention.git
cd trm-latent-history-attention
```

If the repo is already on the machine:

```bash
git fetch origin
git checkout feature/latent-history-attention
git pull origin feature/latent-history-attention
git log -1 --oneline
ls scripts/run_canonical_8h_server.sh scripts/run_lmix_noskip_server.sh
```

If `git pull` leaves you on an old commit (e.g. `d37a107`) or new scripts are
missing, force-sync to GitHub (discards uncommitted repo edits):

```bash
bash scripts/update_from_github.sh
```

Or manually:

```bash
git fetch origin feature/latent-history-attention
git checkout feature/latent-history-attention
git reset --hard origin/feature/latent-history-attention
```

You should see `23e99a1` or newer and `scripts/run_canonical_8h_server.sh`.

If git still fails, download the script directly:

```bash
mkdir -p scripts config/experiment
curl -fsSL -o scripts/run_canonical_8h_server.sh \
  https://raw.githubusercontent.com/Iliyabr/trm-latent-history-attention/feature/latent-history-attention/scripts/run_canonical_8h_server.sh
curl -fsSL -o config/experiment/sudoku_study_canonical_8h.yaml \
  https://raw.githubusercontent.com/Iliyabr/trm-latent-history-attention/feature/latent-history-attention/config/experiment/sudoku_study_canonical_8h.yaml
curl -fsSL -o scripts/update_from_github.sh \
  https://raw.githubusercontent.com/Iliyabr/trm-latent-history-attention/feature/latent-history-attention/scripts/update_from_github.sh
chmod +x scripts/run_canonical_8h_server.sh scripts/update_from_github.sh
```

Confirm the remote URL matches where you push from your laptop:

```bash
git remote -v
# should be https://github.com/Iliyabr/trm-latent-history-attention.git
```

Skip Colab junk: extra nested clones, `.venv`, `__pycache__`. Copy data and
optional checkpoints separately (next section).

## 2. Python environment

Python **3.10–3.12**. Avoid 3.13.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
python3 --version

python3 -m venv .venv
source .venv/bin/activate
python --version
pip install --upgrade pip
```

Install a **CUDA** PyTorch build. CUDA 11.8 wheels are the safe choice for a
1080 Ti:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-colab.txt
```

Check the GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
```

You want `True`, a CUDA wheel (not `+cpu`), and capability `(6, 1)` on a 1080 Ti.

Use `tmux` or `screen`. A 1536-epoch `colab_heavy` run is longer than Colab’s
~1.5–2 hours because this card runs **float32** and is older than a T4.

```bash
tmux new -s trm
# detach: Ctrl-b then d
# reattach: tmux attach -t trm
```

## 3. Dataset

Copy the study split if you already built it (Google Drive
`MyDrive/trm-study/data` and `artifacts`):

```text
data/sudoku-study-v1/{train,dev,test}/
artifacts/data/sudoku_study_v1_manifest.json
```

Or rebuild (900 train bases, 64 augs, 100 dev, 1000 test):

```bash
python dataset/build_sudoku_baseline_v2.py
```

Do not train without that `data/sudoku-study-v1/` tree.

## 4. Train (B0, then P1)

`--preset colab_heavy` loads
`config/experiment/sudoku_study_colab_heavy.yaml` (same D256 / H2 / L4 / ACT6
as Colab). On a 1080 Ti you **must** override dtype and the 120-minute cap.

```bash
python experiments/run_study.py single --preset colab_heavy --variant B0 --seed 0 \
  --override arch.forward_dtype=float32 \
  --override compile_model=false \
  --override max_runtime_minutes=null
```

When B0 finishes, run P1 the same way (`--variant P1`). One job at a time.

### L-mix no-skip compare (B0 vs P1ns vs P1nsMLP)

Matched ACT6 / H3 / L6 / D256 run that contrasts vanilla transformer L against
no-skip history attention on **transformer** and **MLP** L-stacks
(`mlp_t=true`, `pos_encodings=none`). Default wall budget is **18 h total**
(**6 h per arm**; `epochs=8192`, clock stops first):

```bash
bash scripts/run_lmix_noskip_server.sh all
# or stepwise: setup → data → train-all → eval → analyze
# other total: TOTAL_RUNTIME_MINUTES=720 bash scripts/run_lmix_noskip_server.sh train-all
```

Artifacts: `outputs/study-lmix-noskip/`, `results/lmix-noskip/`, analysis under
`results/lmix-noskip/analysis/`.

If CUDA runs out of memory (11 GB vs T4 16 GB), add:

```text
--override global_batch_size=16
```

Print the Hydra command without training:

```bash
python experiments/run_study.py single --preset colab_heavy --variant B0 --seed 0 --dry-run \
  --override arch.forward_dtype=float32 \
  --override compile_model=false \
  --override max_runtime_minutes=null
```

Outputs land in `outputs/study/colab_heavy/<VARIANT>-seed<SEED>/`:

| File | Meaning |
|------|---------|
| `best_dev.pt` | Best development checkpoint (use this for test) |
| `runtime_cap.pt` | Written when a wall-clock cap stops the run |
| `step_<n>.pt` | Periodic checkpoint at each eval |
| `metrics.jsonl` | Train/dev metrics, throughput, peak VRAM |
| `run_metadata.json` | Resolved config and environment |

Do **not** resume a Colab `bfloat16` checkpoint into this float32 run. Start a
fresh `colab_heavy` directory on the server.

### Resume after a crash

Same variant, seed, preset, and overrides:

```bash
python experiments/run_study.py resume --preset colab_heavy --variant B0 --seed 0 \
  --override arch.forward_dtype=float32 \
  --override compile_model=false \
  --override max_runtime_minutes=null
```

Resume picks `runtime_cap.pt` if present, otherwise the latest `step_*.pt`.

## 5. Test

Need `best_dev.pt` (or `runtime_cap.pt`) for each variant you want to compare.
The evaluator reads architecture from the checkpoint when it is stored there,
so a float32 train should evaluate as float32.

```bash
python experiments/evaluate_study.py \
  --config config/experiment/sudoku_study_colab_heavy.yaml \
  --checkpoint B0=outputs/study/colab_heavy/B0-seed0/best_dev.pt \
  --checkpoint P1=outputs/study/colab_heavy/P1-seed0/best_dev.pt \
  --data data/sudoku-study-v1 \
  --split test \
  --seed 0 \
  --device cuda \
  --output results/study-heavy
```

Add `--interventions` only after the main numbers look sane. It runs latent
corruption and P1 attention deletions and takes longer.

Aggregate tables and figures:

```bash
python experiments/analyze_results.py \
  --input results/study-heavy \
  --output results/study-heavy/analysis
```

**exact_accuracy** is a fully correct 81-cell board. It can stay 0 even when
the model is learning. Read **cell_accuracy** in `results/study-heavy/*/metadata.json`
as well.

## 6. Command-line flags

### `experiments/run_study.py`

| Flag | Meaning |
|------|---------|
| `single` / `resume` / `suite` | One job, continue one job, or all 15 Colab-suite jobs |
| `--preset` | Which experiment YAML: `colab` (short), `colab_heavy` (long B0/P1), `publication` (larger model) |
| `--variant` | History mechanism: `B0`, `B1`, `B2`, `B3`, `P1` |
| `--seed` | Training seed: `0`, `1`, or `2` |
| `--output-root` | Parent of run dirs (default `outputs/study`) |
| `--override KEY=VALUE` | Hydra override; repeat the flag. Applied on top of the YAML |
| `--dry-run` | Print the `pretrain.py` command only |
| `--checkpoint PATH` | (`resume` only) Specific `.pt`; default is cap / latest step |

### `experiments/evaluate_study.py`

| Flag | Meaning |
|------|---------|
| `--config PATH` | Experiment YAML (architecture + data defaults) |
| `--checkpoint VARIANT=PATH` | Repeat per variant. Prefer `best_dev.pt` |
| `--data PATH` | Dataset root (`data/sudoku-study-v1`) |
| `--split` | `test` for the held-out set; `dev` for sanity checks |
| `--seed` | Eval RNG (interventions / dataloader). Not the training seed in the path |
| `--device` | `auto`, `cuda`, or `cpu` |
| `--batch-size` | Override eval batch if VRAM is tight |
| `--output` | Artifact directory |
| `--interventions` | Extra corruption / attention probes |

### `experiments/analyze_results.py`

| Flag | Meaning |
|------|---------|
| `--input` | Folder written by `evaluate_study.py` |
| `--output` | Stats JSON, CSVs, and PDFs |
| `--bootstrap-samples` | Paired bootstrap draws (default 10000) |
| `--no-figures` | Skip PDFs |

## 7. Experiment YAML (`sudoku_study_colab_heavy.yaml`)

Edit this file **or** pass `--override`. `--override` wins if both are set.
`epochs` must be divisible by `eval_interval`.

Optimizer steps ≈ `epochs * 900 / global_batch_size` (900 train bases).

| Key | Default (heavy) | Meaning |
|-----|-----------------|---------|
| `epochs` | 1536 | Passes over the train set. 64 ≈ short Colab; 1536 ≈ long run |
| `eval_interval` | 64 | Dev eval and checkpoint every N epochs |
| `max_runtime_minutes` | 120 | Wall-clock stop. Use `null` on a dedicated server |
| `checkpoint_every_eval` | true | Write `step_*.pt` at each eval |
| `global_batch_size` | 32 | Puzzles per optimizer step. Drop to 16 on OOM |
| `compile_model` | false | `torch.compile`. Keep false on 1080 Ti / T4 |
| `dataloader_num_workers` | 1 | Background CPU loaders |
| `lr` | 1e-4 | AdamW learning rate for the backbone |
| `lr_min_ratio` | 0.1 | Cosine-schedule floor as a fraction of `lr` |
| `lr_warmup_steps` | 500 | Linear warmup before cosine decay |
| `beta1` / `beta2` | 0.9 / 0.95 | AdamW moments |
| `weight_decay` | 0.1 | Backbone weight decay |
| `puzzle_emb_lr` | 1e-2 | Separate LR for puzzle embeddings |
| `puzzle_emb_weight_decay` | 0.1 | Weight decay for those embeddings |
| `optimizer` | adamw | PyTorch AdamW (not `adam-atan2`) |
| `seed` | 0 | Overridden by `--seed` from the runner |
| `ema` / `ema_rate` | true / 0.999 | Exponential moving average of weights for eval |
| `device` | auto | CUDA if available, else CPU |
| `wandb_mode` | disabled | No Weights & Biases login required |
| `deterministic` | true | More reproducible kernels where PyTorch allows it |
| `best_dev_metric` | exact_accuracy | Metric that selects `best_dev.pt` |
| `best_dev_mode` | max | Higher is better |
| `plateau_patience_evals` | 20 | Logged plateau flag; training still runs to epochs/cap |
| `data_paths` | `data/sudoku-study-v1` | Train/dev location |
| `eval_split` | dev | Split used **during** training |

## 8. Architecture YAML (`trm_history_colab.yaml`)

Loaded by the heavy preset. Override with `arch.<key>=...`.

| Key | Default | Meaning |
|-----|---------|---------|
| `forward_dtype` | bfloat16 | Compute dtype. **1080 Ti: `float32`**. T4/Ampere: `bfloat16` is fine |
| `hidden_size` | 256 | Residual width (D) |
| `num_heads` | 4 | Spatial attention heads |
| `L_layers` | 2 | Transformer layers inside the L (reasoning) net |
| `H_cycles` | 2 | Outer H recursion cycles |
| `L_cycles` | 4 | Inner L cycles per H cycle |
| `halt_max_steps` | 6 | ACT / adaptive-halt budget (ACT6) |
| `halt_exploration_prob` | 0.1 | Train-time halt exploration |
| `expansion` | 4 | MLP width multiplier |
| `pos_encodings` | rope | Rotary position encodings |
| `puzzle_emb_len` | 16 | Learned puzzle-token length |
| `history_enabled` | false | Set by the runner per variant |
| `history_mode` | B0 | Set by the runner: B0–B3 / P1 |
| `history_rank` | 64 | Low-rank size for P1 temporal attention |
| `history_heads` | 4 | Temporal attention heads (P1) |
| `history_window` | 6 | Max within-H-cycle `z_L` history length |
| `history_gate_init` | -2.0 | P1 residual-gate init (starts near “off”) |

### Variants (`--variant`)

History is **within an H-cycle `z_L`**, not ACT / `z_H`.

| Code | What it does |
|------|----------------|
| `B0` | Identity / vanilla TRM (no history mix) |
| `B1` | RMSNorm residual, no history mix |
| `B2` | Uniform mean of within-cycle history |
| `B3` | Identity reader + extra FFN width ≈ P1 parameter count |
| `P1` | Low-rank multi-head temporal attention over history |

Keep architecture the same when comparing B0 vs P1.

## 9. Common failures

| Symptom | Fix |
|---------|-----|
| `Command 'python' not found` | `python3 -m venv .venv` then `source .venv/bin/activate` |
| `torch.cuda.is_available() == False` | CPU wheel installed; reinstall `cu118` torch |
| BFloat16 / dtype CUDA error | `arch.forward_dtype=float32` |
| CUDA out of memory | `global_batch_size=16` |
| Job dies at ~2 hours | `max_runtime_minutes=null` |
| Training looks done after ~8 minutes | You used preset `colab` (`epochs=64`), not `colab_heavy` |
| `import pretrain` / missing study scripts | Wrong branch (`main`) or nested extra clone; stay on repo root |
| Exact accuracy stays 0 | Check `cell_accuracy`; full-board exact is hard |

## 10. What not to copy back

Do not git-commit checkpoints (`.pt`), `data/`, or `outputs/`. Copy those with
`scp`/`rsync` if you need them elsewhere.
