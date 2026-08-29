# Setup and run: RTX 4090 canonical GPU campaign

This is the step-by-step path for the **RTX 4090 (24 GB)** box. It uses
`--preset canonical` (protocol v1): D256, H3/L6/L2, ACT6, four models, last
checkpoint only.

The other live GPU (1080 Ti / first server) uses a **different seed**. Do not
copy this script onto that machine.

| Machine | Script | Seed | Dtype | Output |
|---------|--------|------|-------|--------|
| First GPU (1080 Ti) | `scripts/run_canonical_server.sh` | **0** | float32 | `outputs/study/` |
| This GPU (4090) | `scripts/run_canonical_4090.sh` | **1** | bfloat16 | `outputs/study-4090/` |

Protocol §22: do **not** treat seed-0 float32 and seed-1 bfloat16 as one matched
multi-seed family. Each card is its own **SCREENING** four-way comparison.
Report them separately.

CPU cap on this box: **16 host threads**, **4** dataloader workers, `OMP=1`
inside workers. Stay off extra GPUs (`CUDA_VISIBLE_DEVICES=0`). Watch
`nvidia-smi`; this D256/batch-32 job should stay well under 24 GB. If the
machine is shared, keep used VRAM in the 15–20 GB band by not raising batch
size.

Do not install `requirements.txt` (`adam-atan2` / `triton`). Use
`requirements-colab.txt` and PyTorch AdamW.

Work on branch `feature/latent-history-attention`. `main` does not contain the
study runner.

---

## 1. Log in and check the GPU

```bash
ssh <user>@<4090-host>
nvidia-smi
```

Confirm one RTX 4090, ~24 GB, and that you are not already saturating the card.
Install `tmux` if missing:

```bash
sudo apt update
sudo apt install -y tmux python3 python3-venv python3-pip git
python3 --version   # need 3.10–3.12, not 3.13
```

The train/eval script already pins itself to cores `0–15` with `taskset` when that command exists. You do not need to wrap it again.

---

## 2. Get the code

```bash
git clone -b feature/latent-history-attention https://github.com/Iliyabr/trm-latent-history-attention.git
cd trm-latent-history-attention
```

If the repo is already there:

```bash
git fetch origin
git checkout feature/latent-history-attention
git pull origin feature/latent-history-attention
ls scripts/run_canonical_4090.sh docs/SERVER_4090.md
```

---

## 3. Python environment (CUDA 12.x, not cu118)

cu118 is for the 1080 Ti. On a 4090:

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-colab.txt
```

If `cu124` wheels fail, try `cu126` at the same PyTorch index. Confirm:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(), torch.cuda.is_bf16_supported())"
```

You want `True`, a CUDA wheel (not `+cpu`), a 4090 name, capability `(8, 9)`,
and `True` for bfloat16.

---

## 4. Detach with tmux

Training is long. Do not run it on a raw SSH session.

```bash
tmux new -s trm4090
source .venv/bin/activate
cd ~/trm-latent-history-attention   # or your clone path
```

Detach: `Ctrl-b` then `d`. Reattach: `tmux attach -t trm4090`.

---

## 5. Hardware check and dataset

```bash
bash scripts/run_canonical_4090.sh setup
bash scripts/run_canonical_4090.sh data
```

`setup` writes `outputs/study-4090/canonical/provenance.txt` (git commit, GPU,
seed 1). `data` builds `data/sudoku-study-v1/` if it is missing (needs network
for Hugging Face). Re-running `data` is a no-op if the split already exists.

---

## 6. Train (seed 1, four models)

Same backbone for every job. Only the history module changes.

| Order | Variant | Model |
|-------|---------|--------|
| 1 | `B0` | Vanilla TRM (no history) |
| 2 | `P1` | Low-rank HistoryAttention |
| 3 | `Gated` | Gated uniform history |
| 4 | `B3` | Parameter-matched no-history |

```bash
bash scripts/run_canonical_4090.sh train
```

That runs, in order:

```text
python experiments/run_study.py single --preset canonical --variant <B0|P1|Gated|B3> --seed 1
  --override checkpoint_every_eval=false
  --override compile_model=false
  --override max_runtime_minutes=null
  --override arch.forward_dtype=bfloat16
  --override dataloader_num_workers=4
```

Checkpoints: one last `step_*.pt` per variant under
`outputs/study-4090/canonical/<VARIANT>-seed1/`. DEV metrics still go to
`metrics.jsonl` (keep those). Evaluate the **last** `step_*.pt`, not
`best_dev.pt`.

If a job dies after the first DEV eval:

```bash
bash scripts/run_canonical_4090.sh resume P1    # or B0 / Gated / B3
```

In another SSH session you can watch:

```bash
watch -n 5 nvidia-smi
htop
```

---

## 7. Test eval and tables

After all four train jobs finish:

```bash
bash scripts/run_canonical_4090.sh eval
```

This scores **TEST** with `--interventions` and writes
`results/canonical-gpu-4090/` plus `results/canonical-gpu-4090/analysis/`.

End-to-end after setup (train then eval):

```bash
bash scripts/run_canonical_4090.sh all
```

---

## 8. What to copy off the machine

Keep, do not git-commit:

- `outputs/study-4090/canonical/*/metrics.jsonl`
- `outputs/study-4090/canonical/*/run_metadata.json`
- `outputs/study-4090/canonical/*/step_*.pt` (last checkpoint)
- `outputs/study-4090/canonical/provenance.txt`
- `results/canonical-gpu-4090/`

---

## 9. Common failures

| Symptom | Fix |
|---------|-----|
| `torch.cuda.is_available() == False` | CPU wheel installed; reinstall `cu124` torch |
| BFloat16 error | Driver too old; or you cloned the 1080 script (float32) by mistake |
| CUDA OOM | Should be rare at batch 32 on 24 GB; do not raise batch |
| Host has 32+ CPUs pegged | Script already sets `OMP_NUM_THREADS=1`; launch with `taskset -c 0-15` |
| `Command 'python' not found` | `source .venv/bin/activate` |
| Training looks done in minutes | Wrong preset; must be `canonical`, not `colab` |
| Exact accuracy stays 0 | Read cell accuracy; full-board exact is hard |

---

## 10. First GPU server (do not mix)

On the **other** machine:

```bash
bash scripts/run_canonical_server.sh setup
bash scripts/run_canonical_server.sh data
bash scripts/run_canonical_server.sh train
bash scripts/run_canonical_server.sh eval
```

That campaign is seed **0**, float32, `outputs/study/`. This 4090 campaign is
seed **1**, bfloat16, `outputs/study-4090/`. Same four models, different seed
and hardware. In the paper, keep two screening tables (or one table with a
hardware column). Do not report a 2-seed mean that mixes the cards.
