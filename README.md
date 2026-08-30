# Latent-History Retrieval for Tiny Recursive Models

This repository is a research fork of **Tiny Recursive Models (TRM)** that
studies whether explicit access to previous recursive latent states improves
reasoning, and whether learned selective retrieval provides value beyond simple
history access or extra model capacity.

The project is built on the upstream TRM implementation from
Samsung SAIL Montreal and adds CPU-friendly execution, latent-history modules,
matched controls, reproducibility tooling, and a canonical GPU scale-transfer
protocol.

## Research question

> **When, why, and under what conditions does explicit latent-history access
> help Tiny Recursive Models?**

The project does **not** assume that HistoryAttention must improve TRM. Positive,
negative, null, and seed-sensitive outcomes are treated as valid scientific
results.

## Method

The primary method is **within-H-cycle Low-Rank HistoryAttention** over `z_L`.

For each H-cycle, the model keeps the complete causal sequence of `z_L` states:

```text
history = [z0]

L-step 1: READ [z0]       -> UPDATE z1 -> APPEND z1
L-step 2: READ [z0,z1]    -> UPDATE z2 -> APPEND z2
...
L-step 6: READ [z0,...z5] -> UPDATE z6 -> APPEND z6
```

The history is reset at the H-cycle boundary and is never carried through ACT.

Canonical retrieval:

```text
q   = W_Q RMSNorm(z)
k_i = W_K RMSNorm(h_i)
v_i = W_V RMSNorm(h_i)

alpha   = softmax((q · k_i) / sqrt(d_head))
context = sum_i alpha_i v_i
memory  = W_O(context)

g      = sigmoid(gate_logit)
z_read = RMSNorm(z + g * memory)
```

The CPU model uses `D=64`, rank `16`, four temporal heads, and
`gate_logit_init=-2`.

## Matched controls

| Variant | Purpose |
|---|---|
| Vanilla TRM | No explicit latent history |
| Gated Uniform History | Tests history access without learned temporal selection |
| Low-Rank HistoryAttention | Tests query-dependent selective retrieval |
| Parameter-Matched No-History | Tests whether extra capacity explains the effect |

## CPU results

The canonical CPU study is complete: **4 models × 5 seeds = 20 runs**, with
eight evaluation checkpoints per run.

| Model | Final token accuracy (mean ± SD) | Delta vs Vanilla |
|---|---:|---:|
| Vanilla | **45.867 ± 0.858%** | — |
| Gated Uniform History | 45.747 ± 0.503% | −0.120 pp |
| HistoryAttention | 45.459 ± 1.520% | −0.407 pp |
| Parameter-Matched No-History | 45.428 ± 0.655% | −0.438 pp |

The final CPU result does **not** show a robust final-accuracy improvement from
HistoryAttention. However, the paired learning curves show a reproducible early
optimization advantage: at step 1250, Attention − Vanilla is `+0.688 pp`,
positive in `5/5` seeds, with a Holm-adjusted checkpoint-level `p=0.040`.


A post-hoc inference-only intervention additionally suppressed the learned
HistoryAttention gate in the five frozen final checkpoints. Gate-off reduced
token accuracy by 0.684 pp on average (4/5 seeds negative; 95% CI
[-2.156,+0.788], p=0.267). The effect was heterogeneous and is treated as
exploratory mechanistic evidence, not as proof of a robust history benefit.

A separate supporting outer-step `z_H` history study found a reproducible
`+0.730 pp` Attention gain at recursion depth 8, with a paired 95% CI of
`[+0.367,+1.110]` pp and `5/5` positive seeds. The effect was not monotonic in
depth.

Exact-grid accuracy is zero in the reduced CPU regime, so CPU conclusions are
mechanistic/token-level rather than evidence of improved complete Sudoku
solving.

## Current status

| Component | Status |
|---|---|
| CPU environment and smoke pipeline | Complete |
| Outer-step latent-history mechanism study | Complete |
| Canonical within-cycle four-model CPU study | Complete / frozen |
| CPU statistical analysis and figures | Complete |
| Post-hoc inference-only gate ablation | Complete / exploratory |
| Unit tests for canonical L-cycle modules | 4 passed |
| Canonical GPU scale-transfer campaign | In progress |
| Final paper | CPU sections ready; GPU result section pending |

## Repository guide

```text
README.md
RUNBOOK.md

docs/
  CPU_STUDY_FINAL.md
  CPU_LCYCLE_FINAL_EVIDENCE_v1.md
  CPU_LCYCLE_STATISTICAL_ANALYSIS_v1.md
  TRM_HISTORY_CANONICAL_PROTOCOL_v1.md
  TRM_HISTORY_DEPLOYMENT_MANIFEST_v1.md
  data/
    CPU_LCYCLE_ALL_METRICS_v1.csv
  figures/
    cpu_paired_delta_vs_vanilla.png
    cpu_learning_curves_accuracy.png
    cpu_final_seed_accuracy.png
  archive/

models/history/
  # Supporting outer-step z_H modules
  attention.py
  gated.py
  last_state.py
  recency.py
  uniform.py

  # Canonical within-cycle z_L modules
  lcycle_lowrank_attention.py
  lcycle_gated.py
  lcycle_param_matched.py

scripts/
  run_lcycle_overnight.ps1
  analyze_cpu_lcycle.py
  extract_lcycle_gates.py

tests/
  test_lcycle_lowrank_history.py
```

## Quick CPU setup

Windows / Conda:

```powershell
conda create -n trm-cpu -c defaults --override-channels python=3.12 -y
conda activate trm-cpu
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
```

For analysis/figure reproduction:

```powershell
python -m pip install -r requirements-analysis.txt
```

Run the canonical module tests:

```powershell
python -m pytest tests/test_lcycle_lowrank_history.py -q
```

Expected:

```text
4 passed
```

Detailed commands, experiment naming, artifact paths, and reproduction policy
are documented in [`RUNBOOK.md`](RUNBOOK.md).

## Scientific protocol

The frozen scientific design is documented in:

```text
docs/TRM_HISTORY_CANONICAL_PROTOCOL_v1.md
```

GPU deployment-specific values are recorded separately in:

```text
docs/TRM_HISTORY_DEPLOYMENT_MANIFEST_v1.md
```

The scientific protocol and hardware deployment manifest are intentionally
separate so runtime feasibility decisions do not silently alter the research
question.

## Reproducibility

Final CPU analysis data are tracked in:

```text
docs/data/CPU_LCYCLE_ALL_METRICS_v1.csv
```

Raw checkpoints, local logs, and large experiment directories are preserved
locally and are intentionally not committed wholesale.

## Attribution

This project is based on the upstream **Tiny Recursive Models** repository and
the paper *Less is More: Recursive Reasoning with Tiny Networks*.

Upstream repository:

<https://github.com/SamsungSAILMontreal/TinyRecursiveModels>

Project fork:

<https://github.com/Iliyabr/trm-latent-history-attention>

Please cite the original TRM and HRM work when using the upstream code or ideas.
