# ACT6 GPU Final Evidence Report

**Scope of this report:** canonical preset `canonical`, ACT `halt_max_steps=6`, GTX 1080 Ti, seed **0 only**. Source: `outputs/canonical/` (training) and `results/canonical-gpu/` (test eval).

**Not mixed in:** Colab/CPU pilots, `canonical_8h` / ACT16 (`results/canonical-gpu-8h/`), extended L_cycle, or 4090 seed-1.

The launch script labels this campaign **SCREENING** (single seed). It is the only completed ACT6 GPU four-variant run in this repository.

---

## 1. Provenance and configuration

| Field | Value | Source |
|-------|--------|--------|
| Git branch | `feature/latent-history-attention` | campaign convention; provenance records hash |
| Git commit (training) | `d37a1071a89c56bc3a741a5b6b2e859ca657e177` | `outputs/canonical/provenance.txt` |
| Git commit (evaluation) | MISSING | no eval provenance file |
| GPU | NVIDIA GeForce GTX 1080 Ti, 11264 MiB, driver 580.173.02 | provenance.txt |
| CUDA | 11.8 | `run_metadata.json` (all four runs) |
| PyTorch | 2.7.1+cu118 | `run_metadata.json` |
| Python | 3.12.3 | `run_metadata.json` |
| NumPy | 2.5.2 | `run_metadata.json` |
| OS | MISSING (Linux host path `/home/mahyar/...` only) | — |
| Dataset path | `data/sudoku-study-v1` | `all_config.yaml` |
| Train / dev / test counts | MISSING in this clone (`data/` gitignored; no `sudoku_study_v1_manifest.json` in repo) | Builder **defaults** used by `run_canonical_server.sh data`: 900 train bases, 100 dev, 1000 test, 64 aug → 58,500 stored train examples — **not re-verified from npy** |
| Leakage checks | MISSING | no manifest in this clone |
| ACT / halt_max_steps | 6 | `all_config.yaml` |
| Physical batch | 32 | `global_batch_size: 32` |
| Gradient accumulation | none (no accumulation code in `pretrain.py`) | code + config |
| Effective batch | 32 | same |
| Optimizer | AdamW | config |
| lr | 1e-4 (0.0001) | config |
| weight decay | 0.1 | config |
| betas | (0.9, 0.95) | config |
| EMA | true, 0.999 | config |
| compile | false | config |
| Training budget | 1024 epochs = **28800** steps, `examples_seen=921600` all four | `run_end` |
| Stopping rule | epoch/step budget (no runtime cap); `runtime_cap_reached: false` | `run_end` |
| Eval checkpoint | last `step_28800.pt` (not `best_dev.pt`) | test `metadata.json` |
| Same budget all models | **yes** — 28800 steps each | `run_end` |

**Training wall-clock (`run_end.runtime_seconds`):**

| Variant | runtime_seconds | examples/sec (train) | peak_vram_bytes (train) |
|--------|-----------------|----------------------|-------------------------|
| Vanilla (B0) | 3812.0966819210007 | 241.75672258542644 | 1069243392 |
| HistoryAttention (P1) | 4813.044401752 | 191.47963805705336 | 1208774656 |
| Gated | 4246.843711241001 | 217.00822131989702 | 1176866304 |
| Parameter-Matched (B3) | 4003.4161828899996 | 230.20339577453382 | 1155308032 |

**Orchestrator UTC (wall calendar):** B0 13:23–14:26, P1 14:26–15:46, Gated 15:46–16:57, B3 16:57–18:04 on 2026-08-29.

**Seeds completed:** **0 only**. Seeds 1 and 2: **NOT RUN** for this ACT6 1080 Ti campaign.

---

## 2. Configuration audit

### MATCHED

D=256; H_cycles=3; L_cycles=6; L_layers=2; spatial heads=4; halt_max_steps=6; history_rank=64; history_heads=4; history_gate_init=−2.0; history_window=0; AdamW; lr=1e-4; wd=0.1; β=(0.9,0.95); EMA=0.999; compile=false; effective batch 32; dataset path `sudoku-study-v1`; same 28800-step budget.

P1 config: `history_enabled=true`, `history_mode=P1`. Pre-QKV RMSNorm is in `models/history/attention.py` (`q = W_Q RMSNorm_D(z)`). Rank ratio r/D = 64/256 = 1/4.

`mlp_t=false`, `pos_encodings=rope`, `forward_dtype=float32`, `no_ACT_continue=true`.

### DEVIATIONS / caveats vs intended protocol list

| Item | Intended | Actual |
|------|----------|--------|
| Primary GPU seed set | 0, 1, 2 | **seed 0 only** |
| Confirmatory vs screening | matched multi-seed | script labels **SCREENING** |
| Eval checkpoint | unspecified in ACT6 prompt; 8h recipe uses `best_dev.pt` | **last step_28800.pt** |
| Dataset counts / leakage | verify from artifacts | **MISSING** in this clone |
| `no_ACT_continue` | not in the ACT6 expected list | **true** (always halt-max / no continue head path) |
| Puzzle-emb lr | not in expected list | 1e-2 (dense lr 1e-4) |

### MISSING

OS distro; eval commit; dataset npy hashes and leakage assertion file; gradient-accumulation hyperparameter (none implemented); learned gate values after training; test-set LM loss / q_halt in eval metadata.

History read-update-append / reset-per-H-cycle: specified in code (`models/history/README.md`, `trm.py`); **not independently logged per run**.

---

## 3. Parameter counts

From test eval `metadata.json` (`parameters.trainable` = `parameters.total`):

| Model | Trainable params | vs Vanilla | % vs Vanilla |
|-------|------------------|------------|--------------|
| Vanilla | 1,710,082 | 0 | 0 |
| Gated Uniform History | 1,710,083 | **+1** | +0.0000585% |
| HistoryAttention | 1,775,619 | **+65,537** | +3.832% |
| Parameter-Matched | 1,775,619 | **+65,537** | +3.832% |

**HistoryAttention and Parameter-Matched added counts are equal:** 4·D·r+1 = 4·256·64+1 = **65537**.

---

## 4. Per-seed final results

**Test split, intervention `clean`, seed 0.** Exact and cell from eval metadata. **Test LM loss, test q_halt: MISSING** (not stored in eval `metadata.json`). Dev last-eval values from training `metrics.jsonl` at step 28800 are listed separately (dev, not test).

| Model | Seed | Checkpoint | Exact-grid | Cell/token | Train wall (s) | Eval wall (s) | Peak VRAM eval (B) | Throughput eval (ex/s) | Gate |
|-------|------|------------|------------|------------|-----------------|---------------|---------------------|--------------------------|------|
| Vanilla | 0 | `.../B0-seed0/step_28800.pt` | 0.009 | 0.6373827160493827 | 3812.0966819210007 | 29.053754566000862 | 121707520 | 54.11548714852564 | n/a |
| Gated | 0 | `.../Gated-seed0/step_28800.pt` | 0.012 | 0.6654567901234568 | 4246.843711241001 | 23.33993965300033 | 137077248 | 78.8313440794983 | MISSING |
| HistoryAttention | 0 | `.../P1-seed0/step_28800.pt` | 0.009 | 0.6389876543209877 | 4813.044401752 | 34.23273052500008 | 113478144 | 42.14370563648292 | MISSING |
| Parameter-Matched | 0 | `.../B3-seed0/step_28800.pt` | 0.001 | 0.6434074074074074 | 4003.4161828899996 | 22.395344807002402 | 124624384 | 84.29687108617983 | n/a |

**Dev last eval (step 28800, eval_split=dev) — not the paper test metric:**

| Model | Dev cell | Dev exact | Dev lm_loss | q_halt_acc | q_halt_loss | ACT steps |
|-------|----------|-----------|-------------|------------|-------------|-----------|
| Vanilla | 0.6323456764221191 | 0.019999999552965164 | 0.8030175566673279 | 1.0 | 4.927961826324463 | 6.0 |
| HistoryAttention | 0.6367901563644409 | 0.0 | 0.8047894835472107 | 1.0 | 3.2117693424224854 | 6.0 |
| Gated | 0.6677777767181396 | 0.009999999776482582 | 0.754443347454071 | 1.0 | 3.7135326862335205 | 6.0 |
| Parameter-Matched | 0.6325925588607788 | 0.0 | 0.8218064308166504 | 1.0 | 1.7244279384613037 | 6.0 |

Dev best exact during training (`best_metric`): Vanilla 0.02; Gated 0.02; P1 0.01; B3 0.01. Test eval used the **final** step, not that best-dev checkpoint.

---

## 5. Aggregate final results

**Seed sets are not matched to the intended {0,1,2}.** Only seed 0 exists. **Do not treat the table below as a multi-seed mean ± SD.** Sample SD is undefined for n=1.

Paper-shaped table using **seed-0 test clean** values (cell/exact as % with 3 decimals for display; raw in §4):

| Model | Exact Acc. (%) | Cell Acc. (%) | LM Loss (test) | Params | Δ Cell vs Vanilla | Δ Exact vs Vanilla |
|-------|----------------|---------------|----------------|--------|-------------------|--------------------|
| Vanilla | 0.900 | 63.738 | MISSING | 1710082 | 0 | 0 |
| Gated | 1.200 | 66.546 | MISSING | 1710083 | +2.807 pp | +0.300 pp |
| HistoryAttention | 0.900 | 63.899 | MISSING | 1775619 | +0.160 pp | 0.000 pp |
| Parameter-Matched | 0.100 | 64.341 | MISSING | 1775619 | +0.602 pp | −0.800 pp |

Exact counts on 1000 test puzzles: Vanilla 9/1000; Gated 12/1000; HistoryAttention 9/1000; Parameter-Matched 1/1000.

---

## 6. Paired comparisons

**Matched seeds: n=1 (seed 0).** No 95% CI, no paired t-test.

Cell-accuracy deltas (percentage points), test clean:

| Comparison | Seed 0 Δ cell (pp) | Mean paired | SD | Positive seeds |
|-----------|--------------------|-------------|-----|----------------|
| HistoryAttention − Vanilla | +0.160493827160495 | same | MISSING | 1/1 |
| Gated − Vanilla | +2.80740740740741 | same | MISSING | 1/1 |
| Parameter-Matched − Vanilla | +0.60246913580247 | same | MISSING | 1/1 |
| HistoryAttention − Gated | −2.64691358024691 | same | MISSING | 0/1 |
| HistoryAttention − Parameter-Matched | −0.44197530864197 | same | MISSING | 0/1 |

Exact-grid deltas (fraction, seed 0):

| Comparison | Δ exact |
|-----------|----------------|
| HistoryAttention − Vanilla | 0.000 |
| Gated − Vanilla | +0.003 |
| Parameter-Matched − Vanilla | −0.008 |
| HistoryAttention − Gated | −0.003 |
| HistoryAttention − Parameter-Matched | +0.008 |

McNemar / per-example paired tests: **not computed** (would need paired per-puzzle correctness vectors; not aggregated here).

**No statistical significance is claimed.**

---

## 7. Optimization trajectory

Dev eval every 1800 steps (64 epochs × 32 examples/step wait: 64 epochs × 900/32 = 1800 steps). **Single seed — no mean ± SD.**

Dev **cell** accuracy:

| Step | Vanilla | Gated | HistoryAttention | Param-Matched | P1−Vanilla (pp) |
|------|---------|-------|-----------------|---------------|-----------------|
| 1800 | 0.44913581013679504 | 0.43518516421318054 | 0.44740742444992065 | 0.43753087520599365 | −0.173 |
| 3600 | 0.4170370101928711 | 0.36580249667167664 | 0.5059259533882141 | 0.5008642077445984 | **+8.889** |
| 5400 | 0.5408642292022705 | 0.4234567880630493 | 0.5464197397232056 | 0.5760494470596313 | +0.556 |
| 7200 | 0.49530863761901855 | 0.4295061528682709 | 0.5986419916152954 | 0.5930864214897156 | **+10.333** |
| 9000 | 0.5829629302024841 | 0.6117283701896667 | 0.6018518805503845 | 0.6046913862228394 | +1.889 |
| 10800 | 0.5839506387710571 | 0.592345654964447 | 0.607654333114624 | 0.6114814877510071 | +2.370 |
| 12600 | 0.6108642220497131 | 0.6218518614768982 | 0.6139506697654724 | 0.6124691367149353 | +0.309 |
| 14400 | 0.6170370578765869 | 0.6235802173614502 | 0.6238271594047546 | 0.6159259080886841 | +0.679 |
| 16200 | 0.6244444847106934 | 0.6165432333946228 | 0.4750617742538452 | 0.621975302696228 | **−14.938** |
| 18000 | 0.6239506602287292 | 0.646049439907074 | 0.627407431602478 | 0.5862963199615479 | +0.346 |
| 19800 | 0.6227160692214966 | 0.6590123772621155 | 0.5435802340507507 | 0.6132099032402039 | −7.914 |
| 21600 | 0.6272839307785034 | 0.6609876155853271 | 0.6316049098968506 | 0.6301234364509583 | +0.432 |
| 23400 | 0.6298765540122986 | 0.6601234674453735 | 0.6329629421234131 | 0.6359260082244873 | +0.309 |
| 25200 | 0.6314814686775208 | 0.6634567975997925 | 0.633456826210022 | 0.6313580870628357 | +0.198 |
| 27000 | 0.6333333253860474 | 0.6629630327224731 | 0.6338271498680115 | 0.6304938197135925 | +0.049 |
| 28800 | 0.6323456764221191 | 0.6677777767181396 | 0.6367901563644409 | 0.6325925588607788 | +0.444 |

- Largest positive HistoryAttention−Vanilla **dev cell** delta: **step 7200, +10.333 pp** (exploratory, one seed; Gated still weak at that step).
- Final checkpoint (28800): +0.444 pp on **dev**; **+0.160 pp on test**.
- Early advantage **exists** on some mid-training evals, **does not persist** as a large gap; two later evals show large **negative** P1 dips (16200, 19800).
- Gated pulls ahead on **dev cell** from ~18000 onward and finishes highest on both **dev** and **test**.

Exact-grid on dev is mostly 0.00–0.02; too sparse for a trajectory claim beyond “near floor.”

---

## 8. Mechanistic diagnostics

**CANONICAL ACT6 (this campaign), P1 seed 0, test clean:**

| Diagnostic | Value |
|----------|--------|
| expected_lookback | 2.3143187450865903 |
| non_adjacent_mass | 0.6021175030618906 |
| entropy | 1.0117093411584694 |
| gate logit / σ(gate) | MISSING |
| history-position histogram | MISSING |
| gate-off ablation | MISSING |

Deletion ablations (same P1 checkpoint, test):

| Intervention | Exact | Cell | Lookback | Non-adj | Entropy |
|--------------|-------|------|----------|---------|---------|
| clean | 0.009 | 0.6389876543209877 | 2.3143187450865903 | 0.6021175030618906 | 1.0117093411584694 |
| delete_most_attended | 0.009 | 0.6394444444444445 | 2.107556518788139 | 0.5637390532841285 | 0.7426138754623631 |
| delete_least_attended | 0.009 | 0.639074074074074 | 2.293452732885877 | 0.6146708337279657 | 0.7435189671814442 |

Deleting most- or least-attended history **does not change exact accuracy** (still 0.009); cell moves by &lt;0.05 pp.

Gaussian σ=0.05/0.10/0.20: exact stays 0.009 for P1; cell stays ~0.639. Gated exact 0.012/0.013/0.013.

**Not used here:** ACT16 / 8h attention stats; Colab diagnostics.

---

## 9. Compute and memory cost

| Model | Params | Train wall (s) | Train slowdown vs Vanilla | Train ex/s | Train peak VRAM (B) | Eval wall (s) | Eval ex/s | Eval slowdown vs Vanilla | Eval peak VRAM (B) |
|-------|--------|----------------|---------------------------|------------|----------------------|---------------|----------|--------------------------|---------------------|
| Vanilla | 1710082 | 3812.0966819210007 | 1.000 | 241.75672258542644 | 1069243392 | 29.053754566000862 | 54.11548714852564 | 1.000 | 121707520 |
| Gated | 1710083 | 4246.843711241001 | 1.114 | 217.00822131989702 | 1176866304 | 23.33993965300033 | 78.8313440794983 | 0.803 | 137077248 |
| HistoryAttention | 1775619 | 4813.044401752 | 1.263 | 191.47963805705336 | 1208774656 | 34.23273052500008 | 42.14370563648292 | 1.284 | 113478144 |
| Parameter-Matched | 1775619 | 4003.4161828899996 | 1.050 | 230.20339577453382 | 1155308032 | 22.395344807002402 | 84.29687108617983 | 0.771 | 124624384 |

Train slowdown = runtime / Vanilla runtime. Eval slowdown = Vanilla eval throughput / variant throughput.

HistoryAttention is the slowest to train (~26% longer) and slowest to eval (~28% lower throughput). Gated eval throughput is **higher** than Vanilla in this measurement (single eval pass; not a claim about architecture FLOPs).

---

## 10. CPU vs GPU comparison

Frozen CPU facts from the prompt (not recomputed). **Do not compare absolute CPU vs GPU accuracy.**

| Behavior | CPU (prompt, Track B) | GPU ACT6 seed 0 (this report) |
|---------|------------------------|--------------------------------|
| Final Attention vs Vanilla cell | −0.407 pp (5 seeds, CI crosses 0) | **+0.160 pp** (1 seed, test) |
| Final Gated vs Vanilla | −0.120 pp | **+2.807 pp** (1 seed, test) |
| Parameter-Matched vs Vanilla | −0.438 pp | +0.602 pp (1 seed) |
| Early Attention advantage | yes (steps 1250/2500) | yes on some **dev** evals (esp. 3600, 7200); **does not persist** as a large final gap |
| Absolute cell | ~45–46% | ~64–67% test (different data/arch/budget) |

GPU Gated leading Vanilla at the end **differs** from CPU Track B (near-null / slight Gated lag). GPU HistoryAttention vs Vanilla is a **tiny** positive, consistent in *direction* with “not a large final win,” unlike CPU Track A outer-history (+0.73 pp).

---

## 11. Scientific interpretation

1. **Exact-grid vs Vanilla:** HistoryAttention **does not improve** final test exact-grid (both 0.009). **Does not support** an exact-solve gain.
2. **Cell vs Vanilla:** HistoryAttention is **+0.16 pp** on one seed. Supports only a **negligible** point estimate, **statistically inconclusive**.
3. **Consistency across seeds:** **Cannot assess** (n=1).
4. **Simple history access:** Gated **+2.81 pp cell** and **+0.3 pp exact** vs Vanilla on this seed **suggests** uniform history can help **in this ACT6 screening run**. Not generalizable without more seeds.
5. **Selectivity beyond uniform:** HistoryAttention **underperforms Gated** (−2.65 pp cell, −0.3 pp exact). **Does not support** learned temporal attention beating gated mean on this run.
6. **Extra capacity:** B3 matches P1 params, **+0.60 pp cell vs Vanilla** but **worse exact** (0.001 vs 0.009). P1 vs B3 cell is **slightly negative**. Capacity **does not explain** Gated’s gain (Gated has +1 param).
7. **vs CPU:** Qualitative **disagreement** on Gated (GPU help vs CPU null). Attention remains **not a clear final winner** on both.
8. **Optimization that vanishes:** **Yes, directionally:** large mid-training P1−Vanilla **dev** gaps shrink by the end; two late P1 crashes on dev. Exploratory only.
9. **History branch active?** Attention mass is spread (lookback ~2.31, ~60% non-adjacent). Deletion ablations **do not** change exact accuracy → **weak evidence** that attended states drive the prediction. Gate value **MISSING**.
10. **Strongest defensible conclusion:** On **one ACT6 GPU seed**, **Gated uniform history** is the only variant with a **clear** test cell (and small exact) edge over Vanilla. **HistoryAttention does not outperform Vanilla on exact-grid and is essentially tied on cell.** Results are **screening / incomplete** relative to a 3-seed protocol. Do not report as a confirmatory GPU finding.

---

## 12. Paper-ready table

**Single-seed ACT6 screening (GTX 1080 Ti, seed 0, test, last checkpoint). Not a multi-seed result.**

| Model | Exact (%) | Cell (%) | Params | Δ cell vs Vanilla (pp) |
|-------|-----------|----------|--------|-------------------------|
| Vanilla TRM | 0.9 | 63.74 | 1.710M | — |
| Gated Uniform History | 1.2 | 66.55 | 1.710M | +2.81 |
| HistoryAttention | 0.9 | 63.90 | 1.776M | +0.16 |
| Parameter-Matched | 0.1 | 64.34 | 1.776M | +0.60 |

---

## 13. Paper-ready paragraph

On a resource-adapted Sudoku split (`sudoku-study-v1`), we trained four D256 TRM variants (H3/L6/L2, ACT6, Transformer+RoPE, AdamW) for 28,800 steps on one GTX 1080 Ti seed. Test exact-grid accuracy remained near 1% (Vanilla 0.9%, HistoryAttention 0.9%, Gated 1.2%, parameter-matched 0.1%). Gated uniform history improved cell accuracy by 2.81 percentage points over Vanilla; HistoryAttention did not improve exact-grid accuracy and differed by only +0.16 pp in cell accuracy. HistoryAttention and the parameter-matched control used the same 65,537 extra parameters. These figures are from a single seed and a last-step checkpoint; they are not a multi-seed confirmatory estimate.

---

## 14. Missing or unresolved evidence

- Seeds **1 and 2** for this ACT6 recipe
- Test **LM loss** and **q_halt** in eval metadata
- Learned **gate logit / sigmoid**
- Gate-off inference ablation
- Dataset **manifest / leakage hashes** in this clone
- Evaluation **git commit**
- OS name
- Per-example McNemar
- 4090 seed-1 ACT6 not merged (separate machine; not in `results/canonical-gpu/`)

ACT6_EVIDENCE_STATUS: INCOMPLETE

Reason: intended primary seed set is {0,1,2}; only seed 0 is present. Test LM loss and gate values are also missing. Seed-0 four-variant test metrics **are** verified.

```json
{
  "configuration": {
    "preset": "canonical",
    "label": "SCREENING",
    "gpu": "NVIDIA GeForce GTX 1080 Ti",
    "seed_completed": [0],
    "halt_max_steps": 6,
    "hidden_size": 256,
    "H_cycles": 3,
    "L_cycles": 6,
    "L_layers": 2,
    "history_rank": 64,
    "history_heads": 4,
    "history_gate_init": -2.0,
    "global_batch_size": 32,
    "epochs": 1024,
    "total_steps": 28800,
    "eval_checkpoint": "step_28800.pt",
    "train_commit": "d37a1071a89c56bc3a741a5b6b2e859ca657e177"
  },
  "per_seed_results": {
    "test_clean": {
      "B0": {"seed": 0, "exact_accuracy": 0.009, "cell_accuracy": 0.6373827160493827},
      "Gated": {"seed": 0, "exact_accuracy": 0.012, "cell_accuracy": 0.6654567901234568},
      "P1": {"seed": 0, "exact_accuracy": 0.009, "cell_accuracy": 0.6389876543209877},
      "B3": {"seed": 0, "exact_accuracy": 0.001, "cell_accuracy": 0.6434074074074074}
    }
  },
  "aggregate_results": {
    "n_seeds": 1,
    "note": "no sample SD"
  },
  "paired_deltas": {
    "P1_minus_B0_cell_pp": 0.160493827160495,
    "Gated_minus_B0_cell_pp": 2.80740740740741,
    "B3_minus_B0_cell_pp": 0.60246913580247,
    "P1_minus_Gated_cell_pp": -2.64691358024691,
    "P1_minus_B3_cell_pp": -0.44197530864197
  },
  "parameter_counts": {
    "B0": 1710082,
    "Gated": 1710083,
    "P1": 1775619,
    "B3": 1775619
  },
  "runtime": {
    "train_seconds": {"B0": 3812.0966819210007, "P1": 4813.044401752, "Gated": 4246.843711241001, "B3": 4003.4161828899996},
    "eval_seconds_clean": {"B0": 29.053754566000862, "Gated": 23.33993965300033, "P1": 34.23273052500008, "B3": 22.395344807002402}
  },
  "vram": {
    "train_peak_bytes": {"B0": 1069243392, "P1": 1208774656, "Gated": 1176866304, "B3": 1155308032},
    "eval_peak_bytes_clean": {"B0": 121707520, "Gated": 137077248, "P1": 113478144, "B3": 124624384}
  },
  "mechanistic_diagnostics": {
    "P1_seed0_clean": {
      "expected_lookback": 2.3143187450865903,
      "non_adjacent_mass": 0.6021175030618906,
      "entropy": 1.0117093411584694
    }
  }
}
```
