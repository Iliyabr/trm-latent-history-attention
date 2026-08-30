# ACT16 Seed-0 Sensitivity Report

**Scope:** exploratory single-seed comparison of ACT `halt_max_steps=16` (`canonical_8h` preset) against matched ACT6 seed-0 screening (`halt_max_steps=6`).  
**Sources:** `results/canonical-gpu-8h/` (ACT16 test eval + analysis), `results/canonical-gpu/` (ACT6 seed-0 test eval), `config/experiment/sudoku_study_canonical_8h.yaml`, `config/experiment/sudoku_study_canonical.yaml`, `outputs/canonical/*/all_config.yaml`, `scripts/run_canonical_8h_server.sh`, `scripts/run_canonical_server.sh`.

**Not mixed in:** multi-seed ACT6 aggregates, Colab/CPU pilots, extended L_cycle (`L_cycles=10`), 4090 seed-1.

**Confound warning:** ACT16 and ACT6 differ in training budget, stopping rule, and eval checkpoint selection (see §2). Treat all ACT6-vs-ACT16 deltas as **single-seed, confounded, exploratory**.

---

## 1. Provenance

| Field | ACT16 seed 0 | Source |
|-------|--------------|--------|
| Git branch | MISSING | no `outputs/study-8h/.../provenance.txt` in repo |
| Git commit (training) | MISSING | — |
| Git commit (evaluation) | MISSING | — |
| GPU | NVIDIA GeForce GTX 1080 Ti (inferred) | checkpoint paths `/home/mahyar/...`; same host as ACT6 `outputs/canonical/provenance.txt` |
| CUDA | MISSING | no `run_metadata.json` for 8h runs |
| PyTorch | MISSING | — |
| Python | MISSING | — |
| Dataset | `data/sudoku-study-v1` | experiment config |
| Dataset split | train / dev / test (paths in config) | config |
| Train / dev / test counts | MISSING in this clone | `data/` gitignored |
| Augmentation | `shuffle_sudoku` (canonical builder default) | dataset builder convention; not re-verified from npy |
| Seed | **0 only** | all eval metadata |
| Physical batch | 32 | `global_batch_size: 32` |
| Gradient accumulation | none | no accumulation in `pretrain.py` |
| Effective batch | 32 | same |
| Training steps / epochs | **MISSING** (runtime-capped; no `metrics.jsonl` in repo) | intended max 8192 epochs; actual steps unknown |
| Runtime cap (intended) | **120 min/model** for `train-all` | `scripts/run_canonical_8h_server.sh` (`TRAIN_ALL_RUNTIME_MINUTES=120`) |
| Optimizer | AdamW | config |
| Learning rate | 1e-4 | config |
| Weight decay | 0.1 | config |
| Betas | (0.9, 0.95) | config |
| EMA | true, 0.999 | config |
| ACT / halt_max_steps | **16** | config + script override |
| Stopping rule | wall-clock cap (`max_runtime_minutes`) + best-dev tracking | config + script |
| Eval checkpoint | **`best_dev.pt`** | test `metadata.json` checkpoint paths |

**Variants completed (ACT16 seed 0, clean test eval present):**

| Protocol name | Variant ID | Status |
|---------------|------------|--------|
| Vanilla | B0 | **COMPLETED** |
| HistoryAttention | P1 | **COMPLETED** |
| Gated Uniform History | Gated | **COMPLETED** |
| Parameter-Matched No-History | B3 | **COMPLETED** |

Seeds 1 and 2: **NOT RUN** for ACT16.

---

## 2. Configuration audit

### ACT16 vs ACT6 seed-0 — field comparison

Architectural fields verified from `outputs/canonical/B0-seed0/all_config.yaml` (ACT6) and `config/experiment/sudoku_study_canonical_8h.yaml` + `config/arch/trm_history_canonical.yaml` + `run_canonical_8h_server.sh` overrides (ACT16).

| Field | ACT6 seed 0 | ACT16 seed 0 | Match? |
|-------|-------------|--------------|--------|
| D (`hidden_size`) | 256 | 256 | yes |
| H_cycles | 3 | 3 | yes |
| L_cycles | 6 | 6 | yes |
| L_layers | 2 | 2 | yes |
| Spatial heads | 4 | 4 | yes |
| History method | per-variant (B0/P1/Gated/B3) | same | yes |
| History rank | 64 | 64 | yes |
| History heads | 4 | 4 | yes |
| History gate init | −2.0 | −2.0 | yes |
| Pre-QKV normalization (P1) | enabled (code) | enabled (code) | yes |
| Optimizer | AdamW | AdamW | yes |
| Learning rate | 1e-4 | 1e-4 | yes |
| Weight decay | 0.1 | 0.1 | yes |
| Betas | (0.9, 0.95) | (0.9, 0.95) | yes |
| EMA | 0.999 | 0.999 | yes |
| Physical batch | 32 | 32 | yes |
| Gradient accumulation | none | none | yes |
| Effective batch | 32 | 32 | yes |
| Dataset path | `data/sudoku-study-v1` | `data/sudoku-study-v1` | yes |
| Augmentation | canonical builder | canonical builder | assumed same |
| Seed | 0 | 0 | yes |
| `mlp_t` | false | false | yes |
| `pos_encodings` | rope | rope | yes |
| `forward_dtype` | float32 | float32 | yes |
| `no_ACT_continue` | true | true | yes |
| **ACT halt_max_steps** | **6** | **16** | **no (intended)** |
| **Training epochs (cap)** | **1024** | **8192** | **no** |
| **max_runtime_minutes** | **null (uncapped)** | **120** (`train-all`) | **no** |
| **Training steps completed** | **28800** (all four) | **MISSING** | **no** |
| **Checkpoint selection (eval)** | **last `step_28800.pt`** | **`best_dev.pt`** | **no** |
| `checkpoint_every_eval` | false | false (script override) | yes |
| `project_name` | trm-latent-history-canonical | trm-latent-history-canonical-8h | no (label only) |

### Confounds beyond ACT budget

The ACT6-vs-ACT16 comparison is **confounded** by at least:

1. **Different eval checkpoints** — ACT6 uses the final training step; ACT16 uses dev-best exact accuracy.
2. **Different training duration** — ACT6 fixed 28 800 steps (~1.1–1.3 h/model); ACT16 runtime-capped (~2 h/model intended) with unknown actual step count.
3. **Different ACT ceiling at train and test time** — models were trained and evaluated under different `halt_max_steps`.

Do **not** attribute observed deltas solely to “more ACT steps at inference.”

---

## 3. ACT16 final results

**Test split, intervention `clean`, seed 0.** Raw values from `results/canonical-gpu-8h/<VARIANT>/seed_0/metadata.json`.

| Model | Seed | Checkpoint | Exact-grid | Cell/token | LM loss (test) | q_halt acc (test) | q_halt loss (test) | Mean inference steps† | Train wall-clock | Eval wall (s) | Throughput (ex/s) | Peak VRAM eval (B) | Params | Gate |
|-------|------|------------|------------|------------|----------------|-------------------|--------------------|-----------------------|------------------|---------------|-------------------|---------------------|--------|------|
| Vanilla (B0) | 0 | `.../B0-seed0/best_dev.pt` | 0.025 | 0.6789382716049382 | MISSING | MISSING | MISSING | 48.0 | MISSING | 57.64701814799628 | 34.14161712462803 | 121707520 | 1710082 | n/a |
| HistoryAttention (P1) | 0 | `.../P1-seed0/best_dev.pt` | 0.019 | 0.6715802469135802 | MISSING | MISSING | MISSING | 48.0 | MISSING | 76.39082665800379 | 20.77835783081847 | 113803776 | 1775619 | MISSING |
| Gated | 0 | `.../Gated-seed0/best_dev.pt` | 0.016 | 0.6426296296296297 | MISSING | MISSING | MISSING | 48.0 | MISSING | 62.35490075200505 | 29.280170858081284 | 137077248 | 1710083 | MISSING |
| Parameter-Matched (B3) | 0 | `.../B3-seed0/best_dev.pt` | 0.016 | 0.6386419753086419 | MISSING | MISSING | MISSING | 48.0 | MISSING | 59.88842526100052 | 31.489246440078794 | 124919296 | 1775619 | n/a |

† **Mean inference steps** = mean of `trajectory.steps` over 1000 test examples in `examples.jsonl` (see §6). This is **not** the same field as training-dev `metrics.steps` (ACT halting count).

Exact puzzle counts: B0 25/1000; P1 19/1000; Gated 16/1000; B3 16/1000.

---

## 4. ACT6 seed0 vs ACT16 seed0

Single-seed deltas: **ACT16 minus ACT6**. No p-values or confidence intervals.

| Model | Metric | ACT6 seed 0 | ACT16 seed 0 | Delta (ACT16 − ACT6) |
|-------|--------|-------------|--------------|----------------------|
| B0 | Exact-grid | 0.009 | 0.025 | **+0.016** |
| B0 | Cell accuracy (pp) | 63.738% | 67.894% | **+4.156 pp** |
| B0 | LM loss | MISSING | MISSING | MISSING |
| B0 | Mean inference steps† | 18.0 | 48.0 | **+30.0** |
| B0 | Eval wall (s) | 29.053754566000862 | 57.64701814799628 | +28.59 s (**×1.984**) |
| B0 | Throughput (ex/s) | 54.11548714852564 | 34.14161712462803 | −19.97 ex/s (**×0.631**) |
| B0 | Peak VRAM eval (B) | 121707520 | 121707520 | 0 |
| P1 | Exact-grid | 0.009 | 0.019 | **+0.010** |
| P1 | Cell accuracy (pp) | 63.899% | 67.158% | **+3.259 pp** |
| P1 | LM loss | MISSING | MISSING | MISSING |
| P1 | Mean inference steps† | 18.0 | 48.0 | **+30.0** |
| P1 | Eval wall (s) | 34.23273052500008 | 76.39082665800379 | +42.16 s (**×2.232**) |
| P1 | Throughput (ex/s) | 42.14370563648292 | 20.77835783081847 | −21.37 ex/s (**×0.493**) |
| P1 | Peak VRAM eval (B) | 113478144 | 113803776 | +325632 |
| Gated | Exact-grid | 0.012 | 0.016 | **+0.004** |
| Gated | Cell accuracy (pp) | 66.546% | 64.263% | **−2.283 pp** |
| Gated | Mean inference steps† | 18.0 | 48.0 | **+30.0** |
| Gated | Eval wall (s) | 23.33993965300033 | 62.35490075200505 | +39.01 s (**×2.670**) |
| Gated | Throughput (ex/s) | 78.8313440794983 | 29.280170858081284 | −49.55 ex/s (**×0.371**) |
| Gated | Peak VRAM eval (B) | 137077248 | 137077248 | 0 |
| B3 | Exact-grid | 0.001 | 0.016 | **+0.015** |
| B3 | Cell accuracy (pp) | 64.341% | 63.864% | **−0.477 pp** |
| B3 | Mean inference steps† | 18.0 | 48.0 | **+30.0** |
| B3 | Eval wall (s) | 22.395344807002402 | 59.88842526100052 | +37.49 s (**×2.674**) |
| B3 | Throughput (ex/s) | 84.29687108617983 | 31.489246440078794 | −52.81 ex/s (**×0.374**) |
| B3 | Peak VRAM eval (B) | 124624384 | 124919296 | +294912 |

**Relative ranking change (cell accuracy):** ACT6 seed 0: Gated > B3 > P1 > B0. ACT16 seed 0: **B0 > P1 > Gated > B3**. Vanilla moves from last to first on cell accuracy under the longer ACT16 recipe.

---

## 5. Within-ACT16 model comparisons

**SINGLE-SEED EXPLORATORY RESULT** — all deltas below.

| Comparison | Δ exact-grid | Δ cell (pp) | Δ LM loss | Eval Δ wall (s) | Eval Δ VRAM (B) |
|------------|--------------|-------------|-----------|-----------------|-----------------|
| HistoryAttention − Vanilla | −0.006 | −0.736 | MISSING | +18.74 (P1 − B0) | −7903744 |
| Gated − Vanilla | −0.009 | −3.631 | MISSING | +4.71 | +15369728 |
| Parameter-Matched − Vanilla | −0.009 | −4.030 | MISSING | +2.24 | +3211776 |
| HistoryAttention − Gated | +0.003 | +2.895 | MISSING | +14.04 | −23273472 |
| HistoryAttention − Parameter-Matched | +0.003 | +3.294 | MISSING | +16.50 | −11115520 |

Under ACT16 seed 0, **Vanilla (B0) leads** both history variants on cell and exact accuracy. HistoryAttention is second on cell accuracy but still below Vanilla.

---

## 6. ACT and halting behavior

Derived from `results/canonical-gpu*/<VARIANT>/seed_0/examples.jsonl` (1000 test puzzles each).

### Test-time trajectory statistics

| Campaign | Variant | Mean `trajectory.steps` | Min | Max | Fraction at max | `len(q_halt_logits)` | Mean final q_halt logit |
|----------|---------|-------------------------|-----|-----|-----------------|----------------------|-------------------------|
| ACT6 | B0 | 18.0 | 18 | 18 | **100%** | 6 | −12.684 |
| ACT6 | P1 | 18.0 | 18 | 18 | **100%** | 6 | −13.615 |
| ACT6 | Gated | 18.0 | 18 | 18 | **100%** | 6 | −14.451 |
| ACT6 | B3 | 18.0 | 18 | 18 | **100%** | 6 | −14.168 |
| ACT16 | B0 | 48.0 | 48 | 48 | **100%** | 16 | −11.658 |
| ACT16 | P1 | 48.0 | 48 | 48 | **100%** | 16 | −12.850 |
| ACT16 | Gated | 48.0 | 48 | 48 | **100%** | 16 | −13.231 |
| ACT16 | B3 | 48.0 | 48 | 48 | **100%** | 16 | −16.219 |

**Interpretation (artifact-consistent):**

- `len(q_halt_logits)` equals `halt_max_steps` (6 vs 16) for every example → models **always run to the ACT ceiling** at test time; **no early halting** observed in either campaign.
- `trajectory.steps` = `halt_max_steps × H_cycles` (18 = 6×3; 48 = 16×3) for all 1000 examples → consistent with full ACT budget use across all H-cycles.
- ACT16 **does** increase inference micro-steps by **2.667×** (48/18) relative to ACT6 seed 0 at test time, because the ceiling rose from 6 to 16.
- Models **do not** halt well before step 16 under ACT16; they hit the maximum every time on this test pass.

### Training-time halting (dev eval)

| Campaign | Source | ACT steps at last dev eval | q_halt accuracy | q_halt loss |
|----------|--------|---------------------------|-----------------|-------------|
| ACT6 B0 | `outputs/canonical/B0-seed0/metrics.jsonl` step 28800 | 6.0 | 1.0 | 4.927961826324463 |
| ACT16 all variants | MISSING | MISSING | MISSING | MISSING |

Test q_halt accuracy / loss: **MISSING** (not stored in eval metadata).

HistoryAttention vs Vanilla halting: at test time both use the same ACT ceiling within each campaign; no differential early-stop pattern is visible in `examples.jsonl`.

---

## 7. Mechanistic diagnostics

**ACT16 HistoryAttention (P1), seed 0, test clean** — from eval metadata:

| Diagnostic | ACT16 seed 0 | ACT6 seed 0 (matched) | Δ (ACT16 − ACT6) |
|------------|--------------|----------------------|------------------|
| Expected lookback | 2.2784015834331512 | 2.3143187450865903 | −0.036 |
| Non-adjacent mass | 0.5965721873799339 | 0.6021175030618906 | −0.006 |
| Entropy | 1.0022417666041292 | 1.0117093411584694 | −0.009 |
| Gate logit / σ(gate) | MISSING | MISSING | — |
| History-position histogram | MISSING | MISSING | — |
| Gate-off ablation | MISSING | MISSING | — |

**ACT16 P1 deletion ablations (test):**

| Intervention | Exact | Cell | Lookback | Non-adj mass | Entropy |
|--------------|-------|------|----------|--------------|---------|
| clean | 0.019 | 0.6715802469135802 | 2.2784015834331512 | 0.5965721873799339 | 1.0022417666041292 |
| delete_most_attended | 0.019 | 0.6711358024691358 | 2.243015233660117 | 0.5885740388766862 | 0.73735032370314 |
| delete_least_attended | 0.019 | 0.6715308641975308 | 2.232315703528002 | 0.5735664624953642 | 0.7295592615846545 |

Deleting most- or least-attended history **does not change exact accuracy** (still 0.019). Cell accuracy moves by ≤0.044 pp.

Attention pattern under ACT16 remains **broad** (entropy ~1.0, non-adjacent mass ~60%, expected lookback ~2.28) and **numerically close** to ACT6 seed 0 — no large diagnostic shift from raising the ACT ceiling alone.

---

## 8. Compute cost

### Evaluation / inference (ACT16 vs ACT6 seed 0)

| Model | ACT6 eval wall (s) | ACT16 eval wall (s) | Slowdown | ACT6 throughput | ACT16 throughput | Throughput ratio |
|-------|-------------------|---------------------|----------|-----------------|------------------|------------------|
| B0 | 29.05 | 57.65 | ×1.98 | 54.12 ex/s | 34.14 ex/s | 0.631 |
| P1 | 34.23 | 76.39 | ×2.23 | 42.14 ex/s | 20.78 ex/s | 0.493 |
| Gated | 23.34 | 62.35 | ×2.67 | 78.83 ex/s | 29.28 ex/s | 0.371 |
| B3 | 22.40 | 59.89 | ×2.67 | 84.30 ex/s | 31.49 ex/s | 0.374 |

Mean inference steps at test: **18 → 48** (+167%). Eval latency scales roughly **2.0–2.7×** depending on variant (HistoryAttention slowest in absolute terms).

Peak eval VRAM: essentially unchanged for B0/Gated; +0.3 MB for P1; +0.3 MB for B3.

### Training

| Model | ACT6 train wall (s) | ACT16 train wall (s) |
|-------|---------------------|----------------------|
| All variants | 3812–4813 (from ACT6 report) | **MISSING** |

Maximum ACT steps (test): 6 → 16. Mean ACT steps used: **always at ceiling** in both campaigns.

**Cost–benefit (single-seed, confounded):** ACT16 buys **+1.0 to +1.6 pp exact** and **+3.3 to +4.2 pp cell** for Vanilla and HistoryAttention vs ACT6 seed 0, at **~2× eval compute**. Gated and Parameter-Matched **lose** cell accuracy vs their ACT6 seed-0 baselines despite similar exact-grid gains. Whether the accuracy lift is worth the compute cannot be settled on one seed and a confounded training recipe.

---

## 9. Scientific interpretation

1. **Does ACT16 seed 0 outperform ACT6 seed 0 for Vanilla?** **Suggestive yes** on this seed: exact +0.016 (9→25 puzzles), cell +4.16 pp. Confounded by longer training, different checkpoint, and higher ACT ceiling.
2. **Does ACT16 seed 0 outperform ACT6 seed 0 for HistoryAttention?** **Suggestive yes** on exact (+0.010) and cell (+3.26 pp), but **Vanilla gains more** — history does not uniquely benefit.
3. **Does the HistoryAttention−Vanilla gap change under ACT16?** **Yes, directionally.** ACT6 seed 0: P1 − B0 = +0.16 pp cell (test). ACT16 seed 0: P1 − B0 = **−0.74 pp** cell. The exploratory ACT16 run **does not** widen a history advantage; if anything Vanilla pulls ahead.
4. **Does additional ACT budget make explicit history more useful?** **Not on this seed.** Gated falls from +2.81 pp vs Vanilla (ACT6) to **−2.28 pp** (ACT16). P1 remains near tied or slightly below Vanilla.
5. **Does ACT16 mainly improve performance, change halting, or only increase compute?** **All three, but halting does not adapt:** models use the full new ceiling (16 steps) on every test example; accuracy moves modestly for some variants; eval compute **~2–2.7×** higher.
6. **Large enough to motivate multi-seed ACT16?** **Hypothesis-generating only.** Vanilla’s gain is the largest single-seed effect, but checkpoint and training-duration confounds block a clean ACT-isolation claim.
7. **Consistent with ACT6 story?** **Partially.** ACT6 seed 0 showed a small Gated cell lead; ACT16 seed 0 **reverses** that ranking. Both campaigns show **near-zero** HistoryAttention deletion sensitivity and **broad** temporal attention — that mechanistic picture is **directionally consistent**.

Language appropriate here: exploratory, single-seed, suggestive, hypothesis-generating. **Not** robust, reproducible, statistically significant, or proof.

---

## 10. Recommendation for final paper

**Recommendation: B — DISCUSSION / LIMITATIONS SENTENCE**

Rationale:

- Single seed only; primary evidence remains the ACT6 multi-seed campaign (when completed).
- ACT6-vs-ACT16 is **confounded** (training duration, checkpoint policy, ACT ceiling at train time).
- ACT16 is **scientifically informative** as a sensitivity probe — longer budget + dev-best checkpoint lifts Vanilla most and **inverts** the Gated screening lead — but too weak and confounded for a dedicated main-text figure.
- Mechanistic diagnostics and ablations remain flat under ACT16, supporting a limitations note rather than a new headline result.

Not recommended as **A (main-text mini-result)** unless multi-seed, matched-checkpoint ACT16 runs are completed. Not **C (repository-only)** because the ranking flip and compute–accuracy trade-off are worth one sentence in discussion.

---

## 11. Paper-ready paragraph

We ran an exploratory seed-0 sensitivity check raising the ACT inference ceiling from 6 to 16 steps (`halt_max_steps=16`) with a longer runtime-capped training schedule and dev-best checkpoint selection (GTX 1080 Ti). On the 1,000-puzzle test split, Vanilla exact accuracy rose from 0.9% to 2.5% and cell accuracy from 63.7% to 67.9%; HistoryAttention moved from 0.9%/63.9% to 1.9%/67.2%, while Gated cell accuracy fell from 66.5% to 64.3%. At evaluation every model used the full ACT budget (16 halting steps; no early stopping on test). Inference cost increased roughly twofold. Because this probe uses **only seed 0** and differs from the primary ACT6 recipe in training length and checkpoint selection, we treat it as hypothesis-generating context rather than confirmatory evidence for latent-history retrieval.

---

## 12. Missing evidence

| Item | Status |
|------|--------|
| ACT16 training `metrics.jsonl` / step count / train wall-clock | MISSING (not in repo) |
| ACT16 git commit, CUDA, PyTorch, Python versions | MISSING |
| Test LM loss, test q_halt accuracy/loss | MISSING |
| Learned gate logit / σ(gate) for Gated / P1 | MISSING |
| History-position distribution, gate-off ablation | MISSING |
| ACT16 dev halting curves | MISSING |
| Dataset counts / leakage verification in clone | MISSING |
| Multi-seed ACT16 runs | NOT RUN |
| Matched-checkpoint ACT6 vs ACT16 ablation (same train budget, only ACT changed) | NOT RUN |

---

```json
{
  "configuration": {
    "act6_seed0": {
      "halt_max_steps": 6,
      "epochs": 1024,
      "training_steps": 28800,
      "max_runtime_minutes": null,
      "eval_checkpoint": "step_28800.pt",
      "seed": 0
    },
    "act16_seed0": {
      "halt_max_steps": 16,
      "epochs": 8192,
      "training_steps": "MISSING",
      "max_runtime_minutes": 120,
      "eval_checkpoint": "best_dev.pt",
      "seed": 0,
      "variants_completed": ["B0", "P1", "Gated", "B3"]
    },
    "confounds": ["eval_checkpoint", "training_duration", "halt_max_steps_at_train"]
  },
  "act16_seed0_results": {
    "B0": {"exact": 0.025, "cell": 0.6789382716049382, "eval_wall_s": 57.64701814799628, "throughput": 34.14161712462803, "params": 1710082, "peak_vram_bytes": 121707520},
    "P1": {"exact": 0.019, "cell": 0.6715802469135802, "eval_wall_s": 76.39082665800379, "throughput": 20.77835783081847, "params": 1775619, "peak_vram_bytes": 113803776, "expected_lookback": 2.2784015834331512, "non_adjacent_mass": 0.5965721873799339, "entropy": 1.0022417666041292},
    "Gated": {"exact": 0.016, "cell": 0.6426296296296297, "eval_wall_s": 62.35490075200505, "throughput": 29.280170858081284, "params": 1710083, "peak_vram_bytes": 137077248},
    "B3": {"exact": 0.016, "cell": 0.6386419753086419, "eval_wall_s": 59.88842526100052, "throughput": 31.489246440078794, "params": 1775619, "peak_vram_bytes": 124919296}
  },
  "act6_seed0_results": {
    "B0": {"exact": 0.009, "cell": 0.6373827160493827, "eval_wall_s": 29.053754566000862, "throughput": 54.11548714852564, "params": 1710082, "peak_vram_bytes": 121707520},
    "P1": {"exact": 0.009, "cell": 0.6389876543209877, "eval_wall_s": 34.23273052500008, "throughput": 42.14370563648292, "params": 1775619, "peak_vram_bytes": 113478144, "expected_lookback": 2.3143187450865903, "non_adjacent_mass": 0.6021175030618906, "entropy": 1.0117093411584694},
    "Gated": {"exact": 0.012, "cell": 0.6654567901234568, "eval_wall_s": 23.33993965300033, "throughput": 78.8313440794983, "params": 1710083, "peak_vram_bytes": 137077248},
    "B3": {"exact": 0.001, "cell": 0.6434074074074074, "eval_wall_s": 22.395344807002402, "throughput": 84.29687108617983, "params": 1775619, "peak_vram_bytes": 124624384}
  },
  "act16_minus_act6": {
    "B0": {"delta_exact": 0.016, "delta_cell_pp": 4.155555555555556, "delta_inference_steps": 30.0, "eval_wall_ratio": 1.984, "throughput_ratio": 0.631},
    "P1": {"delta_exact": 0.010, "delta_cell_pp": 3.259259259259259, "delta_inference_steps": 30.0, "eval_wall_ratio": 2.232, "throughput_ratio": 0.493},
    "Gated": {"delta_exact": 0.004, "delta_cell_pp": -2.282716049382716, "delta_inference_steps": 30.0, "eval_wall_ratio": 2.670, "throughput_ratio": 0.371},
    "B3": {"delta_exact": 0.015, "delta_cell_pp": -0.476543209876543, "delta_inference_steps": 30.0, "eval_wall_ratio": 2.674, "throughput_ratio": 0.374}
  },
  "within_act16_deltas": {
    "P1_minus_B0": {"delta_exact": -0.006, "delta_cell_pp": -0.735802469135802},
    "Gated_minus_B0": {"delta_exact": -0.009, "delta_cell_pp": -3.630864197530864},
    "B3_minus_B0": {"delta_exact": -0.009, "delta_cell_pp": -4.02962962962963},
    "P1_minus_Gated": {"delta_exact": 0.003, "delta_cell_pp": 2.895061728395062},
    "P1_minus_B3": {"delta_exact": 0.003, "delta_cell_pp": 3.293827160493827}
  },
  "halting": {
    "act6_test": {"mean_trajectory_steps": 18.0, "q_halt_logits_len": 6, "fraction_at_max": 1.0},
    "act16_test": {"mean_trajectory_steps": 48.0, "q_halt_logits_len": 16, "fraction_at_max": 1.0},
    "early_halting_observed": false
  },
  "runtime": {
    "act16_train_wall_s": "MISSING",
    "act16_eval_slowdown_vs_act6_range": [1.98, 2.67]
  },
  "vram": {
    "act16_eval_peak_bytes": {"B0": 121707520, "P1": 113803776, "Gated": 137077248, "B3": 124919296}
  },
  "diagnostics": {
    "P1_act16": {"expected_lookback": 2.2784015834331512, "non_adjacent_mass": 0.5965721873799339, "entropy": 1.0022417666041292},
    "P1_act6": {"expected_lookback": 2.3143187450865903, "non_adjacent_mass": 0.6021175030618906, "entropy": 1.0117093411584694},
    "delete_ablation_changes_exact": false
  }
}
```

**ACT16_EVIDENCE_STATUS: INCOMPLETE**

(Incomplete because training logs, provenance, and test LM/q_halt metrics are missing from the repository; eval and halting artifacts for seed 0 are present for all four variants.)
