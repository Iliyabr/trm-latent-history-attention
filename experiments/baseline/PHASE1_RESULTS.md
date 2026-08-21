# Phase 1 — Reduced TRM Baseline Results

## Experimental scope

These experiments establish a reproducible CPU-reduced TRM baseline for later comparison with latent-history methods.

Canonical development configuration:

- Dataset: `sudoku-baseline-v2`
- Train originals: 1000
- Train augmentations: 10 per original
- Dev originals: 200
- Train/dev overlap: 0
- Official test set used for tuning: no
- Seed: 0
- Global batch size: 4
- Optimizer: AdamW
- Default hidden size: 64
- Default L_layers: 1
- Default H_cycles: 1
- Default L_cycles: 1
- Precision: float32
- Device: CPU

These are reduced development experiments and are not direct reproductions of the full paper-scale TRM results.

---

## 1. Outer recursion-depth study

### 20 epochs

| halt_max_steps | Dev accuracy | Exact accuracy | LM loss |
|---:|---:|---:|---:|
| 2 | 0.460123 | 0.0 | 1.313214 |
| 4 | 0.438951 | 0.0 | 1.401505 |
| 8 | 0.424506 | 0.0 | 1.447718 |

### Matched 40-epoch comparison

| halt_max_steps | Dev accuracy | Exact accuracy | LM loss |
|---:|---:|---:|---:|
| 2 | **0.482469** | 0.0 | **1.218538** |
| 4 | 0.458395 | 0.0 | 1.357913 |
| 8 | 0.428148 | 0.0 | 1.444620 |
| 16 | 0.421728 | 0.0 | 1.462849 |

Under the matched 40-epoch budget, development accuracy decreased monotonically as the maximum number of outer reasoning steps increased:

`2 > 4 > 8 > 16`.

This does not establish that deeper recursion is intrinsically harmful. It establishes that, in this reduced CPU regime and under this fixed optimization budget, increasing outer recursion alone did not improve development performance.

The 4-step regime is retained as the primary HistoryAttention development regime because it provides a non-trivial latent history while remaining substantially closer to the strongest 2-step reference than the longer-history settings.

---

## 2. Training-budget control

For the 4-step model:

| Epochs | Dev accuracy | LM loss |
|---:|---:|---:|
| 20 | 0.438951 | 1.401505 |
| 40 | **0.458395** | **1.357913** |

For the 8-step model:

| Epochs | Dev accuracy | LM loss |
|---:|---:|---:|
| 20 | 0.424506 | 1.447718 |
| 40 | **0.428148** | **1.444620** |

Additional optimization substantially improved the 4-step model, but produced only a small improvement for the 8-step model.

Therefore the project does not claim that additional training is generally ineffective.

---

## 3. TRM-4 / 40-epoch controlled tuning sweep

Reference configuration:

- `halt_max_steps = 4`
- `epochs = 40`
- `lr = 1e-3`
- `hidden_size = 64`
- `L_cycles = 1`

Each experiment changed only one factor relative to the reference.

| Experiment | Dev accuracy | LM loss |
|---|---:|---:|
| Reference | **0.458395** | **1.357913** |
| lr = 3e-4 | 0.456296 | 1.370709 |
| lr = 3e-3 | 0.428700 | 1.432477 |
| hidden_size = 96 | 0.428770 | 1.411830 |
| hidden_size = 128 | 0.430490 | 1.411394 |
| L_cycles = 2 | 0.434444 | 1.402203 |

Within the evaluated search space, the reference configuration remained the strongest TRM-4 / 40-epoch development baseline.

---

## 4. Capacity control

Measured trainable parameter counts:

| hidden_size | Trainable parameters | Dev accuracy |
|---:|---:|---:|
| 64 | **67,074** | **0.458395** |
| 96 | 112,898 | 0.428770 |
| 128 | 166,914 | 0.430490 |

Increasing hidden size from 64 to 96 increased the parameter count by approximately 68%, and increasing it from 64 to 128 increased it by approximately 149%.

Within this reduced regime and fixed optimization budget, these increases in model capacity did not improve development accuracy.

This result must not be generalized to the claim that larger TRM models are universally worse.

---

## 5. Baselines retained for later experiments

- **TRM-2 / 40 epochs:** performance-reference baseline.
- **TRM-4 / 40 epochs:** primary matched-history baseline for HistoryAttention.
- **TRM-8 / 40 epochs:** longer-history stress test.
- **TRM-16 / 40 epochs:** paper-aligned maximum-step stress test.

Primary TRM-4 development configuration:

- `halt_max_steps = 4`
- `epochs = 40`
- `lr = 1e-3`
- `hidden_size = 64`
- `L_layers = 1`
- `H_cycles = 1`
- `L_cycles = 1`
- `seed = 0`

---

## 6. Scientific limitations

- All current baseline-selection experiments use seed 0.
- Exact Sudoku accuracy is 0.0 in this reduced regime.
- Token/cell accuracy and LM loss therefore serve as development metrics, not final evidence of puzzle-solving performance.
- The official Sudoku test set has not been used for hyperparameter selection.
- The CPU-reduced architecture and training budget are substantially smaller than the original paper-scale TRM configuration.
- Multi-seed evaluation and final test-set evaluation are deferred until the model variants are frozen.

---

## Phase 1 conclusion

The baseline study found no development improvement from simply increasing outer recursion depth, hidden-state capacity, or internal `L_cycles` under the evaluated matched conditions.

This motivates the next hypothesis:

> Performance may depend not only on adding computation or capacity, but on using recursive latent information more effectively.

Phase 2 therefore introduces a latent-history interface while requiring exact disabled-mode equivalence to vanilla TRM before any HistoryAttention mechanism is enabled.
