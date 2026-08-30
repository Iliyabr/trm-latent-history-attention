# Latent-History Modules

This directory contains **two separate latent-history research tracks**.
They must not be mixed in experiments or paper claims.

## Track A — outer-step `z_H` history

Supporting mechanism study.

Relevant modules include:

```text
base.py
factory.py
none.py
uniform.py
recency.py
last_state.py
gated.py
attention.py
```

Track A aggregates previous **outer/ACT-step** latent states. Historical states
are detached before storage to preserve the truncated-gradient semantics of the
outer recurrent process. The current state is not inserted into history before
retrieval, and history is reset between puzzles.

Track A produced a reproducible positive result at reduced recursion depth 8,
but did not show a monotonic benefit with depth.

## Track B — within-H-cycle `z_L` history

This is the **primary canonical proposal track**.

Relevant modules:

```text
lcycle_lowrank_attention.py
lcycle_gated.py
lcycle_param_matched.py
```

Track-B history is local to each H-cycle.

Canonical semantics:

```text
history = [z_L at H-cycle entry]

for each L-step:
    READ from history
    UPDATE z_L
    APPEND new z_L
```

Frozen invariants:

```text
history location = within-H-cycle z_L
order = READ -> UPDATE -> APPEND
history reset = every H-cycle
history carried through ACT = no
history leakage between puzzles = no
current updated state visible to same-step retrieval = no
```

For `L_cycles=6`, the queried causal history lengths are `1,2,3,4,5,6`.

## Canonical Low-Rank HistoryAttention

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

Normalization is pre-QKV in the D-dimensional latent space.

CPU:

```text
D = 64
rank = 16
temporal_heads = 4
gate_logit_init = -2
added parameters = 4,097
```

Canonical GPU:

```text
D = 256
rank = 64
temporal_heads = 4
gate_logit_init = -2
added parameters = 65,537
```

## Gated Uniform History

`lcycle_gated.py` receives the same causal within-cycle history as
HistoryAttention but removes learned temporal selection. It answers whether
history access helps without query-dependent retrieval.

## Parameter-Matched No-History

`lcycle_param_matched.py` receives **no history tensor**. It uses an added
current-state low-rank branch matched to the parameter overhead of
HistoryAttention. It controls for increased capacity.

## Integration

The Track-B modules are integrated in:

```text
models/recursive_reasoning/trm.py
```

Configuration fields:

```text
lcycle_history_enabled
lcycle_history_method
lcycle_history_rank
lcycle_history_heads
lcycle_history_gate_init
lcycle_history_pre_norm
```

Supported canonical method names:

```text
attention
gated
parameter_matched
param_matched
```

## Tests

Run:

```powershell
python -m pytest tests/test_lcycle_lowrank_history.py -q
```

Frozen observed result:

```text
4 passed
```

## Scientific rule

Track A is supporting mechanistic evidence. Track B is the primary contribution.

Do not average the two tracks, relabel one as a replication of the other, or
use an old outer-step history module as a canonical within-cycle control.
