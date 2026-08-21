# Latent History Module Interface

This directory defines the shared interface for all TRM latent-history aggregation methods.

## Scientific purpose

The TRM core records detached latent states from previous outer reasoning steps.

For outer step t, the valid history contains only:

z_1, z_2, ..., z_(t-1)

The current state z_t is never inserted into the history before aggregation.

This preserves causal history semantics.

## Tensor contract

Every history aggregator receives:

- current_z: [B, L, D]
- history_z: [B, K, L, D]
- history_lengths: [B]

where:

- B = batch size
- K = maximum outer reasoning steps
- L = sequence length
- D = hidden size

For sample b, only the entries from index 0 up to history_lengths[b] are valid historical states.

The aggregator must return:

- shape: [B, L, D]
- dtype: same as current_z

## Gradient policy

Historical states are intentionally detached before storage.

Therefore history modules receive detached previous outer-step states.

This preserves the original TRM truncated-gradient behavior and prevents HistoryAttention from silently introducing backpropagation through the full outer reasoning trajectory.

Changing this policy must be treated as a separate ablation.

## Reset policy

When a batch slot is reused for a new puzzle:

- its history buffer is cleared
- its history length becomes zero before the new state is computed

This prevents history leakage across puzzles.

## Integration contract

History aggregation is called before the newly computed current state is appended to the history.

Conceptually:

1. Compute current z_H
2. Aggregate using strictly previous history
3. Produce updated z_H
4. Detach updated z_H
5. Append it to history
6. Produce and carry outputs

## Baseline implementation

NoHistoryAggregator is the identity control.

Its forward method simply returns current_z.

It:

- ignores history
- adds zero parameters
- introduces no arithmetic
- preserves bitwise-equivalent vanilla TRM logits

This behavior is covered by regression tests.

## Adding a new aggregator

Implement a subclass of HistoryAggregator in a new module.

The forward interface must accept:

current_z
history_z
history_lengths

and return an updated latent tensor with shape [B, L, D].

Then register the implementation in:

models/history/factory.py

Do not modify the TRM recursion logic unless the shared interface is insufficient.

## Team ownership

### HistoryAttention branch

The HistoryAttention implementation should preferably modify only:

- models/history/attention.py
- models/history/factory.py
- tests/test_history_attention.py

Avoid modifying:

- models/recursive_reasoning/trm.py
- pretrain.py
- dataset generation
- Phase-1 baseline configurations

If the current interface is insufficient, coordinate the interface change before modifying the TRM core.

### Non-attention baselines

Alternative history methods such as:

- uniform mean history
- recency-weighted history
- gated history

should implement the same interface.

This ensures all methods are compared through exactly the same TRM integration path.

## Current validated invariants

The following tests already pass:

- history recording does not change vanilla logits
- stored history is detached
- causal history length is correct
- reset isolation prevents cross-puzzle leakage
- NoHistoryAggregator adds zero parameters
- Phase-1 baseline regression still passes
