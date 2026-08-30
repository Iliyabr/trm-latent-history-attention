#!/usr/bin/env python
"""Extract final scalar gate values from frozen L-cycle checkpoints."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch


SPECS = {
    "attention": "model.inner.lcycle_history_attention.gate_logit",
    "gated": "model.inner.lcycle_gated_history.gate_logit",
    "parammatched": "model.inner.lcycle_param_matched.gate_logit",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    args = ap.parse_args()

    init_gate = 1.0 / (1.0 + math.exp(2.0))
    print(f"initial gate: {init_gate:.6f}")

    for method, key in SPECS.items():
        values = []
        print(f"\n===== {method.upper()} =====")
        for seed in range(5):
            path = (
                args.checkpoint_root
                / f"proposal-h3l6-l2-{method}-40ep-seed{seed}"
                / "step_10000"
            )
            state = torch.load(path, map_location="cpu", weights_only=False)
            if key not in state:
                candidates = [k for k in state if "gate_logit" in k.lower()]
                raise KeyError(f"{path}: missing {key}; candidates={candidates}")

            logit = float(state[key])
            gate = 1.0 / (1.0 + math.exp(-logit))
            values.append(gate)
            print(f"seed{seed}: logit={logit:+.6f} gate={gate:.6f}")

        arr = np.asarray(values)
        print(
            f"mean={arr.mean():.6f} "
            f"sample_sd={arr.std(ddof=1):.6f} "
            f"ratio_to_init={arr.mean()/init_gate:.3f}x"
        )


if __name__ == "__main__":
    main()
