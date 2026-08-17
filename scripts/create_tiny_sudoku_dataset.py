"""Create a deterministic toy Sudoku dataset for a local CPU smoke test.

This dataset only checks that preprocessing, training, evaluation, and
checkpointing work. It must not be used for research claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BASE_SOLUTION = np.array(
    [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ],
    dtype=np.uint8,
)


def make_examples(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for _ in range(count):
        digit_map = np.pad(rng.permutation(np.arange(1, 10)), (1, 0))
        solution = digit_map[BASE_SOLUTION]
        puzzle = solution.copy()
        puzzle.flat[rng.choice(81, size=48, replace=False)] = 0
        inputs.append(puzzle.reshape(-1))
        labels.append(solution.reshape(-1))

    # The TRM format reserves token 0 for padding, so Sudoku values are shifted.
    return np.stack(inputs) + 1, np.stack(labels) + 1


def write_split(root: Path, split: str, count: int, seed: int) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    inputs, labels = make_examples(count, seed)

    metadata = {
        "seq_len": 81,
        "vocab_size": 11,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": count,
        "mean_puzzle_examples": 1,
        "total_puzzles": count,
        "sets": ["all"],
    }
    (split_dir / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    arrays = {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": np.zeros(count, dtype=np.int32),
        "puzzle_indices": np.arange(count + 1, dtype=np.int32),
        "group_indices": np.arange(count + 1, dtype=np.int32),
    }
    for name, values in arrays.items():
        np.save(split_dir / f"all__{name}.npy", values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/sudoku-tiny"))
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--test-size", type=int, default=4)
    args = parser.parse_args()

    write_split(args.output_dir, "train", args.train_size, seed=17)
    write_split(args.output_dir, "test", args.test_size, seed=29)
    (args.output_dir / "identifiers.json").write_text('["<blank>"]\n', encoding="utf-8")
    print(f"Created CPU smoke-test dataset at {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

