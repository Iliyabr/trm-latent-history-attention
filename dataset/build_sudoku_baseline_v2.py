"""Build the deterministic Sudoku study dataset.

The split contract is deliberately explicit: select disjoint train/dev bases
before augmentation, keep the official test source isolated, and fail the
build if exact or digit-canonical puzzle identities leak across splits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from huggingface_hub import hf_hub_download

try:
    from dataset.build_sudoku_dataset import shuffle_sudoku
    from dataset.common import PuzzleDatasetMetadata
except ImportError:  # `python dataset/build_sudoku_baseline_v2.py`
    from build_sudoku_dataset import shuffle_sudoku
    from common import PuzzleDatasetMetadata


SOURCE_REPO = "sapientinc/sudoku-extreme"
SCHEMA_VERSION = 2


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            _, question, answer, _ = row
            if len(question) != 81 or len(answer) != 81:
                raise ValueError(f"Malformed Sudoku row in {path}")
            inputs.append(
                np.frombuffer(question.replace(".", "0").encode(), dtype=np.uint8)
                .reshape(9, 9) - ord("0")
            )
            labels.append(
                np.frombuffer(answer.encode(), dtype=np.uint8).reshape(9, 9)
                - ord("0")
            )
    return inputs, labels


def input_solution_hash(board: np.ndarray, solution: np.ndarray) -> str:
    """Hash the exact source pair, independent of numpy dtype/layout."""
    payload = bytes(board.astype(np.uint8).reshape(-1)) + bytes(
        solution.astype(np.uint8).reshape(-1)
    )
    return sha256_bytes(payload)


def canonical_hash(board: np.ndarray, solution: np.ndarray) -> str:
    """Hash after canonical digit relabeling.

    This catches exact puzzles under global digit permutations. Geometric
    Sudoku symmetries are intentionally not collapsed: their augmented forms
    are generated only after leakage checks and never enter dev/test.
    """
    flat_solution = solution.astype(np.uint8).reshape(-1)
    mapping = np.zeros(10, dtype=np.uint8)
    next_digit = 1
    for value in flat_solution:
        value = int(value)
        if mapping[value] == 0:
            mapping[value] = next_digit
            next_digit += 1
    canonical_board = mapping[board.astype(np.uint8).reshape(-1)]
    canonical_solution = mapping[flat_solution]
    return sha256_bytes(bytes(canonical_board) + bytes(canonical_solution))


def identity_records(
    inputs: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    indices: Iterable[int],
) -> list[dict[str, object]]:
    return [
        {
            "source_index": int(index),
            "input_solution_sha256": input_solution_hash(inputs[index], labels[index]),
            "canonical_sha256": canonical_hash(inputs[index], labels[index]),
        }
        for index in indices
    ]


def assert_no_leakage(records: dict[str, list[dict[str, object]]]) -> None:
    for hash_name in ("input_solution_sha256", "canonical_sha256"):
        split_hashes = {
            split: {str(record[hash_name]) for record in values}
            for split, values in records.items()
        }
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            overlap = split_hashes[left] & split_hashes[right]
            if overlap:
                raise AssertionError(
                    f"{hash_name} leakage between {left}/{right}: "
                    f"{len(overlap)} identities"
                )


def write_split(
    root: Path,
    split: str,
    source_inputs: Sequence[np.ndarray],
    source_labels: Sequence[np.ndarray],
    source_indices: Sequence[int],
    num_aug: int,
    augmentation_seed: int,
) -> None:
    results: dict[str, list[object]] = {
        key: []
        for key in (
            "inputs",
            "labels",
            "puzzle_identifiers",
            "puzzle_indices",
            "group_indices",
        )
    }
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    puzzle_id = 0
    example_id = 0

    # shuffle_sudoku uses numpy's legacy global generator.
    previous_state = np.random.get_state()
    np.random.seed(augmentation_seed)
    try:
        for source_index in source_indices:
            original_input = source_inputs[int(source_index)]
            original_label = source_labels[int(source_index)]
            for augmentation_index in range(num_aug + 1):
                if augmentation_index:
                    puzzle_input, puzzle_label = shuffle_sudoku(
                        original_input, original_label
                    )
                else:
                    puzzle_input, puzzle_label = original_input, original_label
                results["inputs"].append(puzzle_input)
                results["labels"].append(puzzle_label)
                example_id += 1
                puzzle_id += 1
                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(0)
            results["group_indices"].append(puzzle_id)
    finally:
        np.random.set_state(previous_state)

    def sequences(values: list[object]) -> np.ndarray:
        array = np.concatenate(values).reshape(len(values), -1)
        if not np.all((array >= 0) & (array <= 9)):
            raise AssertionError(f"Out-of-range value in {split}")
        return array.astype(np.uint8) + 1

    arrays = {
        "inputs": sequences(results["inputs"]),
        "labels": sequences(results["labels"]),
        "group_indices": np.asarray(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.asarray(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.asarray(
            results["puzzle_identifiers"], dtype=np.int32
        ),
    }
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=11,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(source_indices),
        mean_puzzle_examples=1,
        total_puzzles=len(source_indices) * (num_aug + 1),
        sets=["all"],
    )
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "dataset.json").write_text(
        json.dumps(metadata.model_dump(), indent=2) + "\n", encoding="utf-8"
    )
    for name, values in arrays.items():
        np.save(split_dir / f"all__{name}.npy", values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/sudoku-study-v1")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/sudoku_study_v1_manifest.json"),
    )
    parser.add_argument("--train-size", type=int, default=900)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--num-aug", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if min(args.train_size, args.dev_size, args.test_size) <= 0:
        raise ValueError("All split sizes must be positive")
    if args.num_aug < 0:
        raise ValueError("--num-aug must be non-negative")

    train_csv = Path(
        hf_hub_download(SOURCE_REPO, "train.csv", repo_type="dataset")
    )
    test_csv = Path(
        hf_hub_download(SOURCE_REPO, "test.csv", repo_type="dataset")
    )
    train_inputs, train_labels = load_csv(train_csv)
    test_inputs, test_labels = load_csv(test_csv)
    if args.train_size + args.dev_size > len(train_inputs):
        raise ValueError("Requested train/dev bases exceed the source training set")
    if args.test_size > len(test_inputs):
        raise ValueError("Requested test size exceeds the official test set")

    rng = np.random.default_rng(args.seed)
    selected = rng.choice(
        len(train_inputs), args.train_size + args.dev_size, replace=False
    )
    train_indices = np.sort(selected[: args.train_size])
    dev_indices = np.sort(selected[args.train_size :])
    # Bounded official test selection is deterministic and independent.
    test_indices = np.sort(
        np.random.default_rng(args.seed + 1).choice(
            len(test_inputs), args.test_size, replace=False
        )
    )
    if np.intersect1d(train_indices, dev_indices).size:
        raise AssertionError("Source-index leakage between train and dev")

    identities = {
        "train": identity_records(train_inputs, train_labels, train_indices),
        "dev": identity_records(train_inputs, train_labels, dev_indices),
        "test": identity_records(test_inputs, test_labels, test_indices),
    }
    assert_no_leakage(identities)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_split(
        args.output_dir,
        "train",
        train_inputs,
        train_labels,
        train_indices,
        args.num_aug,
        args.seed + 1000,
    )
    write_split(
        args.output_dir,
        "dev",
        train_inputs,
        train_labels,
        dev_indices,
        0,
        args.seed + 2000,
    )
    write_split(
        args.output_dir,
        "test",
        test_inputs,
        test_labels,
        test_indices,
        0,
        args.seed + 3000,
    )
    (args.output_dir / "identifiers.json").write_text(
        '["<blank>"]\n', encoding="utf-8"
    )

    generated_hashes = {
        str(path.relative_to(args.output_dir)).replace("\\", "/"): sha256_file(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": "sudoku-study-v1",
        "source": {
            "repo": SOURCE_REPO,
            "train_sha256": sha256_file(train_csv),
            "official_test_sha256": sha256_file(test_csv),
        },
        "seed": args.seed,
        "split_before_augmentation": True,
        "official_test_used_for_development": False,
        "counts": {
            "train_bases": args.train_size,
            "dev": args.dev_size,
            "test": args.test_size,
            "train_augmentations_per_base": args.num_aug,
            "train_examples_on_disk": args.train_size * (args.num_aug + 1),
        },
        "augmentation_seed": args.seed + 1000,
        "identities": identities,
        "leakage_assertions": {
            "source_train_dev_overlap": 0,
            "input_solution_hash_overlap": 0,
            "canonical_hash_overlap": 0,
        },
        "generated_files_sha256": generated_hashes,
        "numpy_version": np.__version__,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "SUDOKU STUDY BUILD PASS "
        f"(train={args.train_size}x{args.num_aug + 1}, "
        f"dev={args.dev_size}, test={args.test_size})"
    )
    print(f"output={args.output_dir}\nmanifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
