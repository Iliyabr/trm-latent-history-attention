"""Build the deterministic Phase-1 Sudoku baseline dataset.

Scientific contract:
- deterministic source selection
- disjoint train/dev original puzzles
- augmentation only after the split
- official Sudoku-Extreme test set is NOT used for development
- source and generated-file hashes are recorded
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from build_sudoku_dataset import shuffle_sudoku
from common import PuzzleDatasetMetadata


SOURCE_REPO = "sapientinc/sudoku-extreme"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_csv(path: Path):
    inputs = []
    labels = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for _, q, a, _ in reader:
            assert len(q) == 81
            assert len(a) == 81

            inputs.append(
                np.frombuffer(
                    q.replace(".", "0").encode(),
                    dtype=np.uint8,
                ).reshape(9, 9) - ord("0")
            )

            labels.append(
                np.frombuffer(
                    a.encode(),
                    dtype=np.uint8,
                ).reshape(9, 9) - ord("0")
            )

    return inputs, labels


def write_split(
    root: Path,
    split: str,
    source_inputs,
    source_labels,
    source_indices: np.ndarray,
    num_aug: int,
    augmentation_seed: int,
):
    results = {
        k: []
        for k in [
            "inputs",
            "labels",
            "puzzle_identifiers",
            "puzzle_indices",
            "group_indices",
        ]
    }

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    puzzle_id = 0
    example_id = 0

    # The upstream transformation uses np.random directly.
    # Pin its state explicitly for deterministic augmentation.
    np.random.seed(augmentation_seed)

    for source_idx in source_indices:
        orig_inp = source_inputs[int(source_idx)]
        orig_out = source_labels[int(source_idx)]

        for aug_idx in range(1 + num_aug):
            if aug_idx == 0:
                inp = orig_inp
                out = orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            results["inputs"].append(inp)
            results["labels"].append(out)

            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        results["group_indices"].append(puzzle_id)

    def seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1

    arrays = {
        "inputs": seq_to_numpy(results["inputs"]),
        "labels": seq_to_numpy(results["labels"]),
        "group_indices": np.asarray(
            results["group_indices"], dtype=np.int32
        ),
        "puzzle_indices": np.asarray(
            results["puzzle_indices"], dtype=np.int32
        ),
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
        total_puzzles=len(source_indices),
        sets=["all"],
    )

    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    (split_dir / "dataset.json").write_text(
        json.dumps(metadata.model_dump(), indent=2),
        encoding="utf-8",
    )

    for name, values in arrays.items():
        np.save(split_dir / f"all__{name}.npy", values)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sudoku-baseline-v2"),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/data/sudoku_baseline_v2_manifest.json"
        ),
    )

    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--dev-size", type=int, default=200)
    parser.add_argument("--num-aug", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    train_csv = Path(
        hf_hub_download(
            SOURCE_REPO,
            "train.csv",
            repo_type="dataset",
        )
    )

    # Download only to freeze the identity of the official test source.
    # It is NOT used for model development.
    test_csv = Path(
        hf_hub_download(
            SOURCE_REPO,
            "test.csv",
            repo_type="dataset",
        )
    )

    source_inputs, source_labels = load_csv(train_csv)

    requested = args.train_size + args.dev_size
    assert requested <= len(source_inputs)

    rng = np.random.default_rng(args.seed)
    selected = rng.choice(
        len(source_inputs),
        size=requested,
        replace=False,
    )

    # Split BEFORE augmentation.
    train_indices = np.sort(selected[: args.train_size])
    dev_indices = np.sort(selected[args.train_size :])

    assert len(np.intersect1d(train_indices, dev_indices)) == 0

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)

    write_split(
        root=root,
        split="train",
        source_inputs=source_inputs,
        source_labels=source_labels,
        source_indices=train_indices,
        num_aug=args.num_aug,
        augmentation_seed=args.seed + 1000,
    )

    # Development data is never augmented.
    write_split(
        root=root,
        split="dev",
        source_inputs=source_inputs,
        source_labels=source_labels,
        source_indices=dev_indices,
        num_aug=0,
        augmentation_seed=args.seed + 2000,
    )

    (root / "identifiers.json").write_text(
        '["<blank>"]\n',
        encoding="utf-8",
    )

    generated_hashes = {}

    for path in sorted(root.rglob("*")):
        if path.is_file():
            generated_hashes[
                str(path.relative_to(root)).replace("\\", "/")
            ] = sha256_file(path)

    manifest = {
        "schema_version": 1,
        "dataset_name": "sudoku-baseline-v2",
        "source_repo": SOURCE_REPO,
        "source_train_sha256": sha256_file(train_csv),
        "source_official_test_sha256": sha256_file(test_csv),
        "official_test_used_for_development": False,
        "numpy_version": np.__version__,
        "seed": args.seed,
        "augmentation_seed": args.seed + 1000,
        "train_original_count": args.train_size,
        "dev_original_count": args.dev_size,
        "train_augmentations_per_original": args.num_aug,
        "train_source_indices": train_indices.tolist(),
        "dev_source_indices": dev_indices.tolist(),
        "train_dev_overlap": 0,
        "generated_files_sha256": generated_hashes,
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    args.manifest.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("BASELINE DATASET V2 BUILD PASS")
    print(f"output = {root}")
    print(f"manifest = {args.manifest}")
    print(f"train originals = {len(train_indices)}")
    print(f"dev originals = {len(dev_indices)}")
    print("train/dev overlap = 0")
    print("official test used for development = False")


if __name__ == "__main__":
    main()
