"""
Create a reproducible Sudoku baseline dataset for Phase 1.

This dataset is used for comparing:
- Original TRM baseline
- Future TRM + HistoryAttention

It is not the final research-scale dataset.
"""

from pathlib import Path
import subprocess
import sys


OUTPUT_DIR = "data/sudoku-small"


def main():
    cmd = [
        sys.executable,
        "dataset/build_sudoku_dataset.py",
        "--output-dir",
        OUTPUT_DIR,
        "--subsample-size",
        "1000",
        "--num-aug",
        "10",
    ]

    print("Running:")
    print(" ".join(cmd))

    subprocess.run(
        cmd,
        check=True,
    )

    print()
    print(f"Created baseline dataset: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()