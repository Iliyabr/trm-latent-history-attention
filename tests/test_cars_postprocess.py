"""Tests for training-free CARS selection logic."""
from __future__ import annotations

from experiments.cars_postprocess import (
    ACT_STEPS,
    clue_mismatch_count,
    postprocess_example,
    select_cars,
    select_confidence,
    structural_violations,
)


def _blank_inputs() -> list[int]:
    return [1] * 81


def _given_inputs() -> list[int]:
    values = [1] * 81
    values[0] = 5  # given digit 4 encoded as 5
    return values


def test_clue_mismatch_counts_given_cells_only() -> None:
    inputs = _given_inputs()
    predictions = inputs.copy()
    predictions[0] = 6
    assert clue_mismatch_count(inputs, predictions) == 1
    assert clue_mismatch_count(inputs, inputs) == 0


def test_structural_duplicate_excess() -> None:
    tokens = [1] + [3] * 80
    tokens[0] = 3
    tokens[1] = 3
    struct = structural_violations(tokens)
    assert struct["row_duplicate_excess"] >= 1
    assert struct["structural_violations"] >= 1


def test_cars_prefers_fewer_clue_mismatches() -> None:
    candidates = [
        {
            "act_step": 6,
            "clue_mismatch_count": 2,
            "structural_violations": 0,
            "mean_confidence": 0.9,
            "exact": False,
            "cell_accuracy": 0.5,
            "incorrect_cells": 40,
        },
        {
            "act_step": 3,
            "clue_mismatch_count": 0,
            "structural_violations": 5,
            "mean_confidence": 0.1,
            "exact": False,
            "cell_accuracy": 0.4,
            "incorrect_cells": 45,
        },
    ]
    assert select_cars(candidates) == 1


def test_cars_tie_breaks_to_later_step_on_equal_confidence() -> None:
    candidates = [
        {
            "act_step": 2,
            "clue_mismatch_count": 0,
            "structural_violations": 0,
            "mean_confidence": 0.8,
            "exact": False,
            "cell_accuracy": 0.5,
            "incorrect_cells": 40,
        },
        {
            "act_step": 5,
            "clue_mismatch_count": 0,
            "structural_violations": 0,
            "mean_confidence": 0.8,
            "exact": False,
            "cell_accuracy": 0.5,
            "incorrect_cells": 40,
        },
    ]
    assert select_cars(candidates) == 1
    assert select_confidence(candidates) == 1


def test_postprocess_example_shapes() -> None:
    labels = [2] * 81
    act_steps = []
    for step in range(1, ACT_STEPS + 1):
        act_steps.append(
            {
                "act_step": step,
                "predictions": labels,
                "exact": step == ACT_STEPS,
                "cell_accuracy": step / ACT_STEPS,
                "incorrect_cells": 81 - step,
                "clue_mismatch_count": 0,
                "structural_violations": 0,
                "mean_confidence": step / ACT_STEPS,
            }
        )
    row = postprocess_example(
        example_id="test:0:abc:0",
        inputs=_blank_inputs(),
        labels=labels,
        act_steps=act_steps,
    )
    assert row["final_act_step"] == ACT_STEPS
    assert row["cars_act_step"] >= 1
