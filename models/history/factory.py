from __future__ import annotations

from .base import HistoryAggregator
from .none import NoHistoryAggregator
from .uniform import UniformMeanHistory


def build_history_aggregator(name: str) -> HistoryAggregator:
    """Construct a history aggregator by stable configuration name."""

    normalized = name.strip().lower()

    if normalized in {"none", "no_history", "identity"}:
        return NoHistoryAggregator()

    if normalized in {"uniform", "uniform_mean", "mean"}:
        return UniformMeanHistory()

    raise ValueError(
        f"Unknown history aggregator: {name!r}. "
        "Available aggregators: none, uniform"
    )
