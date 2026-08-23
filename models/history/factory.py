from __future__ import annotations

from .base import HistoryAggregator
from .none import NoHistoryAggregator
from .uniform import UniformMeanHistory
from .recency import RecencyWeightedHistory
from .gated import GatedHistory
from .last_state import LastStateHistory
from .attention import HistoryAttention


def build_history_aggregator(name: str) -> HistoryAggregator:
    """Construct a history aggregator by stable configuration name."""

    normalized = name.strip().lower()

    if normalized in {"none", "no_history", "identity"}:
        return NoHistoryAggregator()

    if normalized in {"uniform", "uniform_mean", "mean"}:
        return UniformMeanHistory()

    if normalized in {"recency", "recency_weighted", "exponential_recency"}:
        return RecencyWeightedHistory()

    if normalized in {"gated", "gated_history", "scalar_gate"}:
        return GatedHistory()

    if normalized in {"last_state", "latest", "latest_state"}:
        return LastStateHistory()

    if normalized in {"attention", "history_attention", "temporal_attention"}:
        return HistoryAttention()

    raise ValueError(
        f"Unknown history aggregator: {name!r}. "
        "Available aggregators: none, uniform, recency, gated, "
        "last_state, attention"
    )
