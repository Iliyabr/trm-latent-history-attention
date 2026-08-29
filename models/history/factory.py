from __future__ import annotations

from .attention import HistoryAttention
from .base import HistoryAggregator
from .gated import GatedHistory
from .last_state import LastStateHistory
from .none import NoHistoryAggregator
from .parameter_matched import ParameterMatchedNoHistory
from .residual import ResidualHistory
from .uniform import UniformMeanHistory


def normalize_history_mode(name: str) -> str:
    normalized = name.strip().lower()
    aliases = {
        "no_history": "none",
        "identity": "none",
        "vanilla": "none",
        "b0": "none",
        "residual": "residual",
        "b1": "residual",
        "uniform_mean": "uniform",
        "mean": "uniform",
        "b2": "uniform",
        "gated": "gated",
        "gated_uniform": "gated",
        "gated_history": "gated",
        "history_attention": "attention",
        "temporal_attention": "attention",
        "p1": "attention",
        "attention": "attention",
        "parameter_matched": "parameter_matched",
        "param_matched": "parameter_matched",
        "b3": "parameter_matched",
        "last_state": "static_lag",
        "latest": "static_lag",
    }
    return aliases.get(normalized, normalized)


def build_history_aggregator(
    name: str,
    *,
    hidden_size: int = 0,
    rank: int = 0,
    num_heads: int = 1,
    window: int = 0,
    norm_eps: float = 1e-5,
    gate_init: float = -2.0,
) -> HistoryAggregator:
    """Construct a within-cycle history reader by stable experiment name."""
    mode = normalize_history_mode(name)
    if mode == "none":
        return NoHistoryAggregator()
    if mode == "residual":
        return ResidualHistory(norm_eps)
    if mode == "uniform":
        return UniformMeanHistory(norm_eps)
    if mode == "gated":
        return GatedHistory(norm_eps=norm_eps, gate_init=gate_init)
    if mode == "static_lag":
        return LastStateHistory()
    if mode == "attention":
        return HistoryAttention(
            hidden_size, rank, num_heads, window, norm_eps, gate_init
        )
    if mode == "parameter_matched":
        return ParameterMatchedNoHistory(
            hidden_size, rank, norm_eps=norm_eps, gate_init=gate_init
        )
    raise ValueError(
        f"Unknown history mode {name!r}; expected none/B0, residual/B1, "
        "uniform/B2, gated/Gated, attention/P1, parameter_matched/B3, or static_lag"
    )
