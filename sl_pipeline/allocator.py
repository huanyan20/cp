"""Portfolio allocation interface (S2: RuleBasedAllocator; S5: RLAllocator).

See ``sl_pipeline.rl_allocator.RLAllocator`` for the RL residual spike.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PortfolioState:
    """Read-only snapshot of the portfolio at a given time."""

    positions: dict[str, float] = field(default_factory=dict)
    cash_weight: float = 1.0
    portfolio_value: float = 1.0
    peak_value: float = 1.0
    rolling_mdd: float = 0.0
    position_mdds: dict[str, float] = field(default_factory=dict)
    position_cum_rets: dict[str, float] = field(default_factory=dict)
    position_peaks: dict[str, float] = field(default_factory=dict)
    cooldown_days: dict[str, int] = field(default_factory=dict)


@dataclass
class MarketContext:
    """Optional macro / guard overlay inputs (O6)."""

    macro_guard_level: str = "OK"
    market_volatility: float | None = None


@dataclass
class TargetPortfolio:
    target_weights: dict[str, float]
    cash_weight: float


class PortfolioAllocator(ABC):
    @abstractmethod
    def allocate(
        self,
        scores: dict[str, float],
        vols: dict[str, float],
        state: PortfolioState,
        market_context: MarketContext,
        trends: dict[str, float] | None = None,
        short_trends: dict[str, float] | None = None,
        ma_distances: dict[int, dict[str, float]] | None = None,
    ) -> TargetPortfolio:
        """Calculate target weights for the portfolio."""
