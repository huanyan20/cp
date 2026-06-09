"""Portfolio allocation interface (S2: RuleBasedAllocator; S5: RLAllocator).

See ``sl_pipeline.rl_allocator.RLAllocator`` for the RL residual spike.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PortfolioState:
    """Snapshot passed into allocators (S2+)."""

    positions: dict[str, float] = field(default_factory=dict)
    cash_weight: float = 1.0
    portfolio_value: float = 1.0
    peak_value: float = 1.0
    rolling_mdd: float = 0.0


@dataclass
class MarketContext:
    """Optional macro / guard overlay inputs (O6)."""

    macro_guard_level: str = "OK"


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
        market_context: MarketContext | None = None,
    ) -> TargetPortfolio:
        """Map alpha scores to target weights + cash."""
