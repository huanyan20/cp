"""RLAllocator spike (S5): RuleBased baseline + RL residual weight adjustment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sl_pipeline.allocator import (
    MarketContext,
    PortfolioAllocator,
    PortfolioState,
    TargetPortfolio,
)
from sl_pipeline.rule_based_allocator import (
    RuleBasedAllocator,
    RuleBasedAllocatorConfig,
)


def _softmax(logits: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    temp = max(temperature, 1e-3)
    scaled = logits / temp
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0:
        return np.ones_like(logits) / len(logits)
    return exp / total


@dataclass(frozen=True)
class RLAllocatorConfig:
    mode: str = "residual"  # residual | direct_topk
    residual_scale: float = 0.15
    softmax_temp: float = 0.5
    top_k: int = 5
    max_single_weight: float = 0.35


class RLAllocator(PortfolioAllocator):
    """Blend RuleBasedAllocator baseline with RL logits (residual micro-adjust)."""

    def __init__(
        self,
        config: RLAllocatorConfig | None = None,
        *,
        rule_allocator: RuleBasedAllocator | None = None,
    ) -> None:
        self.config = config or RLAllocatorConfig()
        self.rule_allocator = rule_allocator or RuleBasedAllocator(
            RuleBasedAllocatorConfig(
                top_k=self.config.top_k,
                max_single_weight=self.config.max_single_weight,
            )
        )

    def allocate(
        self,
        scores: dict[str, float],
        vols: dict[str, float],
        state: PortfolioState,
        market_context: MarketContext | None = None,
        trends: dict[str, float] | None = None,
        short_trends: dict[str, float] | None = None,
        ma_distances: dict[int, dict[str, float]] | None = None,
    ) -> TargetPortfolio:
        """Query RL policy for adjustments on top of the RuleBased target."""
        return self.rule_allocator.allocate(scores, vols, state, market_context, trends, short_trends)

    def allocate_from_action(
        self,
        action: np.ndarray,
        scores: dict[str, float],
        vols: dict[str, float],
        state: PortfolioState,
        tickers: list[str],
        *,
        market_context: MarketContext | None = None,
        enable_cash_action: bool = False,
        trends: dict[str, float] | None = None,
        short_trends: dict[str, float] | None = None,
        ma_distances: dict[int, dict[str, float]] | None = None,
    ) -> TargetPortfolio:
        """Map env action logits to target weights on top of RuleBased baseline."""
        action = np.asarray(action, dtype=float).reshape(-1)
        stock_dim = len(tickers) + (1 if enable_cash_action else 0)
        if action.shape[0] < stock_dim:
            raise ValueError(f"Action length {action.shape[0]} < expected {stock_dim}")

        stock_logits = action[: len(tickers)]
        cash_logit = float(action[len(tickers)]) if enable_cash_action else 0.0

        if self.config.mode == "direct_topk":
            return self._direct_topk(stock_logits, cash_logit, tickers, enable_cash_action)

        rule_target = self.rule_allocator.allocate(
            scores, 
            vols, 
            state, 
            market_context=MarketContext(), 
            trends=trends,
            short_trends=short_trends,
            ma_distances=ma_distances,
        )

        return self._residual_blend(
            stock_logits,
            cash_logit,
            rule_target,
            tickers,
            enable_cash_action,
        )

    def _direct_topk(
        self,
        stock_logits: np.ndarray,
        cash_logit: float,
        tickers: list[str],
        enable_cash_action: bool,
    ) -> TargetPortfolio:
        cfg = self.config
        topk_idx = np.argsort(stock_logits)[-cfg.top_k :]
        mask = np.zeros(len(tickers), dtype=float)
        mask[topk_idx] = 1.0
        masked = stock_logits * mask
        weights = _softmax(masked, cfg.softmax_temp)
        stock_total = float(np.sum(weights))
        if enable_cash_action:
            cash_weight = float(
                np.exp(cash_logit / cfg.softmax_temp)
                / (np.exp(cash_logit / cfg.softmax_temp) + stock_total + 1e-8)
            )
            scale = (1.0 - cash_weight) / max(stock_total, 1e-8)
            weights = weights * scale
        else:
            cash_weight = max(0.0, 1.0 - stock_total)
            if stock_total > 1.0:
                weights = weights / stock_total
                cash_weight = 0.0

        capped = {tickers[i]: min(float(weights[i]), cfg.max_single_weight) for i in range(len(tickers)) if weights[i] > 1e-6}
        total = sum(capped.values())
        if total > 1.0 - cash_weight:
            scale = (1.0 - cash_weight) / total
            capped = {t: w * scale for t, w in capped.items()}
        return TargetPortfolio(target_weights=capped, cash_weight=float(cash_weight))

    def _residual_blend(
        self,
        stock_logits: np.ndarray,
        cash_logit: float,
        baseline: TargetPortfolio,
        tickers: list[str],
        enable_cash_action: bool,
    ) -> TargetPortfolio:
        cfg = self.config
        residual = _softmax(stock_logits, cfg.softmax_temp) * cfg.residual_scale
        weights = np.zeros(len(tickers), dtype=float)
        for i, ticker in enumerate(tickers):
            base_w = float(baseline.target_weights.get(ticker, 0.0))
            weights[i] = max(0.0, base_w + residual[i])

        stock_total = float(np.sum(weights))
        cash_weight = float(baseline.cash_weight)
        if enable_cash_action:
            cash_adj = float(np.tanh(cash_logit) * cfg.residual_scale)
            cash_weight = float(np.clip(cash_weight + cash_adj, 0.0, 1.0))

        if stock_total > 0:
            target_stock = min(stock_total, 1.0 - cash_weight)
            weights = weights * (target_stock / stock_total)

        capped = {
            tickers[i]: min(float(weights[i]), cfg.max_single_weight)
            for i in range(len(tickers))
            if weights[i] > 1e-6
        }
        total = sum(capped.values())
        if total > 1.0 - cash_weight:
            scale = (1.0 - cash_weight) / total
            capped = {t: w * scale for t, w in capped.items()}
        return TargetPortfolio(target_weights=capped, cash_weight=cash_weight)
