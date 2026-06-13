"""Rule-based portfolio allocator: Top-K, inv-vol, vol-target, tiered MDD, hysteresis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sl_pipeline.allocator import (
    MarketContext,
    PortfolioAllocator,
    PortfolioState,
    TargetPortfolio,
)


@dataclass(frozen=True)
class RuleBasedAllocatorConfig:
    top_k: int = 5
    hysteresis_rank: int = 10
    weight_band: float = 0.05
    enable_vol_target: bool = False
    target_vol_annual: float = 0.18
    vol_floor: float = 0.05
    weighting_method: str = "inv_vol"  # 'inv_vol' or 'equal'
    yellow_mdd: float = 0.10
    yellow_max_exposure: float = 0.50
    red_mdd: float = 0.15
    red_max_exposure: float = 0.10
    max_single_weight: float = 0.35
    min_score: float = -np.inf


class RuleBasedAllocator(PortfolioAllocator):
    """Top-K + inv-vol + 18% vol-target + tiered MDD + turnover hysteresis."""

    def __init__(self, config: RuleBasedAllocatorConfig | None = None) -> None:
        self.config = config or RuleBasedAllocatorConfig()

    def allocate(
        self,
        scores: dict[str, float],
        vols: dict[str, float],
        state: PortfolioState,
        market_context: MarketContext | None = None,
    ) -> TargetPortfolio:
        cfg = self.config
        if not scores:
            return TargetPortfolio(target_weights={}, cash_weight=1.0)

        ranked = sorted(
            scores,
            key=lambda ticker: scores[ticker],
            reverse=True,
        )
        top_k = ranked[: cfg.top_k]
        top_hysteresis = set(ranked[: cfg.hysteresis_rank])

        selected: set[str] = set(top_k)
        for ticker, weight in state.positions.items():
            if weight > 1e-4 and ticker in top_hysteresis:
                selected.add(ticker)

        selected_sorted = sorted(
            selected,
            key=lambda ticker: scores.get(ticker, -np.inf),
            reverse=True,
        )

        raw_weights: dict[str, float] = {}
        if cfg.weighting_method == "equal":
            n_selected = len(selected_sorted)
            if n_selected > 0:
                raw_weights = {ticker: 1.0 / n_selected for ticker in selected_sorted}
        else:
            inv_vol: dict[str, float] = {}
            for ticker in selected_sorted:
                vol = max(float(vols.get(ticker, cfg.vol_floor)), cfg.vol_floor)
                inv_vol[ticker] = 1.0 / vol

            inv_sum = sum(inv_vol.values())
            if inv_sum > 0:
                raw_weights = {ticker: inv_vol[ticker] / inv_sum for ticker in selected_sorted}

        if not raw_weights:
            return TargetPortfolio(target_weights={}, cash_weight=1.0)

        raw_weights = self._cap_single_names(raw_weights, cfg.max_single_weight)
        raw_weights = self._renormalize(raw_weights)

        exposure = 1.0
        if cfg.enable_vol_target:
            portfolio_vol = self._estimate_portfolio_vol(raw_weights, vols, cfg.vol_floor)
            exposure = min(1.0, cfg.target_vol_annual / max(portfolio_vol, cfg.vol_floor))
            
        exposure = self._apply_mdd_cap(exposure, state.rolling_mdd, cfg)
        exposure = self._apply_macro_cap(exposure, market_context)

        stock_weights = {ticker: weight * exposure for ticker, weight in raw_weights.items()}
        stock_weights = self._cap_single_names(stock_weights, cfg.max_single_weight)
        stock_total = sum(stock_weights.values())
        if stock_total > exposure + 1e-8:
            scale = exposure / stock_total
            stock_weights = {ticker: weight * scale for ticker, weight in stock_weights.items()}

        stock_weights = self._apply_weight_hysteresis(stock_weights, state.positions, cfg.weight_band)
        stock_total = sum(stock_weights.values())
        cash_weight = max(0.0, 1.0 - stock_total)

        return TargetPortfolio(target_weights=stock_weights, cash_weight=float(cash_weight))

    @staticmethod
    def _renormalize(weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {ticker: weight / total for ticker, weight in weights.items()}

    @staticmethod
    def _cap_single_names(weights: dict[str, float], cap: float) -> dict[str, float]:
        return {ticker: min(weight, cap) for ticker, weight in weights.items()}

    @staticmethod
    def _estimate_portfolio_vol(
        weights: dict[str, float],
        vols: dict[str, float],
        vol_floor: float,
    ) -> float:
        if not weights:
            return vol_floor
        # Conservative diagonal estimate: sum(w_i * vol_i).
        return float(
            sum(
                weight * max(float(vols.get(ticker, vol_floor)), vol_floor)
                for ticker, weight in weights.items()
            )
        )

    @staticmethod
    def _apply_mdd_cap(
        exposure: float,
        rolling_mdd: float,
        cfg: RuleBasedAllocatorConfig,
    ) -> float:
        if rolling_mdd >= cfg.red_mdd:
            return min(exposure, cfg.red_max_exposure)
        if rolling_mdd >= cfg.yellow_mdd:
            return min(exposure, cfg.yellow_max_exposure)
        return min(exposure, 1.0)

    @staticmethod
    def _apply_macro_cap(
        exposure: float,
        market_context: MarketContext | None,
    ) -> float:
        if market_context is None:
            return exposure
        level = (market_context.macro_guard_level or "OK").upper()
        if level == "CRITICAL":
            return min(exposure, 0.0)
        if level == "WARN":
            return min(exposure, 0.50)
        return exposure

    @staticmethod
    def _apply_weight_hysteresis(
        target_weights: dict[str, float],
        current_positions: dict[str, float],
        weight_band: float,
    ) -> dict[str, float]:
        adjusted: dict[str, float] = {}
        tickers = set(target_weights) | set(current_positions)
        for ticker in tickers:
            target = float(target_weights.get(ticker, 0.0))
            current = float(current_positions.get(ticker, 0.0))
            if current > 1e-4 and abs(target - current) < weight_band:
                adjusted[ticker] = current
            elif target > 1e-4:
                adjusted[ticker] = target
        return {ticker: weight for ticker, weight in adjusted.items() if weight > 1e-4}
