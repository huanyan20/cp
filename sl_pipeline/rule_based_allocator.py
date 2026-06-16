"""Rule-based portfolio allocator: Top-K, inv-vol, vol-target, tiered MDD, hysteresis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sl_pipeline.allocator import (
    MarketContext,
    PortfolioAllocator,
    PortfolioState,
    TargetPortfolio,
)


@dataclass(frozen=True)
class RiskConfig:
    name: str
    vol_target: float
    yellow_mdd: float
    yellow_max_exposure: float
    red_mdd: float
    red_max_exposure: float
    max_single_weight: float
    weight_band: float = 0.05
    top_k_hold: int = 10
    top_k: int = 5
    recovery_buffer: float = 0.02
    annualization: int = 252
    vol_lookback: int = 20

RISK_V1 = RiskConfig(
    name="v1_baseline",
    vol_target=0.18, yellow_mdd=0.10, yellow_max_exposure=0.50,
    red_mdd=0.15,    red_max_exposure=0.10, max_single_weight=0.35,
)

RISK_V2 = RiskConfig(
    name="v2_mdd_patch",
    vol_target=0.15,
    yellow_mdd=0.07,
    yellow_max_exposure=0.50,
    red_mdd=0.12,
    red_max_exposure=0.10,
    max_single_weight=0.28,
)

RISK_V2B = RiskConfig(
    name="v2b_conservative",
    vol_target=0.15, yellow_mdd=0.08, yellow_max_exposure=0.50,
    red_mdd=0.13,    red_max_exposure=0.10, max_single_weight=0.28,
)

RISK_CONFIGS: dict[str, RiskConfig] = {"v1": RISK_V1, "v2": RISK_V2, "v2b": RISK_V2B}

@dataclass(frozen=True)
class RuleBasedAllocatorConfig:
    top_k: int = 5
    hysteresis_rank: int = 10
    weight_band: float = 0.06
    enable_vol_target: bool = False
    target_vol_annual: float = 0.15
    vol_floor: float = 0.05
    enable_trend_filter: bool = True
    weighting_method: str = "equal"
    yellow_mdd: float = 0.08
    yellow_max_exposure: float = 0.50
    red_mdd: float = 0.13
    red_max_exposure: float = 0.0
    max_single_weight: float = 0.25
    min_score: float = 0.005
    enable_momentum_scaling: bool = False
    momentum_scale_factor: float = 5.0
    enable_trailing_stop: bool = True
    trailing_stop_threshold: float = 0.15
    cooldown_duration: int = 10
    enable_ma_filter: bool = True
    ma_filter_windows: list[int] = field(default_factory=lambda: [20])
class RuleBasedAllocator(PortfolioAllocator):
    """Top-K + inv-vol + 15% vol-target + tiered MDD + turnover hysteresis."""

    def __init__(self, risk_config: RiskConfig = RISK_V2, config: RuleBasedAllocatorConfig | None = None) -> None:
        self.cfg = risk_config
        self.config = config or RuleBasedAllocatorConfig()
        self._regime_state = "normal"
        
    @property
    def config_name(self) -> str:
        return self.cfg.name
        
    def reset_regime(self) -> None:
        self._regime_state = "normal"""

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
        """Allocate based on rules + Abs Vol Parity."""
        cfg = self.config
        risk_cfg = self.cfg
        if not scores:
            return TargetPortfolio(target_weights={}, cash_weight=1.0)

        # Use pure model scores
        adjusted_scores = dict(scores)

        sorted_scores = sorted(
            adjusted_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        ranked = [t for t, s in sorted_scores]
        top_k = ranked[: cfg.top_k]
        top_hysteresis = set(ranked[: cfg.hysteresis_rank])

        selected: set[str] = set(top_k)
        for ticker, weight in state.positions.items():
            if weight > 1e-4 and ticker in top_hysteresis:
                selected.add(ticker)

        dynamic_min_score = cfg.min_score
        if market_context:
            if market_context.macro_guard_level in ("WARN", "CRITICAL"):
                dynamic_min_score = max(dynamic_min_score, 0.015)
            if market_context.market_volatility is not None and market_context.market_volatility > 0.20:
                dynamic_min_score = max(dynamic_min_score, 0.010)

        # Filter available names
        valid_tickers = []
        for t, score in sorted_scores:
            if t not in selected:
                continue
                
            # Apply dynamic min_score only for NEW entries.
            # If already holding, use standard min_score to prevent churn.
            required_score = dynamic_min_score if state.positions.get(t, 0.0) < 1e-4 else cfg.min_score
            if score < required_score:
                continue

            if self.config.enable_trend_filter and trends is not None:
                if trends.get(t, 1.0) < 0.0:
                    continue  # Filter out downward trend stocks

            if self.config.enable_ma_filter and ma_distances is not None:
                passed_ma = True
                for w in self.config.ma_filter_windows:
                    if ma_distances.get(w, {}).get(t, 1.0) < 0.0:
                        passed_ma = False
                        break
                if not passed_ma:
                    continue

            if self.config.enable_trailing_stop:
                if state.position_mdds.get(t, 0.0) >= self.config.trailing_stop_threshold:
                    continue

            if state.cooldown_days.get(t, 0) > 0:
                continue

            valid_tickers.append(t)

        selected_sorted = valid_tickers

        effective_target_vol = risk_cfg.vol_target
        if market_context and market_context.market_volatility is not None:
            if market_context.market_volatility > 0.20:
                # Compress target vol during high market turbulence (VIX/TWII vol > 20%)
                vol_scaler = 0.20 / market_context.market_volatility
                effective_target_vol = risk_cfg.vol_target * max(0.5, vol_scaler)

        raw_weights: dict[str, float] = {}
        if cfg.weighting_method == "equal":
            n_selected = len(selected_sorted)
            if n_selected > 0:
                raw_weights = {ticker: 1.0 / n_selected for ticker in selected_sorted}
        elif cfg.weighting_method == "abs_vol_parity":
            for ticker in selected_sorted:
                vol = max(float(vols.get(ticker, cfg.vol_floor)), cfg.vol_floor)
                # target_weight = (Target_Vol / sqrt(K)) / Asset_Vol
                raw_weight = (effective_target_vol / np.sqrt(cfg.top_k)) / vol
                raw_weights[ticker] = raw_weight
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

        # Apply momentum scaling before renormalization / final sizing
        if cfg.enable_momentum_scaling and short_trends:
            for ticker in list(raw_weights.keys()):
                st = short_trends.get(ticker, 0.0)
                if st < 0.0:
                    scaler = max(0.0, 1.0 + st * cfg.momentum_scale_factor)
                    raw_weights[ticker] *= scaler

        raw_weights = self._apply_single_cap(raw_weights)
        if cfg.weighting_method != "abs_vol_parity":
            raw_weights = self._renormalize(raw_weights)

        exposure = 1.0
        if cfg.enable_vol_target and cfg.weighting_method != "abs_vol_parity":
            portfolio_vol = self._estimate_portfolio_vol(raw_weights, vols, cfg.vol_floor)
            exposure = min(1.0, effective_target_vol / max(portfolio_vol, cfg.vol_floor))
            
        regime, exposure_cap = self._regime_exposure_cap(state.rolling_mdd)
        exposure = min(exposure, exposure_cap)
        exposure = self._apply_macro_cap(exposure, market_context)

        if exposure <= 1e-4:
            return TargetPortfolio(target_weights={}, cash_weight=1.0)

        stock_weights = {ticker: weight * exposure for ticker, weight in raw_weights.items()}
        stock_weights = self._apply_single_cap(stock_weights)
        stock_total = sum(stock_weights.values())
        if stock_total > exposure + 1e-8:
            scale = exposure / stock_total
            stock_weights = {ticker: weight * scale for ticker, weight in stock_weights.items()}

        stock_weights = self._apply_weight_band(stock_weights, state.positions)
        stock_weights = self._apply_single_cap(stock_weights)
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

    def _regime_exposure_cap(self, rolling_mdd: float) -> tuple[str, float]:
        cfg = self.cfg
        buf = cfg.recovery_buffer
        if rolling_mdd >= cfg.red_mdd:
            self._regime_state = "red"
        elif rolling_mdd >= cfg.yellow_mdd:
            if self._regime_state == "red":
                if rolling_mdd < cfg.red_mdd - buf:
                    self._regime_state = "yellow"
            else:
                self._regime_state = "yellow"
        else:
            if self._regime_state in ("yellow", "red"):
                if rolling_mdd < cfg.yellow_mdd - buf:
                    self._regime_state = "normal"
            else:
                self._regime_state = "normal"

        if self._regime_state == "red":
            return "red", cfg.red_max_exposure
        elif self._regime_state == "yellow":
            return "yellow", cfg.yellow_max_exposure
        return "normal", 1.0

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
            exposure = min(exposure, 0.25)
            
        if market_context.market_volatility is not None and market_context.market_volatility > 0.20:
            vol_cap = 0.20 / market_context.market_volatility
            exposure = min(exposure, vol_cap)
            
        return exposure

    def _apply_weight_band(self, target: dict, current: dict) -> dict:
        band = self.cfg.weight_band
        result = {}
        for t, tw in target.items():
            cw = float(current.get(t, 0.0))
            tw = float(tw)
            if cw > 1e-4 and abs(tw - cw) < band:
                result[t] = cw
            elif tw > 1e-4:
                result[t] = tw
        return result

    def _apply_single_cap(self, weights: dict) -> dict:
        cap = self.cfg.max_single_weight
        w = dict(weights)
        for _ in range(5):
            overflow = {t: v - cap for t, v in w.items() if v > cap}
            if not overflow:
                break
            excess = sum(overflow.values())
            for t in overflow:
                w[t] = cap
            under = [t for t in w if w[t] < cap]
            if not under:
                break
            per_stock = excess / len(under)
            for t in under:
                w[t] = min(cap, w[t] + per_stock)
        return w

