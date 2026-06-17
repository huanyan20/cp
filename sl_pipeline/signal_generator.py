"""Pooled 3-class SL scorer (SignalGenerator) - Milestone 3B.

Architecture:
  - LGBMClassifier / XGBClassifier with softmax objective
  - 3-class labels: 0=Bottom 20%, 1=Mid 60%, 2=Top 20%
  - Feature set: IC-filtered (|IC| > 0.02 at 20d)
  - Output score = P(class=2) - P(class=0)  (net upside confidence)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sl_pipeline.labels import (
    build_feature_panel,
    build_labeled_panel,
    default_feature_columns,
    label_column_name,
    split_panel_by_date,
)

# ── Milestone 3B: IC-filtered golden features (|IC| > 0.02 @ 20d) ──────────
# Includes both raw and rank versions; rank versions are more robust cross-sectionally.
IC_FILTERED_FEATURES: list[str] = [
    # Momentum Family (long-term signals proven by IC)
    "ret_60d",
    "rank_ret_60d",
    # Volatility Family (strong predictors)
    "atr_20",
    "rank_atr_20",
    "rolling_std_20",
    "rank_rolling_std_20",
    "atr_60",
    "rank_atr_60",
    "rolling_std_60",
    "rank_rolling_std_60",
    # Trend
    "ADX_14",
    "rank_ADX_14",
    # Oscillators
    "RSI_14",
    "rank_RSI_14",
    "MACD_norm",
    "rank_MACD_norm",
    # Liquidity
    "volume_ma60_ratio",
    "rank_volume_ma60_ratio",
    "price_ma60_ratio",
    "rank_price_ma60_ratio",
    "volume_zscore_60",
    "rank_volume_zscore_60",
    "dollar_volume_log",
    "rank_dollar_volume_log",
    # Market Regime Features (Milestone 3B fix for regime mismatch)
    "price_ma200_ratio",   # Stock: Bull/Bear判斷 (>1=牛市, <1=熊市)
    "trend_slope_60d",     # Stock: 60日趨勢斜率
    "above_ma120",         # Stock: 是否在120日線上
    # TWII Market Index Regime Features (Option A: market-level context)
    "twii_ma60_ratio",     # TWII Close / 60d MA
    "twii_ma120_ratio",    # TWII Close / 120d MA
    "twii_ma200_ratio",    # TWII Close / 200d MA (market-level bull/bear)
    "twii_trend_60d",      # TWII 60d trend slope
    "twii_above_ma120",    # Binary: Market in bull regime
]



DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 5,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 30,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
}


def normalize_model_backend(model_backend: str) -> str:
    b = model_backend.lower().strip()
    if b in ("lgbm", "lightgbm"):
        return "lightgbm"
    if b in ("xgb", "xgboost"):
        return "xgboost"
    raise ValueError(f"Unknown model backend {model_backend!r}; expected 'lightgbm' or 'xgboost'")


@dataclass
class SignalGeneratorConfig:
    horizon: int = 20
    model_backend: str = "lightgbm"
    lgbm_params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LGBM_PARAMS))
    xgb_params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_XGB_PARAMS))
    feature_cols: list[str] | None = None
    use_ic_filtered_features: bool = True


@dataclass
class SignalFitSummary:
    horizon: int
    model_backend: str
    n_train_rows: int
    n_test_rows: int
    n_features: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_class_distribution: dict[str, int]
    feature_importance_top10: dict[str, float]


class SignalGenerator:
    """Train pooled 3-class classifier on cross-sectional labels; emit daily OOS scores.

    Score = P(Top 20%) - P(Bottom 20%)
    Range: [-1, +1]; positive = bullish confidence
    """

    def __init__(self, config: SignalGeneratorConfig | None = None) -> None:
        self.config = config or SignalGeneratorConfig()
        self.backend = normalize_model_backend(self.config.model_backend)

        # Decide feature set: IC-filtered or full default
        if self.config.feature_cols is not None:
            self.feature_cols = self.config.feature_cols
        elif self.config.use_ic_filtered_features:
            self.feature_cols = list(IC_FILTERED_FEATURES)
        else:
            self.feature_cols = default_feature_columns()

        self.model = None
        self.train_median_: pd.Series | None = None
        self._last_prediction_panel: pd.DataFrame | None = None

    @property
    def label_col(self) -> str:
        return label_column_name(self.config.horizon)

    def _prepare_features(self, panel: pd.DataFrame, *, fit_median: bool = False) -> pd.DataFrame:
        # Only use features that exist in the panel
        available_cols = [c for c in self.feature_cols if c in panel.columns]
        X = panel[available_cols].copy().astype(float)
        if fit_median:
            self.train_median_ = X.median()
            self._fit_feature_cols = available_cols
        if self.train_median_ is None:
            raise RuntimeError("Call fit() before predict().")
        # Align columns to training set
        X = X.reindex(columns=self._fit_feature_cols)
        return X.fillna(self.train_median_)

    def fit(self, train_panel: pd.DataFrame) -> None:
        label = self.label_col
        if label not in train_panel.columns:
            raise ValueError(f"Training panel missing label column: {label}")

        X = self._prepare_features(train_panel, fit_median=True)
        y = train_panel[label].astype(int)

        if self.backend == "lightgbm":
            import lightgbm as lgb
            params = dict(self.config.lgbm_params)
            self.model = lgb.LGBMClassifier(**params)
            self.model.fit(X, y)
        else:
            import xgboost as xgb
            params = dict(self.config.xgb_params)
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X, y)

    def predict_score(self, panel: pd.DataFrame) -> pd.Series:
        """Return net upside confidence: P(Top 20%) - P(Bottom 20%)."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        X = self._prepare_features(panel)
        proba = self.model.predict_proba(X)  # shape (N, 3)
        # class 0=bottom, 1=mid, 2=top
        score = proba[:, 2] - proba[:, 0]
        return pd.Series(score, index=panel.index, name="score")

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        """Alias for predict_score (backwards-compatible)."""
        return self.predict_score(panel)

    def predict_proba_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return full softmax probabilities as a DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        X = self._prepare_features(panel)
        proba = self.model.predict_proba(X)
        df = pd.DataFrame(proba, index=panel.index, columns=["p_bot", "p_mid", "p_top"])
        df["score"] = df["p_top"] - df["p_bot"]
        return df

    def fit_period(
        self,
        enriched_train: dict[str, pd.DataFrame],
        enriched_test: dict[str, pd.DataFrame],
        *,
        train_end: str,
        test_start: str,
    ) -> tuple[dict[str, pd.Series], SignalFitSummary]:
        """Fit on train window labels; score OOS test dates."""
        train_panel = build_labeled_panel(
            enriched_train,
            horizon=self.config.horizon,
            feature_cols=self.feature_cols,
        )
        train_panel, _ = split_panel_by_date(train_panel, train_end, test_start)
        if train_panel.empty:
            raise ValueError("Training panel is empty after date split.")

        test_panel = build_feature_panel(enriched_test, feature_cols=self.feature_cols)
        _, test_panel = split_panel_by_date(test_panel, train_end, test_start)
        if test_panel.empty:
            raise ValueError("Test panel is empty after date split.")

        # Class distribution for diagnostics
        label_col = self.label_col
        class_dist = train_panel[label_col].value_counts().to_dict()
        train_class_dist = {str(int(k)): int(v) for k, v in class_dist.items()}

        self.fit(train_panel)
        scored = self.predict_score(test_panel)
        proba_df = self.predict_proba_panel(test_panel)

        test_panel = test_panel.copy()
        test_panel["score"] = scored.values
        test_panel["p_bot"] = proba_df["p_bot"].values
        test_panel["p_mid"] = proba_df["p_mid"].values
        test_panel["p_top"] = proba_df["p_top"].values
        self._last_prediction_panel = test_panel

        scores: dict[str, pd.Series] = {}
        for ticker in sorted(test_panel["ticker"].unique()):
            sub = test_panel.loc[test_panel["ticker"] == ticker].sort_values("date")
            scores[ticker] = pd.Series(
                sub["score"].values,
                index=pd.to_datetime(sub["date"]),
                name=ticker,
            )

        importance = self._top_feature_importance(10)
        summary = SignalFitSummary(
            horizon=self.config.horizon,
            model_backend=self.backend,
            n_train_rows=len(train_panel),
            n_test_rows=len(test_panel),
            n_features=len(self._fit_feature_cols),
            train_start=str(train_panel["date"].min().date()),
            train_end=train_end,
            test_start=test_start,
            test_end=str(test_panel["date"].max().date()),
            train_class_distribution=train_class_dist,
            feature_importance_top10=importance,
        )
        return scores, summary

    def _top_feature_importance(self, top_n: int) -> dict[str, float]:
        if self.model is None:
            return {}
        try:
            if self.backend == "lightgbm":
                importances = self.model.feature_importances_
                cols = self._fit_feature_cols
            else:
                importances = self.model.feature_importances_
                cols = self._fit_feature_cols
            pairs = sorted(
                zip(cols, importances, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
            return {name: float(val) for name, val in pairs[:top_n]}
        except Exception:
            return {}

    def scores_to_wide(self, scores: dict[str, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame({ticker: series for ticker, series in scores.items()}).sort_index()

    def save_scores(self, scores: dict[str, pd.Series], path: Path) -> None:
        wide = self.scores_to_wide(scores)
        path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(path)

    def save_summary(self, summary: SignalFitSummary, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    def save_prediction_panel(self, path: Path) -> None:
        if self._last_prediction_panel is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._last_prediction_panel.to_csv(path, index=False)
