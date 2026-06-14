"""Pooled LightGBM alpha scorer (SignalGenerator)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from sl_pipeline.labels import (
    build_feature_panel,
    build_labeled_panel,
    default_feature_columns,
    label_column_name,
    split_panel_by_date,
)

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "huber",
    "n_estimators": 120,
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}


@dataclass
class SignalGeneratorConfig:
    horizon: int = 5
    lgbm_params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LGBM_PARAMS))
    feature_cols: list[str] | None = None


@dataclass
class SignalFitSummary:
    horizon: int
    n_train_rows: int
    n_test_rows: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    feature_importance_top10: dict[str, float]


class SignalGenerator:
    """Train pooled LightGBM on cross-demean labels; emit daily OOS scores."""

    def __init__(self, config: SignalGeneratorConfig | None = None) -> None:
        self.config = config or SignalGeneratorConfig()
        self.feature_cols = self.config.feature_cols or default_feature_columns()
        self.model: lgb.LGBMRegressor | None = None
        self.train_median_: pd.Series | None = None


    @property
    def label_col(self) -> str:
        return label_column_name(self.config.horizon)

    def _prepare_features(self, panel: pd.DataFrame, *, fit_median: bool = False) -> pd.DataFrame:
        X = panel[self.feature_cols].copy().astype(float)
        
        if fit_median:
            self.train_median_ = X.median()
        if self.train_median_ is None:
            raise RuntimeError("Call fit() before predict().")
        return X.fillna(self.train_median_)

    def fit(self, train_panel: pd.DataFrame) -> None:
        label = self.label_col
        if label not in train_panel.columns:
            raise ValueError(f"Training panel missing label column: {label}")
        X = self._prepare_features(train_panel, fit_median=True)
        y = train_panel[label].astype(float)
        params = dict(self.config.lgbm_params)
        print(f"DEBUG: y.min() = {y.min()}, y.max() = {y.max()}")
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(X, y)

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        X = self._prepare_features(panel)
        preds = self.model.predict(X)
        return pd.Series(preds, index=panel.index, name="score")

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

        self.fit(train_panel)
        scored = self.predict(test_panel)
        test_panel = test_panel.copy()
        test_panel["score"] = scored.values

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
            n_train_rows=len(train_panel),
            n_test_rows=len(test_panel),
            train_start=str(train_panel["date"].min().date()),
            train_end=train_end,
            test_start=test_start,
            test_end=str(test_panel["date"].max().date()),
            feature_importance_top10=importance,
        )
        return scores, summary

    def _top_feature_importance(self, top_n: int) -> dict[str, float]:
        if self.model is None:
            return {}
        cols = self.feature_cols
        pairs = sorted(
            zip(cols, self.model.feature_importances_, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return {name: float(val) for name, val in pairs[:top_n]}

    def scores_to_wide(self, scores: dict[str, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame({ticker: series for ticker, series in scores.items()}).sort_index()

    def save_scores(self, scores: dict[str, pd.Series], path: Path) -> None:
        wide = self.scores_to_wide(scores)
        path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(path)

    def save_summary(self, summary: SignalFitSummary, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
