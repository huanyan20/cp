from dataclasses import dataclass

import numpy as np
import pandas as pd

# 基礎特徵欄位（包含 v8.0 新增的動能特徵）
BASE_FEATURE_COLS = [
    "Open_norm",
    "High_norm",
    "Low_norm",
    "Close_norm",
    "Volume_norm",
    "RSI_norm",
    "MACD_norm",
    "MACDs_norm",
    "MACDh_norm",
    "BBU_norm",
    "BBL_norm",
    "BB_bandwidth",
    "BB_pct_b",
    "ADX_norm",
    "DMP_norm",
    "DMN_norm",
    "ATR_norm",
    "STOCHk_norm",
    "STOCHd_norm",
    "OBV_norm",
    "MFI_norm",
    "log_return",
    "open_return",
    "mom_60d",
    "ma60_bias",
]

CROSS_ASSET_COLS = [
    "peer1_logret",
    "peer2_logret",
    "corr_peer1_20d",
    "corr_peer2_20d",
    "rel_strength",
    "sector_flow",
    "market_breadth",
]

DEFAULT_OVERNIGHT_FEATURE_COLS = [
    "tsm_adr_premium_chg",
    "tsm_adr_premium",
    "TSM_ret",
]


@dataclass(frozen=True)
class FeatureSchema:
    base_features: tuple[str, ...]
    cross_asset_features: tuple[str, ...]
    macro_features: tuple[str, ...] = ()
    overnight_features: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        return (
            self.base_features
            + self.cross_asset_features
            + self.macro_features
            + self.overnight_features
        )

    @property
    def observation_dim(self) -> int:
        return len(self.columns)

    def missing_from(self, frame: pd.DataFrame) -> list[str]:
        return [col for col in self.columns if col not in frame.columns]

    def validate(self, frame: pd.DataFrame) -> None:
        missing = self.missing_from(frame)
        if missing:
            raise ValueError(f"Feature schema missing columns: {missing}")


def build_feature_schema(
    macro_features: list[str] | tuple[str, ...] | None = None,
    overnight_features: list[str] | tuple[str, ...] | None = None,
) -> FeatureSchema:
    return FeatureSchema(
        base_features=tuple(BASE_FEATURE_COLS),
        cross_asset_features=tuple(CROSS_ASSET_COLS),
        macro_features=tuple(macro_features or ()),
        overnight_features=tuple(overnight_features or ()),
    )


def _historical_zscore_clip(series: pd.Series, window: int = 252) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    history = values.shift(1)
    mean = history.rolling(window, min_periods=20).mean()
    std = history.rolling(window, min_periods=20).std().replace(0, np.nan)
    z = (values - mean) / std
    return (z.clip(-3.0, 3.0) / 3.0).fillna(0.0)


def train_val_test_split(
    df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(
        f"資料切分完成 → Train: {len(train_df)} 筆 | Val: {len(val_df)} 筆 | Test: {len(test_df)} 筆"
    )
    return train_df, val_df, test_df
