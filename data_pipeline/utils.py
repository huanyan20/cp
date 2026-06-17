from dataclasses import dataclass

import numpy as np
import pandas as pd

# 基礎特徵欄位（包含 Milestone 3A 新增的 Feature Zoo）
BASE_FEATURE_COLS = [
    # Feature Group 1: Momentum Family
    "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "price_ma20_ratio", "price_ma60_ratio", "price_ma120_ratio",
    
    # Feature Group 2: Volatility Family
    "atr_20", "atr_60",
    "rolling_std_20", "rolling_std_60",
    
    # Feature Group 3: Liquidity Family
    "volume_zscore_20", "volume_zscore_60",
    "dollar_volume_log", "volume_ma60_ratio",
    
    # 保留部分基礎技術指標
    "RSI_14", "MACD_norm", "MACDh_norm", "BB_pct_b", "ADX_14",
    
    # Feature Group 4: Market Regime (Milestone 3B)
    "price_ma200_ratio",   # 200日均線比率：經典牛熊判斷
    "trend_slope_60d",     # 60日趨勢斜率（正=多頭，負=空頭）
    "above_ma120",         # 是否在120日線上（二元 Regime 旗標）
    
    # 用於計算 label 或 rank 的基礎
    "log_return",
]

# 根據 Milestone 3A，暫時移除無效的 sector_flow 等特徵
CROSS_ASSET_COLS = []

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
