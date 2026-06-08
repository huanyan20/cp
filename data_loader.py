"""Backward-compatible wrapper for the `data_pipeline` package.

Use this module as a drop-in replacement for the previous
`data_loader.py` implementation. The heavy lifting now lives in
`data_pipeline.core`, `data_pipeline.multi`, and `data_pipeline.overnight`.
"""

from data_pipeline import (
    BASE_FEATURE_COLS,
    CROSS_ASSET_COLS,
    DEFAULT_OVERNIGHT_FEATURE_COLS,
    FeatureSchema,
    build_feature_schema,
    fetch_and_process_data,
    fetch_multi_asset_data,
    load_overnight_features,
    train_val_test_split,
)

__all__ = [
    "BASE_FEATURE_COLS",
    "CROSS_ASSET_COLS",
    "DEFAULT_OVERNIGHT_FEATURE_COLS",
    "FeatureSchema",
    "build_feature_schema",
    "fetch_and_process_data",
    "fetch_multi_asset_data",
    "load_overnight_features",
    "train_val_test_split",
]


if __name__ == "__main__":
    print("=== 測試 fetch_multi_asset_data ===")
    enriched = fetch_multi_asset_data(
        tickers=["2330.TW", "00919.TW", "00929.TW"],
        start_date="2023-06-01",
        end_date="2024-06-30",
        window_size=20,
    )

    for ticker, df in enriched.items():
        print(f"\n--- {ticker} ---")
        print(f"  Shape: {df.shape}")
        nan_count = df.isna().sum().sum()
        print(
            f"  NaN 檢查：{'[V] 無 NaN' if nan_count == 0 else f'[!] {nan_count} 個 NaN'}"
        )
        print(f"  欄位：{list(df.columns)}")
        print(
            df[["log_return", "peer1_logret", "corr_peer1_20d", "rel_strength"]]
            .tail(3)
            .round(5)
        )
