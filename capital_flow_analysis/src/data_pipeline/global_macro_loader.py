from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from capital_flow_analysis.src.data_pipeline.market_data_provider import (  # noqa: E402
    CompositeProvider,
    extract_symbol_frame,
)
from stock_universe import MACRO_TICKERS_FLOW  # noqa: E402

FLOW_TICKER_LABELS = {
    "^VIX": "VIX",
    "^TNX": "TNX",
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "NQ=F": "NQ_Futures",
    "ES=F": "ES_Futures",
    "JPY=X": "USD_JPY",
    "DX-Y.NYB": "DXY",
}


def fetch_global_macro_data(period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch broad macro data for observation and visualization.

    This loader is intentionally separate from the overnight gap feature pipeline.
    It is useful for dashboards, correlation checks, and quick market context, but
    should not be treated as a direct trading signal.
    """
    tickers = {
        FLOW_TICKER_LABELS.get(symbol, symbol): symbol
        for symbol in MACRO_TICKERS_FLOW
    }

    print(f"Fetching global macro data... Period: {period}, Interval: {interval}")

    provider = CompositeProvider()
    fetch_result = provider.fetch(list(tickers.values()), period=period, interval=interval)

    processed_data = pd.DataFrame()
    for name, symbol in tickers.items():
        symbol_df = extract_symbol_frame(fetch_result.data, symbol)
        if symbol_df.empty:
            continue
        if "Close" in symbol_df:
            processed_data[f"{name}_Close"] = symbol_df["Close"]
        if "Volume" in symbol_df:
            processed_data[f"{name}_Volume"] = symbol_df["Volume"]

    # Different markets have different trading calendars. Forward-fill is acceptable
    # here because this file is for monitoring and visualization, not label creation.
    processed_data = processed_data.ffill()

    output_dir = Path(__file__).resolve().parents[2] / "data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / f"global_macro_data_{interval}.csv"
    processed_data.to_csv(output_file)

    print(f"Data saved to {output_file}")
    if fetch_result.warnings:
        print(f"[WARN] Provider warnings: {fetch_result.warnings}")
    print(f"Provider used: {fetch_result.provider}")
    print("\nData Preview (Last 5 rows):")
    print(processed_data.tail())

    return processed_data


if __name__ == "__main__":
    fetch_global_macro_data(period="ytd", interval="1d")
