from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from capital_flow_analysis.src.data_pipeline.market_data_provider import (  # noqa: E402
    CompositeProvider,
    extract_symbol_frame,
)
from capital_flow_analysis.src.data_pipeline.overnight_gap_features import (  # noqa: E402
    rolling_zscore,
)

CAPITAL_FLOW_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = CAPITAL_FLOW_DIR / "data" / "preopen_macro_check.json"
SYMBOLS = ["TSM", "^SOX", "^IXIC", "^VIX", "JPY=X", "DX-Y.NYB", "BTC-USD"]


def get_log_return_and_zscore(frame: pd.DataFrame, window: int = 120, min_periods: int = 20) -> tuple[float, float]:
    if frame.empty or "Close" not in frame or len(frame.dropna(subset=["Close"])) < 2:
        return np.nan, np.nan
    close = frame["Close"].dropna().astype(float)
    ret = np.log(close / close.shift(1))
    zscore = rolling_zscore(ret, window=window, min_periods=min_periods)
    return float(ret.iloc[-1]), float(zscore.iloc[-1])


def evaluate_preopen_macro(data: pd.DataFrame, provider: str, warnings: list[str]) -> dict:
    frames = {symbol: extract_symbol_frame(data, symbol) for symbol in SYMBOLS}
    
    metrics = {}
    for sym, key_prefix in [
        ("TSM", "tsm"), ("^SOX", "sox"), ("^IXIC", "ixic"), 
        ("^VIX", "vix"), ("DX-Y.NYB", "dxy"), ("BTC-USD", "btc")
    ]:
        ret, z = get_log_return_and_zscore(frames[sym])
        metrics[f"{key_prefix}_ret"] = ret
        metrics[f"{key_prefix}_ret_z"] = z

    jpy_ret, jpy_z = get_log_return_and_zscore(frames["JPY=X"])
    metrics["jpy_strength"] = -jpy_ret if pd.notna(jpy_ret) else np.nan
    metrics["jpy_strength_z"] = -jpy_z if pd.notna(jpy_z) else np.nan

    critical_reasons = []
    warn_reasons = []

    primary_keys = ["sox_ret", "vix_ret", "jpy_strength", "btc_ret"]
    missing = [k for k in primary_keys if pd.isna(metrics[k])]

    if warnings:
        warn_reasons.append("provider warnings present")
    if missing:
        warn_reasons.append(f"missing metrics: {', '.join(missing)}")
        
    if pd.notna(metrics["sox_ret_z"]):
        if metrics["sox_ret_z"] <= -3.0:
            critical_reasons.append(f"SOX Z-score <= -3.0 (ret={metrics['sox_ret']:.2%})")
        elif metrics["sox_ret_z"] <= -2.0:
            warn_reasons.append(f"SOX Z-score <= -2.0 (ret={metrics['sox_ret']:.2%})")

    if pd.notna(metrics["vix_ret_z"]):
        if metrics["vix_ret_z"] >= 3.0:
            critical_reasons.append(f"VIX Z-score >= 3.0 (ret={metrics['vix_ret']:.2%})")
        elif metrics["vix_ret_z"] >= 2.0:
            warn_reasons.append(f"VIX Z-score >= 2.0 (ret={metrics['vix_ret']:.2%})")

    if pd.notna(metrics["jpy_strength_z"]):
        if metrics["jpy_strength_z"] >= 3.0:
            critical_reasons.append(f"JPY Safe-haven Z-score >= 3.0 (ret={metrics['jpy_strength']:.2%})")
        elif metrics["jpy_strength_z"] >= 2.0:
            warn_reasons.append(f"JPY Safe-haven Z-score >= 2.0 (ret={metrics['jpy_strength']:.2%})")

    if pd.notna(metrics["btc_ret_z"]):
        if metrics["btc_ret_z"] <= -3.0:
            critical_reasons.append(f"BTC Z-score <= -3.0 (ret={metrics['btc_ret']:.2%})")
        elif metrics["btc_ret_z"] <= -2.0:
            warn_reasons.append(f"BTC Z-score <= -2.0 (ret={metrics['btc_ret']:.2%})")

    if critical_reasons:
        level = "CRITICAL"
    elif warn_reasons:
        level = "WARN"
    else:
        level = "OK"

    return {
        "level": level,
        "provider": provider,
        "warnings": warnings,
        "metrics": metrics,
        "critical_reasons": critical_reasons,
        "warn_reasons": warn_reasons,
    }


def run_check(period: str = "180d", output_path: Path = OUTPUT_PATH) -> dict:
    provider = CompositeProvider()
    result = provider.fetch(SYMBOLS, period=period, interval="1d")
    status = evaluate_preopen_macro(result.data, result.provider, result.warnings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-open macro risk check.")
    parser.add_argument("--period", default="180d")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = run_check(args.period, Path(args.output))
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
