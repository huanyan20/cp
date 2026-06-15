"""Generate daily signal using SL LightGBM (h10) with daily retraining."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import fetch_multi_asset_data
from data_pipeline.universe_builder import get_universe_builder
from settings import load_settings
from sl_pipeline.allocator import MarketContext, PortfolioState
from sl_pipeline.backtest import build_vols_as_of
from sl_pipeline.rule_based_allocator import RuleBasedAllocator, RuleBasedAllocatorConfig
from sl_pipeline.signal_generator import SignalGenerator, SignalGeneratorConfig
from stock_universe import MACRO_TICKERS_RL

# Attempt to load RPA for current state
try:
    from rpa_pipeline.cmoney_rpa import CMoneyRPA
    HAS_RPA = True
except ImportError:
    HAS_RPA = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EvaluateSLLive")
SETTINGS = load_settings()


def fetch_latest_close(ticker: str) -> float | None:
    """Fetch a recent close for live lot sizing."""
    try:
        hist = yf.Ticker(ticker).history(period="5d")
    except Exception as e:
        logger.warning(f"Failed to fetch latest close for {ticker}: {e}")
        return None
    if hist.empty or "Close" not in hist:
        logger.warning(f"No recent close available for {ticker}")
        return None
    close = float(hist["Close"].dropna().iloc[-1])
    return close if close > 0 else None

def get_current_positions_and_mdd(aid: str | None) -> tuple[dict[str, float], float, float]:
    """Fetch current positions from CMoney and calculate live MDD."""
    if not HAS_RPA or not aid:
        return {}, 0.0, 0.0
    try:
        rpa = CMoneyRPA(aid=aid)
        status = rpa.get_account_status()
        rpa.close()
        
        inventory = status.get("inventory", {})
        total_assets = status.get("total_assets", 0.0)
        
        mdd = 0.0
        if total_assets > 0:
            equity_file = Path("capital_flow_analysis/data/live_equity_curve.json")
            history = {}
            if equity_file.exists():
                try:
                    history = json.loads(equity_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            recent_hist_vals = list(history.values())[-126:] if history else []
            peak_equity = max(recent_hist_vals + [total_assets])
            if peak_equity > 0:
                mdd = (peak_equity - total_assets) / peak_equity

        if total_assets <= 0:
            return {}, 0.0, 0.0
            
        inventory_qty = {}
        for ticker, qty in inventory.items():
            if qty > 0:
                inventory_qty[ticker] = float(qty)
        return inventory_qty, mdd, total_assets
    except Exception as e:
        logger.warning(f"Could not fetch current positions or MDD: {e}")
        return {}, 0.0, 0.0

def main():
    parser = argparse.ArgumentParser(description="Live Signal Generator for SL LightGBM h10")
    parser.add_argument("--horizon", type=int, default=10, help="Prediction horizon")
    parser.add_argument("--top-k", type=int, default=SETTINGS.research.default_topk, help="Number of stocks to select")
    parser.add_argument("--lookback-days", type=int, default=730, help="Days of data to fetch for training")
    parser.add_argument("--output", type=str, default=str(SETTINGS.paths.signal_path), help="Path to write signal.json")
    parser.add_argument("--aid", type=str, default=os.getenv("CMONEY_AID"), help="CMoney Account ID")
    args = parser.parse_args()

    today_str = datetime.now().strftime("%Y-%m-%d")
    start_date_str = (datetime.now() - timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")
    
    logger.info(f"Generating SL live signal for {today_str} (horizon={args.horizon}, train_start={start_date_str})")

    builder = get_universe_builder("dynamic")
    tickers = builder.build_universe(today_str, top_n=45)

    logger.info("Fetching market data...")
    # Fetch data up to today
    data = fetch_multi_asset_data(
        tickers=tickers,
        start_date=start_date_str,
        end_date=today_str,
        macro_tickers=MACRO_TICKERS_RL,
    )

    # Find the actual latest date in the fetched data
    latest_ts = pd.Timestamp("1900-01-01")
    for df in data.values():
        if not df.empty:
            df_max = pd.to_datetime(df.index.max() if "date" not in df.columns else df["date"].max())
            if df_max > latest_ts:
                latest_ts = df_max

    if latest_ts == pd.Timestamp("1900-01-01"):
        logger.error("No data fetched. Cannot proceed.")
        sys.exit(1)

    # Override today_str and yesterday_str based on actual data
    today_str = latest_ts.strftime("%Y-%m-%d")
    yesterday_str = (latest_ts - timedelta(days=1)).strftime("%Y-%m-%d")

    sg_config = SignalGeneratorConfig(horizon=args.horizon)
    generator = SignalGenerator(sg_config)

    logger.info(f"Fitting LightGBM model (train_end={yesterday_str})...")
    scores, summary = generator.fit_period(
        data,
        data,
        train_end=yesterday_str,
        test_start=today_str,
    )

    # Extract today's scores
    today_scores = {}
    last_score_date = None
    for ticker, series in scores.items():
        if not series.empty:
            last_date = series.index[-1]
            if last_score_date is None or last_date > last_score_date:
                last_score_date = last_date
            
            # Use the most recent score (assuming data is fresh)
            if (pd.Timestamp(today_str) - last_date).days <= 4:
                today_scores[ticker] = float(series.iloc[-1])

    if not today_scores:
        logger.error("No scores generated for recent dates. Is market data up-to-date?")
        sys.exit(1)

    logger.info(f"Scored {len(today_scores)} tickers for date {last_score_date.date() if last_score_date else 'unknown'}")
    top_scores = sorted(today_scores.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"Top 3 scores: {top_scores[:3]}")

    # Build vols
    vols = build_vols_as_of(
        data,
        tickers,
        pd.Timestamp(today_str),
        vol_window=20,
        min_vol_obs=5,
        vol_floor=0.05,
    )

    # Get current positions for hysteresis and calculate MDD
    inventory_qty, current_mdd, total_assets = get_current_positions_and_mdd(args.aid)
    logger.info(f"Current live MDD calculated as: {current_mdd*100:.2f}%")
    
    current_positions = {}
    if total_assets > 0:
        for ticker, qty in inventory_qty.items():
            if ticker in data and not data[ticker].empty:
                latest_close = fetch_latest_close(ticker)
                if latest_close is not None:
                    weight = (qty * latest_close) / total_assets
                    current_positions[ticker] = weight

    state = PortfolioState(
        positions=current_positions,
        cash_weight=max(0.0, 1.0 - sum(current_positions.values())),
        portfolio_value=1.0,
        peak_value=1.0,
        rolling_mdd=current_mdd
    )

    # Read macro guard
    market_context = None
    guard_path = SETTINGS.live.macro_guard_path
    guard_level = "OK"
    if guard_path.exists():
        try:
            guard_status = json.loads(guard_path.read_text(encoding="utf-8"))
            guard_level = guard_status.get("level", "OK")
            market_context = MarketContext(macro_guard_level=guard_level)
            logger.info(f"Loaded macro guard level: {guard_level}")
        except Exception as e:
            logger.warning(f"Failed to read macro guard: {e}")

    allocator = RuleBasedAllocator(RuleBasedAllocatorConfig(
        top_k=args.top_k,
        max_single_weight=SETTINGS.risk_limits.max_single_weight,
        target_vol_annual=SETTINGS.research.sl_target_vol,
        trailing_stop_threshold=SETTINGS.research.sl_trailing_stop
    ))

    logger.info("Allocating target weights...")
    target = allocator.allocate(today_scores, vols, state, market_context)
    weights = target.target_weights
    signal_id = f"sl-h{args.horizon}-{datetime.now().strftime('%Y%m%d')}"

    target_lots = {}
    if total_assets > 0:
        for ticker, weight in weights.items():
            if ticker in data and not data[ticker].empty:
                latest_close = fetch_latest_close(ticker)
                if latest_close is not None:
                    target_amt = total_assets * weight
                    lots = int(target_amt / (latest_close * 1000))
                    target_lots[ticker] = lots

    signal = {
        "signal_id": signal_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "aid": args.aid,
        "source": f"SL LightGBM h{args.horizon} Live",
        "target_weights": weights,
        "target_lots": target_lots,
        "metadata": {
            "horizon": args.horizon,
            "train_start": start_date_str,
            "train_end": yesterday_str,
            "top_features": list(summary.feature_importance_top10.keys())[:5],
            "scores": today_scores,
            "vols": vols,
            "hysteresis_held": list(current_positions.keys()),
            "rolling_mdd": current_mdd,
            "macro_guard_level": guard_level
        }
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(signal, indent=2, ensure_ascii=False), encoding="utf-8")
    
    logger.info(f"Live signal saved to {out_path}")
    logger.info(f"Target weights ({len(weights)} stocks): {weights}")
    logger.info(f"Cash weight: {target.cash_weight:.4f}")

if __name__ == "__main__":
    main()
