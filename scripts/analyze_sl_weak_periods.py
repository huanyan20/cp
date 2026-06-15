"""Analyze weak SL h10 periods by weights, PnL contribution, and Rank IC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import fetch_multi_asset_data
from data_pipeline.universe_builder import get_universe_builder
from settings import load_settings
from sl_pipeline.backtest import build_trading_calendar, simulate_period
from sl_pipeline.labels import build_labeled_panel, label_column_name, split_panel_by_date
from sl_pipeline.rule_based_allocator import RuleBasedAllocator, RuleBasedAllocatorConfig
from sl_pipeline.signal_generator import DEFAULT_LGBM_PARAMS, SignalGenerator, SignalGeneratorConfig
from sl_pipeline.walk_forward_sl import resolve_period, test_fetch_start
from stock_universe import MACRO_TICKERS_RL

SETTINGS = load_settings()


def _period_scores(seed: int, horizon: int, period_name: str):
    period = resolve_period(period_name)
    builder = get_universe_builder("dynamic")
    tickers = builder.build_universe(period["train_start"], top_n=45)

    train_data = fetch_multi_asset_data(
        tickers=tickers,
        start_date=period["train_start"],
        end_date=period["train_end"],
        macro_tickers=MACRO_TICKERS_RL,
    )
    test_data = fetch_multi_asset_data(
        tickers=tickers,
        start_date=test_fetch_start(period["test_start"]),
        end_date=period["test_end"],
        macro_tickers=MACRO_TICKERS_RL,
    )

    params = dict(DEFAULT_LGBM_PARAMS)
    params["random_state"] = seed
    generator = SignalGenerator(SignalGeneratorConfig(horizon=horizon, lgbm_params=params))
    scores, summary = generator.fit_period(
        train_data,
        test_data,
        train_end=period["train_end"],
        test_start=period["test_start"],
    )
    return period, tickers, train_data, test_data, scores, summary, generator


def _positions_frame(backtest: dict, tickers: list[str], dates: list[pd.Timestamp]) -> pd.DataFrame:
    position_dates = dates[1 : 1 + len(backtest["positions"])]
    return pd.DataFrame(backtest["positions"], index=position_dates, columns=tickers)


def _pnl_contribution_frame(
    positions: pd.DataFrame,
    enriched: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for date, weights in positions.iterrows():
        row = {}
        for ticker, weight in weights.items():
            if weight <= 1e-10 or ticker not in enriched or date not in enriched[ticker].index:
                row[ticker] = 0.0
                continue
            log_ret = float(enriched[ticker].loc[date, "log_return"])
            row[ticker] = float(weight) * (np.exp(log_ret) - 1.0)
        rows.append(row)
    return pd.DataFrame(rows, index=positions.index).fillna(0.0)


def _rank_ic_frame(
    test_data: dict[str, pd.DataFrame],
    scores: dict[str, pd.Series],
    generator: SignalGenerator,
    *,
    horizon: int,
    train_end: str,
    test_start: str,
) -> pd.DataFrame:
    label_col = label_column_name(horizon)
    panel = build_labeled_panel(test_data, horizon=horizon, feature_cols=generator.feature_cols)
    _, test_panel = split_panel_by_date(panel, train_end, test_start)

    preds = []
    for _, row in test_panel.iterrows():
        ticker = row["ticker"]
        date = row["date"]
        if ticker in scores and date in scores[ticker].index:
            preds.append(float(scores[ticker].loc[date]))
        else:
            preds.append(np.nan)
    test_panel["pred"] = preds
    test_panel = test_panel.dropna(subset=["pred", label_col])

    rows = []
    for date, frame in test_panel.groupby("date"):
        if len(frame) < 3:
            continue
        ic = spearmanr(frame["pred"], frame[label_col])[0]
        if np.isfinite(ic):
            rows.append({"date": pd.Timestamp(date), "rank_ic": float(ic), "n": int(len(frame))})
    return pd.DataFrame(rows).sort_values("date")


def _plot_period(
    *,
    period_name: str,
    seed: int,
    out_dir: Path,
    positions: pd.DataFrame,
    pnl_contrib: pd.DataFrame,
    portfolio_hist: list[float],
    ic: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_index = [positions.index[0] - pd.Timedelta(days=1)] + list(positions.index)
    equity = pd.Series(portfolio_hist[: len(equity_index)], index=equity_index)
    drawdown = equity / equity.cummax() - 1.0

    top_abs_weight = positions.abs().mean().sort_values(ascending=False).head(12).index
    top_loss = pnl_contrib.sum().sort_values().head(12).index

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)
    fig.suptitle(f"SL h10 weak-period diagnostics: {period_name} seed{seed}", fontsize=14)

    axes[0].plot(equity.index, equity.values, label="equity", color="#1f77b4")
    axes[0].set_title("Portfolio equity")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(drawdown.index, drawdown.values * 100.0, label="drawdown", color="#d62728")
    axes[1].set_title("Drawdown (%)")
    axes[1].grid(True, alpha=0.25)

    positions[top_abs_weight].plot.area(ax=axes[2], linewidth=0.0)
    axes[2].set_title("Top average portfolio weights")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    axes[2].grid(True, alpha=0.25)

    pnl_contrib[top_loss].cumsum().plot(ax=axes[3])
    axes[3].set_title("Cumulative PnL contribution: worst tickers")
    axes[3].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / f"{period_name}_seed{seed}_weights_pnl.png", dpi=150)
    plt.close(fig)

    if not ic.empty:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(ic["date"], ic["rank_ic"], width=1.0, color=np.where(ic["rank_ic"] >= 0, "#2ca02c", "#d62728"))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"Daily Rank IC: {period_name} seed{seed}")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{period_name}_seed{seed}_daily_ic.png", dpi=150)
        plt.close(fig)


def analyze_period(period_name: str, seed: int, horizon: int, out_dir: Path) -> dict:
    period, tickers, train_data, test_data, scores, summary, generator = _period_scores(seed, horizon, period_name)
    enriched = {**train_data, **test_data}
    allocator = RuleBasedAllocator(
        RuleBasedAllocatorConfig(
            top_k=SETTINGS.research.default_topk,
            max_single_weight=SETTINGS.risk_limits.max_single_weight,
        )
    )
    backtest = simulate_period(
        enriched,
        scores,
        allocator,
        tickers,
        test_start=period["test_start"],
        test_end=period["test_end"],
    )
    dates = build_trading_calendar(enriched, scores, test_start=period["test_start"], test_end=period["test_end"])
    positions = _positions_frame(backtest, tickers, dates)
    pnl_contrib = _pnl_contribution_frame(positions, enriched)
    ic = _rank_ic_frame(
        test_data,
        scores,
        generator,
        horizon=horizon,
        train_end=period["train_end"],
        test_start=period["test_start"],
    )

    period_dir = out_dir / period_name
    period_dir.mkdir(parents=True, exist_ok=True)
    positions.to_csv(period_dir / f"{period_name}_seed{seed}_weights.csv", encoding="utf-8")
    pnl_contrib.to_csv(period_dir / f"{period_name}_seed{seed}_pnl_contrib.csv", encoding="utf-8")
    ic.to_csv(period_dir / f"{period_name}_seed{seed}_rank_ic.csv", index=False, encoding="utf-8")
    _plot_period(
        period_name=period_name,
        seed=seed,
        out_dir=period_dir,
        positions=positions,
        pnl_contrib=pnl_contrib,
        portfolio_hist=backtest["portfolio_hist"],
        ic=ic,
    )

    total_pnl = pnl_contrib.sum().sort_values()
    summary_payload = {
        "period": period_name,
        "seed": seed,
        "horizon": horizon,
        "portfolio_return": float(backtest["portfolio_hist"][-1] / backtest["portfolio_hist"][0] - 1.0),
        "worst_contributors": {k: float(v) for k, v in total_pnl.head(10).items()},
        "best_contributors": {k: float(v) for k, v in total_pnl.tail(10).sort_values(ascending=False).items()},
        "avg_rank_ic": float(ic["rank_ic"].mean()) if not ic.empty else None,
        "negative_ic_days": int((ic["rank_ic"] < 0).sum()) if not ic.empty else 0,
        "ic_days": int(len(ic)),
        "top_features": summary.feature_importance_top10,
    }
    (period_dir / f"{period_name}_seed{seed}_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", default="2022_BEAR,2025H1", help="Comma-separated periods")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--output-dir", default="results_dir/sl_weak_period_analysis")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    periods = [p.strip() for p in args.periods.split(",") if p.strip()]
    summaries = [analyze_period(period, args.seed, args.horizon, out_dir) for period in periods]
    (out_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
