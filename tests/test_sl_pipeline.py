import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline import BASE_FEATURE_COLS, CROSS_ASSET_COLS, build_feature_schema
from env_config import build_env_config_snapshot
from promotion_gate import run_promotion_gate
from sl_pipeline.allocator import MarketContext, PortfolioState
from sl_pipeline.backtest import (
    build_trading_calendar,
    execute_rebalance,
    metrics_from_backtest,
    simulate_period,
    trade_cost_rate,
)
from sl_pipeline.comparison import build_sl_vs_rl_comparison
from sl_pipeline.gate import (
    build_sl_raw_summary,
    read_sl_metric_files,
    run_sl_promotion_gate,
)
from sl_pipeline.labels import (
    build_cross_demean_frame,
    build_labeled_panel,
    forward_log_return_t1,
    label_column_name,
    split_panel_by_date,
)
from sl_pipeline.rl_allocator import RLAllocator, RLAllocatorConfig
from sl_pipeline.rl_spike import validate_spike
from sl_pipeline.rule_based_allocator import (
    RuleBasedAllocator,
    RuleBasedAllocatorConfig,
)
from sl_pipeline.signal_generator import SignalGenerator, SignalGeneratorConfig
from sl_pipeline.sl_features import (
    SL_FEATURES_PER_STOCK,
    build_sl_feature_arrays,
    cross_sectional_rank_norm,
    cross_sectional_zscore,
)


def make_enriched_frame(days: int = 40, log_slope: float = 0.01) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=days, freq="B")
    data: dict[str, np.ndarray] = {}
    for idx, col in enumerate(BASE_FEATURE_COLS):
        if col == "log_return":
            data[col] = np.full(days, log_slope)
        elif col in ("Volume_norm", "Close_norm"):
            data[col] = np.linspace(0.5, 1.5, days)
        else:
            data[col] = np.full(days, idx / 100.0)
    for idx, col in enumerate(CROSS_ASSET_COLS):
        data[col] = np.full(days, (idx + 1) / 50.0)
    return pd.DataFrame(data, index=index)


def make_enriched_dict(tickers: list[str], **kwargs) -> dict[str, pd.DataFrame]:
    return {ticker: make_enriched_frame(**kwargs) for ticker in tickers}


class SlLabelsTests(unittest.TestCase):
    def test_forward_log_return_t1_sums_post_execution_days(self):
        index = pd.date_range("2024-01-01", periods=8, freq="B")
        log_return = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], index=index)
        out = forward_log_return_t1(log_return, end_day=5)
        # t=0 -> log(P5/P1) = log_return[t+2..t+5] = 0.3+0.4+0.5+0.6 = 1.8
        self.assertAlmostEqual(out.iloc[0], 1.8)
        self.assertTrue(np.isnan(out.iloc[-1]))

    def test_cross_demean_removes_universe_median(self):
        enriched = {
            "A": make_enriched_frame(log_slope=0.02),
            "B": make_enriched_frame(log_slope=0.02),
        }
        demean = build_cross_demean_frame(enriched, horizon=5)
        valid = demean.dropna()
        self.assertGreater(len(valid), 0)
        self.assertTrue(np.allclose(valid["A"], valid["B"], atol=1e-9))
        self.assertTrue(np.allclose(valid.mean(axis=1), 0.0, atol=1e-9))

    def test_build_labeled_panel_has_expected_columns(self):
        enriched = make_enriched_dict(["2330.TW", "2317.TW"])
        panel = build_labeled_panel(enriched, horizon=5)
        label = label_column_name(5)
        schema = build_feature_schema()
        for col in schema.columns:
            self.assertIn(col, panel.columns)
        self.assertIn(label, panel.columns)
        self.assertIn("ticker", panel.columns)
        self.assertIn("date", panel.columns)
        self.assertEqual(panel[label].notna().sum(), len(panel))

    def test_split_panel_by_date_is_time_ordered(self):
        enriched = make_enriched_dict(["2330.TW"])
        panel = build_labeled_panel(enriched, horizon=5)
        train, test = split_panel_by_date(panel, "2024-01-15", "2024-01-20")
        self.assertTrue((train["date"] <= pd.Timestamp("2024-01-15")).all())
        self.assertTrue((test["date"] >= pd.Timestamp("2024-01-20")).all())


class SignalGeneratorTests(unittest.TestCase):
    def test_fit_predict_on_synthetic_panel(self):
        enriched = make_enriched_dict(["2330.TW", "2317.TW"], days=60)
        train_panel = build_labeled_panel(enriched, horizon=5)
        generator = SignalGenerator(
            SignalGeneratorConfig(
                horizon=5,
                lgbm_params={"n_estimators": 10, "verbosity": -1, "random_state": 42},
            )
        )
        generator.fit(train_panel)
        preds = generator.predict(train_panel)
        self.assertEqual(len(preds), len(train_panel))
        self.assertTrue(np.isfinite(preds).all())

    def test_fit_period_produces_per_ticker_series(self):
        tickers = ["2330.TW", "2317.TW"]
        train = make_enriched_dict(tickers, days=80)
        test = make_enriched_dict(tickers, days=40)
        # shift test dates so split works
        for ticker in tickers:
            test[ticker].index = pd.date_range("2024-05-01", periods=40, freq="B")

        generator = SignalGenerator(
            SignalGeneratorConfig(
                horizon=5,
                lgbm_params={"n_estimators": 10, "verbosity": -1, "random_state": 42},
            )
        )
        scores, summary = generator.fit_period(
            train,
            test,
            train_end="2024-04-30",
            test_start="2024-05-01",
        )
        self.assertEqual(set(scores), set(tickers))
        for series in scores.values():
            self.assertGreater(len(series), 0)
            self.assertTrue(np.isfinite(series).all())
        self.assertGreater(summary.n_train_rows, 0)
        self.assertGreater(summary.n_test_rows, 0)

    def test_walk_forward_sl_single_period_offline(self):
        tickers = ["2330.TW", "2317.TW"]
        train_frames = make_enriched_dict(tickers, days=80)
        test_frames = make_enriched_dict(tickers, days=30)
        for ticker in tickers:
            test_frames[ticker].index = pd.date_range("2025-01-01", periods=30, freq="B")

        def fake_fetch(*, tickers, start_date, end_date, **_kwargs):
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            out = {}
            for ticker in tickers:
                src = train_frames[ticker] if end <= pd.Timestamp("2024-12-31") else test_frames[ticker]
                out[ticker] = src.loc[(src.index >= start) & (src.index <= end)].copy()
            return out

        with (
            patch("sl_pipeline.walk_forward_sl.fetch_multi_asset_data", side_effect=fake_fetch),
            patch("sl_pipeline.walk_forward_sl.TICKERS_TECH_EXPANDED", tickers),
        ):
            from sl_pipeline.walk_forward_sl import run_single_period

            result = run_single_period("2025H1", horizon=5, tickers=tickers)
        self.assertEqual(result["period"], "2025H1")
        self.assertGreater(result["n_oos_days"], 0)
        self.assertGreater(result["n_tickers"], 0)


class RuleBasedAllocatorTests(unittest.TestCase):
    def test_top_k_selects_highest_scores(self):
        allocator = RuleBasedAllocator(RuleBasedAllocatorConfig(top_k=2))
        scores = {"A": 0.9, "B": 0.8, "C": 0.1, "D": 0.05}
        vols = {k: 0.2 for k in scores}
        state = PortfolioState()
        target = allocator.allocate(scores, vols, state)
        self.assertEqual(set(target.target_weights), {"A", "B"})
        self.assertAlmostEqual(sum(target.target_weights.values()) + target.cash_weight, 1.0, places=5)

    def test_yellow_mdd_caps_exposure(self):
        allocator = RuleBasedAllocator(
            RuleBasedAllocatorConfig(top_k=2, target_vol_annual=1.0, yellow_mdd=0.10, yellow_max_exposure=0.50)
        )
        scores = {"A": 1.0, "B": 0.9}
        vols = {"A": 0.2, "B": 0.2}
        state = PortfolioState(rolling_mdd=0.11)
        target = allocator.allocate(scores, vols, state)
        self.assertLessEqual(sum(target.target_weights.values()), 0.50 + 1e-6)

    def test_red_mdd_caps_exposure(self):
        allocator = RuleBasedAllocator(
            RuleBasedAllocatorConfig(top_k=2, target_vol_annual=1.0, red_mdd=0.15, red_max_exposure=0.10)
        )
        scores = {"A": 1.0, "B": 0.9}
        vols = {"A": 0.2, "B": 0.2}
        state = PortfolioState(rolling_mdd=0.16)
        target = allocator.allocate(scores, vols, state)
        self.assertLessEqual(sum(target.target_weights.values()), 0.10 + 1e-6)

    def test_hysteresis_keeps_top10_holding(self):
        allocator = RuleBasedAllocator(RuleBasedAllocatorConfig(top_k=2, hysteresis_rank=10))
        scores = {f"S{i}": float(10 - i) for i in range(12)}
        vols = {k: 0.2 for k in scores}
        state = PortfolioState(positions={"S5": 0.5})
        target = allocator.allocate(scores, vols, state)
        self.assertIn("S5", target.target_weights)

    def test_weight_band_preserves_small_delta(self):
        allocator = RuleBasedAllocator(RuleBasedAllocatorConfig(top_k=1, weight_band=0.05, target_vol_annual=1.0))
        scores = {"A": 1.0}
        vols = {"A": 0.2}
        state = PortfolioState(positions={"A": 0.30})
        target = allocator.allocate(scores, vols, state)
        self.assertAlmostEqual(target.target_weights["A"], 0.30, places=3)

    def test_macro_critical_flattens_exposure(self):
        allocator = RuleBasedAllocator(RuleBasedAllocatorConfig(top_k=2, target_vol_annual=1.0))
        scores = {"A": 1.0, "B": 0.9}
        vols = {"A": 0.2, "B": 0.2}
        state = PortfolioState()
        target = allocator.allocate(scores, vols, state, MarketContext(macro_guard_level="CRITICAL"))
        self.assertEqual(sum(target.target_weights.values()), 0.0)
        self.assertAlmostEqual(target.cash_weight, 1.0)


class SlBacktestTests(unittest.TestCase):
    def test_trade_cost_includes_sell_tax(self):
        self.assertGreater(trade_cost_rate(0.2, 0.0), trade_cost_rate(0.0, 0.2))

    def test_execute_rebalance_deducts_cost(self):
        weights, turnover, new_value = execute_rebalance(
            1.0,
            {"A": 0.0},
            {"A": 0.5},
            ["A"],
        )
        self.assertGreater(turnover, 0.0)
        self.assertLess(new_value, 1.0)
        self.assertAlmostEqual(weights["A"], 0.5)

    def test_simulate_period_produces_gate_metrics(self):
        tickers = ["2330.TW", "2317.TW"]
        enriched = make_enriched_dict(tickers, days=50)
        scores = {}
        for ticker in tickers:
            idx = enriched[ticker].index
            scores[ticker] = pd.Series(
                np.linspace(0.1, 0.5, len(idx)),
                index=idx,
                name=ticker,
            )
        allocator = RuleBasedAllocator(RuleBasedAllocatorConfig(top_k=1, target_vol_annual=0.5))
        start = str(enriched[tickers[0]].index[10].date())
        end = str(enriched[tickers[0]].index[-2].date())
        result = simulate_period(
            enriched,
            scores,
            allocator,
            tickers,
            test_start=start,
            test_end=end,
        )
        self.assertGreater(result["n_days"], 0)
        metrics = metrics_from_backtest(
            result,
            tickers,
            period_name="test",
            test_start=start,
            test_end=end,
        )
        for key in ("total_return", "max_drawdown", "sortino", "turnover", "avg_cash_weight"):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))

    def test_build_trading_calendar_intersects_score_dates(self):
        tickers = ["A", "B"]
        enriched = make_enriched_dict(tickers, days=30)
        scores = {
            "A": pd.Series(np.arange(30), index=enriched["A"].index),
            "B": pd.Series(np.arange(30), index=enriched["B"].index),
        }
        dates = build_trading_calendar(
            enriched,
            scores,
            test_start="2024-01-10",
            test_end="2024-01-25",
        )
        self.assertGreater(len(dates), 0)

    def test_walk_forward_sl_single_period_with_allocator(self):
        tickers = ["2330.TW", "2317.TW"]
        train_frames = make_enriched_dict(tickers, days=80)
        test_frames = make_enriched_dict(tickers, days=30)
        for ticker in tickers:
            test_frames[ticker].index = pd.date_range("2025-01-01", periods=30, freq="B")

        def fake_fetch(*, tickers, start_date, end_date, **_kwargs):
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            out = {}
            for ticker in tickers:
                src = train_frames[ticker] if end <= pd.Timestamp("2024-12-31") else test_frames[ticker]
                out[ticker] = src.loc[(src.index >= start) & (src.index <= end)].copy()
            return out

        with (
            patch("sl_pipeline.walk_forward_sl.fetch_multi_asset_data", side_effect=fake_fetch),
            patch("sl_pipeline.walk_forward_sl.TICKERS_TECH_EXPANDED", tickers),
        ):
            from sl_pipeline.walk_forward_sl import run_single_period

            result = run_single_period(
                "2025H1",
                horizon=5,
                allocator_name="rule",
                tickers=tickers,
            )
        self.assertEqual(result["allocator"], "rule")
        self.assertIn("backtest", result)
        metrics = result["backtest"]["metrics"]
        self.assertIn("sortino", metrics)
        self.assertIn("max_drawdown", metrics)


class SlGateTests(unittest.TestCase):
    def _write_sl_metrics(self, directory: Path, seed: int = 42, sortino: float = 1.2) -> Path:
        payload = {
            "strategy": "sl_rule",
            "allocator": "rule",
            "algo": "sl_lightgbm",
            "horizon": 5,
            "seed": seed,
            "cash_mode": "enabled",
            "overall": {
                "sortino": sortino,
                "max_drawdown": 0.12,
                "total_return": 0.15,
                "avg_cash_weight": 0.20,
                "cash_weight_std": 0.05,
                "cash_corr_next_return": -0.1,
                "turnover": 0.03,
                "win_rate": 0.52,
            },
            "periods": {
                "2024H2": {
                    "sortino": sortino,
                    "max_drawdown": 0.10,
                    "total_return": 0.05,
                    "avg_cash_weight": 0.18,
                    "cash_weight_std": 0.04,
                    "turnover": 0.02,
                    "win_rate": 0.51,
                    "test_start": "2024-07-01",
                    "test_end": "2024-12-31",
                },
                "2025H1": {
                    "sortino": sortino,
                    "max_drawdown": 0.08,
                    "total_return": 0.06,
                    "avg_cash_weight": 0.22,
                    "cash_weight_std": 0.06,
                    "turnover": 0.03,
                    "win_rate": 0.53,
                    "test_start": "2025-01-01",
                    "test_end": "2025-06-30",
                },
            },
        }
        path = directory / f"metrics_sl_rule_h5_seed{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_read_sl_metric_files_ignores_rl_namespace(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_sl_metrics(tmp_path)
            (tmp_path / "metrics_sac_enabled_wf_seed42.json").write_text("{}", encoding="utf-8")
            records = read_sl_metric_files(tmp_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["horizon"], 5)

    def test_build_sl_raw_summary_groups_seeds(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_sl_metrics(tmp_path, seed=42, sortino=1.2)
            self._write_sl_metrics(tmp_path, seed=43, sortino=1.0)
            records = read_sl_metric_files(tmp_path)
            summary = build_sl_raw_summary(records)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["seeds"], [42, 43])
        self.assertAlmostEqual(summary[0]["sortino_mean"], 1.1)

    def test_run_sl_promotion_gate_passes_synthetic_metrics(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = self._write_sl_metrics(tmp_path, sortino=1.5)
            result, raw_summary, period_df = run_sl_promotion_gate(path)
        self.assertEqual(len(raw_summary), 1)
        self.assertGreater(len(period_df), 0)
        self.assertTrue(result.can_promote)


class SlRlSpikeTests(unittest.TestCase):
    def test_sl_features_extend_observation_dim(self):
        tickers = ["2330.TW", "2317.TW"]
        enriched = make_enriched_dict(tickers, days=60)
        scores = {
            t: pd.Series(np.linspace(0.1, 0.5, 60), index=enriched[t].index, name=t)
            for t in tickers
        }
        result = validate_spike(enriched, scores, tickers)
        self.assertTrue(result["spike_ok"])
        self.assertEqual(result["obs_dim_delta"], SL_FEATURES_PER_STOCK)

    def test_cross_sectional_features(self):
        scores = {"A": 0.9, "B": 0.1, "C": 0.5}
        z = cross_sectional_zscore(scores)
        ranks = cross_sectional_rank_norm(scores)
        self.assertAlmostEqual(z["A"], max(z.values()))
        self.assertEqual(ranks["A"], 1.0)
        self.assertEqual(ranks["B"], 0.0)

    def test_rl_allocator_residual_stays_near_baseline(self):
        allocator = RLAllocator(RLAllocatorConfig(residual_scale=0.05, top_k=2))
        scores = {"A": 1.0, "B": 0.5, "C": 0.1}
        vols = {k: 0.2 for k in scores}
        state = PortfolioState()
        baseline = allocator.allocate(scores, vols, state)
        action = np.zeros(3, dtype=float)
        target = allocator.allocate_from_action(action, scores, vols, state, list(scores))
        self.assertGreater(len(target.target_weights), 0)
        self.assertLessEqual(sum(target.target_weights.values()) + target.cash_weight, 1.0 + 1e-6)
        self.assertGreaterEqual(len(baseline.target_weights), 1)

    def test_env_config_hash_changes_with_sl_features(self):
        base = build_env_config_snapshot()
        with_sl = build_env_config_snapshot(sl_features="v1")
        self.assertEqual(base["hash"], build_env_config_snapshot()["hash"])
        self.assertNotEqual(base["hash"], with_sl["hash"])
        self.assertEqual(with_sl["sl_features"], "v1")

    def test_build_train_env_accepts_sl_features(self):
        from unittest.mock import patch

        from research_pipeline import build_train_env
        from settings import load_settings

        tickers = ["2330.TW", "2317.TW"]
        enriched = make_enriched_dict(tickers, days=60)
        scores = {
            t: pd.Series(np.linspace(0.1, 0.5, 60), index=enriched[t].index, name=t)
            for t in tickers
        }
        settings = load_settings()
        with patch("research_pipeline.fetch_multi_asset_data", return_value=enriched):
            env, _ = build_train_env(
                tickers=tickers,
                train_start="2024-01-01",
                train_end="2024-03-31",
                window_size=20,
                macro_tickers=[],
                settings=settings,
                enable_sl_features=True,
                sl_scores=scores,
            )
        self.assertTrue(env.enable_sl_features)
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape[1], env._obs_dim_per_stock)
        arrays = build_sl_feature_arrays(enriched, scores, tickers)
        self.assertEqual(arrays["2330.TW"].shape[1], SL_FEATURES_PER_STOCK)


class SlComparisonTests(unittest.TestCase):
    def test_build_sl_vs_rl_comparison_marks_mdd_winner(self):
        raw_summary = [
            {
                "algo": "sac",
                "cash_mode": "enabled",
                "variant": "base",
                "seeds": [42],
                "sortino_mean": 2.0,
                "sortino_std": 0.0,
                "max_drawdown_mean": 0.20,
                "max_drawdown_std": 0.0,
                "total_return_mean": 0.25,
                "total_return_std": 0.0,
                "turnover_mean": 0.04,
                "turnover_std": 0.0,
                "avg_cash_weight_mean": 0.05,
                "cash_weight_std_mean": 0.04,
                "win_rate_mean": 0.55,
                "cash_behavior": "active cash",
            }
        ]
        sl_raw_summary = [
            {
                "algo": "sl_lightgbm",
                "cash_mode": "enabled",
                "variant": "sl_rule_h5",
                "horizon": 5,
                "seeds": [42],
                "sortino_mean": 1.7,
                "sortino_std": 0.0,
                "max_drawdown_mean": 0.12,
                "max_drawdown_std": 0.0,
                "total_return_mean": 0.18,
                "total_return_std": 0.0,
                "turnover_mean": 0.02,
                "turnover_std": 0.0,
                "avg_cash_weight_mean": 0.15,
                "cash_weight_std_mean": 0.05,
                "win_rate_mean": 0.53,
                "cash_behavior": "active cash",
            }
        ]
        period_df = pd.DataFrame(
            [
                {
                    "algo": "sac",
                    "cash_mode": "enabled",
                    "variant": "base",
                    "seed": 42,
                    "period": "2024H2",
                    "sortino": 2.0,
                    "max_drawdown": 0.20,
                    "total_return": 0.10,
                    "turnover": 0.04,
                    "avg_cash_weight": 0.05,
                    "win_rate": 0.55,
                }
            ]
        )
        sl_period_df = pd.DataFrame(
            [
                {
                    "algo": "sl_lightgbm",
                    "cash_mode": "enabled",
                    "variant": "sl_rule_h5",
                    "seed": 42,
                    "period": "2024H2",
                    "sortino": 1.7,
                    "max_drawdown": 0.12,
                    "total_return": 0.08,
                    "turnover": 0.02,
                    "avg_cash_weight": 0.15,
                    "win_rate": 0.53,
                }
            ]
        )
        rl_gate = run_promotion_gate(raw_summary, period_df=period_df, min_seeds=1)
        sl_gate = run_promotion_gate(sl_raw_summary, period_df=sl_period_df, min_seeds=1)
        comparison = build_sl_vs_rl_comparison(
            raw_summary=raw_summary,
            sl_raw_summary=sl_raw_summary,
            period_df=period_df,
            sl_period_df=sl_period_df,
            rl_promotion=rl_gate,
            sl_promotion=sl_gate,
        )
        mdd_row = next(r for r in comparison["overall"] if r["metric"] == "max_drawdown")
        self.assertEqual(mdd_row["winner"], "SL")
        self.assertTrue(any("MDD" in line for line in comparison["verdict"]))


if __name__ == "__main__":
    unittest.main()
