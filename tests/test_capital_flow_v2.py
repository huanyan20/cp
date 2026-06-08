import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capital_flow_analysis.src.data_pipeline.market_calendar import (
    map_available_to_tw_trade_date,
    us_close_available_at_taipei,
)
from capital_flow_analysis.src.data_pipeline.market_data_provider import (
    CompositeProvider,
    FetchResult,
    LastGoodCacheProvider,
    combine_symbol_frames,
)
from capital_flow_analysis.src.modeling.evaluate_gap_model import (
    prepare_dataset,
    read_feature_data,
    run_evaluation,
)
from data_loader import load_overnight_features
from gnn_extractor import GnnFeatureExtractor, TemporalGnnFeatureExtractor


def frame(close_values, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(close_values), freq="B")
    close = np.asarray(close_values, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": np.full(len(close), 1000.0),
        },
        index=idx,
    )


class FakeProvider:
    def __init__(self, name, frames=None, fail_symbols=None):
        self.name = name
        self.frames = frames or {}
        self.fail_symbols = set(fail_symbols or [])

    def fetch(self, symbols, start=None, end=None, period=None, interval="1d"):
        warnings = []
        out = {}
        for symbol in symbols:
            if symbol in self.fail_symbols or symbol not in self.frames:
                warnings.append(f"{symbol}: unavailable")
            else:
                out[symbol] = self.frames[symbol]
        return FetchResult(combine_symbol_frames(out), self.name, warnings)


class CapitalFlowV2Tests(unittest.TestCase):
    def test_composite_provider_falls_back_per_symbol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            health_path = Path(tmpdir) / "health.json"
            cache = LastGoodCacheProvider(Path(tmpdir) / "cache")
            primary = FakeProvider("primary", {"AAA": frame([1, 2, 3])}, fail_symbols={"BBB"})
            secondary = FakeProvider("secondary", {"BBB": frame([4, 5, 6])})
            provider = CompositeProvider(
                providers=[primary, secondary],
                cache_provider=cache,
                health_path=health_path,
            )

            result = provider.fetch(["AAA", "BBB"], start="2026-01-01")
            self.assertEqual(result.provider, "composite")
            self.assertEqual(result.metadata["symbol_providers"], {"AAA": "primary", "BBB": "secondary"})
            self.assertIn("AAA", result.data.columns.get_level_values(0))
            self.assertIn("BBB", result.data.columns.get_level_values(0))
            health = json.loads(health_path.read_text(encoding="utf-8"))
            self.assertEqual(health["symbol_providers"]["BBB"], "secondary")

    def test_cache_provider_rejects_stale_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LastGoodCacheProvider(Path(tmpdir), max_age_days=1)
            stale = frame([1, 2], start="2000-01-03")
            stale.to_csv(Path(tmpdir) / "last_good_AAA.csv")

            result = cache.fetch(["AAA"])
            self.assertTrue(result.data.empty)
            self.assertTrue(any("stale" in warning for warning in result.warnings))

    def test_calendar_maps_after_open_to_next_trade_day(self):
        before_open = pd.Timestamp("2026-06-02 04:00", tz="Asia/Taipei")
        after_open = pd.Timestamp("2026-06-02 09:01", tz="Asia/Taipei")
        dates = pd.to_datetime(["2026-06-02", "2026-06-03"])
        self.assertEqual(map_available_to_tw_trade_date(before_open, dates), pd.Timestamp("2026-06-02"))
        self.assertEqual(map_available_to_tw_trade_date(after_open, dates), pd.Timestamp("2026-06-03"))

        summer = us_close_available_at_taipei(pd.Timestamp("2026-06-01"))
        winter = us_close_available_at_taipei(pd.Timestamp("2026-01-05"))
        self.assertEqual(summer.hour, 4)
        self.assertEqual(winter.hour, 5)

    def test_load_overnight_features_uses_historical_zscore_without_lookahead(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "overnight.csv"
            values = list(np.linspace(0.0, 0.19, 20)) + [10.0]
            pd.DataFrame(
                {
                    "tw_trade_date": pd.date_range("2026-01-01", periods=21, freq="B"),
                    "tsm_adr_premium": values,
                    "target_2330_full_day": np.linspace(0.0, 0.2, 21),
                }
            ).to_csv(path, index=False)

            features = load_overnight_features(path, ["tsm_adr_premium", "baseline_ret_prev"])
            self.assertIn("overnight_tsm_adr_premium", features)
            self.assertAlmostEqual(float(features["overnight_tsm_adr_premium"].iloc[-1]), 1.0)
            self.assertEqual(float(features["overnight_tsm_adr_premium"].iloc[0]), 0.0)

    def test_gap_fade_dataset_and_evaluation_smoke(self):
        rows = 70
        dates = pd.date_range("2025-01-01", periods=rows, freq="B")
        df = pd.DataFrame(
            {
                "tw_trade_date": dates,
                "target_2330_open_gap": np.sin(np.arange(rows) / 3) / 50 + 0.005,
                "target_2330_intraday": -np.sin(np.arange(rows) / 4) / 80,
                "target_2330_full_day": np.sin(np.arange(rows) / 5) / 60,
                "tsm_adr_premium": np.linspace(-0.01, 0.03, rows),
                "tsm_adr_premium_chg": np.sin(np.arange(rows)) / 100,
                "TSM_ret": np.cos(np.arange(rows)) / 100,
                "sox_ret": np.sin(np.arange(rows) / 2) / 100,
                "vix_ret": np.cos(np.arange(rows) / 2) / 100,
                "dxy_ret": np.sin(np.arange(rows) / 6) / 100,
                "jpy_strength": np.cos(np.arange(rows) / 7) / 100,
                "corporate_action_flag": False,
            }
        )
        prepared, target_col = prepare_dataset(df.set_index("tw_trade_date"), "gap_fade")
        self.assertEqual(target_col, "target_gap_fade")
        self.assertIn("target_gap_fade", prepared)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.csv"
            df.to_csv(path, index=False)
            result_df, _, report_path = run_evaluation("gap_fade", path)
            self.assertIn("F1", result_df.columns)
            self.assertTrue(report_path.exists())

    def test_evaluation_missing_dataset_has_actionable_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.csv"
            with self.assertRaisesRegex(FileNotFoundError, "overnight_gap_features.py"):
                read_feature_data(missing)

    def test_feature_extractors_forward_pass(self):
        obs_space = __import__("gymnasium").spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, 5 * 4 + 6),
            dtype=np.float32,
        )
        obs = torch.randn(2, 3, 26)

        gnn = GnnFeatureExtractor(obs_space, features_dim=32)
        temporal = TemporalGnnFeatureExtractor(
            obs_space,
            features_dim=32,
            window_size=5,
            account_features=6,
        )
        self.assertEqual(tuple(gnn(obs).shape), (2, 32))
        self.assertEqual(tuple(temporal(obs).shape), (2, 32))


if __name__ == "__main__":
    unittest.main()
