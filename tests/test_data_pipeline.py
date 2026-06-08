import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_loader
from data_pipeline import BASE_FEATURE_COLS, CROSS_ASSET_COLS, build_feature_schema


def make_feature_frame(days=90, missing_col=None):
    index = pd.date_range("2024-01-01", periods=days, freq="B")
    data = {}
    for idx, col in enumerate(BASE_FEATURE_COLS):
        if col == missing_col:
            continue
        if col == "log_return":
            data[col] = np.linspace(-0.01, 0.01, days)
        elif col == "Volume_norm":
            data[col] = np.linspace(0.5, 1.5, days)
        elif col == "Close_norm":
            data[col] = np.linspace(-0.05, 0.05, days)
        else:
            data[col] = np.full(days, idx / 100)
    return pd.DataFrame(data, index=index)


class DataPipelineTests(unittest.TestCase):
    def test_data_loader_reexports_compatible_facade(self):
        self.assertTrue(callable(data_loader.fetch_multi_asset_data))
        self.assertEqual(tuple(data_loader.BASE_FEATURE_COLS), tuple(BASE_FEATURE_COLS))

    def test_feature_schema_reports_observation_dimension_and_missing_columns(self):
        schema = build_feature_schema(macro_features=("macro_x",), overnight_features=("overnight_y",))
        frame = pd.DataFrame(columns=list(BASE_FEATURE_COLS + CROSS_ASSET_COLS) + ["macro_x"])

        self.assertEqual(schema.observation_dim, len(BASE_FEATURE_COLS) + len(CROSS_ASSET_COLS) + 2)
        self.assertEqual(schema.missing_from(frame), ["overnight_y"])

    def test_fetch_multi_asset_data_adds_cross_asset_features_without_network(self):
        frames = {
            "2330.TW": make_feature_frame(),
            "2317.TW": make_feature_frame(),
            "2454.TW": make_feature_frame(),
        }

        def fake_fetch(ticker, **_kwargs):
            return frames[ticker]

        with patch("data_pipeline.multi.fetch_and_process_data", side_effect=fake_fetch):
            enriched = data_loader.fetch_multi_asset_data(
                tickers=list(frames),
                macro_tickers=[],
                start_date="2024-01-01",
                end_date="2024-04-30",
            )

        schema = build_feature_schema()
        self.assertEqual(set(enriched), set(frames))
        for frame in enriched.values():
            self.assertEqual(list(frame.columns), list(schema.columns))
            self.assertGreater(len(frame), 0)

    def test_fetch_multi_asset_data_fails_fast_on_missing_base_feature(self):
        frames = {
            "2330.TW": make_feature_frame(missing_col="RSI_norm"),
            "2317.TW": make_feature_frame(),
        }

        def fake_fetch(ticker, **_kwargs):
            return frames[ticker]

        with patch("data_pipeline.multi.fetch_and_process_data", side_effect=fake_fetch):
            with self.assertRaisesRegex(ValueError, "Feature schema missing columns"):
                data_loader.fetch_multi_asset_data(
                    tickers=list(frames),
                    macro_tickers=[],
                    start_date="2024-01-01",
                    end_date="2024-04-30",
                )


if __name__ == "__main__":
    unittest.main()
