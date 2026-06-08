import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capital_flow_analysis.overnight_gap_features import (
    TAIPEI_TZ,
    align_us_asset_to_tw_dates,
    build_overnight_gap_features,
    map_available_to_tw_trade_date,
    us_close_available_at_taipei,
    write_feature_report,
)


def make_ohlcv(index, close, open_=None, volume=None, dividends=None):
    dates = pd.to_datetime(index)
    close = np.asarray(close, dtype=float)
    open_values = np.asarray(open_ if open_ is not None else close, dtype=float)
    volume_values = np.asarray(volume if volume is not None else np.full(len(close), 1000), dtype=float)
    df = pd.DataFrame(
        {
            "Open": open_values,
            "High": np.maximum(open_values, close),
            "Low": np.minimum(open_values, close),
            "Close": close,
            "Volume": volume_values,
        },
        index=dates,
    )
    if dividends is not None:
        df["Dividends"] = dividends
    return df


class OvernightGapFeatureTests(unittest.TestCase):
    def test_us_close_available_at_taipei_handles_dst(self):
        summer = us_close_available_at_taipei(pd.Timestamp("2026-06-01"))
        winter = us_close_available_at_taipei(pd.Timestamp("2026-01-05"))

        self.assertEqual(summer.date(), pd.Timestamp("2026-06-02").date())
        self.assertEqual(summer.hour, 4)
        self.assertEqual(winter.date(), pd.Timestamp("2026-01-06").date())
        self.assertEqual(winter.hour, 5)

    def test_available_timestamp_maps_to_next_taiwan_open(self):
        tw_dates = pd.to_datetime(["2026-06-02", "2026-06-03"])
        before_open = pd.Timestamp(datetime(2026, 6, 2, 4, 0, tzinfo=TAIPEI_TZ))
        after_open = pd.Timestamp(datetime(2026, 6, 2, 9, 1, tzinfo=TAIPEI_TZ))

        self.assertEqual(
            map_available_to_tw_trade_date(before_open, tw_dates),
            pd.Timestamp("2026-06-02"),
        )
        self.assertEqual(
            map_available_to_tw_trade_date(after_open, tw_dates),
            pd.Timestamp("2026-06-03"),
        )

    def test_us_holiday_alignment_compresses_multiple_sessions(self):
        us_df = make_ohlcv(
            ["2026-05-28", "2026-05-29", "2026-06-01"],
            [100.0, 110.0, 121.0],
        )
        aligned = align_us_asset_to_tw_dates(
            us_df,
            pd.to_datetime(["2026-06-02"]),
            "TEST",
        )

        row = aligned.loc[pd.Timestamp("2026-06-02")]
        self.assertAlmostEqual(row["TEST_close"], 121.0)
        self.assertAlmostEqual(row["TEST_ret"], np.log(121.0 / 100.0))
        self.assertAlmostEqual(row["TEST_source_age_hours"], 4 + 59 / 60, places=5)

    def test_build_features_calculates_premium_targets_and_risk_flags(self):
        tw_dates = ["2026-05-29", "2026-06-01", "2026-06-02", "2026-06-03"]
        us_dates = ["2026-05-28", "2026-05-29", "2026-06-01", "2026-06-02"]

        frames = {
            "2330.TW": make_ohlcv(
                tw_dates,
                close=[950.0, 1000.0, 1040.0, 1050.0],
                open_=[950.0, 970.0, 1060.0, 1035.0],
                dividends=[0.0, 0.0, 5.0, 0.0],
            ),
            "2303.TW": make_ohlcv(tw_dates, close=[50.0, 51.0, 52.0, 53.0]),
            "3711.TW": make_ohlcv(tw_dates, close=[120.0, 121.0, 122.0, 123.0]),
            "TSM": make_ohlcv(us_dates, close=[150.0, 155.0, 170.0, 171.0], volume=[100, 120, 200, 180]),
            "UMC": make_ohlcv(us_dates, close=[8.0, 8.2, 8.1, 8.0], volume=[80, 90, 95, 100]),
            "ASX": make_ohlcv(us_dates, close=[10.0, 10.1, 10.0, 9.9], volume=[70, 75, 80, 85]),
            "^SOX": make_ohlcv(us_dates, close=[100.0, 100.0, 90.0, 91.0]),
            "^IXIC": make_ohlcv(us_dates, close=[100.0, 100.0, 99.0, 100.0]),
            "^VIX": make_ohlcv(us_dates, close=[20.0, 20.0, 24.0, 23.0]),
            "JPY=X": make_ohlcv(us_dates, close=[150.0, 150.0, 145.0, 146.0]),
            "DX-Y.NYB": make_ohlcv(us_dates, close=[100.0, 101.0, 102.0, 101.0]),
            "USDTWD=X": make_ohlcv(
                ["2026-05-28", "2026-05-29", "2026-06-01", "2026-06-02"],
                close=[30.5, 30.8, 31.0, 31.1],
            ),
        }

        features = build_overnight_gap_features(frames)
        by_date = features.set_index("tw_trade_date")
        row = by_date.loc[pd.Timestamp("2026-06-02")]

        self.assertAlmostEqual(row["target_2330_open_gap"], np.log(1060.0 / 1000.0))
        self.assertAlmostEqual(row["target_2330_intraday"], np.log(1040.0 / 1060.0))
        self.assertAlmostEqual(row["tsm_adr_premium"], np.log((170.0 * 31.0 / 5.0) / 1000.0))
        self.assertAlmostEqual(row["tsm_adr_ret"], np.log(170.0 / 155.0))
        self.assertAlmostEqual(row["sox_ret"], np.log(90.0 / 100.0))
        self.assertAlmostEqual(row["ixic_ret"], np.log(99.0 / 100.0))
        self.assertGreater(row["jpy_strength"], 0.0)
        self.assertTrue(bool(row["corporate_action_flag"]))
        self.assertEqual(row["fx_data_source_risk"], 1)
        self.assertIn("semi_adr_weighted_ret", features.columns)

    def test_feature_report_excludes_targets_from_feature_ranking(self):
        df = pd.DataFrame(
            {
                "tw_trade_date": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"]),
                "target_2330_open_gap": [0.01, -0.02, 0.03],
                "target_2330_intraday": [0.02, -0.01, 0.04],
                "target_2330_full_day": [0.03, -0.03, 0.07],
                "tsm_adr_ret": [0.011, -0.018, 0.029],
                "fx_data_source_risk": [1, 1, 1],
                "corporate_action_flag": [False, False, False],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            write_feature_report(df, report_path)
            report = report_path.read_text(encoding="utf-8")

        top_section = report.split("## Top Open-Gap Correlations", 1)[1].split(
            "## Highest Missing Rates",
            1,
        )[0]
        self.assertNotIn("target_2330_full_day", top_section)
        self.assertNotIn("target_2330_intraday", top_section)
        self.assertIn("tsm_adr_ret", top_section)


if __name__ == "__main__":
    unittest.main()
