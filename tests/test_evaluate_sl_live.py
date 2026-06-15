import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.evaluate_sl_live as eval_live


def test_evaluate_sl_live_calculates_lots_correctly():
    # Setup mocks
    with patch("scripts.evaluate_sl_live.fetch_latest_close") as mock_fetch_close, \
         patch("scripts.evaluate_sl_live.get_current_positions_and_mdd") as mock_get_pos, \
         patch("scripts.evaluate_sl_live.fetch_multi_asset_data") as mock_fetch_data, \
         patch("scripts.evaluate_sl_live.get_universe_builder") as mock_get_universe, \
         patch("scripts.evaluate_sl_live.SignalGenerator.fit_period") as mock_fit_period, \
         patch("scripts.evaluate_sl_live.build_vols_as_of") as mock_build_vols, \
         patch("scripts.evaluate_sl_live.Path.write_text") as mock_write, \
         patch("scripts.evaluate_sl_live.Path.mkdir"):

        # Mock universe
        mock_builder = MagicMock()
        mock_builder.build_universe.return_value = ["2330", "2317"]
        mock_get_universe.return_value = mock_builder

        # Mock data (need actual dates)
        today = pd.Timestamp("2026-06-15")
        df = pd.DataFrame({"Close": [100.0]}, index=[today])
        mock_fetch_data.return_value = {"2330": df, "2317": df}

        # Mock latest close
        def mock_latest_close(ticker):
            return 900.0 if ticker == "2330" else 200.0
        mock_fetch_close.side_effect = mock_latest_close

        # Mock positions & total assets
        mock_get_pos.return_value = ({}, 0.0, 1_000_000.0) # total_assets = 1M

        # Mock fit_period scores
        scores = {
            "2330": pd.Series([1.0], index=[today]),
            "2317": pd.Series([0.5], index=[today])
        }
        mock_summary = MagicMock()
        mock_summary.feature_importance_top10 = {"f1": 1.0}
        mock_fit_period.return_value = (scores, mock_summary)

        # Mock vols
        mock_build_vols.return_value = {"2330": 0.2, "2317": 0.2}

        # Instead of running the whole main(), we can run main() with sys.argv mocked
        with patch.object(sys, 'argv', ['evaluate_sl_live.py', '--output', 'dummy.json']):
            eval_live.main()
            
        # Verify the write_text call
        assert mock_write.called
        import json
        written_content = json.loads(mock_write.call_args[0][0])
        
        target_weights = written_content["target_weights"]
        target_lots = written_content["target_lots"]
        metadata = written_content["metadata"]
        
        # Verify lots logic
        assert len(target_weights) > 0
        assert "2330" in target_weights
        
        expected_amt_2330 = 1_000_000.0 * target_weights["2330"]
        expected_lots_2330 = int(expected_amt_2330 / (900.0 * 1000))
        assert target_lots.get("2330", 0) == expected_lots_2330

        # Verify metadata is updated
        assert "strategy_config" in metadata
        assert "config_hash" in metadata
        assert "gate_status" in metadata
        assert "metrics_source" in metadata
