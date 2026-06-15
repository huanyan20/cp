import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import pandas as pd

from scripts.evaluate_sl_live import main

def test_evaluate_sl_live_fail_closed(tmp_path):
    # Set up mock command line arguments
    test_args = [
        "scripts/evaluate_sl_live.py",
        "--output", str(tmp_path / "signal.json"),
        "--top-k", "5",
        "--horizon", "10"
    ]
    
    mock_data = {
        "2330.TW": pd.DataFrame({"Close_norm": [500], "log_return": [0.01], "macro_^TWII_log_return": [0.0], "macro_^IXIC_log_return": [0.0]}, index=[pd.Timestamp("2024-06-01")]),
        "^TWII": pd.DataFrame({"Close_norm": [20000], "log_return": [0.01]}, index=[pd.Timestamp("2024-06-01")]),
        "^IXIC": pd.DataFrame({"Close_norm": [15000], "log_return": [0.01]}, index=[pd.Timestamp("2024-06-01")]),
        "USDTWD=X": pd.DataFrame({"Close_norm": [32]}, index=[pd.Timestamp("2024-06-01")])
    }
    
    with patch("sys.argv", test_args):
        with patch("scripts.evaluate_sl_live.fetch_multi_asset_data", return_value=mock_data):
            with patch("scripts.evaluate_sl_live.fetch_latest_close", return_value=500.0):
                with patch("scripts.evaluate_sl_live.get_current_positions_and_mdd", return_value=({}, 0.0, 1000.0)):
                    # Target weights will be non-empty (assuming generator predicts something)
                    # But total_assets=1000 and close=500 -> target_amt = 1000 * 0.2 = 200
                    # lots = 200 / (500 * 1000) = 0
                    
                    # We need to mock the generator so it predicts weights reliably
                    mock_generator = MagicMock()
                    mock_generator.predict_today.return_value = {"2330.TW": 0.05}
                    mock_summary = MagicMock()
                    mock_summary.feature_importance_top10 = {"feat1": 0.1}
                    mock_scores = {"2330.TW": pd.Series([0.05], index=[pd.Timestamp("2024-06-01")])}
                    mock_generator.fit_period.return_value = (mock_scores, mock_summary)
                    
                    with patch("scripts.evaluate_sl_live.SignalGenerator", return_value=mock_generator):
                        with pytest.raises(RuntimeError, match="FAIL-CLOSED: Target weights are non-empty"):
                            main()

def test_evaluate_sl_live_success(tmp_path):
    test_args = [
        "scripts/evaluate_sl_live.py",
        "--output", str(tmp_path / "signal.json"),
        "--top-k", "5",
        "--horizon", "10"
    ]
    
    mock_data = {
        "2330.TW": pd.DataFrame({"Close_norm": [500], "log_return": [0.01], "macro_^TWII_log_return": [0.0], "macro_^IXIC_log_return": [0.0]}, index=[pd.Timestamp("2024-06-01")]),
        "^TWII": pd.DataFrame({"Close_norm": [20000], "log_return": [0.01]}, index=[pd.Timestamp("2024-06-01")]),
        "^IXIC": pd.DataFrame({"Close_norm": [15000], "log_return": [0.01]}, index=[pd.Timestamp("2024-06-01")]),
        "USDTWD=X": pd.DataFrame({"Close_norm": [32]}, index=[pd.Timestamp("2024-06-01")])
    }
    
    with patch("sys.argv", test_args):
        with patch("scripts.evaluate_sl_live.fetch_multi_asset_data", return_value=mock_data):
            with patch("scripts.evaluate_sl_live.fetch_latest_close", return_value=500.0):
                with patch("scripts.evaluate_sl_live.get_current_positions_and_mdd", return_value=({}, 0.0, 10000000.0)):
                    mock_generator = MagicMock()
                    mock_generator.predict_today.return_value = {"2330.TW": 0.05}
                    mock_summary = MagicMock()
                    mock_summary.feature_importance_top10 = {"feat1": 0.1}
                    mock_scores = {"2330.TW": pd.Series([0.05], index=[pd.Timestamp("2024-06-01")])}
                    mock_generator.fit_period.return_value = (mock_scores, mock_summary)
                    
                    with patch("scripts.evaluate_sl_live.SignalGenerator", return_value=mock_generator):
                        main()
                    
                    # Verify output
                    assert (tmp_path / "signal.json").exists()
                    with open(tmp_path / "signal.json", "r", encoding="utf-8") as f:
                        signal = json.load(f)
                    
                    assert "target_lots" in signal
                    assert "2330.TW" in signal["target_lots"]
                    
                    # 10,000,000 * 0.20 weight max = 2,000,000
                    # 2,000,000 / (500 * 1000) = 4 lots
                    assert signal["target_lots"]["2330.TW"] == 4
