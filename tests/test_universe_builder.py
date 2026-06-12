import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
from data_pipeline.universe_builder import get_universe_builder

def test_static_builder():
    builder = get_universe_builder("static")
    tickers = builder.build_universe("2024-01-01", top_n=5)
    assert len(tickers) > 0
    assert "2330.TW" in tickers

def test_dynamic_builder_fallback():
    # 測試給定假資料或極端日期時的 Fallback
    builder = get_universe_builder("dynamic", base_pool=["2330.TW", "2454.TW"])
    tickers = builder.build_universe("2024-01-01", top_n=1)
    
    # 至少應該回傳 top_n 個股票，或者原 pool 大小
    assert len(tickers) == 1
    assert tickers[0] in ["2330.TW", "2454.TW"]
