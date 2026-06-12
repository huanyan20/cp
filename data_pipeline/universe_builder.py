import logging
from datetime import datetime, timedelta
from typing import Protocol

import pandas as pd
import yfinance as yf

from stock_universe import TICKERS_TECH_EXPANDED

logger = logging.getLogger(__name__)


class UniverseBuilder(Protocol):
    def build_universe(self, target_date: str, top_n: int = 50) -> list[str]:
        """給定日期，回傳當下應使用的 Ticker 列表"""
        ...


class StaticUniverseBuilder:
    """退化版本的 Builder，直接回傳常數列表（相容舊版）"""
    def build_universe(self, target_date: str, top_n: int = 50) -> list[str]:
        return TICKERS_TECH_EXPANDED


class DynamicVolumeUniverseBuilder:
    """
    基於 yfinance 歷史成交額 (Dollar Volume) 的動態篩選器。
    
    注意：此實作仍然受限於 yfinance 缺乏完整的下市股票資料，
    但它展示了「每個時間段動態篩選 Top N 標的」的機制。
    要徹底解決存活者偏差，需擴充 self.base_pool 包含下市股票清單。
    """

    def __init__(self, base_pool: list[str] | None = None):
        # 這裡應該要是「包含所有已下市股票的歷史大全集」
        # 由於 yfinance 限制，我們暫時以 expanded list 模擬
        self.base_pool = base_pool or TICKERS_TECH_EXPANDED

    def build_universe(self, target_date: str, top_n: int = 50) -> list[str]:
        """
        篩選邏輯：計算 target_date 往前推 60 天的日均成交額 (Volume * Close)，
        取成交額最高的前 top_n 檔股票。
        """
        end_dt = pd.to_datetime(target_date)
        start_dt = end_dt - timedelta(days=90)  # 多抓幾天確保有 60 個交易日

        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        logger.info(f"正在為 {target_date} 篩選動態股票池，基準期間: {start_str} ~ {end_str}")

        # 使用 yfinance 批量下載以節省時間
        try:
            data = yf.download(
                self.base_pool,
                start=start_str,
                end=end_str,
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            logger.error(f"下載歷史資料失敗: {e}")
            return self.base_pool[:top_n]

        if data.empty or "Close" not in data or "Volume" not in data:
            return self.base_pool[:top_n]

        close_df = data["Close"]
        vol_df = data["Volume"]

        # 處理 multi-index (當 base_pool 只有一檔股票時 yfinance 不會回傳 multi-index)
        if isinstance(close_df, pd.Series):
            close_df = close_df.to_frame(self.base_pool[0])
            vol_df = vol_df.to_frame(self.base_pool[0])

        dollar_volume = close_df * vol_df
        avg_dollar_volume = dollar_volume.mean()

        # 剔除在該期間完全沒有交易量（例如尚未上市）的標的
        avg_dollar_volume = avg_dollar_volume.dropna()
        avg_dollar_volume = avg_dollar_volume[avg_dollar_volume > 0]

        # 取 Top N
        top_tickers = avg_dollar_volume.nlargest(top_n).index.tolist()

        if not top_tickers:
            logger.warning(f"{target_date} 無法計算有效成交額，回退至預設名單。")
            return self.base_pool[:top_n]

        logger.info(f"{target_date} 動態篩選完成，選出 {len(top_tickers)} 檔股票。")
        return top_tickers


class FinMindUniverseBuilder:
    """
    基於 FinMind 的 Universe Builder，支援獲取台灣股市歷史狀態。
    注意：這需要真實的 FinMind Token 才能大量呼叫。
    """
    def __init__(self, token: str = ""):
        try:
            from FinMind.data import DataLoader
            self.dl = DataLoader()
            if token:
                self.dl.login_by_token(api_token=token)
        except ImportError:
            raise ImportError("請先安裝 FinMind: pip install FinMind")

    def build_universe(self, target_date: str, top_n: int = 50) -> list[str]:
        # TODO: 實作透過 FinMind 獲取當時所有上市櫃清單與成交額
        # 由於 FinMind 限制，實務上通常需在本地建立 SQLite Cache 避免 API Rate Limit
        raise NotImplementedError("FinMind 動態篩選器需配合本地 Cache 使用以避免超時。")

# 預設對外提供的工廠方法
def get_universe_builder(strategy: str = "dynamic", **kwargs) -> UniverseBuilder:
    if strategy == "static":
        return StaticUniverseBuilder()
    elif strategy == "dynamic":
        return DynamicVolumeUniverseBuilder(base_pool=kwargs.get("base_pool"))
    elif strategy == "finmind":
        return FinMindUniverseBuilder(token=kwargs.get("token", ""))
    else:
        raise ValueError(f"未知的 UniverseBuilder 策略: {strategy}")
