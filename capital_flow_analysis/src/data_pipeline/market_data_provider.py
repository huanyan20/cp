from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd
import yfinance as yf

try:  # Optional fallback source.
    from pandas_datareader import data as pdr_data
except Exception:  # pragma: no cover - depends on local environment
    pdr_data = None

try:  # Optional fallback source.
    import requests
except Exception:  # pragma: no cover - requests is expected in this project
    requests = None


ROOT_DIR = Path(__file__).resolve().parents[3]
CAPITAL_FLOW_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = CAPITAL_FLOW_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
SOURCE_HEALTH_PATH = DATA_DIR / "source_health.json"

REQUIRED_FIELDS = {"Close"}
OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


@dataclass
class FetchResult:
    data: pd.DataFrame
    provider: str
    warnings: list[str] = field(default_factory=list)
    from_cache: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.data.empty and not self.warnings


class MarketDataProvider(Protocol):
    name: str

    def fetch(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> FetchResult:
        ...


def safe_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol)


def ensure_multiindex(df: pd.DataFrame, symbols: Iterable[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        return out.sort_index(axis=1)
    symbols = list(symbols)
    if len(symbols) == 1:
        out.columns = pd.MultiIndex.from_product([symbols, out.columns])
        return out.sort_index(axis=1)
    return out


def normalize_single_symbol_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(-1)
    keep = [col for col in OHLCV_FIELDS if col in out.columns]
    out = out[keep].copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def extract_symbol_frame(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if symbol not in data.columns.get_level_values(0):
            return pd.DataFrame()
        return normalize_single_symbol_frame(data[symbol])
    return normalize_single_symbol_frame(data)


def validate_fetch_result(
    result: FetchResult,
    symbols: list[str],
    min_rows: int = 2,
) -> FetchResult:
    warnings = list(result.warnings)
    if result.data.empty:
        warnings.append("empty data")
    for symbol in symbols:
        frame = extract_symbol_frame(result.data, symbol)
        if frame.empty:
            warnings.append(f"{symbol}: missing frame")
            continue
        missing = REQUIRED_FIELDS - set(frame.columns)
        if missing:
            warnings.append(f"{symbol}: missing columns {sorted(missing)}")
        if len(frame.dropna(subset=["Close"])) < min_rows:
            warnings.append(f"{symbol}: insufficient rows")
    result.warnings = warnings
    return result


def combine_symbol_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = {}
    for symbol, frame in frames.items():
        clean = normalize_single_symbol_frame(frame)
        if not clean.empty:
            pieces[symbol] = clean
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, axis=1).sort_index(axis=1)


class YFinanceProvider:
    name = "yfinance"

    def fetch(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> FetchResult:
        kwargs = {
            "tickers": symbols,
            "interval": interval,
            "group_by": "ticker",
            "auto_adjust": False,
            "actions": True,
            "progress": False,
        }
        if period:
            kwargs["period"] = period
        else:
            kwargs["start"] = start
            kwargs["end"] = end
        try:
            data = yf.download(**kwargs)
            data = ensure_multiindex(data, symbols)
            return validate_fetch_result(FetchResult(data, self.name), symbols)
        except Exception as exc:
            return FetchResult(pd.DataFrame(), self.name, [f"download failed: {exc}"])


class StooqProvider:
    name = "stooq"

    SYMBOL_MAP = {
        "TSM": "TSM.US",
        "UMC": "UMC.US",
        "ASX": "ASX.US",
        "^IXIC": "^NDQ",
        "^VIX": "^VIX",
        "DX-Y.NYB": "DX.F",
        "BTC-USD": "BTCUSD",
    }

    def fetch(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> FetchResult:
        if pdr_data is None:
            return FetchResult(pd.DataFrame(), self.name, ["pandas_datareader unavailable"])
        if interval != "1d":
            return FetchResult(pd.DataFrame(), self.name, ["only 1d interval supported"])

        frames = {}
        warnings = []
        for symbol in symbols:
            stooq_symbol = self.SYMBOL_MAP.get(symbol)
            if not stooq_symbol:
                warnings.append(f"{symbol}: unsupported by stooq provider")
                continue
            try:
                frame = pdr_data.DataReader(stooq_symbol, "stooq", start=start, end=end)
                frame = frame.sort_index()
                frames[symbol] = frame
            except Exception as exc:
                warnings.append(f"{symbol}: stooq failed: {exc}")
        data = combine_symbol_frames(frames)
        return validate_fetch_result(FetchResult(data, self.name, warnings), symbols)


class AlphaVantageProvider:
    name = "alpha_vantage"

    SYMBOL_MAP = {
        "TSM": "TSM",
        "UMC": "UMC",
        "ASX": "ASX",
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")

    def fetch(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> FetchResult:
        if not self.api_key:
            return FetchResult(pd.DataFrame(), self.name, ["ALPHA_VANTAGE_API_KEY not set"])
        if requests is None:
            return FetchResult(pd.DataFrame(), self.name, ["requests unavailable"])
        if interval != "1d":
            return FetchResult(pd.DataFrame(), self.name, ["only 1d interval supported"])

        frames = {}
        warnings = []
        for symbol in symbols:
            av_symbol = self.SYMBOL_MAP.get(symbol)
            if not av_symbol:
                warnings.append(f"{symbol}: unsupported by alpha vantage provider")
                continue
            try:
                response = requests.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "TIME_SERIES_DAILY_ADJUSTED",
                        "symbol": av_symbol,
                        "outputsize": "full",
                        "apikey": self.api_key,
                    },
                    timeout=20,
                )
                payload = response.json()
                series = payload.get("Time Series (Daily)", {})
                if not series:
                    warnings.append(f"{symbol}: alpha vantage empty response")
                    continue
                frame = pd.DataFrame.from_dict(series, orient="index")
                frame.index = pd.to_datetime(frame.index)
                frame = frame.rename(
                    columns={
                        "1. open": "Open",
                        "2. high": "High",
                        "3. low": "Low",
                        "4. close": "Close",
                        "5. adjusted close": "Adj Close",
                        "6. volume": "Volume",
                    }
                )
                frame = frame[[col for col in OHLCV_FIELDS if col in frame.columns]]
                frame = frame.apply(pd.to_numeric, errors="coerce")
                if start:
                    frame = frame.loc[frame.index >= pd.Timestamp(start)]
                if end:
                    frame = frame.loc[frame.index < pd.Timestamp(end)]
                frames[symbol] = frame.sort_index()
            except Exception as exc:
                warnings.append(f"{symbol}: alpha vantage failed: {exc}")
        data = combine_symbol_frames(frames)
        return validate_fetch_result(FetchResult(data, self.name, warnings), symbols)


class LastGoodCacheProvider:
    name = "last_good_cache"

    def __init__(self, cache_dir: Path = CACHE_DIR, max_age_days: int = 10):
        self.cache_dir = Path(cache_dir)
        self.max_age_days = max_age_days

    def cache_result(self, result: FetchResult, symbols: list[str], interval: str = "1d") -> None:
        if result.data.empty:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for symbol in symbols:
            frame = extract_symbol_frame(result.data, symbol)
            if frame.empty:
                continue
            frame.to_csv(self.cache_dir / f"last_good_{interval}_{safe_symbol(symbol)}.csv")

    def fetch(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> FetchResult:
        frames = {}
        warnings = []
        now = pd.Timestamp.now().normalize()
        for symbol in symbols:
            interval_path = self.cache_dir / f"last_good_{interval}_{safe_symbol(symbol)}.csv"
            legacy_path = self.cache_dir / f"last_good_{safe_symbol(symbol)}.csv"
            path = interval_path if interval_path.exists() else legacy_path
            if not path.exists():
                warnings.append(f"{symbol}: cache missing")
                continue
            try:
                frame = pd.read_csv(path, index_col=0, parse_dates=True)
                frame = normalize_single_symbol_frame(frame)
                if frame.empty:
                    warnings.append(f"{symbol}: cache empty")
                    continue
                age = (now - frame.index.max().normalize()).days
                if age > self.max_age_days:
                    warnings.append(f"{symbol}: cache stale ({age} days)")
                    continue
                if start:
                    frame = frame.loc[frame.index >= pd.Timestamp(start)]
                if end:
                    frame = frame.loc[frame.index < pd.Timestamp(end)]
                frames[symbol] = frame
            except Exception as exc:
                warnings.append(f"{symbol}: cache read failed: {exc}")
        data = combine_symbol_frames(frames)
        result = FetchResult(data, self.name, warnings, from_cache=True)
        return validate_fetch_result(result, symbols)


class CompositeProvider:
    name = "composite"

    def __init__(
        self,
        providers: list[MarketDataProvider] | None = None,
        cache_provider: LastGoodCacheProvider | None = None,
        health_path: Path = SOURCE_HEALTH_PATH,
    ):
        self.cache_provider = cache_provider or LastGoodCacheProvider()
        self.providers = providers or [
            YFinanceProvider(),
            StooqProvider(),
            AlphaVantageProvider(),
        ]
        self.health_path = Path(health_path)

    def fetch(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> FetchResult:
        frames: dict[str, pd.DataFrame] = {}
        symbol_providers: dict[str, str] = {}
        attempts: list[dict] = []
        warnings: list[str] = []

        # Resolve each symbol independently. One flaky ticker should not force
        # the whole multi-asset request onto cache or fail the daily run.
        for symbol in symbols:
            selected: FetchResult | None = None
            for provider in [*self.providers, self.cache_provider]:
                result = provider.fetch(
                    [symbol],
                    start=start,
                    end=end,
                    period=period,
                    interval=interval,
                )
                symbol_frame = extract_symbol_frame(result.data, symbol)
                attempts.append(
                    {
                        "symbol": symbol,
                        "provider": result.provider,
                        "from_cache": result.from_cache,
                        "warnings": result.warnings,
                        "rows": int(len(symbol_frame)),
                    }
                )
                if not result.warnings and not symbol_frame.empty:
                    selected = result
                    frames[symbol] = symbol_frame
                    symbol_providers[symbol] = result.provider
                    if not result.from_cache:
                        self.cache_provider.cache_result(result, [symbol], interval=interval)
                    break
            if selected is None:
                warnings.append(f"{symbol}: all providers failed")

        data = combine_symbol_frames(frames)
        provider_name = (
            "composite"
            if len(set(symbol_providers.values())) > 1
            else next(iter(symbol_providers.values()), "none")
        )
        result = FetchResult(data, provider_name, warnings)
        result.metadata["attempts"] = attempts
        result.metadata["symbol_providers"] = symbol_providers
        self.write_health(result, symbols)
        return result

    def write_health(self, result: FetchResult, symbols: list[str]) -> None:
        self.health_path.parent.mkdir(parents=True, exist_ok=True)
        symbol_health = {}
        for symbol in symbols:
            frame = extract_symbol_frame(result.data, symbol)
            symbol_health[symbol] = {
                "rows": int(len(frame)),
                "last_date": str(frame.index.max().date()) if not frame.empty else None,
                "has_close": "Close" in frame.columns if not frame.empty else False,
            }
        payload = {
            "provider": result.provider,
            "from_cache": result.from_cache,
            "warnings": result.warnings,
            "symbol_providers": result.metadata.get("symbol_providers", {}),
            "symbols": symbol_health,
            "attempts": result.metadata.get("attempts", []),
        }
        self.health_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
