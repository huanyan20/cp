from __future__ import annotations

from datetime import datetime, time
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import pandas_market_calendars as mcal
except Exception:  # pragma: no cover - optional dependency
    mcal = None


TAIPEI_TZ = ZoneInfo("Asia/Taipei")
NEW_YORK_TZ = ZoneInfo("America/New_York")


@lru_cache(maxsize=32)
def _nyse_schedule(start: str, end: str) -> pd.DataFrame:
    if mcal is None:
        return pd.DataFrame()
    calendar = mcal.get_calendar("NYSE")
    return calendar.schedule(start_date=start, end_date=end)


def default_us_close_available_at_taipei(us_session_date: pd.Timestamp) -> pd.Timestamp:
    session_date = pd.Timestamp(us_session_date).date()
    ny_close = datetime.combine(session_date, time(16, 0), tzinfo=NEW_YORK_TZ)
    return pd.Timestamp(ny_close.astimezone(TAIPEI_TZ))


def us_close_available_at_taipei(us_session_date: pd.Timestamp) -> pd.Timestamp:
    """Return US market close availability in Taipei time, using NYSE calendar when available."""
    session = pd.Timestamp(us_session_date).normalize()
    if mcal is not None:
        schedule = _nyse_schedule(
            (session - pd.Timedelta(days=7)).date().isoformat(),
            (session + pd.Timedelta(days=7)).date().isoformat(),
        )
        if not schedule.empty and session in pd.DatetimeIndex(schedule.index).normalize():
            row = schedule.loc[schedule.index.normalize() == session].iloc[-1]
            close_ts = pd.Timestamp(row["market_close"])
            if close_ts.tzinfo is None:
                close_ts = close_ts.tz_localize("UTC")
            return close_ts.tz_convert(TAIPEI_TZ)
    return default_us_close_available_at_taipei(session)


def taipei_market_open(trade_date: pd.Timestamp) -> pd.Timestamp:
    date_value = pd.Timestamp(trade_date).date()
    return pd.Timestamp(datetime.combine(date_value, time(9, 0), tzinfo=TAIPEI_TZ))


def taipei_feature_cutoff(trade_date: pd.Timestamp) -> pd.Timestamp:
    date_value = pd.Timestamp(trade_date).date()
    return pd.Timestamp(datetime.combine(date_value, time(8, 59), tzinfo=TAIPEI_TZ))


def normalize_trade_dates(trade_dates: pd.DatetimeIndex | list) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(trade_dates)).normalize().sort_values()


def map_available_to_tw_trade_date(
    available_at_taipei: pd.Timestamp,
    tw_trade_dates: pd.DatetimeIndex | list,
) -> pd.Timestamp:
    """Map a known timestamp to the first TW trading date whose 09:00 open is later."""
    dates = normalize_trade_dates(tw_trade_dates)
    for trade_date in dates:
        if taipei_market_open(trade_date) > available_at_taipei:
            return trade_date
    return pd.NaT


def source_age_hours(trade_date: pd.Timestamp, available_at_taipei: pd.Timestamp) -> float:
    return max(
        (taipei_feature_cutoff(trade_date) - available_at_taipei).total_seconds() / 3600.0,
        0.0,
    )
