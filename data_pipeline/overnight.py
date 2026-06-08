import numpy as np
import pandas as pd

from .utils import DEFAULT_OVERNIGHT_FEATURE_COLS, _historical_zscore_clip


def load_overnight_features(
    path: str, feature_cols: list[str] | None = None
) -> pd.DataFrame:
    feature_cols = feature_cols or DEFAULT_OVERNIGHT_FEATURE_COLS
    overnight = pd.read_csv(path)
    if "tw_trade_date" in overnight.columns:
        overnight["tw_trade_date"] = pd.to_datetime(overnight["tw_trade_date"])
        overnight = overnight.set_index("tw_trade_date")
    else:
        overnight.index = pd.to_datetime(overnight.iloc[:, 0])
        overnight = overnight.iloc[:, 1:]
    overnight.index = pd.DatetimeIndex(overnight.index).normalize()
    overnight = overnight.sort_index()

    if "baseline_ret_prev" in feature_cols and "baseline_ret_prev" not in overnight:
        if "target_2330_full_day" in overnight:
            overnight["baseline_ret_prev"] = overnight["target_2330_full_day"].shift(1)
        else:
            overnight["baseline_ret_prev"] = np.nan

    normalized = pd.DataFrame(index=overnight.index)
    for col in feature_cols:
        out_col = f"overnight_{col}"
        if col in overnight:
            normalized[out_col] = _historical_zscore_clip(overnight[col])
        else:
            normalized[out_col] = 0.0
    return normalized
