from __future__ import annotations

import pandas as pd


def to_month_end(index: pd.Index | pd.Series) -> pd.DatetimeIndex:
    dates = pd.to_datetime(index)
    return pd.DatetimeIndex(dates).to_period("M").to_timestamp("M")


def to_quarter_end(index: pd.Index | pd.Series) -> pd.DatetimeIndex:
    dates = pd.to_datetime(index)
    return pd.DatetimeIndex(dates).to_period("Q").to_timestamp("Q")
