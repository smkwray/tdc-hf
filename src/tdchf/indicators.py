from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .calendar import to_month_end
from .proxy import COMPONENT_ORDER


def read_wide_time_series_csv(path: str | Path, *, date_column: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_column])
    if date_column not in df.columns:
        raise KeyError(f"Missing date column: {date_column}")
    out = df.set_index(date_column).sort_index()
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def load_monthly_indicator_csv(
    path: str | Path,
    *,
    date_column: str = "date",
    require_all: bool = True,
) -> dict[str, pd.Series]:
    df = read_wide_time_series_csv(path, date_column=date_column)
    missing = [component for component in COMPONENT_ORDER if component not in df.columns]
    if missing and require_all:
        raise KeyError(f"Missing monthly indicator columns: {missing}")
    df.index = to_month_end(df.index)
    return {component: df[component].rename(component) for component in COMPONENT_ORDER if component in df.columns}


def aggregate_flows_to_monthly(series: pd.Series, *, how: str = "sum") -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").copy()
    values.index = pd.to_datetime(values.index)
    if how == "sum":
        out = values.resample("ME").sum(min_count=1)
    elif how == "mean":
        out = values.resample("ME").mean()
    else:
        raise ValueError(f"Unsupported flow aggregation: {how}")
    out.name = series.name
    return out


def aggregate_levels_to_monthly(series: pd.Series, *, how: str = "last") -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").copy()
    values.index = pd.to_datetime(values.index)
    if how == "last":
        out = values.resample("ME").last()
    elif how == "mean":
        out = values.resample("ME").mean()
    else:
        raise ValueError(f"Unsupported level aggregation: {how}")
    out.name = series.name
    return out


def level_change_to_monthly_flow(series: pd.Series, *, level_agg: str = "last") -> pd.Series:
    monthly_level = aggregate_levels_to_monthly(series, how=level_agg)
    out = monthly_level.diff()
    out.name = series.name
    return out


def positive_only(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").clip(lower=0.0)
    out.name = series.name
    return out


def no_indicator_equal_months(quarterly_anchors: Mapping[str, pd.Series]) -> dict[str, pd.Series]:
    """Allocate each quarterly anchor equally across the three months.

    This is a placebo / bootstrap path. It is useful for exercising downstream
    contracts before live monthly indicators are available.
    """
    monthly: dict[str, pd.Series] = {}
    for component in COMPONENT_ORDER:
        if component not in quarterly_anchors:
            raise KeyError(f"Missing quarterly anchor: {component}")
        anchor = quarterly_anchors[component].dropna().sort_index()
        rows: list[tuple[pd.Timestamp, float]] = []
        for quarter_end, value in anchor.items():
            q = pd.Timestamp(quarter_end).to_period("Q")
            for month in pd.period_range(q.start_time, q.end_time, freq="M"):
                rows.append((month.to_timestamp("M"), float(value) / 3.0))
        monthly[component] = pd.Series(
            [value for _, value in rows],
            index=pd.DatetimeIndex([date for date, _ in rows]),
            name=component,
            dtype="float64",
        )
    return monthly


def fill_indicator_gaps_from_equal_months(
    monthly_indicators: Mapping[str, pd.Series],
    quarterly_anchors: Mapping[str, pd.Series],
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Fill missing component-months with explicit equal-month fallbacks."""
    fallback = no_indicator_equal_months(quarterly_anchors)
    filled: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []

    for component in COMPONENT_ORDER:
        base = fallback[component].copy()
        observed = monthly_indicators.get(component)
        if observed is None:
            filled[component] = base
            rows.append(
                {
                    "component": component,
                    "observed_months": 0,
                    "fallback_months": int(base.notna().sum()),
                    "first_observed_month": "",
                    "last_observed_month": "",
                    "source": "equal_month_fallback",
                }
            )
            continue

        obs = observed.copy()
        obs.index = to_month_end(obs.index)
        obs = pd.to_numeric(obs, errors="coerce").astype("float64").sort_index()
        if obs.index.has_duplicates:
            obs = obs.groupby(level=0).sum(min_count=1)
        obs = obs.dropna()
        overlap = obs.index.intersection(base.index)
        combined = base.copy()
        combined.loc[overlap] = obs.loc[overlap]
        filled[component] = combined.rename(component)
        rows.append(
            {
                "component": component,
                "observed_months": int(len(overlap)),
                "observed_months_outside_anchor_window": int(len(obs) - len(overlap)),
                "fallback_months": int(combined.notna().sum() - len(overlap)),
                "first_observed_month": obs.index.min().date().isoformat() if not obs.empty else "",
                "last_observed_month": obs.index.max().date().isoformat() if not obs.empty else "",
                "source": "observed_with_equal_month_fallback",
            }
        )

    return filled, pd.DataFrame(rows)
