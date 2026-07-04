from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar import to_month_end, to_quarter_end


@dataclass(frozen=True)
class BenchmarkDiagnostics:
    component: str
    max_abs_quarterly_error: float
    quarters_checked: int
    method: str


def _as_monthly_series(series: pd.Series, *, name: str) -> pd.Series:
    out = series.copy()
    out.index = to_month_end(out.index)
    out = pd.to_numeric(out, errors="coerce").astype("float64").sort_index()
    out.name = name
    if out.index.has_duplicates:
        out = out.groupby(level=0).sum(min_count=1)
    return out


def _as_quarterly_series(series: pd.Series, *, name: str) -> pd.Series:
    out = series.copy()
    out.index = to_quarter_end(out.index)
    out = pd.to_numeric(out, errors="coerce").astype("float64").sort_index()
    out.name = name
    if out.index.has_duplicates:
        out = out.groupby(level=0).sum(min_count=1)
    return out


def additive_quarterly_residual_spread(
    monthly_indicator: pd.Series,
    quarterly_anchor: pd.Series,
    *,
    component: str,
) -> pd.Series:
    """Benchmark monthly values to quarterly anchors by spreading residuals.

    This is the first exact benchmarking contract. For each quarter, preserve
    the indicator's within-quarter pattern and add an equal residual adjustment
    to each observed month so the monthly sum equals the quarterly anchor.
    """
    monthly = _as_monthly_series(monthly_indicator, name=component)
    quarterly = _as_quarterly_series(quarterly_anchor, name=component)

    if monthly.empty:
        raise ValueError(f"{component}: monthly indicator is empty")
    if quarterly.empty:
        raise ValueError(f"{component}: quarterly anchor is empty")

    out = monthly.copy()
    month_quarters = to_quarter_end(out.index)
    for quarter, anchor in quarterly.items():
        mask = month_quarters == quarter
        if not mask.any():
            continue
        observed = out.loc[mask]
        valid_count = int(observed.notna().sum())
        if valid_count == 0:
            continue
        residual = float(anchor) - float(observed.sum(skipna=True))
        out.loc[mask & out.notna()] = observed.dropna() + residual / valid_count

    out.name = component
    return out


def additive_denton(
    monthly_indicator: pd.Series,
    quarterly_anchor: pd.Series,
    *,
    component: str,
) -> pd.Series:
    """Additive Denton benchmarking with first-difference adjustment penalty."""
    monthly = _as_monthly_series(monthly_indicator, name=component)
    quarterly = _as_quarterly_series(quarterly_anchor, name=component)

    valid = monthly.dropna()
    if valid.empty:
        raise ValueError(f"{component}: monthly indicator is empty")

    month_quarters = to_quarter_end(valid.index)
    quarters = [quarter for quarter in quarterly.index if (month_quarters == quarter).any()]
    if not quarters:
        raise ValueError(f"{component}: no overlapping monthly/quarterly observations")

    x = valid.to_numpy(dtype=float)
    n = len(x)
    q = len(quarters)
    c = np.zeros((q, n), dtype=float)
    target = np.zeros(q, dtype=float)
    for row, quarter in enumerate(quarters):
        mask = month_quarters == quarter
        c[row, mask] = 1.0
        target[row] = float(quarterly.loc[quarter])

    residual = target - c @ x
    if n == 1:
        adjustment = np.array([residual[0]], dtype=float)
    else:
        d = np.zeros((n - 1, n), dtype=float)
        for idx in range(n - 1):
            d[idx, idx] = -1.0
            d[idx, idx + 1] = 1.0
        penalty = d.T @ d
        ridge = np.eye(n, dtype=float) * 1e-10
        kkt = np.block(
            [
                [penalty + ridge, c.T],
                [c, np.zeros((q, q), dtype=float)],
            ]
        )
        rhs = np.r_[np.zeros(n, dtype=float), residual]
        solved = np.linalg.solve(kkt, rhs)
        adjustment = solved[:n]

    benchmarked = valid + adjustment
    out = monthly.copy()
    out.loc[valid.index] = benchmarked
    out.name = component
    return out


def validate_quarterly_identity(
    monthly_component: pd.Series,
    quarterly_anchor: pd.Series,
    *,
    component: str,
    atol: float = 1e-8,
    method: str = "additive_quarterly_residual_spread",
) -> BenchmarkDiagnostics:
    monthly = _as_monthly_series(monthly_component, name=component)
    quarterly = _as_quarterly_series(quarterly_anchor, name=component)
    summed = monthly.groupby(to_quarter_end(monthly.index)).sum(min_count=1)
    aligned = pd.concat([summed.rename("monthly_sum"), quarterly.rename("anchor")], axis=1, sort=False).dropna()
    if aligned.empty:
        raise ValueError(f"{component}: no overlapping monthly/quarterly observations")
    errors = aligned["monthly_sum"] - aligned["anchor"]
    max_error = float(np.nanmax(np.abs(errors.to_numpy(dtype=float))))
    if max_error > atol:
        raise AssertionError(f"{component}: max quarterly benchmark error {max_error:g} exceeds {atol:g}")
    return BenchmarkDiagnostics(
        component=component,
        max_abs_quarterly_error=max_error,
        quarters_checked=int(len(aligned)),
        method=method,
    )
