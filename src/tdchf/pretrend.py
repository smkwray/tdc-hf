from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .indicators import read_wide_time_series_csv


def add_pretrend_controls(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    lags: Sequence[int] = (2, 3),
    windows: Sequence[int] = (3, 6),
) -> pd.DataFrame:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing pretrend columns: {missing}")

    out = df.copy()
    for column in columns:
        series = pd.to_numeric(out[column], errors="coerce")
        for lag in lags:
            if lag < 1:
                raise ValueError("Pretrend lags must be positive")
            out[f"lag{lag}_{column}"] = series.shift(lag)
        for window in windows:
            if window < 1:
                raise ValueError("Pretrend windows must be positive")
            out[f"pretrend{window}_{column}"] = series.shift(1).rolling(window=window, min_periods=window).sum()

    out.index.name = "date"
    return out


def add_pretrend_controls_csv(
    data_csv: str | Path,
    *,
    columns: Sequence[str],
    out_csv: str | Path,
    lags: Sequence[int] = (2, 3),
    windows: Sequence[int] = (3, 6),
) -> dict[str, object]:
    df = read_wide_time_series_csv(data_csv)
    out = add_pretrend_controls(df, columns=columns, lags=lags, windows=windows)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")
    added = [
        column
        for column in out.columns
        if any(column.startswith(f"lag{lag}_") for lag in lags) or any(column.startswith(f"pretrend{window}_") for window in windows)
    ]
    return {"status": "ok", "out": str(path), "rows": int(len(out)), "added_columns": added}


def add_lagged_factor_controls(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    n_factors: int = 3,
    lag: int = 1,
    prefix: str = "factor",
) -> pd.DataFrame:
    if n_factors < 1:
        raise ValueError("n_factors must be positive")
    if lag < 1:
        raise ValueError("lag must be positive")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing factor columns: {missing}")

    out = df.copy()
    lagged = out.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").shift(lag)
    complete = lagged.dropna()
    if complete.empty:
        for factor in range(1, n_factors + 1):
            out[f"{prefix}{factor}_lag{lag}"] = np.nan
        out.index.name = "date"
        return out

    standardized = (complete - complete.mean()) / complete.std(ddof=0).replace(0, 1)
    _, singular_values, vt = np.linalg.svd(standardized.to_numpy(dtype=float), full_matrices=False)
    usable = min(n_factors, vt.shape[0])
    scores = standardized.to_numpy(dtype=float) @ vt[:usable].T
    # Scale scores to unit sample standard deviation so coefficients are stable
    # across alternative factor sets.
    score_frame = pd.DataFrame(scores, index=complete.index, columns=[f"{prefix}{i}_lag{lag}" for i in range(1, usable + 1)])
    score_frame = score_frame / score_frame.std(ddof=0).replace(0, 1)

    for factor in range(1, n_factors + 1):
        name = f"{prefix}{factor}_lag{lag}"
        out[name] = score_frame[name] if name in score_frame else np.nan

    out.index.name = "date"
    return out


def add_lagged_factor_controls_csv(
    data_csv: str | Path,
    *,
    columns: Sequence[str],
    out_csv: str | Path,
    n_factors: int = 3,
    lag: int = 1,
    prefix: str = "factor",
) -> dict[str, object]:
    df = read_wide_time_series_csv(data_csv)
    out = add_lagged_factor_controls(df, columns=columns, n_factors=n_factors, lag=lag, prefix=prefix)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")
    added = [f"{prefix}{factor}_lag{lag}" for factor in range(1, n_factors + 1)]
    nonmissing = {column: int(out[column].notna().sum()) for column in added if column in out.columns}
    return {
        "status": "ok",
        "out": str(path),
        "rows": int(len(out)),
        "factor_columns": list(columns),
        "added_columns": added,
        "nonmissing": nonmissing,
    }
