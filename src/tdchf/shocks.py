from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import read_wide_time_series_csv


def add_residual_design_terms(
    df: pd.DataFrame,
    predictors: Sequence[str],
    *,
    month_dummies: bool = False,
    trend: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    expanded = list(predictors)
    if month_dummies:
        months = pd.to_datetime(out.index).month
        for month in range(2, 13):
            column = f"shock_month_{month:02d}"
            out[column] = (months == month).astype(float)
            expanded.append(column)
    if trend:
        column = "shock_trend"
        out[column] = np.arange(len(out), dtype=float)
        expanded.append(column)
    return out, expanded


def expanding_window_residual(
    df: pd.DataFrame,
    *,
    target: str,
    predictors: Sequence[str],
    min_train_obs: int = 24,
    residual_column: str | None = None,
    fitted_column: str | None = None,
    z_column: str | None = None,
) -> pd.DataFrame:
    if target not in df.columns:
        raise KeyError(f"Missing target column: {target}")
    missing = [col for col in predictors if col not in df.columns]
    if missing:
        raise KeyError(f"Missing predictor columns: {missing}")

    out = df.copy()
    resid_name = residual_column or f"{target}_residual"
    fitted_name = fitted_column or f"{target}_fitted"
    z_name = z_column or f"{target}_residual_z"

    fitted = np.full(len(out), np.nan, dtype=float)
    residual = np.full(len(out), np.nan, dtype=float)
    z = np.full(len(out), np.nan, dtype=float)

    for idx in range(min_train_obs, len(out)):
        train = out.iloc[:idx][[target, *predictors]].dropna()
        if len(train) < max(min_train_obs, len(predictors) + 3):
            continue
        row = out.iloc[idx][list(predictors)]
        if row.isna().any() or pd.isna(out.iloc[idx][target]):
            continue

        x_train = np.column_stack([np.ones(len(train)), train[list(predictors)].to_numpy(dtype=float)])
        y_train = train[target].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        x_now = np.r_[1.0, row.to_numpy(dtype=float)]
        fitted[idx] = float(x_now @ beta)
        residual[idx] = float(out.iloc[idx][target] - fitted[idx])
        train_resid = y_train - x_train @ beta
        sd = float(np.std(train_resid, ddof=1)) if len(train_resid) > 1 else np.nan
        if np.isfinite(sd) and sd > 0:
            z[idx] = residual[idx] / sd

    out[fitted_name] = fitted
    out[resid_name] = residual
    out[z_name] = z
    return out


def build_residual_shock_csv(
    data_csv: str | Path,
    *,
    target: str,
    predictors: Sequence[str],
    out_csv: str | Path,
    min_train_obs: int = 24,
    month_dummies: bool = False,
    trend: bool = False,
) -> dict[str, object]:
    df = read_wide_time_series_csv(data_csv)
    df, design_predictors = add_residual_design_terms(df, predictors, month_dummies=month_dummies, trend=trend)
    out = expanding_window_residual(df, target=target, predictors=design_predictors, min_train_obs=min_train_obs)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")
    return {
        "status": "ok",
        "out": str(path),
        "rows": int(len(out)),
        "target": target,
        "predictors": list(design_predictors),
        "base_predictors": list(predictors),
        "month_dummies": month_dummies,
        "trend": trend,
    }


def build_named_residual_shock_csv(
    data_csv: str | Path,
    *,
    target: str,
    predictors: Sequence[str],
    out_csv: str | Path,
    residual_column: str,
    fitted_column: str,
    z_column: str,
    min_train_obs: int = 24,
    month_dummies: bool = False,
    trend: bool = False,
) -> dict[str, object]:
    df = read_wide_time_series_csv(data_csv)
    df, design_predictors = add_residual_design_terms(df, predictors, month_dummies=month_dummies, trend=trend)
    out = expanding_window_residual(
        df,
        target=target,
        predictors=design_predictors,
        min_train_obs=min_train_obs,
        residual_column=residual_column,
        fitted_column=fitted_column,
        z_column=z_column,
    )
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")
    return {
        "status": "ok",
        "out": str(path),
        "rows": int(len(out)),
        "target": target,
        "predictors": list(design_predictors),
        "base_predictors": list(predictors),
        "residual_column": residual_column,
        "z_column": z_column,
        "month_dummies": month_dummies,
        "trend": trend,
    }
