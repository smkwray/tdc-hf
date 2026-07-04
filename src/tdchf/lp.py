from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .indicators import read_wide_time_series_csv
from .units import add_same_unit_columns


def cumulative_forward_sum(series: pd.Series, horizon: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    for idx in range(len(values) - horizon):
        window = values[idx : idx + horizon + 1]
        if not np.isnan(window).any():
            out[idx] = float(window.sum())
    return pd.Series(out, index=series.index, name=series.name)


def cumulative_backward_sum(series: pd.Series, horizon: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    for idx in range(horizon, len(values)):
        window = values[idx - horizon : idx + 1]
        if not np.isnan(window).any():
            out[idx] = float(window.sum())
    return pd.Series(out, index=series.index, name=series.name)


def run_local_projections(
    df: pd.DataFrame,
    *,
    shock_col: str,
    outcome_cols: Sequence[str],
    controls: Sequence[str] = (),
    horizons: Sequence[int] = (0, 1, 2, 3, 6, 12),
    cumulative: bool = True,
    nw_lags: int = 4,
    spec_name: str = "baseline",
) -> pd.DataFrame:
    required = [shock_col, *outcome_cols, *controls]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    for outcome in outcome_cols:
        for horizon in horizons:
            dep = cumulative_forward_sum(df[outcome], horizon) if cumulative else df[outcome].shift(-horizon)
            sample = pd.DataFrame({"dep": dep, shock_col: df[shock_col]})
            for control in controls:
                sample[control] = df[control]
            sample = sample.dropna()
            if len(sample) < len(controls) + 8:
                continue
            x = sm.add_constant(sample[[shock_col, *controls]], has_constant="add")
            fit = sm.OLS(sample["dep"], x).fit(cov_type="HAC", cov_kwds={"maxlags": max(nw_lags, horizon)})
            beta = float(fit.params[shock_col])
            se = float(fit.bse[shock_col])
            rows.append(
                {
                    "spec_name": spec_name,
                    "outcome": outcome,
                    "shock": shock_col,
                    "horizon": int(horizon),
                    "beta": beta,
                    "se": se,
                    "lower95": beta - 1.96 * se,
                    "upper95": beta + 1.96 * se,
                    "n": int(fit.nobs),
                    "response_type": "cumulative_sum_h0_to_h" if cumulative else "lead_h",
                }
            )
    return pd.DataFrame(rows)


def run_local_projections_csv(
    data_csv: str | Path,
    *,
    shock_col: str,
    outcome_cols: Sequence[str],
    controls: Sequence[str] = (),
    horizons: Sequence[int] = (0, 1, 2, 3, 6, 12),
    out_csv: str | Path,
    cumulative: bool = True,
    nw_lags: int = 4,
    spec_name: str = "baseline",
) -> dict[str, object]:
    df = read_wide_time_series_csv(data_csv)
    result = run_local_projections(
        df,
        shock_col=shock_col,
        outcome_cols=outcome_cols,
        controls=controls,
        horizons=horizons,
        cumulative=cumulative,
        nw_lags=nw_lags,
        spec_name=spec_name,
    )
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return {"status": "ok", "out": str(path), "rows": int(len(result))}


def run_lp_iv(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    instrument_cols: Sequence[str],
    outcome_cols: Sequence[str],
    controls: Sequence[str] = (),
    horizons: Sequence[int] = (0, 1, 2, 3, 6, 12),
    cumulative: bool = True,
    nw_lags: int = 4,
    spec_name: str = "lp_iv_2sls",
) -> pd.DataFrame:
    required = [treatment_col, *instrument_cols, *outcome_cols, *controls]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    for outcome in outcome_cols:
        for horizon in horizons:
            dep = cumulative_forward_sum(df[outcome], horizon) if cumulative else df[outcome].shift(-horizon)
            sample = pd.DataFrame({"dep": dep, treatment_col: df[treatment_col]})
            for column in [*instrument_cols, *controls]:
                sample[column] = df[column]
            sample = sample.dropna()
            if len(sample) < len(instrument_cols) + len(controls) + 10:
                continue

            first_x = sm.add_constant(sample[[*instrument_cols, *controls]], has_constant="add")
            first_fit = sm.OLS(sample[treatment_col], first_x).fit()
            fitted_name = f"{treatment_col}_hat"
            sample[fitted_name] = first_fit.fittedvalues
            second_x = sm.add_constant(sample[[fitted_name, *controls]], has_constant="add")
            second_fit = sm.OLS(sample["dep"], second_x).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": max(nw_lags, horizon)},
            )
            beta = float(second_fit.params[fitted_name])
            se = float(second_fit.bse[fitted_name])
            restriction = " = 0, ".join(instrument_cols) + " = 0"
            ftest = first_fit.f_test(restriction)
            rows.append(
                {
                    "spec_name": spec_name,
                    "outcome": outcome,
                    "treatment": treatment_col,
                    "instruments": ",".join(instrument_cols),
                    "horizon": int(horizon),
                    "beta": beta,
                    "se": se,
                    "lower95": beta - 1.96 * se,
                    "upper95": beta + 1.96 * se,
                    "n": int(second_fit.nobs),
                    "first_stage_f": float(ftest.fvalue),
                    "first_stage_pvalue": float(ftest.pvalue),
                    "response_type": "cumulative_sum_h0_to_h" if cumulative else "lead_h",
                    "estimator": "manual_2sls_generated_regressor",
                }
            )
    return add_same_unit_columns(pd.DataFrame(rows), treatment_col=treatment_col)


def run_lp_iv_placebo(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    instrument_cols: Sequence[str],
    outcome_cols: Sequence[str],
    controls: Sequence[str] = (),
    placebo_horizons: Sequence[int] = (1, 2, 3, 6, 12),
    nw_lags: int = 4,
    spec_name: str = "lp_iv_placebo",
) -> pd.DataFrame:
    required = [treatment_col, *instrument_cols, *outcome_cols, *controls]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    for outcome in outcome_cols:
        for horizon in placebo_horizons:
            dep = cumulative_backward_sum(df[outcome], int(horizon))
            sample = pd.DataFrame({"dep": dep, treatment_col: df[treatment_col]})
            for column in [*instrument_cols, *controls]:
                sample[column] = df[column]
            sample = sample.dropna()
            if len(sample) < len(instrument_cols) + len(controls) + 10:
                continue

            first_x = sm.add_constant(sample[[*instrument_cols, *controls]], has_constant="add")
            first_fit = sm.OLS(sample[treatment_col], first_x).fit()
            fitted_name = f"{treatment_col}_hat"
            sample[fitted_name] = first_fit.fittedvalues
            second_x = sm.add_constant(sample[[fitted_name, *controls]], has_constant="add")
            second_fit = sm.OLS(sample["dep"], second_x).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": max(nw_lags, int(horizon))},
            )
            beta = float(second_fit.params[fitted_name])
            se = float(second_fit.bse[fitted_name])
            restriction = " = 0, ".join(instrument_cols) + " = 0"
            ftest = first_fit.f_test(restriction)
            rows.append(
                {
                    "spec_name": spec_name,
                    "outcome": outcome,
                    "treatment": treatment_col,
                    "instruments": ",".join(instrument_cols),
                    "placebo_horizon": int(horizon),
                    "beta": beta,
                    "se": se,
                    "lower95": beta - 1.96 * se,
                    "upper95": beta + 1.96 * se,
                    "n": int(second_fit.nobs),
                    "first_stage_f": float(ftest.fvalue),
                    "first_stage_pvalue": float(ftest.pvalue),
                    "response_type": "backward_cumulative_sum_h0_to_minus_h",
                    "estimator": "manual_2sls_generated_regressor_placebo",
                    "placebo_sig_95": bool(beta - 1.96 * se > 0 or beta + 1.96 * se < 0),
                }
            )
    return add_same_unit_columns(pd.DataFrame(rows), treatment_col=treatment_col)


def run_lp_iv_csv(
    data_csv: str | Path,
    *,
    treatment_col: str,
    instrument_cols: Sequence[str],
    outcome_cols: Sequence[str],
    controls: Sequence[str] = (),
    horizons: Sequence[int] = (0, 1, 2, 3, 6, 12),
    out_csv: str | Path,
    cumulative: bool = True,
    nw_lags: int = 4,
    spec_name: str = "lp_iv_2sls",
) -> dict[str, object]:
    df = read_wide_time_series_csv(data_csv)
    result = run_lp_iv(
        df,
        treatment_col=treatment_col,
        instrument_cols=instrument_cols,
        outcome_cols=outcome_cols,
        controls=controls,
        horizons=horizons,
        cumulative=cumulative,
        nw_lags=nw_lags,
        spec_name=spec_name,
    )
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return {"status": "ok", "out": str(path), "rows": int(len(result))}


def run_lp_iv_placebo_csv(
    data_csv: str | Path,
    *,
    treatment_col: str,
    instrument_cols: Sequence[str],
    outcome_cols: Sequence[str],
    controls: Sequence[str] = (),
    placebo_horizons: Sequence[int] = (1, 2, 3, 6, 12),
    out_csv: str | Path,
    nw_lags: int = 4,
    spec_name: str = "lp_iv_placebo",
) -> dict[str, object]:
    df = read_wide_time_series_csv(data_csv)
    result = run_lp_iv_placebo(
        df,
        treatment_col=treatment_col,
        instrument_cols=instrument_cols,
        outcome_cols=outcome_cols,
        controls=controls,
        placebo_horizons=placebo_horizons,
        nw_lags=nw_lags,
        spec_name=spec_name,
    )
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return {
        "status": "ok",
        "out": str(path),
        "rows": int(len(result)),
        "significant_rows": int(result["placebo_sig_95"].sum()) if "placebo_sig_95" in result.columns else 0,
    }
