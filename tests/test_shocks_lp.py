from __future__ import annotations

import numpy as np
import pandas as pd

from tdchf.lp import cumulative_forward_sum, run_local_projections
from tdchf.shocks import build_named_residual_shock_csv, build_residual_shock_csv, expanding_window_residual


def test_expanding_window_residual_adds_columns() -> None:
    df = pd.DataFrame(
        {
            "target": np.arange(20, dtype=float),
            "lag": np.arange(20, dtype=float) * 0.5,
        }
    )

    out = expanding_window_residual(df, target="target", predictors=["lag"], min_train_obs=8)

    assert "target_residual" in out.columns
    assert out["target_fitted"].notna().sum() > 0


def test_cumulative_forward_sum() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    out = cumulative_forward_sum(series, 2)

    assert out.iloc[0] == 6.0
    assert pd.isna(out.iloc[-1])


def test_run_local_projections_returns_rows() -> None:
    n = 30
    shock = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "shock": shock,
            "outcome": 2.0 * shock,
            "control": np.ones(n),
        }
    )

    out = run_local_projections(
        df,
        shock_col="shock",
        outcome_cols=["outcome"],
        controls=["control"],
        horizons=[0, 1],
        nw_lags=1,
    )

    assert set(out["horizon"]) == {0, 1}
    assert out["beta"].notna().all()


def test_build_residual_shock_csv(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=20, freq="ME"),
            "target": np.arange(20, dtype=float),
            "lag": np.arange(20, dtype=float) * 0.5,
        }
    )
    data = tmp_path / "shock.csv"
    df.to_csv(data, index=False)

    report = build_residual_shock_csv(data, target="target", predictors=["lag"], min_train_obs=8, out_csv=tmp_path / "out.csv")
    out = pd.read_csv(tmp_path / "out.csv")

    assert report["status"] == "ok"
    assert "target_residual" in out.columns


def test_build_named_residual_shock_csv(tmp_path) -> None:
    n = 20
    data = tmp_path / "shock.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=n, freq="ME"),
            "target": range(n),
            "lag": range(n),
        }
    ).to_csv(data, index=False)

    report = build_named_residual_shock_csv(
        data,
        target="target",
        predictors=["lag"],
        residual_column="custom_surprise",
        fitted_column="custom_expected",
        z_column="custom_surprise_z",
        min_train_obs=8,
        out_csv=tmp_path / "out.csv",
    )
    out = pd.read_csv(tmp_path / "out.csv")

    assert report["z_column"] == "custom_surprise_z"
    assert "custom_surprise" in out.columns
    assert "custom_surprise_z" in out.columns


def test_build_named_residual_shock_csv_with_seasonal_design(tmp_path) -> None:
    n = 48
    data = tmp_path / "seasonal.csv"
    dates = pd.date_range("2020-01-31", periods=n, freq="ME")
    pd.DataFrame(
        {
            "date": dates,
            "target": [100.0 + 10.0 * (date.month == 4) + i for i, date in enumerate(dates)],
            "lag": range(n),
        }
    ).to_csv(data, index=False)

    report = build_named_residual_shock_csv(
        data,
        target="target",
        predictors=["lag"],
        residual_column="seasonal_surprise",
        fitted_column="seasonal_expected",
        z_column="seasonal_surprise_z",
        min_train_obs=24,
        month_dummies=True,
        trend=True,
        out_csv=tmp_path / "seasonal_out.csv",
    )
    out = pd.read_csv(tmp_path / "seasonal_out.csv")

    assert report["month_dummies"] is True
    assert report["trend"] is True
    assert "shock_month_04" in out.columns
    assert "shock_trend" in out.columns
