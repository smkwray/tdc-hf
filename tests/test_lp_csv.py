from __future__ import annotations

import numpy as np
import pandas as pd

from tdchf.lp import run_local_projections_csv, run_lp_iv_csv, run_lp_iv_placebo_csv


def test_run_local_projections_csv(tmp_path) -> None:
    n = 30
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=n, freq="ME"),
            "shock": np.arange(n, dtype=float),
            "outcome": np.arange(n, dtype=float) * 2.0,
        }
    )
    data = tmp_path / "lp.csv"
    df.to_csv(data, index=False)

    report = run_local_projections_csv(data, shock_col="shock", outcome_cols=["outcome"], horizons=[0, 1], out_csv=tmp_path / "out.csv")

    assert report["status"] == "ok"
    assert report["rows"] == 2


def test_run_lp_iv_placebo_csv(tmp_path) -> None:
    n = 40
    instrument = np.arange(n, dtype=float)
    treatment = 2.0 * instrument
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=n, freq="ME"),
            "instrument": instrument,
            "treatment": treatment,
            "outcome": np.arange(n, dtype=float),
            "control": np.ones(n),
        }
    )
    data = tmp_path / "placebo.csv"
    df.to_csv(data, index=False)

    report = run_lp_iv_placebo_csv(
        data,
        treatment_col="treatment",
        instrument_cols=["instrument"],
        outcome_cols=["outcome"],
        controls=["control"],
        placebo_horizons=[1, 2],
        out_csv=tmp_path / "out.csv",
    )
    out = pd.read_csv(tmp_path / "out.csv")

    assert report["status"] == "ok"
    assert report["rows"] == 2
    assert set(out["placebo_horizon"]) == {1, 2}
    assert "placebo_sig_95" in out.columns


def test_run_lp_iv_csv(tmp_path) -> None:
    n = 36
    instrument = np.arange(n, dtype=float)
    treatment = 2.0 * instrument
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=n, freq="ME"),
            "instrument": instrument,
            "treatment": treatment,
            "outcome": 3.0 * treatment,
            "control": np.ones(n),
        }
    )
    data = tmp_path / "lpiv.csv"
    df.to_csv(data, index=False)

    report = run_lp_iv_csv(
        data,
        treatment_col="treatment",
        instrument_cols=["instrument"],
        outcome_cols=["outcome"],
        controls=["control"],
        horizons=[0, 1],
        out_csv=tmp_path / "out.csv",
    )

    assert report["status"] == "ok"
    assert report["rows"] == 2


def test_run_lp_iv_csv_adds_same_unit_pass_through_for_tdc(tmp_path) -> None:
    n = 36
    instrument = np.arange(n, dtype=float)
    treatment = 2.0 * instrument
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=n, freq="ME"),
            "instrument": instrument,
            "tdc_monthly": treatment,
            "deposits": 0.5 * treatment / 1000.0,
            "control": np.ones(n),
        }
    )
    data = tmp_path / "lpiv_tdc.csv"
    df.to_csv(data, index=False)

    run_lp_iv_csv(
        data,
        treatment_col="tdc_monthly",
        instrument_cols=["instrument"],
        outcome_cols=["deposits"],
        controls=["control"],
        horizons=[0],
        out_csv=tmp_path / "out.csv",
    )
    out = pd.read_csv(tmp_path / "out.csv")

    assert "same_unit_beta" in out.columns
    assert round(float(out.loc[0, "same_unit_beta"]), 3) == 0.5
    assert out.loc[0, "same_unit_interpretation"] == "outcome dollars per 1 TDC dollar"


def test_run_lp_iv_csv_keeps_million_dollar_reserves_in_same_units(tmp_path) -> None:
    n = 36
    instrument = np.arange(n, dtype=float)
    treatment = 2.0 * instrument
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=n, freq="ME"),
            "instrument": instrument,
            "tdc_monthly": treatment,
            "reserves": 0.25 * treatment,
            "control": np.ones(n),
        }
    )
    data = tmp_path / "lpiv_tdc_reserves.csv"
    df.to_csv(data, index=False)

    run_lp_iv_csv(
        data,
        treatment_col="tdc_monthly",
        instrument_cols=["instrument"],
        outcome_cols=["reserves"],
        controls=["control"],
        horizons=[0],
        out_csv=tmp_path / "out.csv",
    )
    out = pd.read_csv(tmp_path / "out.csv")

    assert round(float(out.loc[0, "same_unit_multiplier"]), 3) == 1.0
    assert round(float(out.loc[0, "same_unit_beta"]), 3) == 0.25
    assert out.loc[0, "same_unit_interpretation"] == "outcome dollars per 1 TDC dollar"
