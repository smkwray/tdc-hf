from __future__ import annotations

import pandas as pd

from tdchf.pretrend import add_lagged_factor_controls, add_pretrend_controls


def test_add_pretrend_controls_adds_deeper_lags_and_rolling_sums() -> None:
    idx = pd.date_range("2024-01-31", periods=6, freq="ME")
    df = pd.DataFrame({"deposits": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=idx)

    out = add_pretrend_controls(df, columns=["deposits"], lags=[2], windows=[3])

    assert out.loc[pd.Timestamp("2024-03-31"), "lag2_deposits"] == 1.0
    assert out.loc[pd.Timestamp("2024-04-30"), "pretrend3_deposits"] == 6.0


def test_add_lagged_factor_controls_uses_only_lagged_inputs() -> None:
    idx = pd.date_range("2024-01-31", periods=8, freq="ME")
    df = pd.DataFrame(
        {
            "deposits": [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0],
            "bank_credit": [2.0, 3.0, 5.0, 8.0, 12.0, 17.0, 23.0, 30.0],
            "yield_2y": [4.0, 4.1, 4.0, 3.9, 4.2, 4.3, 4.1, 4.0],
        },
        index=idx,
    )

    out = add_lagged_factor_controls(df, columns=["deposits", "bank_credit", "yield_2y"], n_factors=2, lag=1)

    assert {"factor1_lag1", "factor2_lag1"}.issubset(out.columns)
    assert pd.isna(out.loc[pd.Timestamp("2024-01-31"), "factor1_lag1"])
    assert out["factor1_lag1"].notna().sum() == 7
