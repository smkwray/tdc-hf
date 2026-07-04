from __future__ import annotations

import numpy as np
import pandas as pd

from tdchf.disbursement import FLOW_BUCKETS
from tdchf.reabsorption import (
    build_reabsorption_state_weekly,
    estimate_state_interacted_retention,
    fit_exponential_decay,
)


def test_reabsorption_state_alignment_lags_and_fdic_break(tmp_path) -> None:
    raw = tmp_path / "fred_state.csv"
    pd.DataFrame(
        {
            "date": [
                "2012-12-26",
                "2013-01-02",
                "2021-03-22",
                "2021-03-24",
                "2021-03-29",
                "2021-03-31",
                "2021-04-01",
                "2021-04-07",
                "2021-04-14",
            ],
            "DGS3MO": [0.05, 0.06, np.nan, 0.02, np.nan, 0.03, np.nan, 0.04, 0.05],
            "SAVNRNJ": [np.nan, np.nan, 0.04, np.nan, 0.04, np.nan, np.nan, np.nan, np.nan],
            "SNDR": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.05, np.nan, np.nan],
            "RRPONTSYD": [100.0, 120.0, np.nan, 10.0, np.nan, 11.0, np.nan, 12.0, 13.0],
            "WRESBAL": [1000.0] * 9,
            "GDP": [20000.0] * 9,
            "MMMFFAQ027S": [5000.0] * 9,
        }
    ).to_csv(raw, index=False)

    state = build_reabsorption_state_weekly(raw_state_csv=raw, out_csv=tmp_path / "state.csv", start="2021-03-24", end="2021-04-14")
    pre_rrp = build_reabsorption_state_weekly(raw_state_csv=raw, out_csv=tmp_path / "state_pre.csv", start="2012-12-26", end="2013-01-02")

    assert state.loc[pd.Timestamp("2021-03-31"), "deposit_rate_proxy_source"] == "SAVNRNJ_weekly_discontinued"
    assert state.loc[pd.Timestamp("2021-04-07"), "deposit_rate_proxy_source"] == "SNDR_monthly"
    assert int(state["fdic_methodology_break_2021_04"].sum()) == 1
    assert state.loc[pd.Timestamp("2021-04-07"), "fdic_methodology_break_2021_04"] == 1
    assert state.loc[pd.Timestamp("2021-04-07"), "yield_gradient_full_lag1"] == state.loc[pd.Timestamp("2021-03-31"), "yield_gradient_full"]
    assert state.loc[pd.Timestamp("2021-04-07"), "rrp_balance_lag1"] == state.loc[pd.Timestamp("2021-03-31"), "rrp_balance"]
    assert (pre_rrp["rrp_balance"] == 0.0).all()


def test_decay_fit_recovers_known_half_life() -> None:
    lambda_true = 0.2
    horizons = np.arange(14)
    beta = 10.0 * np.exp(-lambda_true * horizons)
    fit = fit_exponential_decay(pd.DataFrame({"horizon": horizons, "beta": beta}), treatment="du_core_outflows_bn")

    assert fit.meaningful
    assert abs(fit.lambda_hat - lambda_true) < 1e-9
    assert abs(fit.half_life - (np.log(2.0) / lambda_true)) < 1e-9
    assert fit.r2 > 0.999


def test_seeded_random_walk_placebo_interaction_is_not_flagged() -> None:
    n = 180
    dates = pd.date_range("2016-01-06", periods=n, freq="W-WED")
    rng = np.random.default_rng(20260704)
    flows = pd.DataFrame(index=dates)
    for col in FLOW_BUCKETS:
        flows[col] = rng.normal(0.0, 1.0, n)
    deposits = 100.0 + (0.6 * flows["du_core_outflows_bn"] - 0.4 * flows["tax_receipts_bn"]).cumsum()
    weekly = pd.DataFrame(
        {
            "broad_deposits_nsa": deposits + rng.normal(0.0, 0.05, n),
            "reserves": 1000.0,
            "onrrp": 0.0,
        },
        index=dates,
    )
    calendar = pd.DataFrame({"date": dates})
    state = pd.DataFrame({"random_walk_placebo_state_lag1": np.cumsum(rng.normal(0.0, 1.0, n))}, index=dates)

    estimates = estimate_state_interacted_retention(
        flows,
        calendar,
        weekly,
        state,
        state_columns={"random_walk_placebo": "random_walk_placebo_state_lag1"},
        samples=["ex_pandemic"],
        outcomes={"deposits_dpsacb": "broad_deposits_nsa"},
        interaction_bootstrap_reps=19,
    )
    ref = estimates.loc[estimates["horizon"].isin([2, 4, 8])]

    assert not ref.empty
    assert ref["p_moving_block_wild"].notna().all()
    assert (ref["p_moving_block_wild"] >= 0.05).all()
