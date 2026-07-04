from __future__ import annotations

import numpy as np
import pandas as pd

from tdchf.reabsorption import (
    build_reabsorption_state_weekly,
    calibrate_random_walk_placebo,
    estimate_state_interacted_retention,
    fit_exponential_decay,
)
from tdchf.qra_event_lp import normalize_weekly_panel_units


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


def test_seeded_multi_seed_placebo_calibration_reproduces_false_positive_rates(tmp_path) -> None:
    flows = pd.read_csv("data/processed/dts_weekly_flow_decomposition.csv", parse_dates=["date"]).set_index("date")
    calendar = pd.read_csv("data/processed/fiscal_calendar_weekly.csv", parse_dates=["date"])
    weekly = normalize_weekly_panel_units(pd.read_csv("data/processed/tdc_weekly_channel_panel.csv", parse_dates=["date"]).set_index("date"))
    state = build_reabsorption_state_weekly(
        raw_state_csv="data/raw/fred_reabsorption_state_sources.csv",
        out_csv=tmp_path / "reabsorption_state_test.csv",
        start=flows.index.min(),
        end=flows.index.max(),
    )
    estimates = estimate_state_interacted_retention(
        flows,
        calendar,
        weekly,
        state,
        interaction_bootstrap_reps=0,
        compute_moving_block_wild=False,
    )
    calibration = calibrate_random_walk_placebo(flows, calendar, weekly, state, estimates)
    by_h = calibration.set_index("horizon")

    assert by_h.loc[4, "placebo_seed_count"] == 20
    assert by_h.loc[8, "placebo_seed_count"] == 20
    assert abs(by_h.loc[4, "placebo_false_positive_rate"] - 0.45) < 1e-9
    assert abs(by_h.loc[8, "placebo_false_positive_rate"] - 0.40) < 1e-9
    assert abs(by_h.loc[4, "placebo_effective_p"] - 0.25) < 1e-9
    assert abs(by_h.loc[8, "placebo_effective_p"] - 0.05) < 1e-9
