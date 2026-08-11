from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tdchf.state_response_diagnostics import (
    build_state_response_weekly,
    write_state_response_gate_status,
)


def test_state_response_alignment_names_lags_and_fdic_break(tmp_path) -> None:
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
            "RRPONTSYD": [100.0, 120.0, np.nan, 10.0, np.nan, 60.0, np.nan, 12.0, 13.0],
        }
    ).to_csv(raw, index=False)

    state = build_state_response_weekly(
        raw_state_csv=raw,
        out_csv=tmp_path / "state.csv",
        start="2021-03-24",
        end="2021-04-14",
    )
    pre_rrp = build_state_response_weekly(
        raw_state_csv=raw,
        out_csv=tmp_path / "state_pre.csv",
        start="2012-12-26",
        end="2013-01-02",
    )

    assert state.loc[pd.Timestamp("2021-03-31"), "fdic_savings_rate_proxy_source"] == "SAVNRNJ_weekly_discontinued"
    assert state.loc[pd.Timestamp("2021-04-07"), "fdic_savings_rate_proxy_source"] == "SNDR_monthly"
    assert int(state["fdic_methodology_break_2021_04"].sum()) == 1
    assert state.loc[pd.Timestamp("2021-04-07"), "bill3m_yield_level_lag1"] == state.loc[
        pd.Timestamp("2021-03-31"), "bill3m_yield_level"
    ]
    assert state.loc[pd.Timestamp("2021-04-07"), "on_rrp_balance_ge_50bn_lag1_indicator"] == 1.0
    assert (pre_rrp["on_rrp_balance"] == 0.0).all()


def test_state_response_source_fails_closed_on_missing_required_series(tmp_path) -> None:
    raw = tmp_path / "incomplete_state.csv"
    pd.DataFrame({"date": ["2021-03-24"], "DGS3MO": [0.02]}).to_csv(raw, index=False)

    with pytest.raises(ValueError, match="missing required series"):
        build_state_response_weekly(
            raw_state_csv=raw,
            out_csv=tmp_path / "state.csv",
            start="2021-03-24",
            end="2021-03-24",
        )


def test_state_response_gate_status_contains_only_coverage_and_failed_gates(tmp_path) -> None:
    dates = pd.date_range("2021-03-24", periods=4, freq="W-WED")
    state = pd.DataFrame(
        {
            "bill3m_yield_level": [0.02, 0.03, 0.04, 0.05],
            "bill_minus_fdic_savings_rate_proxy": [0.01, 0.02, -0.01, 0.0],
            "on_rrp_balance_ge_50bn_lag1_indicator": [0.0, 1.0, 1.0, 0.0],
            "fdic_methodology_break_2021_04": [0, 0, 1, 0],
        },
        index=dates,
    )
    calibration = pd.DataFrame(
        {
            "horizon": [4, 8],
            "placebo_seed_count": [20, 20],
            "placebo_false_positive_rate": [0.45, 0.40],
            "placebo_effective_p": [0.25, 0.05],
        }
    )

    status = write_state_response_gate_status(
        state,
        placebo_calibration=calibration,
        stability_failed_horizons=2,
        out_csv=tmp_path / "state_response_gate_status.csv",
    )

    assert set(status["row_type"]) == {"coverage", "failed_gate"}
    assert (status.loc[status["row_type"].eq("failed_gate"), "admission_status"] == "failed").all()
    assert not status["promotion_ready"].any()
    assert {"random_walk_placebo_h4", "random_walk_placebo_h8"}.issubset(set(status["diagnostic_id"]))
    assert set(status.columns) == {
        "row_type",
        "diagnostic_id",
        "state_variable",
        "admission_status",
        "promotion_ready",
        "coverage_start",
        "coverage_end",
        "observed_weeks",
        "total_weeks",
        "horizon",
        "observed_value",
        "gate_threshold",
        "evidence_scope",
        "reason",
    }
