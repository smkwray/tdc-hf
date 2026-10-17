from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tdchf.qra_event_lp as qra_event_lp
from tdchf.qra_event_lp import HORIZONS, OUTCOME_SPECS, EventWeek, construct_qra_event_panel, estimate_qra_event_lps, normalize_weekly_panel_units


def test_qra_event_panel_alignment_has_no_lookahead(tmp_path) -> None:
    qra = tmp_path / "qra.csv"
    weekly = tmp_path / "weekly.csv"
    gdp = tmp_path / "gdp.csv"
    pd.DataFrame(
        {
            "event_id": ["e1"],
            "quarter": ["2024Q1"],
            "release_date": ["2024-01-22"],
            "surprise_bn": [100.0],
            "prior_estimate_bn": [500.0],
            "tga_assumption_announced_bn": [750.0],
            "tga_assumption_prior_bn": [700.0],
        }
    ).to_csv(qra, index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-03", periods=12, freq="W-WED"),
            "broad_deposits_nsa": np.arange(12, dtype=float),
        }
    ).to_csv(weekly, index=False)
    pd.DataFrame({"observation_date": ["2024-01-01"], "GDP": [20000.0]}).to_csv(gdp, index=False)

    panel = construct_qra_event_panel(
        qra,
        weekly,
        gdp_csv=gdp,
        outcome_specs={"deposits_dpsacb": "broad_deposits_nsa"},
        horizons=[1],
    )

    row = panel.iloc[0]
    assert row["week_date"] == "2024-01-17"
    assert row["base_date"] == "2024-01-10"
    assert pd.Timestamp(row["base_date"]) < pd.Timestamp(row["release_date"])
    assert row["y_change_bn"] == 2.0


def test_qra_event_lp_normalizes_h4_series_to_billions() -> None:
    panel = pd.DataFrame({"reserves": [2_500_000.0], "fed_total_assets": [6_700_000.0], "broad_deposits_nsa": [19_000.0]})

    out = normalize_weekly_panel_units(panel)

    assert out.loc[0, "reserves"] == 2500.0
    assert out.loc[0, "fed_total_assets"] == 6700.0
    assert out.loc[0, "broad_deposits_nsa"] == 19000.0


def test_qra_event_lp_uses_rrpontsyd_as_headline_on_rrp_and_drops_dead_placebo() -> None:
    assert OUTCOME_SPECS["on_rrp_rrpontsyd"] == "onrrp"
    assert OUTCOME_SPECS["on_rrp_total_incl_foreign_wlrral"] == "fed_liquidity_facilities"
    assert -1 not in HORIZONS


def test_qra_event_lp_smoke_recovers_planted_beta() -> None:
    n = 36
    rng = np.random.default_rng(123)
    surprise = np.linspace(-150.0, 200.0, n)
    prior = rng.normal(500.0, 40.0, n)
    pretrend = rng.normal(0.0, 8.0, n)
    panel = pd.DataFrame(
        {
            "quarter": [f"201{i // 4}Q{(i % 4) + 1}" for i in range(n)],
            "release_date": pd.date_range("2010-01-01", periods=n, freq="90D").astype(str),
            "outcome": ["deposits_dpsacb"] * n,
            "horizon": [1] * n,
            "y_change_bn": 0.25 * surprise + 0.01 * prior + 0.02 * pretrend,
            "surprise_bn": surprise,
            "surprise_pct_gdp": surprise / 20000.0 * 100.0,
            "gdp_bn": [20000.0] * n,
            "prior_estimate_bn": prior,
            "pretrend_4w_bn": pretrend,
            "exclude_2020_outlier": [False] * n,
            "exclude_pandemic_block": [False] * n,
            "post_2020": [False] * n,
            "rrp_active": [False] * n,
            "tga_target_component_bn": [np.nan] * n,
            "deficit_component_bn": [np.nan] * n,
        }
    )

    estimates = estimate_qra_event_lps(panel)
    row = estimates.loc[
        estimates["outcome"].eq("deposits_dpsacb")
        & estimates["horizon"].eq(1)
        & estimates["sample"].eq("full")
        & estimates["scaling"].eq("bn")
    ].iloc[0]

    assert round(float(row["beta"]), 2) == 0.25
    assert round(float(row["beta_per_100bn"]), 1) == 25.0


def _write_state_fixture(tmp_path, release_dates: list[str], onrrp: dict[str, float]):
    qra = tmp_path / "qra.csv"
    weekly = tmp_path / "weekly.csv"
    gdp = tmp_path / "gdp.csv"
    pd.DataFrame(
        {
            "event_id": [f"e{idx}" for idx in range(len(release_dates))],
            "quarter": [pd.Timestamp(value).to_period("Q").strftime("%YQ%q") for value in release_dates],
            "release_date": release_dates,
            "surprise_bn": [100.0] * len(release_dates),
            "prior_estimate_bn": [500.0] * len(release_dates),
        }
    ).to_csv(qra, index=False)
    dates = pd.date_range("2023-11-01", periods=40, freq="W-WED")
    pd.DataFrame(
        {
            "date": dates,
            "broad_deposits_nsa": np.arange(len(dates), dtype=float),
            "onrrp": [onrrp.get(day.date().isoformat(), 20.0) for day in dates],
        }
    ).to_csv(weekly, index=False)
    pd.DataFrame({"observation_date": ["2023-10-01", "2024-01-01", "2024-04-01"], "GDP": [20000.0] * 3}).to_csv(gdp, index=False)
    return qra, weekly, gdp


def test_qra_on_rrp_state_is_read_at_the_pre_window_base_week(tmp_path) -> None:
    # Outcomes are changes from the base week. If the state were read at the
    # event week, an ON RRP move inside the measured window (10bn -> 100bn here)
    # would reclassify the event as RRP-active.
    qra, weekly, gdp = _write_state_fixture(tmp_path, ["2024-01-22"], {"2024-01-10": 10.0, "2024-01-17": 100.0})

    panel = construct_qra_event_panel(qra, weekly, gdp_csv=gdp, outcome_specs={"deposits_dpsacb": "broad_deposits_nsa"}, horizons=[1])

    row = panel.iloc[0]
    assert (row["base_date"], row["week_date"]) == ("2024-01-10", "2024-01-17")
    assert row["rrp_lag_bn"] == 10.0
    assert not row["rrp_active"]
    assert row["rrp_state_date"] == "2024-01-10"


def test_qra_state_strictly_precedes_every_announcement(tmp_path) -> None:
    releases = ["2024-01-29", "2024-04-29", "2024-07-29"]
    qra, weekly, gdp = _write_state_fixture(tmp_path, releases, {})

    panel = construct_qra_event_panel(qra, weekly, gdp_csv=gdp, outcome_specs={"deposits_dpsacb": "broad_deposits_nsa"}, horizons=[1, 4])

    events = panel[["release_date", "week_date", "base_date", "rrp_state_date"]].drop_duplicates()
    assert sorted(events["release_date"]) == releases
    state = pd.to_datetime(events["rrp_state_date"])
    assert (state <= pd.to_datetime(events["base_date"])).all()
    assert (state < pd.to_datetime(events["week_date"])).all()
    assert (state < pd.to_datetime(events["release_date"])).all()


def test_qra_event_panel_rejects_a_window_that_does_not_precede_the_release(tmp_path, monkeypatch) -> None:
    qra, weekly, gdp = _write_state_fixture(tmp_path, ["2024-01-22"], {})

    def misaligned(release_date, weekly_dates):
        return EventWeek(release_date=release_date, week_date=pd.Timestamp("2024-01-24"), base_date=pd.Timestamp("2024-01-17"))

    monkeypatch.setattr(qra_event_lp, "find_event_week", misaligned)
    with pytest.raises(ValueError, match="not predetermined"):
        construct_qra_event_panel(qra, weekly, gdp_csv=gdp, outcome_specs={"deposits_dpsacb": "broad_deposits_nsa"}, horizons=[1])
