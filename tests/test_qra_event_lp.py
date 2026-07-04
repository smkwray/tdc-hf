from __future__ import annotations

import numpy as np
import pandas as pd

from tdchf.qra_event_lp import construct_qra_event_panel, estimate_qra_event_lps, normalize_weekly_panel_units


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
            "post_2020": [False] * n,
            "rrp_active": [False] * n,
            "tga_target_surprise_bn": [np.nan] * n,
            "deficit_surprise_bn": [np.nan] * n,
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
