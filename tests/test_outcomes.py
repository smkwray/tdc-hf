from __future__ import annotations

import pandas as pd

from tdchf.outcomes import build_monthly_outcomes_from_fred_frame


def test_build_monthly_outcomes_from_fred_frame() -> None:
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    df = pd.DataFrame(
        {
            "DPSACBM027SBOG": [100.0, 110.0, 125.0],
            "DPSDCBW027SBOG": [90.0, 95.0, 100.0],
            "LTDACBW027SBOG": [20.0, 24.0, 30.0],
            "TOTBKCR": [200.0, 210.0, 225.0],
            "WRESBAL": [50.0, 60.0, 55.0],
            "RRPONTSYD": [10.0, 12.0, 11.0],
        },
        index=idx,
    )

    out, meta = build_monthly_outcomes_from_fred_frame(df)

    assert out.loc[pd.Timestamp("2024-02-29"), "broad_deposits"] == 10.0
    assert out.loc[pd.Timestamp("2024-02-29"), "domestic_deposits"] == 5.0
    assert out.loc[pd.Timestamp("2024-02-29"), "deposits"] == 1.0
    assert out.loc[pd.Timestamp("2024-03-31"), "large_time_deposits"] == 6.0
    assert out.loc[pd.Timestamp("2024-03-31"), "broad_non_large_time_deposits"] == 9.0
    assert out.loc[pd.Timestamp("2024-03-31"), "domestic_non_large_time_deposits"] == -1.0
    assert out.loc[pd.Timestamp("2024-02-29"), "reserves"] == 10.0
    assert out.loc[pd.Timestamp("2024-02-29"), "onrrp"] == 2.0
    assert out.loc[pd.Timestamp("2024-02-29"), "reserves_level"] == 60.0
    assert "deposits" in set(meta["column"])
    deposit_meta = meta.loc[meta["column"] == "deposits"].iloc[0]
    assert deposit_meta["source_series"] == "domestic_non_large_time_deposits"


def test_build_monthly_outcomes_from_fred_frame_adds_mmf_flows() -> None:
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    df = pd.DataFrame(
        {
            "WRMFNS": [20.0, 22.0, 25.0],
            "WIMFNS": [30.0, 33.0, 31.0],
        },
        index=idx,
    )

    out, meta = build_monthly_outcomes_from_fred_frame(df)

    assert out.loc[pd.Timestamp("2024-02-29"), "retail_mmf"] == 2.0
    assert out.loc[pd.Timestamp("2024-02-29"), "institutional_mmf"] == 3.0
    assert out.loc[pd.Timestamp("2024-02-29"), "total_mmf"] == 5.0
    assert {"retail_mmf", "institutional_mmf", "total_mmf"}.issubset(set(meta["column"]))


def test_build_monthly_outcomes_from_fred_frame_adds_loan_and_spread_outcomes() -> None:
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    df = pd.DataFrame(
        {
            "TOTCINSA": [100.0, 105.0, 103.0],
            "CLSACBW027SBOG": [50.0, 55.0, 58.0],
            "CCLACBW027SBOG": [20.0, 22.0, 25.0],
            "CRLACBW027SBOG": [80.0, 82.0, 81.0],
            "BAA": [6.0, 6.2, 6.3],
            "AAA": [5.0, 5.1, 5.0],
            "DGS10": [4.0, 4.2, 4.1],
            "DGS2": [3.0, 3.1, 3.2],
        },
        index=idx,
    )

    out, meta = build_monthly_outcomes_from_fred_frame(df)

    assert out.loc[pd.Timestamp("2024-02-29"), "commercial_industrial_loans"] == 5.0
    assert out.loc[pd.Timestamp("2024-02-29"), "consumer_loans"] == 5.0
    assert out.loc[pd.Timestamp("2024-02-29"), "credit_card_revolving_loans"] == 2.0
    assert out.loc[pd.Timestamp("2024-02-29"), "closed_end_residential_loans"] == 2.0
    assert round(out.loc[pd.Timestamp("2024-02-29"), "d_baa_aaa_spread"], 6) == 0.1
    assert {"commercial_industrial_loans", "d_baa_aaa_spread"}.issubset(set(meta["column"]))
