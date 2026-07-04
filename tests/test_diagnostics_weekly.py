from __future__ import annotations

import pandas as pd

from tdchf.diagnostics import raw_indicator_quarterly_fit
from tdchf.weekly import build_weekly_channel_panel, build_weekly_state_from_fred_frame


def test_raw_indicator_quarterly_fit_reports_error() -> None:
    months = pd.date_range("2024-01-31", periods=3, freq="ME")
    monthly = {"fed_tsy": pd.Series([1.0, 2.0, 3.0], index=months)}
    anchors = {"fed_tsy": pd.Series([12.0], index=[pd.Timestamp("2024-03-31")])}

    out = raw_indicator_quarterly_fit(monthly, anchors)

    assert out.loc[0, "component"] == "fed_tsy"
    assert out.loc[0, "mean_error"] == -6.0
    assert out.loc[0, "quarters"] == 1


def test_build_weekly_state_from_fred_frame() -> None:
    idx = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-10"])
    df = pd.DataFrame(
        {
            "WTREGEN": [1.0, 2.0, 3.0],
            "WDTGAL": [4.0, 5.0, 6.0],
            "TREAST": [7.0, 8.0, 9.0],
            "RRPONTSYD": [10.0, 11.0, 12.0],
        },
        index=idx,
    )

    out = build_weekly_state_from_fred_frame(df)

    assert "tga_week_avg" in out.columns
    assert "onrrp" in out.columns
    assert out.loc[pd.Timestamp("2024-01-03"), "tga_week_avg"] == 2.0


def test_build_weekly_channel_panel_adds_changes_and_deposit_target(tmp_path) -> None:
    path = tmp_path / "weekly.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-03", periods=3, freq="W-WED"),
            "tga_wednesday": [10.0, 12.0, 11.0],
            "domestic_deposits": [100.0, 103.0, 105.0],
            "large_time_deposits": [20.0, 21.0, 21.5],
            "onrrp": [50.0, 49.0, 47.0],
            "retail_mmf": [40.0, 41.0, 43.0],
            "institutional_mmf": [60.0, 60.5, 61.0],
        }
    ).to_csv(path, index=False)

    out = build_weekly_channel_panel([path])

    assert "domestic_non_large_time_deposits" in out.columns
    assert "total_mmf" in out.columns
    assert "d_total_mmf" in out.columns
    assert "d_domestic_non_large_time_deposits" in out.columns
    assert "lag_d_onrrp" in out.columns
    assert out.loc[pd.Timestamp("2024-01-10"), "d_tga_wednesday"] == 2.0
