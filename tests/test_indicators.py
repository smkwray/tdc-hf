from __future__ import annotations

import pandas as pd

from tdchf.indicators import (
    aggregate_flows_to_monthly,
    aggregate_levels_to_monthly,
    fill_indicator_gaps_from_equal_months,
    level_change_to_monthly_flow,
    no_indicator_equal_months,
    positive_only,
)


def test_aggregate_daily_flows_to_monthly_sum() -> None:
    series = pd.Series(
        [1.0, 2.0, 5.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01"]),
        name="flow",
    )

    out = aggregate_flows_to_monthly(series)

    assert out.loc[pd.Timestamp("2024-01-31")] == 3.0
    assert out.loc[pd.Timestamp("2024-02-29")] == 5.0


def test_level_change_to_monthly_flow_uses_month_end_level() -> None:
    series = pd.Series(
        [10.0, 13.0, 20.0, 18.0],
        index=pd.to_datetime(["2024-01-03", "2024-01-31", "2024-02-15", "2024-02-29"]),
        name="level",
    )

    levels = aggregate_levels_to_monthly(series)
    flow = level_change_to_monthly_flow(series)

    assert levels.loc[pd.Timestamp("2024-01-31")] == 13.0
    assert flow.loc[pd.Timestamp("2024-02-29")] == 5.0


def test_positive_only_clips_negative_values() -> None:
    series = pd.Series([-2.0, 0.0, 3.0])

    out = positive_only(series)

    assert list(out) == [0.0, 0.0, 3.0]


def test_no_indicator_equal_months_allocates_quarterly_total() -> None:
    anchors = {
        component: pd.Series([9.0], index=[pd.Timestamp("2024-03-31")], name=component)
        for component in ["fed_tsy", "banks_tsy", "row_tsy", "minus_toc", "fed_remit_positive"]
    }

    out = no_indicator_equal_months(anchors)

    assert out["fed_tsy"].sum() == 9.0
    assert list(out["fed_tsy"]) == [3.0, 3.0, 3.0]


def test_fill_indicator_gaps_from_equal_months_keeps_observed_values() -> None:
    anchors = {
        component: pd.Series([9.0], index=[pd.Timestamp("2024-03-31")], name=component)
        for component in ["fed_tsy", "banks_tsy", "row_tsy", "minus_toc", "fed_remit_positive"]
    }
    observed = {"fed_tsy": pd.Series([5.0], index=[pd.Timestamp("2024-02-29")], name="fed_tsy")}

    filled, coverage = fill_indicator_gaps_from_equal_months(observed, anchors)

    assert filled["fed_tsy"].loc[pd.Timestamp("2024-02-29")] == 5.0
    assert filled["row_tsy"].sum() == 9.0
    assert set(coverage["component"]) == set(anchors)
