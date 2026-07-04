from __future__ import annotations

import pandas as pd

from tdchf.benchmarking import additive_denton, additive_quarterly_residual_spread, validate_quarterly_identity
from tdchf.proxy import benchmark_components, build_monthly_proxy


def test_additive_quarterly_residual_spread_matches_anchor() -> None:
    months = pd.date_range("2024-01-31", periods=3, freq="ME")
    indicator = pd.Series([1.0, 2.0, 3.0], index=months)
    anchor = pd.Series([12.0], index=[pd.Timestamp("2024-03-31")])

    out = additive_quarterly_residual_spread(indicator, anchor, component="fed_tsy")

    assert list(out.round(6)) == [3.0, 4.0, 5.0]
    diag = validate_quarterly_identity(out, anchor, component="fed_tsy")
    assert diag.max_abs_quarterly_error == 0.0
    assert diag.quarters_checked == 1


def test_additive_denton_matches_anchor() -> None:
    months = pd.date_range("2024-01-31", periods=6, freq="ME")
    indicator = pd.Series([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], index=months)
    anchor = pd.Series([12.0, 9.0], index=[pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")])

    out = additive_denton(indicator, anchor, component="fed_tsy")
    diag = validate_quarterly_identity(out, anchor, component="fed_tsy", method="additive_denton")

    assert diag.max_abs_quarterly_error < 1e-8
    assert round(out.iloc[:3].sum(), 8) == 12.0
    assert round(out.iloc[3:].sum(), 8) == 9.0


def test_build_monthly_proxy_sums_components() -> None:
    months = pd.date_range("2024-01-31", periods=3, freq="ME")
    quarters = [pd.Timestamp("2024-03-31")]
    monthly = {
        "fed_tsy": pd.Series([1.0, 1.0, 1.0], index=months),
        "banks_tsy": pd.Series([2.0, 2.0, 2.0], index=months),
        "row_tsy": pd.Series([3.0, 3.0, 3.0], index=months),
        "minus_toc": pd.Series([-1.0, -1.0, -1.0], index=months),
        "fed_remit_positive": pd.Series([0.5, 0.5, 0.5], index=months),
    }
    anchors = {
        "fed_tsy": pd.Series([3.0], index=quarters),
        "banks_tsy": pd.Series([6.0], index=quarters),
        "row_tsy": pd.Series([9.0], index=quarters),
        "minus_toc": pd.Series([-3.0], index=quarters),
        "fed_remit_positive": pd.Series([1.5], index=quarters),
    }

    components, diagnostics = benchmark_components(monthly, anchors)
    proxy = build_monthly_proxy(components)

    assert len(diagnostics) == 5
    assert proxy["tdc_monthly"].sum() == 16.5
    assert all(diag.max_abs_quarterly_error == 0.0 for diag in diagnostics)
