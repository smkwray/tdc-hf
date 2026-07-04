from __future__ import annotations

import json

import pandas as pd

from tdchf.pipeline import run_monthly_proxy_pipeline


def test_run_monthly_proxy_pipeline_from_tdcest_anchors(tmp_path) -> None:
    report = run_monthly_proxy_pipeline(out_dir=tmp_path)

    assert report["status"] == "ok"
    assert (tmp_path / "tdc_monthly_components.csv").exists()
    assert (tmp_path / "tdc_monthly_proxy.csv").exists()
    validation = json.loads((tmp_path / "tdc_validation_report.json").read_text())
    assert validation["status"] == "ok"


def test_run_monthly_proxy_pipeline_from_indicator_csv(tmp_path) -> None:
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    indicators = pd.DataFrame(
        {
            "date": dates,
            "fed_tsy": [1.0, 1.0, 1.0],
            "banks_tsy": [1.0, 1.0, 1.0],
            "row_tsy": [1.0, 1.0, 1.0],
            "minus_toc": [1.0, 1.0, 1.0],
            "fed_remit_positive": [1.0, 1.0, 1.0],
        }
    )
    anchors = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-03-31")],
            "fed_tsy_tx": [6.0],
            "bank_depository_tsy_tx": [6.0],
            "row_tsy_tx": [6.0],
            "minus_treasury_operating_cash_tx": [6.0],
            "fed_remit_positive": [6.0],
        }
    )
    indicator_path = tmp_path / "indicators.csv"
    anchor_path = tmp_path / "anchors.csv"
    indicators.to_csv(indicator_path, index=False)
    anchors.to_csv(anchor_path, index=False)

    report = run_monthly_proxy_pipeline(
        out_dir=tmp_path / "out",
        monthly_indicators_path=indicator_path,
        quarterly_anchors_path=anchor_path,
    )

    assert report["status"] == "ok"
    proxy = pd.read_csv(tmp_path / "out" / "tdc_monthly_proxy.csv")
    assert proxy["tdc_monthly"].sum() == 30.0


def test_run_monthly_proxy_pipeline_fills_partial_indicator_csv(tmp_path) -> None:
    indicators = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-02-29")],
            "fed_tsy": [10.0],
        }
    )
    anchors = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-03-31")],
            "fed_tsy_tx": [9.0],
            "bank_depository_tsy_tx": [6.0],
            "row_tsy_tx": [6.0],
            "minus_treasury_operating_cash_tx": [6.0],
            "fed_remit_positive": [6.0],
        }
    )
    indicator_path = tmp_path / "partial.csv"
    anchor_path = tmp_path / "anchors.csv"
    indicators.to_csv(indicator_path, index=False)
    anchors.to_csv(anchor_path, index=False)

    report = run_monthly_proxy_pipeline(
        out_dir=tmp_path / "out_partial",
        monthly_indicators_path=indicator_path,
        quarterly_anchors_path=anchor_path,
    )

    assert report["status"] == "ok"
    assert (tmp_path / "out_partial" / "tdc_indicator_source_coverage.csv").exists()
    assert (tmp_path / "out_partial" / "tdc_raw_indicator_quarterly_fit.csv").exists()
    assert (tmp_path / "out_partial" / "tdc_allocation_error_proxy.csv").exists()
