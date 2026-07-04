from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .proxy import COMPONENT_ORDER, benchmark_components, build_monthly_proxy
from .validation import write_validation_report


def _demo_series() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    months = pd.date_range("2024-01-31", periods=6, freq="ME")
    quarters = pd.date_range("2024-03-31", periods=2, freq="QE")

    monthly_indicators: dict[str, pd.Series] = {}
    quarterly_anchors: dict[str, pd.Series] = {}
    raw_values = {
        "fed_tsy": [10, 12, 8, 9, 11, 10],
        "banks_tsy": [4, 5, 6, 7, 6, 5],
        "row_tsy": [3, 2, 5, 1, 4, 3],
        "minus_toc": [-6, -4, 2, -3, -1, 4],
        "fed_remit_positive": [1, 1, 1, 0, 0, 0],
    }
    anchors = {
        "fed_tsy": [33, 36],
        "banks_tsy": [18, 21],
        "row_tsy": [12, 10],
        "minus_toc": [-9, 2],
        "fed_remit_positive": [3, 0],
    }
    for component in COMPONENT_ORDER:
        monthly_indicators[component] = pd.Series(raw_values[component], index=months, name=component)
        quarterly_anchors[component] = pd.Series(anchors[component], index=quarters, name=component)
    return monthly_indicators, quarterly_anchors


def run_demo(out_dir: str | Path) -> dict[str, object]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    monthly_indicators, quarterly_anchors = _demo_series()
    components, diagnostics = benchmark_components(monthly_indicators, quarterly_anchors)
    proxy = build_monthly_proxy(components)

    components.to_csv(out_path / "tdc_monthly_components.csv", index_label="date")
    proxy[["tdc_monthly"]].to_csv(out_path / "tdc_monthly_proxy.csv", index_label="date")
    report = write_validation_report(
        out_path / "tdc_validation_report.json",
        components=components,
        anchors=quarterly_anchors,
        diagnostics=diagnostics,
        method="synthetic_demo_additive_quarterly_residual_spread",
        notes=["Synthetic data only."],
    )
    report["outputs"] = {
        "components": str(out_path / "tdc_monthly_components.csv"),
        "proxy": str(out_path / "tdc_monthly_proxy.csv"),
    }
    (out_path / "tdc_validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
