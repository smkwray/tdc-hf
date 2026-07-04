from __future__ import annotations

from pathlib import Path

import pandas as pd

from .indicators import fill_indicator_gaps_from_equal_months, load_monthly_indicator_csv, no_indicator_equal_months
from .diagnostics import raw_indicator_quarterly_fit
from .proxy import benchmark_components, build_monthly_proxy
from .upstream import load_tdcest_quarterly_anchors
from .validation import (
    allocation_error_proxy,
    component_validation_table,
    loqo_month_share_diagnostics,
    quarterly_component_sums,
    write_validation_report,
)


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="date")


def run_monthly_proxy_pipeline(
    *,
    out_dir: str | Path = "data/processed",
    monthly_indicators_path: str | Path | None = None,
    quarterly_anchors_path: str | Path | None = None,
    benchmark_method: str = "residual_spread",
    fill_missing: bool = True,
    method_label: str | None = None,
) -> dict[str, object]:
    anchors = load_tdcest_quarterly_anchors(quarterly_anchors_path)
    source_coverage: pd.DataFrame | None = None
    raw_fit: pd.DataFrame | None = None
    if monthly_indicators_path is None:
        monthly_indicators = no_indicator_equal_months(anchors)
        raw_monthly_indicators = monthly_indicators
        method = method_label or "no_indicator_equal_months"
        notes = ["Bootstrap/placebo build: quarterly anchors are spread equally across months."]
    else:
        raw_monthly_indicators = load_monthly_indicator_csv(monthly_indicators_path, require_all=not fill_missing)
        raw_fit = raw_indicator_quarterly_fit(raw_monthly_indicators, anchors)
        monthly_indicators = raw_monthly_indicators
        if fill_missing:
            monthly_indicators, source_coverage = fill_indicator_gaps_from_equal_months(monthly_indicators, anchors)
        method = method_label or f"indicator_{benchmark_method}"
        notes = [f"Monthly indicators loaded from {Path(monthly_indicators_path).name}."]

    components, diagnostics = benchmark_components(monthly_indicators, anchors, method=benchmark_method)
    proxy = build_monthly_proxy(components)
    quarterly_sums = quarterly_component_sums(components)
    validation = component_validation_table(components, anchors, diagnostics)
    allocation_errors = allocation_error_proxy(components, raw_monthly_indicators)
    loqo_errors = loqo_month_share_diagnostics(components, raw_monthly_indicators)

    out_path = Path(out_dir)
    _write_frame(components, out_path / "tdc_monthly_components.csv")
    _write_frame(proxy[["tdc_monthly"]], out_path / "tdc_monthly_proxy.csv")
    _write_frame(quarterly_sums, out_path / "tdc_monthly_quarterly_sums.csv")
    validation.to_csv(out_path / "tdc_validation_table.csv", index=False)
    if raw_fit is not None:
        raw_fit.to_csv(out_path / "tdc_raw_indicator_quarterly_fit.csv", index=False)
    if not allocation_errors.empty:
        allocation_errors.to_csv(out_path / "tdc_allocation_error_proxy.csv", index=False)
    if not loqo_errors.empty:
        loqo_errors.to_csv(out_path / "tdc_loqo_month_share_diagnostics.csv", index=False)
    if source_coverage is not None:
        source_coverage.to_csv(out_path / "tdc_indicator_source_coverage.csv", index=False)
    report = write_validation_report(
        out_path / "tdc_validation_report.json",
        components=components,
        anchors=anchors,
        diagnostics=diagnostics,
        method=method,
        notes=notes,
    )

    return {
        "status": report["status"],
        "method": method,
        "rows": int(len(proxy)),
        "outputs": {
            "components": str(out_path / "tdc_monthly_components.csv"),
            "proxy": str(out_path / "tdc_monthly_proxy.csv"),
            "quarterly_sums": str(out_path / "tdc_monthly_quarterly_sums.csv"),
            "validation_table": str(out_path / "tdc_validation_table.csv"),
            "validation_report": str(out_path / "tdc_validation_report.json"),
            "source_coverage": str(out_path / "tdc_indicator_source_coverage.csv") if source_coverage is not None else "",
            "raw_indicator_fit": str(out_path / "tdc_raw_indicator_quarterly_fit.csv") if raw_fit is not None else "",
            "allocation_error_proxy": str(out_path / "tdc_allocation_error_proxy.csv") if not allocation_errors.empty else "",
            "loqo_month_share_diagnostics": str(out_path / "tdc_loqo_month_share_diagnostics.csv") if not loqo_errors.empty else "",
        },
    }
