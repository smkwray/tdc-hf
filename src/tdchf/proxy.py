from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .benchmarking import (
    BenchmarkDiagnostics,
    additive_denton,
    additive_quarterly_residual_spread,
    validate_quarterly_identity,
)
from .calendar import to_month_end


COMPONENT_ORDER = ["fed_tsy", "banks_tsy", "row_tsy", "minus_toc", "fed_remit_positive"]


def benchmark_components(
    monthly_indicators: Mapping[str, pd.Series],
    quarterly_anchors: Mapping[str, pd.Series],
    *,
    method: str = "residual_spread",
) -> tuple[pd.DataFrame, list[BenchmarkDiagnostics]]:
    frames: list[pd.Series] = []
    diagnostics: list[BenchmarkDiagnostics] = []
    method_fns = {
        "residual_spread": additive_quarterly_residual_spread,
        "denton": additive_denton,
    }
    if method not in method_fns:
        raise ValueError(f"Unsupported benchmark method: {method}")
    method_fn = method_fns[method]
    method_label = {
        "residual_spread": "additive_quarterly_residual_spread",
        "denton": "additive_denton",
    }[method]

    for component in COMPONENT_ORDER:
        if component not in monthly_indicators:
            raise KeyError(f"Missing monthly indicator: {component}")
        if component not in quarterly_anchors:
            raise KeyError(f"Missing quarterly anchor: {component}")
        benchmarked = method_fn(
            monthly_indicators[component],
            quarterly_anchors[component],
            component=component,
        )
        diagnostics.append(
            validate_quarterly_identity(
                benchmarked,
                quarterly_anchors[component],
                component=component,
                method=method_label,
            )
        )
        frames.append(benchmarked)

    components = pd.concat(frames, axis=1, sort=False).sort_index()
    components.index = to_month_end(components.index)
    return components, diagnostics


def build_monthly_proxy(components: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in COMPONENT_ORDER if col not in components.columns]
    if missing:
        raise KeyError(f"Missing component columns: {missing}")
    out = components.copy()
    out["tdc_monthly"] = out[COMPONENT_ORDER].sum(axis=1, min_count=len(COMPONENT_ORDER))
    return out
