from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .benchmarking import BenchmarkDiagnostics
from .calendar import to_quarter_end
from .proxy import COMPONENT_ORDER


def quarterly_component_sums(components: pd.DataFrame) -> pd.DataFrame:
    out = components[COMPONENT_ORDER].groupby(to_quarter_end(components.index)).sum(min_count=1)
    out.index.name = "date"
    return out


def component_validation_table(
    components: pd.DataFrame,
    anchors: dict[str, pd.Series],
    diagnostics: list[BenchmarkDiagnostics],
) -> pd.DataFrame:
    quarterly = quarterly_component_sums(components)
    diag_by_component = {diag.component: diag for diag in diagnostics}
    rows: list[dict[str, object]] = []
    for component in COMPONENT_ORDER:
        anchor = anchors[component].rename("anchor").dropna()
        summed = quarterly[component].rename("monthly_sum")
        aligned = pd.concat([summed, anchor], axis=1).dropna()
        errors = aligned["monthly_sum"] - aligned["anchor"]
        diag = diag_by_component.get(component)
        rows.append(
            {
                "component": component,
                "quarters_checked": int(len(aligned)),
                "first_quarter": aligned.index.min().date().isoformat() if not aligned.empty else "",
                "last_quarter": aligned.index.max().date().isoformat() if not aligned.empty else "",
                "max_abs_quarterly_error": float(errors.abs().max()) if not aligned.empty else float("nan"),
                "mean_abs_quarterly_error": float(errors.abs().mean()) if not aligned.empty else float("nan"),
                "method": diag.method if diag else "",
                "status": "ok" if diag and diag.max_abs_quarterly_error <= 1e-8 else "check",
            }
        )
    return pd.DataFrame(rows)


def write_validation_report(
    path: str | Path,
    *,
    components: pd.DataFrame,
    anchors: dict[str, pd.Series],
    diagnostics: list[BenchmarkDiagnostics],
    method: str,
    notes: list[str] | None = None,
) -> dict[str, object]:
    table = component_validation_table(components, anchors, diagnostics)
    report = {
        "status": "ok" if (table["status"] == "ok").all() else "check",
        "method": method,
        "component_order": COMPONENT_ORDER,
        "notes": notes or [],
        "diagnostics": table.to_dict(orient="records"),
    }
    Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def allocation_error_proxy(
    benchmarked_components: pd.DataFrame,
    raw_monthly_indicators: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compare benchmarked monthly allocation to raw indicator allocation shares.

    This is not a true leave-one-quarter-out validation, but it is a useful
    pre-LOQO diagnostic: it reports how much benchmarking changes the
    within-quarter month shares implied by raw indicators.
    """
    rows: list[dict[str, object]] = []
    for component in COMPONENT_ORDER:
        if component not in benchmarked_components.columns or component not in raw_monthly_indicators:
            continue
        raw = raw_monthly_indicators[component].dropna().copy()
        if raw.empty:
            continue
        raw.index = pd.to_datetime(raw.index).to_period("M").to_timestamp("M")
        bench = benchmarked_components[component].dropna().copy()
        aligned = pd.concat([bench.rename("bench"), raw.rename("raw")], axis=1, sort=False).dropna()
        if aligned.empty:
            continue
        quarters = to_quarter_end(aligned.index)
        for quarter, group in aligned.groupby(quarters):
            raw_sum = group["raw"].sum()
            bench_sum = group["bench"].sum()
            if raw_sum == 0 or bench_sum == 0:
                continue
            raw_share = group["raw"] / raw_sum
            bench_share = group["bench"] / bench_sum
            share_error = bench_share - raw_share
            rows.append(
                {
                    "component": component,
                    "quarter": pd.Timestamp(quarter).date().isoformat(),
                    "months": int(len(group)),
                    "mean_abs_share_error": float(share_error.abs().mean()),
                    "max_abs_share_error": float(share_error.abs().max()),
                }
            )
    return pd.DataFrame(rows)


def loqo_month_share_diagnostics(
    benchmarked_components: pd.DataFrame,
    raw_monthly_indicators: dict[str, pd.Series],
) -> pd.DataFrame:
    """Leave-one-quarter-out diagnostic for within-quarter month shares.

    For each component, raw indicator month shares are used to estimate average
    month-of-quarter shares excluding the held-out quarter. Those shares are
    compared with the full benchmarked component's within-quarter shares.
    """
    rows: list[dict[str, object]] = []
    for component in COMPONENT_ORDER:
        if component not in benchmarked_components.columns or component not in raw_monthly_indicators:
            continue
        raw = raw_monthly_indicators[component].dropna().copy()
        raw.index = pd.to_datetime(raw.index).to_period("M").to_timestamp("M")
        bench = benchmarked_components[component].dropna().copy()
        aligned = pd.concat([bench.rename("bench"), raw.rename("raw")], axis=1, sort=False).dropna()
        if aligned.empty:
            continue
        aligned["quarter"] = to_quarter_end(aligned.index)
        aligned["month_in_quarter"] = ((aligned.index.month - 1) % 3) + 1

        share_rows: list[pd.DataFrame] = []
        for quarter, group in aligned.groupby("quarter"):
            if len(group) < 2:
                continue
            raw_sum = group["raw"].sum()
            bench_sum = group["bench"].sum()
            if raw_sum == 0 or bench_sum == 0:
                continue
            tmp = group[["quarter", "month_in_quarter"]].copy()
            tmp["raw_share"] = group["raw"] / raw_sum
            tmp["bench_share"] = group["bench"] / bench_sum
            share_rows.append(tmp)
        if not share_rows:
            continue
        shares = pd.concat(share_rows, axis=0, sort=False)
        quarters = sorted(shares["quarter"].unique())
        for quarter in quarters:
            train = shares[shares["quarter"] != quarter]
            test = shares[shares["quarter"] == quarter]
            if train.empty or test.empty:
                continue
            avg = train.groupby("month_in_quarter")["raw_share"].mean()
            pred = test["month_in_quarter"].map(avg)
            if pred.isna().any():
                continue
            pred_sum = pred.sum()
            if pred_sum == 0:
                continue
            pred = pred / pred_sum
            error = test["bench_share"].to_numpy(dtype=float) - pred.to_numpy(dtype=float)
            rows.append(
                {
                    "component": component,
                    "quarter": pd.Timestamp(quarter).date().isoformat(),
                    "months": int(len(test)),
                    "mean_abs_share_error": float(pd.Series(error).abs().mean()),
                    "max_abs_share_error": float(pd.Series(error).abs().max()),
                }
            )
    return pd.DataFrame(rows)
