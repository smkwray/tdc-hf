from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .indicators import read_wide_time_series_csv


def summarize_proxy_run(run_dir: str | Path, *, out_json: str | Path | None = None, out_md: str | Path | None = None) -> dict[str, object]:
    root = Path(run_dir)
    validation_path = root / "tdc_validation_report.json"
    proxy_path = root / "tdc_monthly_proxy.csv"
    coverage_path = root / "tdc_indicator_source_coverage.csv"
    raw_fit_path = root / "tdc_raw_indicator_quarterly_fit.csv"
    envelope_path = root / "tdc_monthly_method_envelope.csv"

    summary: dict[str, object] = {"run_dir": str(root), "files": {}}
    for path in [validation_path, proxy_path, coverage_path, raw_fit_path, envelope_path]:
        summary["files"][path.name] = path.exists()  # type: ignore[index]

    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        summary["validation_status"] = validation.get("status")
        summary["method"] = validation.get("method")
        summary["components"] = validation.get("diagnostics", [])

    if proxy_path.exists():
        proxy = pd.read_csv(proxy_path, parse_dates=["date"])
        summary["proxy_rows"] = int(len(proxy))
        summary["proxy_first_month"] = proxy["date"].min().date().isoformat() if not proxy.empty else ""
        summary["proxy_last_month"] = proxy["date"].max().date().isoformat() if not proxy.empty else ""
        summary["proxy_nonnull"] = int(proxy["tdc_monthly"].notna().sum()) if "tdc_monthly" in proxy.columns else 0

    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        summary["source_coverage"] = coverage.to_dict(orient="records")

    if raw_fit_path.exists():
        raw_fit = pd.read_csv(raw_fit_path)
        summary["raw_indicator_fit"] = raw_fit.to_dict(orient="records")

    if out_json is not None:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if out_md is not None:
        path = Path(out_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# TDC-HF Run Summary",
            "",
            f"- Run dir: `{root}`",
            f"- Status: `{summary.get('validation_status', 'unknown')}`",
            f"- Method: `{summary.get('method', 'unknown')}`",
            f"- Rows: `{summary.get('proxy_rows', 0)}`",
            f"- Window: `{summary.get('proxy_first_month', '')}` to `{summary.get('proxy_last_month', '')}`",
            "",
            "## Files",
            "",
        ]
        for file_name, exists in summary["files"].items():  # type: ignore[union-attr]
            lines.append(f"- `{file_name}`: {'yes' if exists else 'no'}")
        if "source_coverage" in summary:
            lines.extend(["", "## Source Coverage", ""])
            for row in summary["source_coverage"]:  # type: ignore[union-attr]
                lines.append(
                    f"- `{row.get('component')}`: observed `{row.get('observed_months', 0)}`, fallback `{row.get('fallback_months', 0)}`"
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary


def summarize_proxy_run_cli(run_dir: str | Path, *, out_json: str | Path, out_md: str | Path | None = None) -> dict[str, object]:
    summary = summarize_proxy_run(run_dir, out_json=out_json, out_md=out_md)
    return {"status": "ok", "out_json": str(out_json), "out_md": str(out_md) if out_md else "", "run_dir": str(run_dir), "proxy_rows": summary.get("proxy_rows", 0)}


def summarize_panel_csv(data_csv: str | Path, *, out_csv: str | Path) -> dict[str, object]:
    panel = read_wide_time_series_csv(data_csv)
    rows: list[dict[str, object]] = []
    for column in panel.columns:
        nonnull = panel[column].dropna()
        rows.append(
            {
                "column": column,
                "nonnull": int(len(nonnull)),
                "first_date": nonnull.index.min().date().isoformat() if not nonnull.empty else "",
                "last_date": nonnull.index.max().date().isoformat() if not nonnull.empty else "",
                "mean": float(nonnull.mean()) if not nonnull.empty else float("nan"),
                "std": float(nonnull.std()) if len(nonnull) > 1 else float("nan"),
            }
        )
    summary = pd.DataFrame(rows)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return {"status": "ok", "out": str(path), "rows": int(len(summary)), "panel_rows": int(len(panel))}


def summarize_estimates_bundle(
    lp_csv: str | Path,
    *,
    first_stage_csv: str | Path | None = None,
    out_dir: str | Path,
) -> dict[str, object]:
    lp = pd.read_csv(lp_csv)
    required = {"outcome", "horizon", "beta", "se", "lower95", "upper95", "n"}
    missing = required.difference(lp.columns)
    if missing:
        raise KeyError(f"LP result is missing columns: {sorted(missing)}")

    rows: list[pd.Series] = []
    for _, group in lp.sort_values(["outcome", "horizon"]).groupby("outcome", sort=True):
        ranked = group.assign(abs_beta=group["beta"].abs()).sort_values(["abs_beta", "horizon"], ascending=[False, True])
        rows.append(ranked.iloc[0].drop(labels=["abs_beta"]))
    peak = pd.DataFrame(rows)
    peak["significant_95"] = (peak["lower95"] > 0) | (peak["upper95"] < 0)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    peak_path = out_root / "lp_peak_responses.csv"
    peak.to_csv(peak_path, index=False)

    first_stage_path = None
    first_stage_summary: dict[str, object] = {}
    if first_stage_csv is not None:
        first = pd.read_csv(first_stage_csv)
        first_stage_path = out_root / "first_stage_summary.csv"
        first.to_csv(first_stage_path, index=False)
        joint = first.loc[first["excluded_instrument_f"].notna()] if "excluded_instrument_f" in first.columns else pd.DataFrame()
        if not joint.empty:
            first_stage_summary = {
                "excluded_instrument_f": float(joint.iloc[0]["excluded_instrument_f"]),
                "excluded_instrument_pvalue": float(joint.iloc[0]["excluded_instrument_pvalue"]),
                "n": int(joint.iloc[0]["n"]),
            }

    md_path = out_root / "estimate_summary.md"
    lines = [
        "# Estimate Summary",
        "",
        f"- LP rows: `{len(lp)}`",
        f"- Outcomes: `{', '.join(str(value) for value in sorted(lp['outcome'].dropna().unique()))}`",
    ]
    if first_stage_summary:
        lines.extend(
            [
                f"- First-stage excluded-instrument F: `{first_stage_summary['excluded_instrument_f']:.3f}`",
                f"- First-stage p-value: `{first_stage_summary['excluded_instrument_pvalue']:.3g}`",
                f"- First-stage n: `{first_stage_summary['n']}`",
            ]
        )
    lines.extend(["", "## Peak Absolute Responses", ""])
    for row in peak.sort_values("outcome").to_dict(orient="records"):
        stars = "*" if row.get("significant_95") else ""
        lines.append(
            f"- `{row['outcome']}` h={int(row['horizon'])}: beta `{float(row['beta']):.6g}` "
            f"[`{float(row['lower95']):.6g}`, `{float(row['upper95']):.6g}`], n `{int(row['n'])}`{stars}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": "ok",
        "out_dir": str(out_root),
        "lp_peak": str(peak_path),
        "first_stage_summary": str(first_stage_path) if first_stage_path else "",
        "markdown": str(md_path),
        "outcomes": int(peak["outcome"].nunique()),
        "lp_rows": int(len(lp)),
    }


def compare_monthly_proxies(
    left_csv: str | Path,
    right_csv: str | Path,
    *,
    out_dir: str | Path,
    left_label: str = "left",
    right_label: str = "right",
    column: str = "tdc_monthly",
) -> dict[str, object]:
    left = read_wide_time_series_csv(left_csv)
    right = read_wide_time_series_csv(right_csv)
    if column not in left.columns:
        raise KeyError(f"Missing {left_label} column: {column}")
    if column not in right.columns:
        raise KeyError(f"Missing {right_label} column: {column}")

    combined = pd.concat(
        [
            left[[column]].rename(columns={column: left_label}),
            right[[column]].rename(columns={column: right_label}),
        ],
        axis=1,
        sort=False,
    ).dropna()
    combined["difference"] = combined[left_label] - combined[right_label]
    combined["abs_difference"] = combined["difference"].abs()
    combined["quarter"] = combined.index.to_period("Q").astype(str)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    monthly_path = out_root / "monthly_proxy_comparison.csv"
    combined.drop(columns=["quarter"]).to_csv(monthly_path, index_label="date")

    quarterly = combined.groupby("quarter", as_index=False).agg(
        left_sum=(left_label, "sum"),
        right_sum=(right_label, "sum"),
        difference_sum=("difference", "sum"),
        mean_abs_difference=("abs_difference", "mean"),
        max_abs_difference=("abs_difference", "max"),
    )
    quarterly_path = out_root / "quarterly_proxy_comparison.csv"
    quarterly.to_csv(quarterly_path, index=False)

    summary = {
        "status": "ok",
        "out_dir": str(out_root),
        "monthly": str(monthly_path),
        "quarterly": str(quarterly_path),
        "rows": int(len(combined)),
        "mean_abs_difference": float(combined["abs_difference"].mean()) if not combined.empty else float("nan"),
        "max_abs_difference": float(combined["abs_difference"].max()) if not combined.empty else float("nan"),
        "quarterly_identity_max_abs_difference": float(quarterly["difference_sum"].abs().max()) if not quarterly.empty else float("nan"),
    }

    md_path = out_root / "proxy_comparison.md"
    lines = [
        "# Monthly Proxy Method Comparison",
        "",
        f"- Left: `{left_label}`",
        f"- Right: `{right_label}`",
        f"- Overlap months: `{summary['rows']}`",
        f"- Mean absolute monthly difference: `{summary['mean_abs_difference']:.6g}`",
        f"- Max absolute monthly difference: `{summary['max_abs_difference']:.6g}`",
        f"- Max absolute quarterly sum difference: `{summary['quarterly_identity_max_abs_difference']:.6g}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["markdown"] = str(md_path)
    return summary


def _fmt_float(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    numeric = float(value)
    if numeric == 0:
        return "0"
    if abs(numeric) < 0.001 or abs(numeric) >= 10000:
        return f"{numeric:.{digits}e}"
    return f"{numeric:.{digits}f}"


def _stars(lower: object, upper: object) -> str:
    if pd.isna(lower) or pd.isna(upper):
        return ""
    if float(lower) > 0 or float(upper) < 0:
        return "*"
    return ""


def export_publication_tables(
    *,
    lp_csv: str | Path,
    out_dir: str | Path,
    bootstrap_csv: str | Path | None = None,
    first_stage_csv: str | Path | None = None,
    validation_csv: str | Path | None = None,
    proxy_comparison_csv: str | Path | None = None,
) -> dict[str, object]:
    lp = pd.read_csv(lp_csv)
    required = {"outcome", "horizon", "beta", "lower95", "upper95", "n"}
    missing = required.difference(lp.columns)
    if missing:
        raise KeyError(f"LP result is missing columns: {sorted(missing)}")

    table = lp.copy()
    table = table.rename(columns={"lower95": "hac_lower95", "upper95": "hac_upper95", "se": "hac_se"})
    if bootstrap_csv is not None and Path(bootstrap_csv).exists():
        boot = pd.read_csv(bootstrap_csv)
        boot_columns = [
            column
            for column in [
                "outcome",
                "horizon",
                "bootstrap_se",
                "bootstrap_lower95",
                "bootstrap_upper95",
                "same_unit_bootstrap_lower95",
                "same_unit_bootstrap_upper95",
                "draws",
                "block_length",
            ]
            if column in boot.columns
        ]
        table = table.merge(
            boot[boot_columns],
            on=["outcome", "horizon"],
            how="left",
        )
    else:
        table["bootstrap_se"] = pd.NA
        table["bootstrap_lower95"] = pd.NA
        table["bootstrap_upper95"] = pd.NA
        table["draws"] = pd.NA
        table["block_length"] = pd.NA

    keep = [
        "outcome",
        "horizon",
        "beta",
        "same_unit_beta",
        "same_unit_lower95",
        "same_unit_upper95",
        "same_unit_interpretation",
        "hac_se",
        "hac_lower95",
        "hac_upper95",
        "bootstrap_se",
        "bootstrap_lower95",
        "bootstrap_upper95",
        "same_unit_bootstrap_lower95",
        "same_unit_bootstrap_upper95",
        "n",
        "draws",
        "block_length",
    ]
    for column in keep:
        if column not in table.columns:
            table[column] = pd.NA
    table = table[keep].sort_values(["outcome", "horizon"]).reset_index(drop=True)
    table["hac_sig_95"] = [_stars(row.hac_lower95, row.hac_upper95) for row in table.itertuples()]
    table["bootstrap_sig_95"] = [_stars(row.bootstrap_lower95, row.bootstrap_upper95) for row in table.itertuples()]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    lp_table_path = out_root / "table_lp_iv_with_bootstrap.csv"
    table.to_csv(lp_table_path, index=False)

    pretty = table.copy()
    for column in [
        "beta",
        "same_unit_beta",
        "same_unit_lower95",
        "same_unit_upper95",
        "hac_se",
        "hac_lower95",
        "hac_upper95",
        "bootstrap_se",
        "bootstrap_lower95",
        "bootstrap_upper95",
        "same_unit_bootstrap_lower95",
        "same_unit_bootstrap_upper95",
    ]:
        pretty[column] = pretty[column].map(_fmt_float)
    pretty_path = out_root / "table_lp_iv_with_bootstrap_pretty.csv"
    pretty.to_csv(pretty_path, index=False)

    peak_rows = []
    for _, group in table.groupby("outcome", sort=True):
        ranked = group.assign(abs_beta=group["beta"].abs()).sort_values(["abs_beta", "horizon"], ascending=[False, True])
        peak_rows.append(ranked.iloc[0].drop(labels=["abs_beta"]))
    peak = pd.DataFrame(peak_rows)
    peak_path = out_root / "table_peak_responses.csv"
    peak.to_csv(peak_path, index=False)

    first_stage_path = ""
    first_stage_md: list[str] = []
    if first_stage_csv is not None and Path(first_stage_csv).exists():
        first = pd.read_csv(first_stage_csv)
        first_stage_path = str(out_root / "table_first_stage.csv")
        first.to_csv(first_stage_path, index=False)
        joint = first.loc[first["excluded_instrument_f"].notna()] if "excluded_instrument_f" in first.columns else pd.DataFrame()
        if not joint.empty:
            row = joint.iloc[0]
            first_stage_md = [
                f"- First-stage F: `{_fmt_float(row['excluded_instrument_f'])}`",
                f"- First-stage p-value: `{_fmt_float(row['excluded_instrument_pvalue'])}`",
                f"- First-stage n: `{int(row['n'])}`",
            ]

    validation_path = ""
    validation_md: list[str] = []
    if validation_csv is not None and Path(validation_csv).exists():
        validation = pd.read_csv(validation_csv)
        compact = validation[
            [
                "component",
                "quarters_checked",
                "first_quarter",
                "last_quarter",
                "max_abs_quarterly_error",
                "method",
                "status",
            ]
        ].copy()
        validation_path = str(out_root / "table_proxy_validation.csv")
        compact.to_csv(validation_path, index=False)
        validation_md = [
            f"- Proxy validation components: `{compact['component'].nunique()}`",
            f"- Max quarterly identity error: `{_fmt_float(compact['max_abs_quarterly_error'].abs().max())}`",
        ]

    comparison_path = ""
    comparison_md: list[str] = []
    if proxy_comparison_csv is not None and Path(proxy_comparison_csv).exists():
        comparison = pd.read_csv(proxy_comparison_csv)
        summary = pd.DataFrame(
            [
                {
                    "quarters": int(len(comparison)),
                    "mean_abs_monthly_difference": float(comparison["mean_abs_difference"].mean()),
                    "max_abs_monthly_difference": float(comparison["max_abs_difference"].max()),
                    "max_abs_quarterly_sum_difference": float(comparison["difference_sum"].abs().max()),
                }
            ]
        )
        comparison_path = str(out_root / "table_method_comparison_summary.csv")
        summary.to_csv(comparison_path, index=False)
        row = summary.iloc[0]
        comparison_md = [
            f"- Method-comparison quarters: `{int(row['quarters'])}`",
            f"- Mean monthly method difference: `{_fmt_float(row['mean_abs_monthly_difference'])}`",
            f"- Max monthly method difference: `{_fmt_float(row['max_abs_monthly_difference'])}`",
            f"- Max quarterly method-sum difference: `{_fmt_float(row['max_abs_quarterly_sum_difference'])}`",
        ]

    md_path = out_root / "publication_tables.md"
    lines = [
        "# Publication Tables",
        "",
        f"- LP-IV rows: `{len(table)}`",
        f"- Outcomes: `{', '.join(str(value) for value in sorted(table['outcome'].dropna().unique()))}`",
        f"- Bootstrap rows with draws: `{int(table['draws'].notna().sum())}`",
    ]
    lines.extend(first_stage_md)
    lines.extend(validation_md)
    lines.extend(comparison_md)
    lines.extend(["", "## Peak Responses", ""])
    for row in peak.sort_values("outcome").to_dict(orient="records"):
        interval = (
            f"bootstrap [`{_fmt_float(row.get('bootstrap_lower95'))}`, `{_fmt_float(row.get('bootstrap_upper95'))}`]"
            if pd.notna(row.get("bootstrap_lower95"))
            else f"HAC [`{_fmt_float(row.get('hac_lower95'))}`, `{_fmt_float(row.get('hac_upper95'))}`]"
        )
        same_unit = ""
        if pd.notna(row.get("same_unit_beta")) and row.get("same_unit_interpretation") == "outcome dollars per 1 TDC dollar":
            same_unit = f", same-unit pass-through `{_fmt_float(row.get('same_unit_beta'))}`"
        lines.append(f"- `{row['outcome']}` h={int(row['horizon'])}: beta `{_fmt_float(row['beta'])}`{same_unit}, {interval}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    outputs = {
        "lp_table": str(lp_table_path),
        "pretty_lp_table": str(pretty_path),
        "peak_table": str(peak_path),
        "first_stage": first_stage_path,
        "proxy_validation": validation_path,
        "method_comparison": comparison_path,
        "markdown": str(md_path),
    }
    return {
        "status": "ok",
        "out_dir": str(out_root),
        "outputs": outputs,
        "lp_rows": int(len(table)),
        "peak_rows": int(len(peak)),
    }


def compare_iv_robustness_summaries(
    baseline_csv: str | Path,
    controlled_csv: str | Path,
    *,
    out_dir: str | Path,
) -> dict[str, object]:
    baseline = pd.read_csv(baseline_csv)
    controlled = pd.read_csv(controlled_csv)
    keys = ["iv_spec", "outcome"]
    merged = baseline.merge(
        controlled,
        on=keys,
        how="inner",
        suffixes=("_baseline", "_controlled"),
    )
    merged["delta_peak_beta"] = merged["peak_beta_controlled"] - merged["peak_beta_baseline"]
    merged["delta_first_stage_f"] = merged["first_stage_f_controlled"] - merged["first_stage_f_baseline"]
    merged["same_peak_horizon"] = merged["peak_horizon_baseline"] == merged["peak_horizon_controlled"]
    merged["same_hac_significance"] = merged["peak_hac_sig_95_baseline"] == merged["peak_hac_sig_95_controlled"]

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    comparison_path = root / "iv_robustness_control_comparison.csv"
    merged.to_csv(comparison_path, index=False)

    md_path = root / "iv_robustness_control_comparison.md"
    lines = [
        "# IV Robustness: Baseline vs Calendar Controls",
        "",
        f"- Matched rows: `{len(merged)}`",
        f"- Specs: `{', '.join(sorted(merged['iv_spec'].dropna().unique()))}`",
        "",
        "## Peak Response Changes",
        "",
    ]
    for row in merged.sort_values(["outcome", "iv_spec"]).to_dict(orient="records"):
        lines.append(
            f"- `{row['outcome']}` / `{row['iv_spec']}`: "
            f"baseline h={int(row['peak_horizon_baseline'])} beta `{float(row['peak_beta_baseline']):.6g}`, "
            f"controlled h={int(row['peak_horizon_controlled'])} beta `{float(row['peak_beta_controlled']):.6g}`, "
            f"delta `{float(row['delta_peak_beta']):.6g}`, "
            f"F delta `{float(row['delta_first_stage_f']):.6g}`"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "out_dir": str(root),
        "comparison": str(comparison_path),
        "markdown": str(md_path),
        "rows": int(len(merged)),
    }


def compare_lp_results(
    left_csv: str | Path,
    right_csv: str | Path,
    *,
    out_dir: str | Path,
    left_label: str = "left",
    right_label: str = "right",
) -> dict[str, object]:
    left = pd.read_csv(left_csv)
    right = pd.read_csv(right_csv)
    merged = left.merge(right, on=["outcome", "horizon"], how="inner", suffixes=(f"_{left_label}", f"_{right_label}"))
    merged["delta_beta"] = merged[f"beta_{left_label}"] - merged[f"beta_{right_label}"]
    merged["same_hac_significance"] = (
        (merged[f"lower95_{left_label}"] > 0) | (merged[f"upper95_{left_label}"] < 0)
    ) == ((merged[f"lower95_{right_label}"] > 0) | (merged[f"upper95_{right_label}"] < 0))

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    comparison_path = root / "lp_result_comparison.csv"
    merged.to_csv(comparison_path, index=False)

    peak_rows = []
    for _, group in merged.groupby("outcome", sort=True):
        ranked = group.assign(abs_delta=group["delta_beta"].abs()).sort_values(["abs_delta", "horizon"], ascending=[False, True])
        peak_rows.append(ranked.iloc[0].drop(labels=["abs_delta"]))
    peak = pd.DataFrame(peak_rows)
    peak_path = root / "lp_result_comparison_peak_deltas.csv"
    peak.to_csv(peak_path, index=False)

    md_path = root / "lp_result_comparison.md"
    lines = [
        "# LP-IV Result Comparison",
        "",
        f"- Left: `{left_label}`",
        f"- Right: `{right_label}`",
        f"- Matched rows: `{len(merged)}`",
        "",
        "## Largest Absolute Beta Differences by Outcome",
        "",
    ]
    for row in peak.sort_values("outcome").to_dict(orient="records"):
        lines.append(
            f"- `{row['outcome']}` h={int(row['horizon'])}: "
            f"{left_label} `{float(row[f'beta_{left_label}']):.6g}`, "
            f"{right_label} `{float(row[f'beta_{right_label}']):.6g}`, "
            f"delta `{float(row['delta_beta']):.6g}`"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "out_dir": str(root),
        "comparison": str(comparison_path),
        "peak_deltas": str(peak_path),
        "markdown": str(md_path),
        "rows": int(len(merged)),
        "outcomes": int(merged["outcome"].nunique()),
    }


def export_thesis_status_report(
    *,
    out_md: str | Path,
    publication_md: str | Path | None = None,
    iv_robustness_md: str | Path | None = None,
    fiscal_flow_iv_md: str | Path | None = None,
    category_flow_iv_md: str | Path | None = None,
    controlled_iv_md: str | Path | None = None,
    lp_comparison_md: str | Path | None = None,
    method_comparison_md: str | Path | None = None,
    identification_md: str | Path | None = None,
) -> dict[str, object]:
    sections: list[tuple[str, str | Path | None]] = [
        ("Publication Tables", publication_md),
        ("IV Robustness", iv_robustness_md),
        ("DTS Fiscal-Flow IV Robustness", fiscal_flow_iv_md),
        ("DTS Category-Flow IV Robustness", category_flow_iv_md),
        ("Calendar-Controlled IV Robustness", controlled_iv_md),
        ("Denton vs Residual-Spread LP-IV Comparison", lp_comparison_md),
        ("Proxy Method Comparison", method_comparison_md),
        ("Identification Risks", identification_md),
    ]
    lines = [
        "# TDC-HF Thesis Status Report",
        "",
        "This report is generated from the current Denton analysis artifacts.",
        "",
        "## Current Status",
        "",
        "- Headline temporal disaggregation: additive Denton.",
        "- Canonical deposit outcome: domestic non-large-time deposits.",
        "- Current working instrument: TGA rebuild surprise.",
        "- DTS net-withdrawal fiscal-flow surprise is available as a short-window robustness instrument.",
        "- DTS category-flow tax/payment surprises are available as source-backed robustness instruments.",
        "- Auction-size surprise remains weak as a standalone instrument.",
        "- Calendar/regime controls are now available as a robustness path.",
        "",
    ]
    included = 0
    for title, source in sections:
        if source is None or not Path(source).exists():
            continue
        text = Path(source).read_text(encoding="utf-8").strip()
        if not text:
            continue
        included += 1
        lines.extend([f"## {title}", "", text, ""])
    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {"status": "ok", "out_md": str(path), "sections": included}
