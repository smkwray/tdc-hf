from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .benchmarking import additive_denton, validate_quarterly_identity
from .bootstrap import bootstrap_lp_iv
from .calendar import to_month_end
from .first_stage import run_first_stage
from .indicators import read_wide_time_series_csv
from .lp import run_lp_iv, run_lp_iv_placebo
from .upstream import resolve_repos


@dataclass(frozen=True)
class AnchorVariantSpec:
    key: str
    anchor_column: str
    indicator: str
    role: str
    note: str


DEFAULT_ANCHOR_VARIANTS = [
    AnchorVariantSpec(
        key="tier1_bank_only",
        anchor_column="tdc_tier1_fed_corrected_bank_only_ru_flow",
        indicator="ru_component_sum",
        role="current_ru_reference",
        note="Current Tier 1 reserve-user-side bank-only quarterly estimate.",
    ),
    AnchorVariantSpec(
        key="du_gr_narrow",
        anchor_column="tdc_du_fiscal_flow_first_pass_narrow",
        indicator="dts_core_less_tax",
        role="du_g_minus_r_proxy",
        note="DU G-minus-R first-pass narrow quarterly estimate timed with core-payment withdrawals less tax deposits.",
    ),
    AnchorVariantSpec(
        key="du_gr_broad",
        anchor_column="tdc_du_fiscal_flow_first_pass_broad",
        indicator="dts_net_withdrawals",
        role="du_g_minus_r_proxy",
        note="DU G-minus-R first-pass broad quarterly estimate timed with total DTS withdrawals less deposits.",
    ),
    AnchorVariantSpec(
        key="du_domestic_nonfinancial",
        anchor_column="tdc_du_selected_domestic_nonfinancial_proxy",
        indicator="dts_core_less_tax",
        role="du_nonfinancial_proxy",
        note="Selected domestic nonfinancial DU proxy, using core fiscal timing.",
    ),
    AnchorVariantSpec(
        key="du_residual_bank_only",
        anchor_column="tdc_du_residual_proxy_bank_only_ru",
        indicator="ru_component_sum",
        role="du_residual_proxy",
        note="DU residual proxy using bank-only RU perimeter timing.",
    ),
    AnchorVariantSpec(
        key="du_residual_np_cu",
        anchor_column="tdc_du_residual_proxy_np_cu_ru",
        indicator="ru_component_sum",
        role="du_residual_proxy",
        note="DU residual proxy using natural-person credit-union-inclusive RU perimeter timing.",
    ),
]


def _default_tdcest_estimates_path(config_path: str | Path = "config/upstream_sources.yml") -> Path:
    repos = resolve_repos(config_path)
    return repos["tdcest"].preferred_outputs["estimates"]


def _quarterly_anchor(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"Missing TDC estimate column: {column}")
    out = pd.to_numeric(df[column], errors="coerce").dropna()
    out.index = pd.to_datetime(out.index).to_period("Q").to_timestamp("Q")
    out = out.groupby(level=0).sum(min_count=1).sort_index()
    out.name = "tdc_monthly"
    return out


def _equal_months_for_total(anchor: pd.Series) -> pd.Series:
    rows: list[tuple[pd.Timestamp, float]] = []
    for quarter_end, value in anchor.dropna().items():
        q = pd.Timestamp(quarter_end).to_period("Q")
        for month in pd.period_range(q.start_time, q.end_time, freq="M"):
            rows.append((month.to_timestamp("M"), float(value) / 3.0))
    return pd.Series(
        [value for _, value in rows],
        index=pd.DatetimeIndex([date for date, _ in rows]),
        name="tdc_monthly",
        dtype="float64",
    ).sort_index()


def _monthly_indicator(frame: pd.DataFrame, name: str, anchor: pd.Series) -> pd.Series:
    index = to_month_end(frame.index)
    df = frame.copy()
    df.index = index
    if name == "dts_net_withdrawals" and "dts_net_withdrawals" in df.columns:
        return pd.to_numeric(df["dts_net_withdrawals"], errors="coerce").rename("tdc_monthly")
    if name == "dts_core_less_tax":
        missing = [col for col in ["dts_core_payment_withdrawals", "dts_tax_deposits"] if col not in df.columns]
        if missing:
            raise KeyError(f"Cannot build `dts_core_less_tax`; missing {missing}")
        core = pd.to_numeric(df.get("dts_core_payment_withdrawals"), errors="coerce")
        tax = pd.to_numeric(df.get("dts_tax_deposits"), errors="coerce")
        return (core - tax).rename("tdc_monthly")
    if name == "ru_component_sum":
        cols = ["fed_tsy", "banks_tsy", "minus_toc", "fed_remit_positive"]
        present = [col for col in cols if col in df.columns]
        if present:
            return df[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1).rename("tdc_monthly")
    if name == "equal_months":
        return _equal_months_for_total(anchor)
    raise KeyError(f"Cannot build monthly indicator `{name}` from available columns")


def _benchmark_total_proxy(indicator: pd.Series, anchor: pd.Series, *, method: str = "denton") -> tuple[pd.Series, float, int]:
    fallback = _equal_months_for_total(anchor)
    observed = indicator.copy()
    observed.index = to_month_end(observed.index)
    observed = pd.to_numeric(observed, errors="coerce").dropna().sort_index()
    monthly = fallback.copy()
    overlap = monthly.index.intersection(observed.index)
    if len(overlap) > 0:
        monthly.loc[overlap] = observed.loc[overlap]
    if method != "denton":
        raise ValueError("Anchor-variant total proxies currently support additive Denton only")
    benchmarked = additive_denton(monthly, anchor, component="tdc_monthly")
    diag = validate_quarterly_identity(benchmarked, anchor, component="tdc_monthly", method="additive_denton")
    return benchmarked.rename("tdc_monthly"), diag.max_abs_quarterly_error, int(len(overlap))


def _parse_variant_specs(raw: Sequence[str] | None) -> list[AnchorVariantSpec]:
    if not raw:
        return DEFAULT_ANCHOR_VARIANTS
    specs: list[AnchorVariantSpec] = []
    for item in raw:
        parts = [part.strip() for part in item.split(":")]
        if len(parts) < 3:
            raise ValueError("Variant specs must be `key:anchor_column:indicator[:role[:note]]`")
        specs.append(
            AnchorVariantSpec(
                key=parts[0],
                anchor_column=parts[1],
                indicator=parts[2],
                role=parts[3] if len(parts) > 3 else "custom",
                note=parts[4] if len(parts) > 4 else "",
            )
        )
    return specs


def run_anchor_variant_robustness(
    *,
    tdcest_estimates: pd.DataFrame,
    monthly_indicators: pd.DataFrame,
    base_panel: pd.DataFrame,
    variant_specs: Sequence[AnchorVariantSpec],
    treatment: str,
    instruments: Sequence[str],
    outcomes: Sequence[str],
    controls: Sequence[str],
    horizons: Sequence[int],
    bootstrap_outcomes: Sequence[str],
    bootstrap_replications: int,
    block_length: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    proxies: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, object]] = []
    first_stage_frames: list[pd.DataFrame] = []
    lp_frames: list[pd.DataFrame] = []
    placebo_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []

    for offset, spec in enumerate(variant_specs):
        anchor = _quarterly_anchor(tdcest_estimates, spec.anchor_column)
        indicator = _monthly_indicator(monthly_indicators, spec.indicator, anchor)
        proxy, max_error, observed_months = _benchmark_total_proxy(indicator, anchor)
        proxies[spec.key] = proxy.rename(spec.key)

        panel = base_panel.copy()
        panel.index = to_month_end(panel.index)
        panel[treatment] = proxy.reindex(panel.index)

        first_stage = run_first_stage(panel, treatment=treatment, instruments=instruments, controls=controls)
        first_stage.insert(0, "anchor_variant", spec.key)
        first_stage.insert(1, "anchor_column", spec.anchor_column)
        first_stage_frames.append(first_stage)

        lp = run_lp_iv(
            panel,
            treatment_col=treatment,
            instrument_cols=instruments,
            outcome_cols=outcomes,
            controls=controls,
            horizons=horizons,
        )
        lp.insert(0, "anchor_variant", spec.key)
        lp.insert(1, "anchor_column", spec.anchor_column)
        lp_frames.append(lp)

        placebo = run_lp_iv_placebo(
            panel,
            treatment_col=treatment,
            instrument_cols=instruments,
            outcome_cols=outcomes,
            controls=controls,
            placebo_horizons=[h for h in horizons if h > 0],
        )
        placebo.insert(0, "anchor_variant", spec.key)
        placebo.insert(1, "anchor_column", spec.anchor_column)
        placebo_frames.append(placebo)

        boot = bootstrap_lp_iv(
            panel,
            treatment=treatment,
            instruments=instruments,
            outcomes=bootstrap_outcomes,
            controls=controls,
            horizons=horizons,
            replications=bootstrap_replications,
            block_length=block_length,
            seed=seed + offset,
        )
        boot.insert(0, "anchor_variant", spec.key)
        boot.insert(1, "anchor_column", spec.anchor_column)
        bootstrap_frames.append(boot)

        nonmissing = proxy.dropna()
        summary_rows.append(
            {
                "anchor_variant": spec.key,
                "anchor_column": spec.anchor_column,
                "indicator": spec.indicator,
                "role": spec.role,
                "note": spec.note,
                "quarterly_anchor_obs": int(anchor.notna().sum()),
                "monthly_proxy_obs": int(nonmissing.size),
                "observed_indicator_months_used": int(observed_months),
                "first_month": nonmissing.index.min().date().isoformat() if not nonmissing.empty else "",
                "last_month": nonmissing.index.max().date().isoformat() if not nonmissing.empty else "",
                "max_abs_quarterly_error": float(max_error),
            }
        )

    proxy_frame = pd.concat(proxies.values(), axis=1, sort=False).sort_index()
    summary = pd.DataFrame(summary_rows)
    first_stage_all = pd.concat(first_stage_frames, ignore_index=True)
    lp_all = pd.concat(lp_frames, ignore_index=True)
    placebo_all = pd.concat(placebo_frames, ignore_index=True)
    bootstrap_all = pd.concat(bootstrap_frames, ignore_index=True)
    dashboard = _dashboard(lp_all, bootstrap_all, placebo_all, first_stage_all, horizons=[12])
    short_dashboard = _dashboard(lp_all, bootstrap_all, placebo_all, first_stage_all, horizons=[0, 1, 2, 3, 4])

    return {
        "monthly_proxies": proxy_frame,
        "summary": summary,
        "first_stage": first_stage_all,
        "lp_iv": lp_all,
        "placebo": placebo_all,
        "bootstrap": bootstrap_all,
        "dashboard": dashboard,
        "short_dashboard": short_dashboard,
    }


def _dashboard(
    lp: pd.DataFrame,
    bootstrap: pd.DataFrame,
    placebo: pd.DataFrame,
    first_stage: pd.DataFrame,
    *,
    horizons: Sequence[int],
) -> pd.DataFrame:
    selected = lp.loc[lp["horizon"].isin(horizons)].copy()
    selected = selected[
        [
            "anchor_variant",
            "anchor_column",
            "outcome",
            "horizon",
            "same_unit_beta",
            "same_unit_lower95",
            "same_unit_upper95",
            "n",
            "first_stage_f",
        ]
    ]
    boot_selected = bootstrap.loc[bootstrap["horizon"].isin(horizons)].copy()
    boot_cols = [
        "anchor_variant",
        "outcome",
        "horizon",
        "same_unit_bootstrap_lower95",
        "same_unit_bootstrap_upper95",
        "draws",
        "failures",
    ]
    if not boot_selected.empty:
        selected = selected.merge(boot_selected[boot_cols], on=["anchor_variant", "outcome", "horizon"], how="left")
    else:
        for column in boot_cols:
            if column not in {"anchor_variant", "outcome", "horizon"}:
                selected[column] = pd.NA
    placebo_counts = (
        placebo.groupby(["anchor_variant", "outcome"], dropna=False)
        .agg(placebo_rows=("placebo_sig_95", "size"), placebo_sig_rows=("placebo_sig_95", "sum"))
        .reset_index()
    )
    selected = selected.merge(placebo_counts, on=["anchor_variant", "outcome"], how="left")
    fs = first_stage.loc[first_stage["excluded_instrument_f"].notna(), ["anchor_variant", "excluded_instrument_f"]].rename(
        columns={"excluded_instrument_f": "full_sample_first_stage_f"}
    )
    selected = selected.merge(fs, on="anchor_variant", how="left")
    selected["hac_sig_95"] = (selected["same_unit_lower95"] > 0) | (selected["same_unit_upper95"] < 0)
    selected["bootstrap_sig_95"] = (selected["same_unit_bootstrap_lower95"] > 0) | (
        selected["same_unit_bootstrap_upper95"] < 0
    )
    selected["placebo_clean"] = selected["placebo_sig_rows"].fillna(0).eq(0)
    return selected.sort_values(["horizon", "anchor_variant", "outcome"]).reset_index(drop=True)


def _write_markdown(outputs: Mapping[str, pd.DataFrame], path: Path) -> None:
    summary = outputs["summary"]
    dashboard = outputs["dashboard"]
    lines = [
        "# Anchor Variant Robustness",
        "",
        "This artifact reruns the long-sample TGA LP-IV design after replacing the monthly TDC treatment with alternative quarterly TDC anchors benchmarked to monthly timing indicators.",
        "",
        "## Variants",
        "",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"- `{row['anchor_variant']}`: `{row['anchor_column']}`; indicator `{row['indicator']}`; "
            f"monthly obs `{row['monthly_proxy_obs']}`; observed indicator months `{row['observed_indicator_months_used']}`."
        )
    lines.extend(["", "## Short-Run Dashboard", ""])
    short_dashboard = outputs.get("short_dashboard", dashboard)
    short_key = short_dashboard.loc[
        short_dashboard["outcome"].isin(["deposits", "bank_credit", "onrrp", "total_mmf", "commercial_industrial_loans"])
    ]
    for row in short_key.to_dict("records"):
        lines.append(
            f"- h=`{int(row['horizon'])}` `{row['anchor_variant']}` / `{row['outcome']}`: "
            f"same-unit `{row['same_unit_beta']:.3g}` "
            f"[`{row['same_unit_lower95']:.3g}`, `{row['same_unit_upper95']:.3g}`], "
            f"F `{row['first_stage_f']:.3g}`, placebo clean `{bool(row['placebo_clean'])}`."
        )
    lines.extend(["", "## H=12 Dashboard", ""])
    key = dashboard.loc[dashboard["outcome"].isin(["deposits", "bank_credit", "onrrp", "total_mmf", "commercial_industrial_loans"])]
    for row in key.to_dict("records"):
        lines.append(
            f"- `{row['anchor_variant']}` / `{row['outcome']}`: same-unit `{row['same_unit_beta']:.3g}` "
            f"[`{row['same_unit_lower95']:.3g}`, `{row['same_unit_upper95']:.3g}`], "
            f"F `{row['first_stage_f']:.3g}`, placebo clean `{bool(row['placebo_clean'])}`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_anchor_variant_robustness_csv(
    *,
    tdcest_estimates_csv: str | Path | None = None,
    monthly_indicators_csv: str | Path,
    base_panel_csv: str | Path,
    out_dir: str | Path,
    variant_specs: Sequence[str] | None = None,
    treatment: str = "tdc_monthly",
    instruments: Sequence[str] = ("tga_long_surprise_z",),
    outcomes: Sequence[str] = ("deposits", "bank_credit", "onrrp", "total_mmf", "commercial_industrial_loans"),
    controls: Sequence[str] = (),
    horizons: Sequence[int] = (0, 1, 2, 3, 6, 12),
    bootstrap_outcomes: Sequence[str] = ("deposits", "bank_credit", "onrrp", "total_mmf", "commercial_industrial_loans"),
    bootstrap_replications: int = 100,
    block_length: int = 6,
    seed: int = 20260502,
) -> dict[str, object]:
    estimates_path = Path(tdcest_estimates_csv) if tdcest_estimates_csv is not None else _default_tdcest_estimates_path()
    estimates = read_wide_time_series_csv(estimates_path)
    monthly_indicators = read_wide_time_series_csv(monthly_indicators_csv)
    base_panel = read_wide_time_series_csv(base_panel_csv)
    specs = _parse_variant_specs(variant_specs)

    outputs = run_anchor_variant_robustness(
        tdcest_estimates=estimates,
        monthly_indicators=monthly_indicators,
        base_panel=base_panel,
        variant_specs=specs,
        treatment=treatment,
        instruments=instruments,
        outcomes=outcomes,
        controls=controls,
        horizons=horizons,
        bootstrap_outcomes=bootstrap_outcomes,
        bootstrap_replications=bootstrap_replications,
        block_length=block_length,
        seed=seed,
    )

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "monthly_proxies": root / "anchor_variant_monthly_proxies.csv",
        "summary": root / "anchor_variant_summary.csv",
        "first_stage": root / "anchor_variant_first_stage.csv",
        "lp_iv": root / "anchor_variant_lp_iv.csv",
        "placebo": root / "anchor_variant_placebo.csv",
        "bootstrap": root / "anchor_variant_bootstrap.csv",
        "dashboard": root / "anchor_variant_h12_dashboard.csv",
        "short_dashboard": root / "anchor_variant_short_dashboard.csv",
        "markdown": root / "anchor_variant_robustness.md",
    }
    outputs["monthly_proxies"].to_csv(paths["monthly_proxies"], index_label="date")
    for key in ["summary", "first_stage", "lp_iv", "placebo", "bootstrap", "dashboard", "short_dashboard"]:
        outputs[key].to_csv(paths[key], index=False)
    _write_markdown(outputs, paths["markdown"])

    return {
        "status": "ok",
        "out_dir": str(root),
        **{key: str(value) for key, value in paths.items()},
        "variants": [spec.key for spec in specs],
        "dashboard_rows": int(len(outputs["dashboard"])),
        "short_dashboard_rows": int(len(outputs["short_dashboard"])),
    }
