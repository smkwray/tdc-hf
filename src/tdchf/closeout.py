from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .indicators import read_wide_time_series_csv
from .lp import run_local_projections
from .units import add_same_unit_columns


CORE_OUTCOMES = [
    "deposits",
    "broad_deposits",
    "bank_credit",
    "commercial_industrial_loans",
    "reserves",
    "onrrp",
    "total_mmf",
]
CORE_HORIZONS = [0, 1, 2, 3, 4, 12]
DEFAULT_ANCHOR_CANDIDATES = (
    "tdc_base_bank_only_ru_flow",
    "tdc_tier1_fed_corrected_bank_only_ru_flow",
    "tdc_du_fiscal_flow_first_pass_narrow",
    "tdc_du_fiscal_flow_first_pass_broad",
    "tdc_du_selected_domestic_nonfinancial_proxy",
    "tdc_du_residual_proxy_bank_only_ru",
    "tdc_du_residual_proxy_np_cu_ru",
)


def _tdcest_default_path() -> Path:
    return Path.home() / "malus" / "proj" / "tdcest" / "data" / "processed" / "tdc_estimates.csv"


def _sig(row: pd.Series, *, low: str = "same_unit_lower95", high: str = "same_unit_upper95") -> bool:
    return pd.notna(row.get(low)) and pd.notna(row.get(high)) and (float(row[low]) > 0 or float(row[high]) < 0)


def _write_md_table(table: pd.DataFrame, path: Path, *, title: str, max_rows: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    show = table.head(max_rows)
    if show.empty:
        lines.append("_No rows._")
    else:
        cols = [str(col) for col in show.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in show.to_dict("records"):
            values = [str(row.get(col, "")).replace("|", "\\|") for col in cols]
            lines.append("| " + " | ".join(values) + " |")
        if len(table) > len(show):
            lines.append("")
            lines.append(f"_Showing first {len(show)} of {len(table)} rows._")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_anchor_contract_audit(
    *,
    monthly_proxy_csv: str | Path,
    tdcest_estimates_csv: str | Path | None = None,
    out_csv: str | Path,
    out_md: str | Path | None = None,
    proxy_column: str = "tdc_monthly",
    candidate_anchors: Sequence[str] | None = DEFAULT_ANCHOR_CANDIDATES,
) -> dict[str, object]:
    proxy = read_wide_time_series_csv(monthly_proxy_csv)
    if proxy_column not in proxy.columns:
        raise KeyError(f"Missing proxy column `{proxy_column}` in {monthly_proxy_csv}")
    quarterly_proxy = proxy[proxy_column].resample("QE").sum(min_count=1).rename("quarterly_proxy_sum")
    estimates = read_wide_time_series_csv(tdcest_estimates_csv or _tdcest_default_path())

    rows: list[dict[str, object]] = []
    anchors = list(candidate_anchors or DEFAULT_ANCHOR_CANDIDATES)
    for anchor in anchors:
        if anchor not in estimates.columns:
            rows.append({"candidate_anchor": anchor, "interpretation": "missing_from_tdcest_estimates"})
            continue
        frame = pd.concat([quarterly_proxy, estimates[anchor].rename("anchor")], axis=1).dropna()
        if frame.empty:
            rows.append({"candidate_anchor": anchor, "interpretation": "no_overlap"})
            continue
        diff = frame["quarterly_proxy_sum"] - frame["anchor"]
        latest = diff.dropna().iloc[-1] if not diff.dropna().empty else pd.NA
        max_abs = float(diff.abs().max())
        mean_abs = float(diff.abs().mean())
        near_exact = int(diff.abs().le(1e-6).sum())
        if max_abs <= 1e-6:
            interpretation = "near_exact_default_anchor_match"
        elif near_exact > 0:
            interpretation = "partial_match_only"
        else:
            interpretation = "not_default_anchor"
        rows.append(
            {
                "candidate_anchor": anchor,
                "n_quarters_compared": int(len(frame)),
                "max_abs_quarterly_diff": max_abs,
                "mean_abs_quarterly_diff": mean_abs,
                "near_exact_match_count": near_exact,
                "latest_quarter": frame.index.max().date().isoformat(),
                "latest_quarter_diff": float(latest) if pd.notna(latest) else pd.NA,
                "interpretation": interpretation,
            }
        )

    out = pd.DataFrame(rows).sort_values(["max_abs_quarterly_diff", "candidate_anchor"], na_position="last")
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    if out_md:
        _write_md_table(out, Path(out_md), title="Anchor Contract Audit")
    return {"status": "ok", "out": str(path), "out_md": str(out_md or ""), "rows": int(len(out))}


def export_noniv_tdc_lp_closeout(
    *,
    pretrend_panel_csv: str | Path,
    calendar_panel_csv: str | Path,
    out_csv: str | Path,
    out_md: str | Path | None = None,
    outcomes: Sequence[str] | None = CORE_OUTCOMES,
    horizons: Sequence[int] | None = CORE_HORIZONS,
    pretrend_controls: Sequence[str] = (),
    calendar_controls: Sequence[str] = (),
) -> dict[str, object]:
    outcomes = list(outcomes or CORE_OUTCOMES)
    horizons = list(horizons or CORE_HORIZONS)
    frames = []
    for model, panel_csv, controls in [
        ("noniv_tdc_pretrend", pretrend_panel_csv, pretrend_controls),
        ("noniv_tdc_calendar", calendar_panel_csv, calendar_controls),
    ]:
        panel = read_wide_time_series_csv(panel_csv)
        result = run_local_projections(
            panel,
            shock_col="tdc_monthly",
            outcome_cols=outcomes,
            controls=controls,
            horizons=horizons,
            spec_name=model,
        )
        result = add_same_unit_columns(result, treatment_col="tdc_monthly")
        result["model"] = model
        result["causal_interpretation"] = "descriptive_not_causal"
        frames.append(result)
    out = pd.concat(frames, ignore_index=True)
    out["hac_sig_95"] = (out["same_unit_lower95"] > 0) | (out["same_unit_upper95"] < 0)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    if out_md:
        keep = [
            "model",
            "outcome",
            "horizon",
            "same_unit_beta",
            "same_unit_lower95",
            "same_unit_upper95",
            "hac_sig_95",
            "n",
            "causal_interpretation",
        ]
        _write_md_table(out[keep], Path(out_md), title="Non-IV TDC LP Closeout")
    return {"status": "ok", "out": str(path), "out_md": str(out_md or ""), "rows": int(len(out))}


def export_tga_reduced_form_closeout(
    *,
    pretrend_panel_csv: str | Path,
    calendar_panel_csv: str | Path,
    out_csv: str | Path,
    out_md: str | Path | None = None,
    outcomes: Sequence[str] | None = CORE_OUTCOMES,
    horizons: Sequence[int] | None = CORE_HORIZONS,
    pretrend_controls: Sequence[str] = (),
    calendar_controls: Sequence[str] = (),
) -> dict[str, object]:
    outcomes = list(outcomes or CORE_OUTCOMES)
    horizons = list(horizons or CORE_HORIZONS)
    frames = []
    for model, panel_csv, controls in [
        ("tga_reduced_form_pretrend", pretrend_panel_csv, pretrend_controls),
        ("tga_reduced_form_calendar", calendar_panel_csv, calendar_controls),
    ]:
        panel = read_wide_time_series_csv(panel_csv)
        result = run_local_projections(
            panel,
            shock_col="tga_long_surprise_z",
            outcome_cols=outcomes,
            controls=controls,
            horizons=horizons,
            spec_name=model,
        )
        result["model"] = model
        result["coefficient_interpretation"] = "native outcome units per 1 sd TGA surprise"
        frames.append(result)
    out = pd.concat(frames, ignore_index=True)
    out["hac_sig_95"] = (out["lower95"] > 0) | (out["upper95"] < 0)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    if out_md:
        keep = ["model", "outcome", "horizon", "beta", "lower95", "upper95", "hac_sig_95", "n", "coefficient_interpretation"]
        _write_md_table(out[keep], Path(out_md), title="TGA Reduced-Form Closeout")
    return {"status": "ok", "out": str(path), "out_md": str(out_md or ""), "rows": int(len(out))}


def export_placebo_summary(
    *,
    placebo_csv: str | Path,
    out_csv: str | Path,
    out_md: str | Path | None = None,
    model: str = "long_tga_pretrend_iv",
) -> dict[str, object]:
    placebo = pd.read_csv(placebo_csv)
    beta_col = "same_unit_beta" if "same_unit_beta" in placebo.columns else "beta"
    summary = (
        placebo.groupby("outcome", dropna=False)
        .agg(
            n_placebo_horizons=("placebo_sig_95", "size"),
            n_significant_placebos=("placebo_sig_95", "sum"),
            max_abs_placebo_beta=(beta_col, lambda x: float(pd.to_numeric(x, errors="coerce").abs().max())),
        )
        .reset_index()
    )
    summary["model"] = model
    summary["warning_label"] = summary["n_significant_placebos"].map(lambda n: "placebo_warning" if int(n) else "placebo_clean")
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    if out_md:
        _write_md_table(summary, Path(out_md), title="Placebo Summary")
    return {"status": "ok", "out": str(path), "out_md": str(out_md or ""), "rows": int(len(summary))}


def _standardize_iv(frame: pd.DataFrame, *, model: str, bootstrap: pd.DataFrame | None = None, placebo: pd.DataFrame | None = None) -> pd.DataFrame:
    out = frame.copy()
    out["model"] = model
    if bootstrap is not None and not bootstrap.empty:
        boot_cols = ["outcome", "horizon", "same_unit_bootstrap_lower95", "same_unit_bootstrap_upper95", "draws", "block_length"]
        present = [col for col in boot_cols if col in bootstrap.columns]
        out = out.merge(bootstrap[present], on=["outcome", "horizon"], how="left")
    if placebo is not None and not placebo.empty:
        counts = (
            placebo.groupby("outcome", dropna=False)
            .agg(placebo_sig_count=("placebo_sig_95", "sum"), placebo_rows=("placebo_sig_95", "size"))
            .reset_index()
        )
        out = out.merge(counts, on="outcome", how="left")
    return out


def _claim_label(row: pd.Series, calendar_sig_lookup: dict[tuple[str, int], bool] | None = None) -> str:
    outcome = str(row.get("outcome", ""))
    model = str(row.get("model", ""))
    horizon = int(row.get("horizon", -1))
    raw_placebo_sig = row.get("placebo_sig_count", 0)
    placebo_sig = 0 if pd.isna(raw_placebo_sig) else int(raw_placebo_sig)
    sig = _sig(row)
    if outcome in {"deposits", "broad_deposits"} and horizon <= 4 and not sig:
        return "null_core"
    if outcome in {"onrrp", "total_mmf"} and placebo_sig > 0:
        return "placebo_contaminated"
    if outcome == "onrrp" and sig:
        return "plumbing_response"
    if model == "long_tga_pretrend_iv" and calendar_sig_lookup is not None:
        if sig and not calendar_sig_lookup.get((outcome, horizon), False):
            return "calendar_sensitive"
    if not sig:
        return "imprecise"
    return "appendix_only"


def export_short_run_core_closeout(
    *,
    pretrend_iv_csv: str | Path,
    calendar_iv_csv: str | Path,
    bootstrap_csv: str | Path | None,
    placebo_csv: str | Path | None,
    out_csv: str | Path,
    out_md: str | Path | None = None,
    outcomes: Sequence[str] | None = CORE_OUTCOMES,
    horizons: Sequence[int] | None = (0, 1, 2, 3, 4),
) -> dict[str, object]:
    outcomes = list(outcomes or CORE_OUTCOMES)
    horizons = list(horizons or (0, 1, 2, 3, 4))
    pre = pd.read_csv(pretrend_iv_csv)
    cal = pd.read_csv(calendar_iv_csv)
    boot = pd.read_csv(bootstrap_csv) if bootstrap_csv and Path(bootstrap_csv).exists() else None
    placebo = pd.read_csv(placebo_csv) if placebo_csv and Path(placebo_csv).exists() else None
    pre = pre.loc[pre["outcome"].isin(outcomes) & pre["horizon"].isin(horizons)].copy()
    cal = cal.loc[cal["outcome"].isin(outcomes) & cal["horizon"].isin(horizons)].copy()
    if boot is not None:
        boot = boot.loc[boot["outcome"].isin(outcomes) & boot["horizon"].isin(horizons)].copy()
    frames = [
        _standardize_iv(pre, model="long_tga_pretrend_iv", bootstrap=boot, placebo=placebo),
        _standardize_iv(cal, model="long_tga_calendar_iv", placebo=placebo),
    ]
    out = pd.concat(frames, ignore_index=True)
    cal_lookup = {
        (str(row["outcome"]), int(row["horizon"])): _sig(row)
        for row in frames[1].to_dict("records")
    }
    out["claim_label"] = out.apply(lambda row: _claim_label(row, calendar_sig_lookup=cal_lookup), axis=1)
    cols = [
        "model",
        "outcome",
        "horizon",
        "same_unit_beta",
        "same_unit_lower95",
        "same_unit_upper95",
        "same_unit_bootstrap_lower95",
        "same_unit_bootstrap_upper95",
        "first_stage_f",
        "n",
        "placebo_sig_count",
        "placebo_rows",
        "claim_label",
    ]
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[cols].sort_values(["outcome", "horizon", "model"]).reset_index(drop=True)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    if out_md:
        _write_md_table(out, Path(out_md), title="Short-Run Core LP-IV Closeout", max_rows=120)
    return {"status": "ok", "out": str(path), "out_md": str(out_md or ""), "rows": int(len(out))}


def export_method_envelope_closeout(
    *,
    proxy_monthly_comparison_csv: str | Path,
    proxy_quarterly_comparison_csv: str | Path,
    out_csv: str | Path,
    out_md: str | Path | None = None,
) -> dict[str, object]:
    monthly = pd.read_csv(proxy_monthly_comparison_csv)
    quarterly = pd.read_csv(proxy_quarterly_comparison_csv)
    rows = [
        {
            "comparison": "monthly_denton_vs_residual_spread",
            "n": int(len(monthly)),
            "mean_abs_difference": float(monthly["abs_difference"].mean()),
            "max_abs_difference": float(monthly["abs_difference"].max()),
            "interpretation": "monthly_timing_differs_materially_even_when_quarterly_sums_match",
        },
        {
            "comparison": "quarterly_sum_denton_vs_residual_spread",
            "n": int(len(quarterly)),
            "mean_abs_difference": float(quarterly["difference_sum"].abs().mean()),
            "max_abs_difference": float(quarterly["difference_sum"].abs().max()),
            "interpretation": "quarterly_identity_preserved",
        },
    ]
    out = pd.DataFrame(rows)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    if out_md:
        _write_md_table(out, Path(out_md), title="Core Method Envelope Closeout")
    return {"status": "ok", "out": str(path), "out_md": str(out_md or ""), "rows": int(len(out))}


def export_core_iv_vs_noniv_profiles(
    *,
    short_run_iv_csv: str | Path,
    noniv_csv: str | Path,
    tga_rf_csv: str | Path,
    out_dir: str | Path,
    outcomes: Sequence[str] | None = ("deposits", "bank_credit", "onrrp", "total_mmf"),
    horizons: Sequence[int] | None = (0, 1, 2, 3, 4, 12),
) -> dict[str, object]:
    outcomes = list(outcomes or ("deposits", "bank_credit", "onrrp", "total_mmf"))
    horizons = list(horizons or (0, 1, 2, 3, 4, 12))
    iv = pd.read_csv(short_run_iv_csv)
    noniv = pd.read_csv(noniv_csv)
    rf = pd.read_csv(tga_rf_csv)

    iv_plot = iv.rename(columns={"same_unit_beta": "beta_plot"})
    iv_plot = iv_plot.loc[iv_plot["model"].isin(["long_tga_pretrend_iv", "long_tga_calendar_iv"])]
    noniv_plot = noniv.rename(columns={"same_unit_beta": "beta_plot"})
    rf_plot = rf.rename(columns={"beta": "beta_plot"})
    plot = pd.concat(
        [
            iv_plot[["model", "outcome", "horizon", "beta_plot"]],
            noniv_plot[["model", "outcome", "horizon", "beta_plot"]],
            rf_plot[["model", "outcome", "horizon", "beta_plot"]],
        ],
        ignore_index=True,
    )
    plot = plot.loc[plot["outcome"].isin(outcomes) & plot["horizon"].isin(horizons)].copy()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "core_iv_vs_noniv_profiles_data.csv"
    fig_path = root / "core_iv_vs_noniv_profiles.png"
    plot.to_csv(data_path, index=False)

    fig, axes = plt.subplots(len(outcomes), 1, figsize=(12, 2.4 * len(outcomes)), sharex=True)
    if len(outcomes) == 1:
        axes = [axes]
    model_order = [
        "long_tga_pretrend_iv",
        "long_tga_calendar_iv",
        "noniv_tdc_pretrend",
        "noniv_tdc_calendar",
        "tga_reduced_form_pretrend",
        "tga_reduced_form_calendar",
    ]
    for ax, outcome in zip(axes, outcomes, strict=False):
        sub = plot.loc[plot["outcome"].eq(outcome)]
        ax.axhline(0, color="black", linewidth=0.8)
        for model in model_order:
            line = sub.loc[sub["model"].eq(model)].sort_values("horizon")
            if line.empty:
                continue
            ax.plot(line["horizon"], line["beta_plot"], marker="o", linewidth=1.2, label=model)
        ax.set_title(outcome)
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(list(horizons))
    axes[-1].set_xlabel("Horizon, months")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return {"status": "ok", "out_dir": str(root), "data": str(data_path), "figure": str(fig_path), "rows": int(len(plot))}
