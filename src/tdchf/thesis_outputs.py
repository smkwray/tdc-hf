from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


CORE_OUTCOMES = {
    "deposits",
    "broad_deposits",
    "bank_credit",
    "commercial_industrial_loans",
}
MECHANISM_OUTCOMES = {"onrrp", "retail_mmf", "institutional_mmf", "total_mmf", "reserves"}


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def _claim_status(row: pd.Series) -> tuple[str, str]:
    outcome = str(row.get("outcome", ""))
    hac = bool(row.get("hac_sig_95", False))
    boot = bool(row.get("bootstrap_sig_95", False))
    placebo = bool(row.get("placebo_clean", False))
    regime_pass = row.get("regime_pass_count")
    regime_rows = row.get("regime_rows")
    weak_regime = pd.notna(regime_pass) and pd.notna(regime_rows) and float(regime_rows) >= 3 and float(regime_pass) < 3

    if outcome in {"deposits", "broad_deposits"}:
        return (
            "imprecise_core",
            "Positive but not tightly estimated; use as the main pass-through target but do not claim a precise multiplier.",
        )
    if outcome == "commercial_industrial_loans" and hac and placebo and weak_regime:
        return (
            "supportive_core_regime_sensitive",
            "Positive in the anchor-variant dashboard and placebo-clean, but weakened by regime/calendar robustness; use as secondary credit-channel evidence.",
        )
    if outcome in {"bank_credit", "commercial_industrial_loans"} and hac and placebo and not weak_regime:
        if boot:
            return ("core_evidence", "HAC, bootstrap, and placebo evidence support this as a core balance-sheet response.")
        return ("supportive_core", "HAC and placebo evidence support this response, but bootstrap intervals remain wide.")
    if outcome in MECHANISM_OUTCOMES:
        if not placebo:
            return (
                "mechanism_evidence_placebo_contaminated",
                "Use as mechanism evidence only; backward-placebo failures block headline causal interpretation.",
            )
        if hac or boot:
            return ("mechanism_evidence", "Mechanism response with cleaner placebo behavior, not a primary pass-through estimate.")
    return ("appendix_only", "Useful sensitivity or context outcome; do not foreground in the thesis claim.")


def build_claim_status_table(
    *,
    anchor_dashboard: pd.DataFrame,
    regime_summary: pd.DataFrame | None = None,
    outcome_order: list[str] | None = None,
) -> pd.DataFrame:
    table = anchor_dashboard.copy()
    if outcome_order is not None:
        table = table.loc[table["outcome"].isin(outcome_order)].copy()

    table["hac_sig_95"] = _bool_series(table, "hac_sig_95")
    table["bootstrap_sig_95"] = _bool_series(table, "bootstrap_sig_95")
    table["placebo_clean"] = _bool_series(table, "placebo_clean")

    if regime_summary is not None and not regime_summary.empty:
        reg = regime_summary.copy()
        if "hac_sig_95" in reg.columns:
            reg["hac_sig_95"] = reg["hac_sig_95"].fillna(False).astype(bool)
        reg_counts = (
            reg.loc[~reg["sample"].eq("full_sample")]
            .groupby("outcome")
            .agg(
                regime_rows=("sample", "size"),
                regime_pass_count=("hac_sig_95", "sum"),
                regime_min_beta=("same_unit_beta", "min"),
                regime_max_beta=("same_unit_beta", "max"),
            )
            .reset_index()
        )
        table = table.merge(reg_counts, on="outcome", how="left")

    statuses = table.apply(_claim_status, axis=1, result_type="expand")
    table["claim_status"] = statuses[0]
    table["thesis_language"] = statuses[1]
    cols = [
        "anchor_variant",
        "anchor_column",
        "outcome",
        "horizon",
        "same_unit_beta",
        "same_unit_lower95",
        "same_unit_upper95",
        "same_unit_bootstrap_lower95",
        "same_unit_bootstrap_upper95",
        "first_stage_f",
        "full_sample_first_stage_f",
        "placebo_rows",
        "placebo_sig_rows",
        "placebo_clean",
        "regime_rows",
        "regime_pass_count",
        "regime_min_beta",
        "regime_max_beta",
        "claim_status",
        "thesis_language",
    ]
    for col in cols:
        if col not in table.columns:
            table[col] = pd.NA
    return table[cols].sort_values(["outcome", "anchor_variant"]).reset_index(drop=True)


def export_claim_status_table(
    *,
    anchor_dashboard_csv: str | Path,
    out_csv: str | Path,
    out_md: str | Path | None = None,
    regime_summary_csv: str | Path | None = None,
    outcomes: list[str] | None = None,
) -> dict[str, object]:
    dashboard = pd.read_csv(anchor_dashboard_csv)
    regime = pd.read_csv(regime_summary_csv) if regime_summary_csv and Path(regime_summary_csv).exists() else None
    out = build_claim_status_table(anchor_dashboard=dashboard, regime_summary=regime, outcome_order=outcomes)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    md_path = None
    if out_md is not None:
        md_path = Path(out_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        _write_claim_status_md(out, md_path)
    return {"status": "ok", "out": str(path), "out_md": str(md_path) if md_path else "", "rows": int(len(out))}


def _write_claim_status_md(table: pd.DataFrame, path: Path) -> None:
    horizons = sorted(table["horizon"].dropna().astype(int).unique().tolist()) if "horizon" in table.columns else []
    title = "H=12 Claim Status" if horizons == [12] else f"Short-Run Claim Status, h={','.join(map(str, horizons))}"
    lines = [f"# {title}", ""]
    for status, group in table.groupby("claim_status", dropna=False):
        lines.append(f"## {status}")
        lines.append("")
        for row in group.to_dict("records"):
            lines.append(
                f"- h=`{int(row['horizon'])}` `{row['anchor_variant']}` / `{row['outcome']}`: "
                f"`{float(row['same_unit_beta']):.3g}` "
                f"[`{float(row['same_unit_lower95']):.3g}`, `{float(row['same_unit_upper95']):.3g}`]; "
                f"placebo clean `{row['placebo_clean']}`. {row['thesis_language']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_short_run_profile_plot(
    *,
    anchor_dashboard_csv: str | Path,
    out_dir: str | Path,
    outcomes: list[str] | None = None,
    horizons: list[int] | None = None,
) -> dict[str, object]:
    dashboard = pd.read_csv(anchor_dashboard_csv)
    outcomes = outcomes or ["deposits", "bank_credit", "commercial_industrial_loans", "onrrp", "total_mmf"]
    horizons = horizons or [0, 1, 2, 3]
    plot_data = dashboard.loc[dashboard["outcome"].isin(outcomes) & dashboard["horizon"].isin(horizons)].copy()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "anchor_variant_short_run_profile_data.csv"
    plot_data.to_csv(data_path, index=False)
    png_path = root / "anchor_variant_short_run_profile.png"

    variants = list(dict.fromkeys(plot_data["anchor_variant"].dropna()))
    fig, axes = plt.subplots(len(outcomes), 1, figsize=(12, 2.1 * len(outcomes)), sharex=True)
    if len(outcomes) == 1:
        axes = [axes]
    for ax, outcome in zip(axes, outcomes, strict=False):
        subset = plot_data.loc[plot_data["outcome"].eq(outcome)].copy()
        ax.axhline(0.0, color="black", linewidth=0.8)
        for variant in variants:
            line = subset.loc[subset["anchor_variant"].eq(variant)].sort_values("horizon")
            if line.empty:
                continue
            ax.plot(line["horizon"], line["same_unit_beta"], marker="o", linewidth=1.2, label=variant)
            ax.fill_between(
                line["horizon"],
                line["same_unit_lower95"],
                line["same_unit_upper95"],
                alpha=0.08,
            )
        ax.set_title(outcome)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylabel("same-unit beta")
    axes[-1].set_xlabel("Horizon, months")
    axes[-1].set_xticks(horizons)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    return {"status": "ok", "out_dir": str(root), "data": str(data_path), "figure": str(png_path), "rows": int(len(plot_data))}


def export_anchor_variant_forest_plot(
    *,
    anchor_dashboard_csv: str | Path,
    out_dir: str | Path,
    outcomes: list[str] | None = None,
) -> dict[str, object]:
    dashboard = pd.read_csv(anchor_dashboard_csv)
    outcomes = outcomes or ["deposits", "bank_credit", "commercial_industrial_loans", "onrrp", "total_mmf"]
    plot_data = dashboard.loc[dashboard["outcome"].isin(outcomes)].copy()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "anchor_variant_forest_plot_data.csv"
    plot_data.to_csv(data_path, index=False)
    png_path = root / "anchor_variant_h12_forest.png"

    variants = list(dict.fromkeys(plot_data["anchor_variant"].dropna()))
    fig, axes = plt.subplots(len(outcomes), 1, figsize=(12, 2.2 * len(outcomes)), sharex=True)
    if len(outcomes) == 1:
        axes = [axes]
    for ax, outcome in zip(axes, outcomes, strict=False):
        subset = plot_data.loc[plot_data["outcome"].eq(outcome)].set_index("anchor_variant").reindex(variants).reset_index()
        y = range(len(subset))
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.errorbar(
            subset["same_unit_beta"],
            y,
            xerr=[
                subset["same_unit_beta"] - subset["same_unit_lower95"],
                subset["same_unit_upper95"] - subset["same_unit_beta"],
            ],
            fmt="o",
            color="#1f4e79",
            ecolor="#6c8ebf",
            capsize=3,
            label="HAC 95%",
        )
        boot_ok = subset[["same_unit_bootstrap_lower95", "same_unit_bootstrap_upper95"]].notna().all(axis=1)
        if boot_ok.any():
            ax.errorbar(
                subset.loc[boot_ok, "same_unit_beta"],
                [idx for idx, ok in enumerate(boot_ok) if ok],
                xerr=[
                    subset.loc[boot_ok, "same_unit_beta"] - subset.loc[boot_ok, "same_unit_bootstrap_lower95"],
                    subset.loc[boot_ok, "same_unit_bootstrap_upper95"] - subset.loc[boot_ok, "same_unit_beta"],
                ],
                fmt="none",
                color="#a23b3b",
                ecolor="#d98a8a",
                capsize=2,
                linewidth=1.0,
                label="Bootstrap 95%",
            )
        for idx, row in subset.iterrows():
            if int(row.get("placebo_sig_rows", 0) or 0) > 0:
                ax.text(row["same_unit_upper95"], idx, f"  pbo {int(row['placebo_sig_rows'])}", va="center", fontsize=8)
        ax.set_yticks(list(y), variants)
        ax.set_title(outcome)
        ax.grid(True, axis="x", alpha=0.25)
    axes[-1].set_xlabel("Outcome dollars per 1 TDC dollar")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    return {"status": "ok", "out_dir": str(root), "data": str(data_path), "figure": str(png_path), "rows": int(len(plot_data))}
