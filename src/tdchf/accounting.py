from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_ACCOUNTING_OUTCOMES = [
    "deposits",
    "large_time_deposits",
    "broad_deposits",
    "bank_credit",
    "reserves",
    "onrrp",
    "retail_mmf",
    "institutional_mmf",
    "total_mmf",
    "commercial_industrial_loans",
    "consumer_loans",
    "credit_card_revolving_loans",
    "auto_loans",
    "other_consumer_loans",
    "closed_end_residential_loans",
    "heloc_loans",
    "construction_land_development_loans",
    "cre_multifamily_loans",
    "cre_nonfarm_nonresidential_loans",
    "loans_to_nondepository_financial_institutions",
]


def build_accounting_decomposition(
    lp: pd.DataFrame,
    *,
    horizons: list[int],
    outcomes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "same_unit_beta" not in lp.columns:
        raise KeyError("LP table must contain same_unit_beta")
    selected = outcomes or DEFAULT_ACCOUNTING_OUTCOMES
    table = lp.loc[lp["outcome"].isin(selected) & lp["horizon"].isin(horizons)].copy()
    table["same_unit_hac_sig_95"] = (table["same_unit_lower95"] > 0) | (table["same_unit_upper95"] < 0)
    table["included_in_simple_sum"] = ~table["outcome"].isin(["broad_deposits", "total_mmf", "bank_credit"])
    table["abs_same_unit_beta"] = table["same_unit_beta"].abs()

    summary_rows: list[dict[str, object]] = []
    for horizon, group in table.groupby("horizon"):
        simple = group.loc[group["included_in_simple_sum"]]
        summary_rows.append(
            {
                "horizon": int(horizon),
                "included_outcomes": int(simple["outcome"].nunique()),
                "simple_sum_same_unit_beta": float(simple["same_unit_beta"].sum()),
                "positive_sum_same_unit_beta": float(simple.loc[simple["same_unit_beta"] > 0, "same_unit_beta"].sum()),
                "negative_sum_same_unit_beta": float(simple.loc[simple["same_unit_beta"] < 0, "same_unit_beta"].sum()),
                "largest_abs_outcome": str(group.sort_values("abs_same_unit_beta", ascending=False).iloc[0]["outcome"]) if not group.empty else "",
                "largest_abs_same_unit_beta": float(group["abs_same_unit_beta"].max()) if not group.empty else float("nan"),
                "significant_positive_count": int(((group["same_unit_lower95"] > 0) & (group["same_unit_upper95"] > 0)).sum()),
                "significant_negative_count": int(((group["same_unit_lower95"] < 0) & (group["same_unit_upper95"] < 0)).sum()),
            }
        )
    return table.sort_values(["horizon", "outcome"]), pd.DataFrame(summary_rows).sort_values("horizon")


def export_accounting_decomposition(
    lp_csv: str | Path,
    *,
    out_dir: str | Path,
    horizons: list[int],
    outcomes: list[str] | None = None,
) -> dict[str, object]:
    lp = pd.read_csv(lp_csv)
    table, summary = build_accounting_decomposition(lp, horizons=horizons, outcomes=outcomes)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    table_path = root / "accounting_decomposition.csv"
    summary_path = root / "accounting_decomposition_summary.csv"
    table.to_csv(table_path, index=False)
    summary.to_csv(summary_path, index=False)

    md_path = root / "accounting_decomposition.md"
    lines = [
        "# Same-Unit Accounting Decomposition",
        "",
        "Coefficients are outcome dollars per one TDC dollar.",
        "",
        "## Horizon Summary",
        "",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"- h={int(row['horizon'])}: simple sum `{float(row['simple_sum_same_unit_beta']):.3g}`, "
            f"largest `{row['largest_abs_outcome']}` `{float(row['largest_abs_same_unit_beta']):.3g}`, "
            f"significant + `{int(row['significant_positive_count'])}`, significant - `{int(row['significant_negative_count'])}`"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "out_dir": str(root),
        "table": str(table_path),
        "summary": str(summary_path),
        "markdown": str(md_path),
        "rows": int(len(table)),
    }
