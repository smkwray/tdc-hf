from __future__ import annotations

from pathlib import Path

import pandas as pd

from .indicators import aggregate_levels_to_monthly, level_change_to_monthly_flow, read_wide_time_series_csv

FRED_OUTCOME_SERIES = [
    "DPSACBM027SBOG",
    "DPSACBM027NBOG",
    "DPSACBW027SBOG",
    "DPSDCBW027SBOG",
    "LTDACBW027SBOG",
    "TOTBKCR",
    "TOTBKCRNSA",
    "WRESBAL",
    "RRPONTSYD",
    "DGS2",
    "DGS10",
    "EFFR",
    "IORB",
    "SOFR",
    "WRMFNS",
    "WIMFNS",
    "TOTCINSA",
    "CLDACBW027SBOG",
    "SMPACBW027SBOG",
    "SNFACBW027SBOG",
    "CLSACBW027SBOG",
    "CCLACBW027SBOG",
    "CARACBW027SBOG",
    "OCLACBW027SBOG",
    "RHEACBW027SBOG",
    "CRLACBW027SBOG",
    "LCBACBW027SBOG",
    "LNFACBW027SBOG",
    "BAA",
    "AAA",
    "BAMLC0A0CM",
    "BAMLH0A0HYM2",
    "DGS3MO",
]


def _first_available(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns and df[candidate].notna().any():
            return candidate
    return None


def build_monthly_outcomes_from_fred_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.DataFrame()
    meta: list[dict[str, object]] = []

    broad_col = _first_available(df, ["DPSACBM027SBOG", "DPSACBM027NBOG", "DPSACBW027SBOG"])
    if broad_col:
        out["broad_deposits"] = level_change_to_monthly_flow(df[broad_col].rename("broad_deposits"))
        out["broad_deposits_level"] = aggregate_levels_to_monthly(df[broad_col].rename("broad_deposits_level"))
        meta.append(
            {
                "column": "broad_deposits",
                "source_series": broad_col,
                "transform": "month_end_level_diff",
                "definition": "broad all-commercial-bank deposits; retained for comparison, not the preferred thesis target",
            }
        )

    domestic_col = _first_available(df, ["DPSDCBW027SBOG"])
    if domestic_col:
        out["domestic_deposits"] = level_change_to_monthly_flow(df[domestic_col].rename("domestic_deposits"))
        out["domestic_deposits_level"] = aggregate_levels_to_monthly(df[domestic_col].rename("domestic_deposits_level"))
        meta.append(
            {
                "column": "domestic_deposits",
                "source_series": domestic_col,
                "transform": "month_end_level_diff",
                "definition": "domestic-bank deposits; excludes foreign-related institutions when FRED source is available",
            }
        )

    large_time_col = _first_available(df, ["LTDACBW027SBOG"])
    if large_time_col:
        out["large_time_deposits"] = level_change_to_monthly_flow(df[large_time_col].rename("large_time_deposits"))
        out["large_time_deposits_level"] = aggregate_levels_to_monthly(df[large_time_col].rename("large_time_deposits_level"))
        meta.append(
            {
                "column": "large_time_deposits",
                "source_series": large_time_col,
                "transform": "month_end_level_diff",
                "definition": "large time deposit flow removed from the preferred non-large-time target",
            }
        )
        if broad_col:
            out["broad_non_large_time_deposits"] = out["broad_deposits"] - out["large_time_deposits"]
            out["broad_non_large_time_deposits_level"] = out["broad_deposits_level"] - out["large_time_deposits_level"]
            meta.append(
                {
                    "column": "broad_non_large_time_deposits",
                    "source_series": f"{broad_col}-{large_time_col}",
                    "transform": "derived_difference",
                    "definition": "broad deposit flow less large time deposits; comparison series",
                }
            )
        if domestic_col:
            out["domestic_non_large_time_deposits"] = out["domestic_deposits"] - out["large_time_deposits"]
            out["domestic_non_large_time_deposits_level"] = out["domestic_deposits_level"] - out["large_time_deposits_level"]
            meta.append(
                {
                    "column": "domestic_non_large_time_deposits",
                    "source_series": f"{domestic_col}-{large_time_col}",
                    "transform": "derived_difference",
                    "definition": "preferred thesis deposit target: domestic deposits less large time deposits",
                }
            )

    if "domestic_non_large_time_deposits" in out.columns:
        out["deposits"] = out["domestic_non_large_time_deposits"]
        out["deposits_level"] = out["domestic_non_large_time_deposits_level"]
        meta.append(
            {
                "column": "deposits",
                "source_series": "domestic_non_large_time_deposits",
                "transform": "preferred_alias",
                "definition": "canonical modeling target: domestic non-large-time deposits",
            }
        )
    elif "domestic_deposits" in out.columns:
        out["deposits"] = out["domestic_deposits"]
        out["deposits_level"] = out["domestic_deposits_level"]
        meta.append(
            {
                "column": "deposits",
                "source_series": domestic_col,
                "transform": "preferred_alias",
                "definition": "fallback modeling target: domestic deposits; large-time subtraction unavailable",
            }
        )
    elif "broad_deposits" in out.columns:
        out["deposits"] = out["broad_deposits"]
        out["deposits_level"] = out["broad_deposits_level"]
        meta.append(
            {
                "column": "deposits",
                "source_series": broad_col,
                "transform": "fallback_alias",
                "definition": "fallback only: broad deposits because domestic source was unavailable",
            }
        )

    bank_credit_col = _first_available(df, ["TOTBKCR", "TOTBKCRNSA"])
    if bank_credit_col:
        out["bank_credit"] = level_change_to_monthly_flow(df[bank_credit_col].rename("bank_credit"))
        out["bank_credit_level"] = aggregate_levels_to_monthly(df[bank_credit_col].rename("bank_credit_level"))
        meta.append({"column": "bank_credit", "source_series": bank_credit_col, "transform": "month_end_level_diff"})

    for source, flow_target, level_target, definition in [
        ("WRESBAL", "reserves", "reserves_level", "reserve balances with Federal Reserve Banks"),
        ("RRPONTSYD", "onrrp", "onrrp_level", "overnight reverse repurchase agreements with the Treasury securities pool"),
        ("WRMFNS", "retail_mmf", "retail_mmf_level", "retail money market funds"),
        ("WIMFNS", "institutional_mmf", "institutional_mmf_level", "institutional money market funds"),
    ]:
        if source in df.columns:
            out[flow_target] = level_change_to_monthly_flow(df[source].rename(flow_target))
            out[level_target] = aggregate_levels_to_monthly(df[source].rename(level_target))
            meta.append(
                {
                    "column": flow_target,
                    "source_series": source,
                    "transform": "month_end_level_diff",
                    "definition": definition,
                }
            )
            meta.append(
                {
                    "column": level_target,
                    "source_series": source,
                    "transform": "month_end_level",
                    "definition": definition,
                }
            )

    if "retail_mmf" in out.columns and "institutional_mmf" in out.columns:
        out["total_mmf"] = out["retail_mmf"] + out["institutional_mmf"]
        out["total_mmf_level"] = out["retail_mmf_level"] + out["institutional_mmf_level"]
        meta.append(
            {
                "column": "total_mmf",
                "source_series": "WRMFNS+WIMFNS",
                "transform": "derived_sum",
                "definition": "retail plus institutional money market fund flow",
            }
        )
        meta.append(
            {
                "column": "total_mmf_level",
                "source_series": "WRMFNS+WIMFNS",
                "transform": "derived_sum",
                "definition": "retail plus institutional money market fund level",
            }
        )

    for source, flow_target, level_target, definition in [
        ("TOTCINSA", "commercial_industrial_loans", "commercial_industrial_loans_level", "commercial and industrial loans"),
        (
            "CLDACBW027SBOG",
            "construction_land_development_loans",
            "construction_land_development_loans_level",
            "construction and land development loans",
        ),
        ("SMPACBW027SBOG", "cre_multifamily_loans", "cre_multifamily_loans_level", "multifamily CRE loans"),
        (
            "SNFACBW027SBOG",
            "cre_nonfarm_nonresidential_loans",
            "cre_nonfarm_nonresidential_loans_level",
            "nonfarm nonresidential CRE loans",
        ),
        ("CLSACBW027SBOG", "consumer_loans", "consumer_loans_level", "consumer loans"),
        ("CCLACBW027SBOG", "credit_card_revolving_loans", "credit_card_revolving_loans_level", "credit card revolving loans"),
        ("CARACBW027SBOG", "auto_loans", "auto_loans_level", "auto loans"),
        ("OCLACBW027SBOG", "other_consumer_loans", "other_consumer_loans_level", "other consumer loans"),
        ("RHEACBW027SBOG", "heloc_loans", "heloc_loans_level", "home equity lines of credit"),
        (
            "CRLACBW027SBOG",
            "closed_end_residential_loans",
            "closed_end_residential_loans_level",
            "closed-end residential real estate loans",
        ),
        ("LCBACBW027SBOG", "loans_to_commercial_banks", "loans_to_commercial_banks_level", "loans to commercial banks"),
        (
            "LNFACBW027SBOG",
            "loans_to_nondepository_financial_institutions",
            "loans_to_nondepository_financial_institutions_level",
            "loans to nondepository financial institutions",
        ),
    ]:
        if source in df.columns:
            out[flow_target] = level_change_to_monthly_flow(df[source].rename(flow_target))
            out[level_target] = aggregate_levels_to_monthly(df[source].rename(level_target))
            meta.append(
                {
                    "column": flow_target,
                    "source_series": source,
                    "transform": "month_end_level_diff",
                    "definition": definition,
                }
            )
            meta.append(
                {
                    "column": level_target,
                    "source_series": source,
                    "transform": "month_end_level",
                    "definition": definition,
                }
            )

    if "BAA" in df.columns and "AAA" in df.columns:
        out["baa_aaa_spread"] = aggregate_levels_to_monthly(df["BAA"] - df["AAA"], how="mean")
        out["d_baa_aaa_spread"] = out["baa_aaa_spread"].diff()
        meta.append(
            {
                "column": "d_baa_aaa_spread",
                "source_series": "BAA-AAA",
                "transform": "monthly_mean_spread_diff",
                "definition": "change in Moody's BAA minus AAA corporate yield spread",
            }
        )
    for source, target, definition in [
        ("BAMLC0A0CM", "d_investment_grade_oas", "change in ICE BofA US Corporate Master option-adjusted spread"),
        ("BAMLH0A0HYM2", "d_high_yield_oas", "change in ICE BofA US High Yield option-adjusted spread"),
    ]:
        if source in df.columns:
            level = aggregate_levels_to_monthly(df[source].rename(target.removeprefix("d_")), how="mean")
            out[target.removeprefix("d_")] = level
            out[target] = level.diff()
            meta.append({"column": target, "source_series": source, "transform": "monthly_mean_diff", "definition": definition})
    if "DGS10" in df.columns and "DGS2" in df.columns:
        out["term_spread_10y_2y"] = aggregate_levels_to_monthly(df["DGS10"] - df["DGS2"], how="mean")
        out["d_term_spread_10y_2y"] = out["term_spread_10y_2y"].diff()
        meta.append({"column": "d_term_spread_10y_2y", "source_series": "DGS10-DGS2", "transform": "monthly_mean_spread_diff"})
    if "DGS10" in df.columns and "DGS3MO" in df.columns:
        out["term_spread_10y_3m"] = aggregate_levels_to_monthly(df["DGS10"] - df["DGS3MO"], how="mean")
        out["d_term_spread_10y_3m"] = out["term_spread_10y_3m"].diff()
        meta.append({"column": "d_term_spread_10y_3m", "source_series": "DGS10-DGS3MO", "transform": "monthly_mean_spread_diff"})
    if "EFFR" in df.columns and "IORB" in df.columns:
        out["effr_iorb_spread"] = aggregate_levels_to_monthly(df["EFFR"] - df["IORB"], how="mean")
        out["d_effr_iorb_spread"] = out["effr_iorb_spread"].diff()
        meta.append({"column": "d_effr_iorb_spread", "source_series": "EFFR-IORB", "transform": "monthly_mean_spread_diff"})
    if "SOFR" in df.columns and "IORB" in df.columns:
        out["sofr_iorb_spread"] = aggregate_levels_to_monthly(df["SOFR"] - df["IORB"], how="mean")
        out["d_sofr_iorb_spread"] = out["sofr_iorb_spread"].diff()
        meta.append({"column": "d_sofr_iorb_spread", "source_series": "SOFR-IORB", "transform": "monthly_mean_spread_diff"})

    for source, target in [
        ("DGS2", "yield_2y"),
        ("DGS10", "yield_10y"),
        ("EFFR", "effr"),
        ("IORB", "iorb"),
        ("SOFR", "sofr"),
    ]:
        if source in df.columns:
            out[target] = aggregate_levels_to_monthly(df[source].rename(target), how="mean")
            meta.append({"column": target, "source_series": source, "transform": "monthly_mean"})

    out.index.name = "date"
    return out.sort_index(), pd.DataFrame(meta)


def build_monthly_outcomes_csv(
    raw_fred_csv: str | Path,
    *,
    out_csv: str | Path,
    metadata_csv: str | Path | None = None,
) -> dict[str, object]:
    raw = read_wide_time_series_csv(raw_fred_csv)
    outcomes, metadata = build_monthly_outcomes_from_fred_frame(raw)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(out_path, index_label="date")
    metadata_path = None
    if metadata_csv is not None:
        metadata_path = Path(metadata_csv)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(metadata_path, index=False)
    return {
        "status": "ok",
        "out": str(out_path),
        "metadata": str(metadata_path) if metadata_path else "",
        "columns": [col for col in outcomes.columns if outcomes[col].notna().any()],
        "rows": int(len(outcomes)),
    }
