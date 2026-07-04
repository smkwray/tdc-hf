from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .indicators import read_wide_time_series_csv
from .shocks import expanding_window_residual


def _numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _security_bucket(security_type: object, security_term: object, cmb: object) -> str:
    type_value = str(security_type or "").strip().lower()
    term_value = str(security_term or "").strip().lower()
    cmb_value = str(cmb or "").strip().lower()
    if cmb_value in {"yes", "true", "1"} or "cash management" in term_value:
        return "cmb"
    if "bill" in type_value:
        return "bill"
    if "note" in type_value or "frn" in type_value:
        return "note"
    if "bond" in type_value:
        return "bond"
    if "tips" in type_value or "inflation" in type_value:
        return "tips"
    return type_value or "all"


def _term_months(value: object) -> float:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return float("nan")
    parts = re.findall(r"(\d+)\s*-\s*(day|week|month|year)|(\d+)\s*(day|week|month|year)", text)
    total = 0.0
    for first_num, first_unit, second_num, second_unit in parts:
        num = float(first_num or second_num)
        unit = first_unit or second_unit
        if unit == "day":
            total += num / 30.0
        elif unit == "week":
            total += num / 4.345
        elif unit == "month":
            total += num
        elif unit == "year":
            total += num * 12.0
    return total if total else float("nan")


def _add_auction_design(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    out["lag_auction_amount"] = out["auction_amount"].shift(1)
    out["trail3_auction_amount"] = out["auction_amount"].shift(1).rolling(3, min_periods=2).mean()
    out["trail6_auction_amount"] = out["auction_amount"].shift(1).rolling(6, min_periods=3).mean()
    out["trend"] = np.arange(len(out), dtype=float)
    out["month"] = out.index.month.astype(float)
    out["quarter_refunding_month"] = out.index.month.isin([2, 5, 8, 11]).astype(float)
    return out


def _fit_group_shock(group: pd.DataFrame, *, predictors: list[str], min_train_obs: int) -> pd.DataFrame:
    available = [column for column in predictors if column in group.columns]
    if len(group[["auction_amount", *available]].dropna()) >= min_train_obs + 3:
        return expanding_window_residual(
            group,
            target="auction_amount",
            predictors=available,
            min_train_obs=min_train_obs,
            residual_column="auction_size_surprise",
            fitted_column="auction_size_expected",
            z_column="auction_size_surprise_z",
        )
    out = group.copy()
    out["auction_size_expected"] = pd.NA
    out["auction_size_surprise"] = pd.NA
    out["auction_size_surprise_z"] = pd.NA
    return out


def build_auction_size_shock(
    auction_csv: str | Path,
    *,
    out_csv: str | Path,
    date_column: str = "issue_date",
    amount_column: str = "offering_amt",
    tenor_column: str = "security_term",
    min_train_obs: int = 12,
) -> dict[str, object]:
    df = pd.read_csv(auction_csv, parse_dates=[date_column], low_memory=False)
    missing = [col for col in [date_column, amount_column] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing auction columns: {missing}")
    if tenor_column not in df.columns:
        df[tenor_column] = "all"
    work = df[[date_column, amount_column, tenor_column]].copy()
    work[amount_column] = pd.to_numeric(work[amount_column], errors="coerce")
    work["security_type"] = df["security_type"] if "security_type" in df.columns else ""
    work["cash_management_bill_cmb"] = df["cash_management_bill_cmb"] if "cash_management_bill_cmb" in df.columns else ""
    work["reopening"] = df["reopening"] if "reopening" in df.columns else ""
    work["term_months"] = df[tenor_column].map(_term_months)
    work["soma_accepted"] = _numeric_column(df, "soma_accepted")
    work["total_accepted"] = _numeric_column(df, "total_accepted")
    work = work.dropna(subset=[date_column, amount_column])
    work["month"] = work[date_column].dt.to_period("M").dt.to_timestamp("M")
    work["security_bucket"] = [
        _security_bucket(sec_type, term, cmb)
        for sec_type, term, cmb in zip(work["security_type"], work[tenor_column], work["cash_management_bill_cmb"])
    ]
    work["is_reopening"] = work["reopening"].astype("string").str.lower().isin(["yes", "true", "1"]).astype(float)
    work["private_auction_amount"] = (work[amount_column] - work["soma_accepted"]).clip(lower=0)
    monthly = (
        work.groupby(["month", "security_bucket", tenor_column], as_index=False)
        .agg(
            auction_amount=(amount_column, "sum"),
            private_auction_amount=("private_auction_amount", "sum"),
            mean_term_months=("term_months", "mean"),
            reopening_share=("is_reopening", "mean"),
        )
        .rename(columns={"month": "date", amount_column: "auction_amount"})
        .sort_values(["date", "security_bucket", tenor_column])
    )

    shock_frames: list[pd.DataFrame] = []
    predictors = [
        "lag_auction_amount",
        "trail3_auction_amount",
        "trail6_auction_amount",
        "trend",
        "month",
        "quarter_refunding_month",
        "mean_term_months",
        "reopening_share",
    ]
    for (bucket, tenor), group in monthly.groupby(["security_bucket", tenor_column]):
        group = group.set_index("date").sort_index()
        group = _add_auction_design(group)
        shocked = _fit_group_shock(group, predictors=predictors, min_train_obs=min_train_obs)
        shocked["security_bucket"] = bucket
        shocked[tenor_column] = tenor
        shock_frames.append(shocked.reset_index())

    by_tenor = pd.concat(shock_frames, ignore_index=True, sort=False)
    aggregate = by_tenor.groupby("date", as_index=False).agg(
        auction_amount=("auction_amount", "sum"),
        private_auction_amount=("private_auction_amount", "sum"),
        auction_size_expected=("auction_size_expected", lambda x: x.sum(min_count=1)),
        auction_size_surprise=("auction_size_surprise", lambda x: x.sum(min_count=1)),
    )
    bucket_surprises = (
        by_tenor.pivot_table(index="date", columns="security_bucket", values="auction_size_surprise", aggfunc=lambda x: x.sum(min_count=1))
        .add_prefix("auction_")
        .add_suffix("_surprise")
    )
    aggregate = aggregate.set_index("date").join(bucket_surprises, how="left").sort_index()
    coupon_cols = [col for col in ["auction_note_surprise", "auction_bond_surprise", "auction_tips_surprise"] if col in aggregate.columns]
    if coupon_cols:
        aggregate["auction_coupon_surprise"] = aggregate[coupon_cols].sum(axis=1, min_count=1)
    sd = aggregate["auction_size_surprise"].std(skipna=True)
    aggregate["auction_size_surprise_z"] = aggregate["auction_size_surprise"] / sd if pd.notna(sd) and sd else pd.NA
    for column in [col for col in aggregate.columns if col.startswith("auction_") and col.endswith("_surprise")]:
        col_sd = aggregate[column].std(skipna=True)
        aggregate[f"{column}_z"] = aggregate[column] / col_sd if pd.notna(col_sd) and col_sd else pd.NA
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(path, index_label="date")
    return {
        "status": "ok",
        "out": str(path),
        "rows": int(len(aggregate)),
        "tenors": int(by_tenor[tenor_column].nunique()),
        "security_buckets": sorted(str(value) for value in by_tenor["security_bucket"].dropna().unique()),
    }


def build_tga_rebuild_shock_csv(
    data_csv: str | Path,
    *,
    target: str = "minus_toc",
    predictors: list[str] | None = None,
    out_csv: str | Path,
    min_train_obs: int = 24,
) -> dict[str, object]:
    predictors = predictors or []
    df = read_wide_time_series_csv(data_csv)
    out = expanding_window_residual(
        df,
        target=target,
        predictors=predictors,
        min_train_obs=min_train_obs,
        residual_column="tga_rebuild_surprise",
        fitted_column="tga_rebuild_expected",
        z_column="tga_rebuild_surprise_z",
    )
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")
    return {"status": "ok", "out": str(path), "rows": int(len(out)), "target": target, "predictors": predictors}


def build_shock_bundle_csv(
    inputs: list[str | Path],
    *,
    out_csv: str | Path,
    columns: list[str] | None = None,
) -> dict[str, object]:
    frames = [read_wide_time_series_csv(path) for path in inputs]
    if not frames:
        raise ValueError("At least one shock input is required")
    bundle = pd.concat(frames, axis=1, sort=False)
    bundle = bundle.loc[:, ~bundle.columns.duplicated(keep="last")].sort_index()
    if columns:
        missing = [column for column in columns if column not in bundle.columns]
        if missing:
            raise KeyError(f"Missing requested shock bundle columns: {missing}")
        bundle = bundle[columns]
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle.to_csv(path, index_label="date")
    return {"status": "ok", "out": str(path), "rows": int(len(bundle)), "columns": list(bundle.columns)}
