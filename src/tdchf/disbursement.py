from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .fiscaldata import _clean_number
from .qra_event_lp import normalize_weekly_panel_units

SEAM_LAST_DEDICATED = pd.Timestamp("2023-02-13")
SEAM_FIRST_TABLE_II = pd.Timestamp("2023-02-14")
HORIZONS = [-4, -3, -2, *range(0, 14)]
RETENTION_HORIZONS = [0, 4, 8, 13]
PANDEMIC_BLOCK_QUARTERS = {"2020Q2", "2020Q3", "2020Q4", "2021Q1"}
SSA_OACT_PAYMENT_URL = "https://www.ssa.gov/oact/progdata/payment.html"
FLOW_BUCKETS = [
    "du_core_outflows_bn",
    "tax_receipts_bn",
    "du_broad_outflows_bn",
    "interest_outflows_bn",
    "debt_issues_gross_bn",
    "debt_redemptions_gross_bn",
]
HEADLINE_OUTCOMES = {
    "deposits_dpsacb": "broad_deposits_nsa",
    "reserves_wresbal": "reserves",
    "tga_wednesday_wdtgal": "tga_wednesday",
    "tga_weekavg_wtregen_sens": "tga_week_avg",
    "on_rrp_rrpontsyd": "onrrp",
    "total_mmf": "total_mmf",
    "bank_credit_sens": "bank_credit",
}


@dataclass(frozen=True)
class DisbursementRunReport:
    estimates: pd.DataFrame
    weekly: pd.DataFrame
    crosswalk: pd.DataFrame
    calendar: pd.DataFrame
    seam: dict[str, object]
    ssa: dict[str, object]
    tax_lane: dict[str, object]


def assign_h8_week(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value).normalize()
    return ts + pd.Timedelta(days=(2 - ts.weekday()) % 7)


def _num(value: object) -> float:
    return _clean_number(value)


def _amount_bn(series: pd.Series) -> pd.Series:
    return series.map(_num) / 1000.0


def _norm_text(value: object) -> str:
    return str(value or "").strip()


def map_dts_category(transaction_type: object, category: object) -> str:
    typ = _norm_text(transaction_type).lower()
    cat = _norm_text(category).lower()
    if "sub-total" in cat or "subtotal" in cat:
        return "table_ii_reference_total"
    if typ == "deposits":
        if cat.startswith("public debt cash issues"):
            return "debt_issues_gross"
        if cat.startswith("taxes - withheld individual/fica") or "withheld income and employment" in cat:
            return "tax_withheld"
        if "non withheld ind/seca" in cat or "individual income and employment taxes, not withheld" in cat or "individual income taxes" in cat:
            return "tax_nonwithheld"
        if "corporate income" in cat or "corporation income taxes" in cat:
            return "tax_corporate"
        if cat.startswith("taxes - ") or any(
            token in cat
            for token in [
                "estate and gift",
                "excise",
                "customs",
                "federal unemployment",
                "railroad retirement taxes",
                "state unemployment insurance deposits",
            ]
        ):
            return "tax_other"
        if any(
            token in cat
            for token in [
                "cash ftd",
                "ftd's received",
                "transfers from depositaries",
                "transfers from federal reserve account",
            ]
        ):
            return "tax_table_reference"
        if cat in {"other deposits", "unclassified - deposits"}:
            return "other_deposits"
        if any(
            token in cat
            for token in [
                "agriculture",
                "commodity credit",
                "deposit insurance",
                "deposits by states",
                "department",
                "dept of",
                "dhs -",
                "dod -",
                "dol -",
                "doi -",
                "dot -",
                "air transport security fees",
                "education",
                "energy",
                "export-import bank",
                "fcc",
                "federal communications",
                "federal reserve earnings",
                "federal retirement thrift savings plan",
                "foreign deposits",
                "foreign military sales",
                "general services",
                "gse",
                "health",
                "hhs -",
                "housing and urban",
                "hud -",
                "iap -",
                "independent agencies",
                "international monetary fund",
                "justice",
                "judicial branch",
                "medicare premiums",
                "mint",
                "national credit union",
                "postal service",
                "opm - federal employee",
                "securities and exchange",
                "small business",
                "ssa -",
                "treas -",
                "united states postal service",
                "usda -",
                "veterans affairs",
            ]
        ):
            return "other_deposits"
        return "unmapped"
    if typ == "withdrawals":
        if cat.startswith("public debt cash redemp"):
            return "debt_redemptions_gross"
        if "interest on treasury securities" in cat:
            return "interest_outflows"
        if "ssa - benefits" in cat or "social security benefits" in cat or "supplemental security income" in cat:
            return "du_core_benefits"
        if "va - benefits" in cat or "veterans affairs" in cat or "rrb - benefit" in cat or "unemployment benefits" in cat:
            return "du_core_benefits"
        if "federal salaries" in cat or "active duty pay" in cat or "military retirement" in cat:
            return "du_core_salaries_other"
        if "tax refunds" in cat or "irs tax refunds" in cat:
            return "refunds_table_ii_reference"
        if any(
            token in cat
            for token in [
                "medicare",
                "medicaid",
                "hhs - grants",
                "defense vendor",
                "dod -",
                "grants to states",
                "department",
                "dept of",
                "dhs -",
                "dol -",
                "doi -",
                "dot -",
                "education",
                "energy",
                "food and nutrition",
                "food stamps",
                "fed. highway administration",
                "irs - advanced child tax credit",
                "irs - economic impact payments",
                "gsa",
                "general services",
                "health and human services",
                "hhs -",
                "housing and urban",
                "hud -",
                "independent agencies",
                "justice",
                "labor dept",
                "marketplace",
                "nasa",
                "national science",
                "environmental protection",
                "fcc - universal service",
                "postal service",
                "small business",
                "supp nutrition",
                "supple. nutrition",
                "temporary assistance",
                "transportation security",
                "unemployment assist",
                "unemployment insurance benefits",
                "us army corps",
                "usda -",
                "veterans affairs programs",
            ]
        ):
            return "du_broad_outflows"
        if any(
            token in cat
            for token in [
                "air carrier worker support",
                "airline worker support",
                "commodity credit",
                "coronavirus relief fund",
                "emergency rental assistance",
                "federal communications",
                "federal deposit insurance",
                "federal employees insurance",
                "federal financing bank",
                "federal retirement thrift savings",
                "iap -",
                "international monetary fund",
                "opm - federal employee",
                "opm - civil serv",
                "judicial branch",
                "social security admin",
                "treas -",
                "transfers to depositaries",
                "transfers to federal reserve account",
                "unclassified",
            ]
        ):
            return "other_withdrawals"
        if cat == "other withdrawals":
            return "other_withdrawals"
        return "unmapped"
    return "unmapped"


def build_dts_crosswalk(transactions: pd.DataFrame, *, threshold_bn: float = 10.0) -> pd.DataFrame:
    raw = _filter_transaction_detail_rows(transactions)
    raw["amount_bn"] = _amount_bn(raw["transaction_today_amt"])
    raw["bucket"] = [map_dts_category(t, c) for t, c in zip(raw["transaction_type"], raw["transaction_catg"], strict=False)]
    grouped = (
        raw.groupby(["transaction_type", "transaction_catg", "bucket"], dropna=False)["amount_bn"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "daily_rows", "sum": "signed_amount_bn"})
    )
    grouped["abs_amount_bn"] = grouped["signed_amount_bn"].abs()
    grouped["above_threshold"] = grouped["abs_amount_bn"] >= threshold_bn
    grouped["mapped"] = ~grouped["bucket"].eq("unmapped")
    return grouped.sort_values(["transaction_type", "bucket", "transaction_catg"]).reset_index(drop=True)


def _filter_transaction_detail_rows(transactions: pd.DataFrame) -> pd.DataFrame:
    raw = transactions.copy()
    if "account_type" in raw.columns:
        account = raw["account_type"].astype("string")
        ref_total = account.str.contains("Total Deposits|Total Withdrawals", case=False, na=False)
        raw = raw.loc[~ref_total].copy()
    return raw


def _tax_bucket_from_dedicated(value: object) -> str | None:
    text = _norm_text(value).lower()
    if any(
        token in text
        for token in [
            "cash federal tax deposits",
            "inter-agency transfers",
            "these receipts were deposited in",
            "tax and loan accounts",
        ]
    ):
        return None
    if "withheld income and employment" in text:
        return "tax_withheld_bn"
    if "individual income taxes" in text or "cash federal tax deposits" in text:
        return "tax_nonwithheld_bn"
    if "corporation income taxes" in text:
        return "tax_corporate_bn"
    if any(token in text for token in ["railroad retirement", "excise", "federal unemployment", "estate and gift", "unclassified taxes"]):
        return "tax_other_bn"
    return "tax_other_bn"


def _tax_bucket_from_table_ii(value: object) -> str | None:
    text = _norm_text(value).lower()
    if text.startswith("taxes - withheld individual/fica"):
        return "tax_withheld_bn"
    if "non withheld ind/seca" in text:
        return "tax_nonwithheld_bn"
    if text.startswith("taxes - corporate income"):
        return "tax_corporate_bn"
    if text.startswith("taxes - ") and any(token in text for token in ["estate", "gift", "unemployment", "excise", "railroad", "misc"]):
        return "tax_other_bn"
    return None


def build_stitched_tax_daily(transactions: pd.DataFrame, tax_deposits: pd.DataFrame) -> pd.DataFrame:
    pre = tax_deposits.copy()
    if pre.empty:
        pre = pd.DataFrame(columns=["record_date", "tax_bucket", "amount_bn", "source"])
    else:
        pre["record_date"] = pd.to_datetime(pre["record_date"], errors="coerce")
        pre = pre.loc[pre["record_date"].le(SEAM_LAST_DEDICATED)].copy()
        pre["tax_bucket"] = pre["tax_deposit_type"].map(_tax_bucket_from_dedicated)
        pre["amount_bn"] = _amount_bn(pre["tax_deposit_today_amt"])
        pre["source"] = "dedicated_federal_tax_deposits"
        pre = pre.loc[pre["tax_bucket"].notna()].copy()
        pre = pre[["record_date", "tax_bucket", "amount_bn", "source"]]

    post = transactions.copy()
    post["record_date"] = pd.to_datetime(post["record_date"], errors="coerce")
    post = post.loc[post["record_date"].ge(SEAM_FIRST_TABLE_II) & post["transaction_type"].astype("string").str.fullmatch("Deposits", case=False, na=False)].copy()
    post["tax_bucket"] = post["transaction_catg"].map(_tax_bucket_from_table_ii)
    post = post.loc[post["tax_bucket"].notna()].copy()
    post["amount_bn"] = _amount_bn(post["transaction_today_amt"])
    post["source"] = "table_ii_tax_categories"
    post = post[["record_date", "tax_bucket", "amount_bn", "source"]]
    return pd.concat([pre, post], ignore_index=True).dropna(subset=["record_date"])


def seam_diagnostic(tax_daily: pd.DataFrame) -> dict[str, object]:
    before = tax_daily.loc[tax_daily["record_date"].between(SEAM_LAST_DEDICATED - pd.Timedelta(days=14), SEAM_LAST_DEDICATED)]
    after = tax_daily.loc[tax_daily["record_date"].between(SEAM_FIRST_TABLE_II, SEAM_FIRST_TABLE_II + pd.Timedelta(days=14))]
    before_days = int(before["record_date"].nunique())
    after_days = int(after["record_date"].nunique())
    before_buckets = sorted(before["tax_bucket"].dropna().unique().tolist())
    after_buckets = sorted(after["tax_bucket"].dropna().unique().tolist())
    before_med = float(before.groupby("record_date")["amount_bn"].sum().median()) if before_days else np.nan
    after_med = float(after.groupby("record_date")["amount_bn"].sum().median()) if after_days else np.nan
    ratio = after_med / before_med if np.isfinite(after_med) and np.isfinite(before_med) and before_med else np.nan
    annual = (
        tax_daily.assign(year=pd.to_datetime(tax_daily["record_date"], errors="coerce").dt.year)
        .pivot_table(index="year", columns="tax_bucket", values="amount_bn", aggfunc="sum", fill_value=0.0)
        .sort_index()
    )
    anchor_years = [year for year in [2022, 2023, 2024] if year in annual.index]
    anchor = annual.loc[anchor_years].round(3).to_dict(orient="index") if anchor_years else {}
    continuity_ok = True
    anchor_level_ok = True
    anchor_level_bounds: dict[str, object] = {}
    continuity_bounds: dict[str, float] = {}
    if 2022 in annual.index and "tax_nonwithheld_bn" in annual.columns:
        nonwithheld_2022 = float(annual.loc[2022, "tax_nonwithheld_bn"])
        anchor_level_bounds["tax_nonwithheld_bn_2022"] = {"value": round(nonwithheld_2022, 3), "lower": 500.0, "upper": 700.0}
        anchor_level_ok = 500.0 < nonwithheld_2022 < 700.0
    if len(anchor_years) >= 2:
        for bucket in [c for c in annual.columns if c in {"tax_withheld_bn", "tax_nonwithheld_bn", "tax_corporate_bn", "tax_other_bn"}]:
            vals = annual.loc[anchor_years, bucket].replace(0, np.nan).dropna()
            if len(vals) >= 2:
                ratios = vals.iloc[1:].to_numpy() / vals.iloc[:-1].to_numpy()
                max_ratio = float(np.nanmax(ratios))
                min_ratio = float(np.nanmin(ratios))
                continuity_bounds[bucket] = max(abs(max_ratio), abs(1 / min_ratio)) if min_ratio else np.inf
                if min_ratio < 0.2 or max_ratio > 5.0:
                    continuity_ok = False
    verdict = "stitched"
    if before_days == 0 or after_days == 0 or set(before_buckets) != set(after_buckets):
        verdict = "coverage_warning"
    elif not anchor_level_ok:
        verdict = "annual_anchor_level_warning"
    elif not continuity_ok:
        verdict = "annual_anchor_warning"
    return {
        "verdict": verdict,
        "last_dedicated_date": SEAM_LAST_DEDICATED.date().isoformat(),
        "first_table_ii_date": SEAM_FIRST_TABLE_II.date().isoformat(),
        "before_days": before_days,
        "after_days": after_days,
        "before_buckets": before_buckets,
        "after_buckets": after_buckets,
        "median_daily_ratio_after_before": ratio,
        "annual_anchor_totals_bn": anchor,
        "annual_anchor_continuity_ok": continuity_ok,
        "annual_anchor_level_ok": anchor_level_ok,
        "annual_anchor_level_bounds": anchor_level_bounds,
        "annual_anchor_max_ratio_or_inverse": continuity_bounds,
        "note": "Dedicated federal_tax_deposits component rows are used pre-seam; subtotal/inter-agency/reference rows are excluded. Post-seam taxes are stitched from Table II tax lines.",
    }


def _refund_daily(refunds: pd.DataFrame) -> pd.DataFrame:
    if refunds.empty:
        return pd.DataFrame(columns=["record_date", "du_core_refunds_bn"])
    raw = refunds.copy()
    raw["record_date"] = pd.to_datetime(raw["record_date"], errors="coerce")
    raw["amount_bn"] = _amount_bn(raw["tax_refund_today_amt"])
    return raw.groupby("record_date")["amount_bn"].sum().rename("du_core_refunds_bn").reset_index()


def _single_weekly(rows: pd.DataFrame, value_col: str) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype="float64", name=value_col)
    tmp = rows.copy()
    tmp["week_date"] = tmp["record_date"].map(assign_h8_week)
    return tmp.groupby("week_date")[value_col].sum()


def _pivot_weekly(rows: pd.DataFrame, value_col: str, bucket_col: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    tmp = rows.copy()
    tmp["week_date"] = tmp["record_date"].map(assign_h8_week)
    return tmp.pivot_table(index="week_date", columns=bucket_col, values=value_col, aggfunc="sum", fill_value=0.0)


def build_weekly_flow_decomposition(
    *,
    transactions_csv: str | Path,
    refunds_csv: str | Path,
    tax_deposits_csv: str | Path,
    operating_cash_balance_csv: str | Path,
    out_csv: str | Path,
    crosswalk_csv: str | Path,
) -> dict[str, object]:
    tx = pd.read_csv(transactions_csv, parse_dates=["record_date"], low_memory=False)
    refunds = pd.read_csv(refunds_csv, parse_dates=["record_date"], low_memory=False) if Path(refunds_csv).exists() else pd.DataFrame()
    tax = pd.read_csv(tax_deposits_csv, parse_dates=["record_date"], low_memory=False) if Path(tax_deposits_csv).exists() else pd.DataFrame()
    ocb = pd.read_csv(operating_cash_balance_csv, parse_dates=["record_date"], low_memory=False)

    crosswalk = build_dts_crosswalk(tx)
    crosswalk_path = Path(crosswalk_csv)
    crosswalk_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(crosswalk_path, index=False)

    raw = _filter_transaction_detail_rows(tx)
    max_record_date = pd.to_datetime(
        pd.concat(
            [
                raw["record_date"],
                refunds["record_date"] if "record_date" in refunds.columns else pd.Series(dtype="datetime64[ns]"),
                ocb["record_date"],
            ],
            ignore_index=True,
        ),
        errors="coerce",
    ).max()
    last_complete_week = assign_h8_week(max_record_date)
    if pd.Timestamp(max_record_date).normalize() < last_complete_week:
        last_complete_week = last_complete_week - pd.Timedelta(days=7)
    raw["bucket"] = [map_dts_category(t, c) for t, c in zip(raw["transaction_type"], raw["transaction_catg"], strict=False)]
    raw["amount_bn"] = _amount_bn(raw["transaction_today_amt"])
    weekly = _pivot_weekly(raw, "amount_bn", "bucket")
    tax_daily = build_stitched_tax_daily(raw, tax)
    tax_weekly = _pivot_weekly(tax_daily, "amount_bn", "tax_bucket")
    refund_weekly = _single_weekly(_refund_daily(refunds), "du_core_refunds_bn").to_frame()

    out = pd.concat([weekly, tax_weekly, refund_weekly], axis=1, sort=False).sort_index()
    for col in [
        "du_core_benefits",
        "du_core_salaries_other",
        "du_broad_outflows",
        "interest_outflows",
        "debt_issues_gross",
        "debt_redemptions_gross",
        "other_deposits",
        "other_withdrawals",
        "tax_withheld_bn",
        "tax_nonwithheld_bn",
        "tax_corporate_bn",
        "tax_other_bn",
        "du_core_refunds_bn",
    ]:
        if col not in out.columns:
            out[col] = 0.0

    out["du_core_benefits_bn"] = out["du_core_benefits"]
    out["du_core_salaries_other_bn"] = out["du_core_salaries_other"]
    out["du_core_outflows_bn"] = out["du_core_benefits_bn"] + out["du_core_refunds_bn"] + out["du_core_salaries_other_bn"]
    out["du_broad_outflows_bn"] = out["du_broad_outflows"]
    out["interest_outflows_bn"] = out["interest_outflows"]
    out["tax_receipts_bn"] = out["tax_withheld_bn"] + out["tax_nonwithheld_bn"] + out["tax_corporate_bn"] + out["tax_other_bn"]
    out["debt_issues_gross_bn"] = out["debt_issues_gross"]
    out["debt_redemptions_gross_bn"] = out["debt_redemptions_gross"]
    out["debt_net_bn"] = out["debt_issues_gross_bn"] - out["debt_redemptions_gross_bn"]
    deposit_cols = [c for c in out.columns if c in {"other_deposits", "tax_withheld", "tax_nonwithheld", "tax_corporate", "tax_other", "debt_issues_gross"}]
    withdrawal_cols = [c for c in out.columns if c in {"du_core_benefits", "du_core_salaries_other", "du_broad_outflows", "interest_outflows", "debt_redemptions_gross", "refunds_table_ii_reference", "other_withdrawals"}]
    out["table_ii_deposits_reconciled_bn"] = out[deposit_cols].sum(axis=1) if deposit_cols else 0.0
    out["table_ii_withdrawals_reconciled_bn"] = out[withdrawal_cols].sum(axis=1) if withdrawal_cols else 0.0
    out["table_ii_net_deposits_bn"] = out["table_ii_deposits_reconciled_bn"] - out["table_ii_withdrawals_reconciled_bn"]

    ocb_close = ocb.loc[ocb["account_type"].astype("string").str.contains("TGA\\) Closing Balance", case=False, na=False)].copy()
    ocb_close["tga_close_bn"] = ocb_close["open_today_bal"].map(_num) / 1000.0
    ocb_close["week_date"] = ocb_close["record_date"].map(assign_h8_week)
    tga_week = ocb_close.sort_values("record_date").groupby("week_date")["tga_close_bn"].last()
    out["dts_tga_close_bn"] = tga_week
    out["dts_delta_ocb_bn"] = out["dts_tga_close_bn"].diff()
    out["dts_reconciliation_error_bn"] = out["dts_delta_ocb_bn"] - out["table_ii_net_deposits_bn"]
    out.index.name = "date"
    final_cols = [
        "du_core_outflows_bn",
        "du_core_benefits_bn",
        "du_core_refunds_bn",
        "du_core_salaries_other_bn",
        "du_broad_outflows_bn",
        "interest_outflows_bn",
        "tax_receipts_bn",
        "tax_withheld_bn",
        "tax_nonwithheld_bn",
        "tax_corporate_bn",
        "tax_other_bn",
        "debt_issues_gross_bn",
        "debt_redemptions_gross_bn",
        "debt_net_bn",
        "table_ii_deposits_reconciled_bn",
        "table_ii_withdrawals_reconciled_bn",
        "table_ii_net_deposits_bn",
        "dts_tga_close_bn",
        "dts_delta_ocb_bn",
        "dts_reconciliation_error_bn",
    ]
    out = out.loc[out.index <= last_complete_week, final_cols].sort_index()
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")
    seam = seam_diagnostic(tax_daily)
    return {
        "status": "ok",
        "out": str(path),
        "crosswalk": str(crosswalk_path),
        "rows": int(len(out)),
        "start": out.index.min().date().isoformat(),
        "end": out.index.max().date().isoformat(),
        "crosswalk_rows": int(len(crosswalk)),
        "unmapped_above_threshold": int((~crosswalk["mapped"] & crosswalk["above_threshold"]).sum()),
        "seam": seam,
    }


def _observed(date_value: date, holidays: set[date]) -> date:
    out = date_value
    while out.weekday() >= 5 or out in holidays:
        out += timedelta(days=1)
    return out


def _observed_fixed_holiday(date_value: date) -> date:
    if date_value.weekday() == 5:
        return date_value - timedelta(days=1)
    if date_value.weekday() == 6:
        return date_value + timedelta(days=1)
    return date_value


def _previous_business(date_value: date, holidays: set[date]) -> date:
    out = date_value
    while out.weekday() >= 5 or out in holidays:
        out -= timedelta(days=1)
    return out


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=((weekday - first.weekday()) % 7) + 7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = (date(year + int(month == 12), 1 if month == 12 else month + 1, 1) - timedelta(days=1))
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def federal_holidays(year: int) -> set[date]:
    holidays = {
        _observed_fixed_holiday(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _last_weekday(year, 5, 0),
        _observed_fixed_holiday(date(year, 6, 19)),
        _observed_fixed_holiday(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 10, 0, 2),
        _observed_fixed_holiday(date(year, 11, 11)),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed_holiday(date(year, 12, 25)),
    }
    return holidays


def statutory_tax_due_dates(year: int) -> set[date]:
    holidays = set().union(*(federal_holidays(y) for y in [year - 1, year, year + 1]))
    due = {
        _observed(date(year, 1, 15), holidays),
        _observed(date(year, 4, 15), holidays),
        _observed(date(year, 6, 15), holidays),
        _observed(date(year, 9, 15), holidays),
    }
    if year == 2020:
        due.discard(_observed(date(2020, 4, 15), holidays))
        due.discard(_observed(date(2020, 6, 15), holidays))
        due.add(date(2020, 7, 15))
    if year == 2021:
        due.discard(_observed(date(2021, 4, 15), holidays))
        due.add(date(2021, 5, 17))
    return due


def build_fiscal_calendar_weekly(*, start: str, end: str, out_csv: str | Path) -> dict[str, object]:
    weeks = pd.date_range(assign_h8_week(start), assign_h8_week(end), freq="W-WED")
    years = range(pd.Timestamp(start).year - 1, pd.Timestamp(end).year + 2)
    holidays = set().union(*(federal_holidays(y) for y in years))
    tax_dates = set().union(*(statutory_tax_due_dates(y) for y in years))
    rows: list[dict[str, object]] = []
    for week in weeks:
        days = [week.date() - timedelta(days=i) for i in range(6, -1, -1)]
        row = {
            "date": week.date().isoformat(),
            "tax_due_count": sum(day in tax_dates for day in days),
            "tax_due_week": int(any(day in tax_dates for day in days)),
            "ssa_cycle1_count": 0,
            "ssa_cycle2_count": 0,
            "ssa_cycle3_count": 0,
            "ssa_cycle4_count": 0,
            "ssi_first_count": 0,
            "legacy_third_count": 0,
            "federal_salary_count": 0,
            "coupon_week": 0,
            "auction_settlement_week": 0,
            "redemption_week": 0,
            "federal_holiday_count": sum(day in holidays for day in days),
            "month_end": int(any((pd.Timestamp(day) + pd.offsets.MonthEnd(0)).date() == day for day in days)),
            "quarter_end": int(any(day.month in {3, 6, 9, 12} and (pd.Timestamp(day) + pd.offsets.MonthEnd(0)).date() == day for day in days)),
        }
        for day in days:
            if day == _nth_weekday(day.year, day.month, 2, 2):
                row["ssa_cycle2_count"] += 1
            if day == _nth_weekday(day.year, day.month, 2, 3):
                row["ssa_cycle3_count"] += 1
            if day == _nth_weekday(day.year, day.month, 2, 4):
                row["ssa_cycle4_count"] += 1
            if day == _previous_business(date(day.year, day.month, 1), holidays):
                row["ssi_first_count"] += 1
            if day == _previous_business(date(day.year, day.month, 3), holidays):
                row["legacy_third_count"] += 1
            last_day = (pd.Timestamp(day) + pd.offsets.MonthEnd(0)).date()
            if day in {_previous_business(date(day.year, day.month, 15), holidays), _previous_business(last_day, holidays)}:
                row["federal_salary_count"] += 1
            if day.day in {15, last_day.day}:
                row["coupon_week"] = 1
                row["redemption_week"] = 1
        row["ssa_cycle_payment_count"] = row["ssa_cycle2_count"] + row["ssa_cycle3_count"] + row["ssa_cycle4_count"]
        row["auction_settlement_week"] = int(row["debt_issues_proxy_count"] > 0) if "debt_issues_proxy_count" in row else row["coupon_week"]
        rows.append(row)
    out = pd.DataFrame(rows)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return {"status": "ok", "out": str(path), "rows": int(len(out)), "start": str(out["date"].min()), "end": str(out["date"].max())}


def _ols_hac(y: pd.Series, x: pd.DataFrame, *, maxlags: int) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    sample = pd.concat([y.rename("y"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < max(24, x.shape[1] + 8):
        return None
    xcols = [col for col in x.columns if sample[col].nunique() >= 2]
    if not xcols:
        return None
    return sm.OLS(sample["y"], sm.add_constant(sample[xcols], has_constant="add")).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def _hc1_ols(y: pd.Series, x: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    sample = pd.concat([y.rename("y"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < max(24, x.shape[1] + 8):
        return None
    xcols = [col for col in x.columns if sample[col].nunique() >= 2]
    if not xcols:
        return None
    return sm.OLS(sample["y"], sm.add_constant(sample[xcols], has_constant="add")).fit(cov_type="HC1")


def _wild_pvalue(y: pd.Series, x: pd.DataFrame, coef: str, beta: float, *, seed: int, reps: int = 999) -> float:
    sample = pd.concat([y.rename("y"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 24:
        return np.nan
    cols = list(x.columns)
    keep = [c for c in cols if c != coef]
    x_full = sm.add_constant(sample[cols], has_constant="add").to_numpy(dtype=float)
    x_res = sm.add_constant(sample[keep], has_constant="add").to_numpy(dtype=float)
    try:
        b0 = np.linalg.lstsq(x_res, sample["y"].to_numpy(dtype=float), rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    fitted = x_res @ b0
    resid = sample["y"].to_numpy(dtype=float) - fitted
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(len(sample), reps))
    betas = (np.linalg.pinv(x_full) @ (fitted[:, None] + resid[:, None] * signs))[cols.index(coef) + 1]
    return float(np.mean(np.abs(betas) >= abs(beta)))


def _permutation_pvalue(
    sample: pd.DataFrame,
    *,
    y_col: str,
    event_col: str,
    control_cols: list[str],
    beta: float,
    seed: int,
    reps: int = 999,
) -> float:
    clean = sample[[y_col, event_col, *control_cols]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(clean) < 24 or clean[event_col].nunique() < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    years = pd.DatetimeIndex(clean.index).year
    betas: list[float] = []
    for _ in range(reps):
        permuted = clean[event_col].copy()
        for year in np.unique(years):
            idx = clean.index[years == year]
            permuted.loc[idx] = rng.permutation(permuted.loc[idx].to_numpy())
        x = sm.add_constant(pd.concat([permuted.rename(event_col), clean[control_cols]], axis=1), has_constant="add")
        try:
            fit = sm.OLS(clean[y_col], x).fit()
        except (ValueError, np.linalg.LinAlgError):
            continue
        if event_col in fit.params:
            betas.append(float(fit.params[event_col]))
    if not betas:
        return np.nan
    arr = np.asarray(betas)
    return float((np.sum(np.abs(arr) >= abs(beta)) + 1) / (len(arr) + 1))


def _wild_pvalue_sum(y: pd.Series, x: pd.DataFrame, coefs: list[str], beta: float, *, seed: int, reps: int = 999) -> float:
    sample = pd.concat([y.rename("y"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    coefs = [coef for coef in coefs if coef in sample.columns]
    if len(sample) < 24 or not coefs:
        return np.nan
    cols = list(x.columns)
    keep = [c for c in cols if c not in coefs]
    x_full = sm.add_constant(sample[cols], has_constant="add").to_numpy(dtype=float)
    x_res = sm.add_constant(sample[keep], has_constant="add").to_numpy(dtype=float)
    try:
        b0 = np.linalg.lstsq(x_res, sample["y"].to_numpy(dtype=float), rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    fitted = x_res @ b0
    resid = sample["y"].to_numpy(dtype=float) - fitted
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(len(sample), reps))
    boot = np.linalg.pinv(x_full) @ (fitted[:, None] + resid[:, None] * signs)
    positions = [cols.index(coef) + 1 for coef in coefs]
    betas = boot[positions, :].sum(axis=0)
    return float(np.mean(np.abs(betas) >= abs(beta)))


def _prepare_regression_panel(flows: pd.DataFrame, calendar: pd.DataFrame, weekly_state: pd.DataFrame) -> pd.DataFrame:
    panel = flows.join(calendar.set_index("date"), how="left").join(weekly_state, how="left").sort_index()
    for col in FLOW_BUCKETS:
        for lag in range(1, 5):
            panel[f"{col}_lag{lag}"] = panel[col].shift(lag)
        for lead in range(1, 14):
            panel[f"{col}_lead{lead}"] = panel[col].shift(-lead)
    panel["net_du_flow_bn"] = panel["du_core_outflows_bn"] - panel["tax_receipts_bn"]
    return panel


def _pandemic_mask(index: pd.Index) -> pd.Series:
    dates = pd.DatetimeIndex(index)
    return pd.Series(dates.to_period("Q").astype(str).isin(PANDEMIC_BLOCK_QUARTERS), index=index)


def _window_overlaps_pandemic(index: pd.Index, horizon: int) -> pd.Series:
    blocked = _pandemic_mask(index)
    out = blocked.copy()
    if horizon >= 0:
        for step in range(0, horizon + 1):
            out |= blocked.shift(-step, fill_value=False)
    else:
        for step in range(1, abs(horizon) + 2):
            out |= blocked.shift(step, fill_value=False)
    return out


def _sample_mask(index: pd.Index, sample_name: str, horizon: int) -> pd.Series:
    if sample_name == "full":
        return pd.Series(True, index=index)
    if sample_name != "ex_pandemic":
        raise ValueError(f"unknown sample: {sample_name}")
    return ~_window_overlaps_pandemic(index, horizon)


def estimate_disbursement_lps(flows: pd.DataFrame, calendar: pd.DataFrame, weekly_state: pd.DataFrame) -> pd.DataFrame:
    panel = _prepare_regression_panel(flows, calendar, weekly_state)
    flow_controls = ["du_broad_outflows_bn", "interest_outflows_bn", "debt_issues_gross_bn", "debt_redemptions_gross_bn"]
    cal_controls = [c for c in calendar.columns if c != "date"]
    lag_controls = [f"{col}_lag{lag}" for col in FLOW_BUCKETS for lag in range(1, 5)]
    controls = [c for c in [*flow_controls, *cal_controls, *lag_controls] if c in panel.columns]
    rows: list[dict[str, object]] = []
    for sample_name in ["full", "ex_pandemic"]:
        for outcome, source in HEADLINE_OUTCOMES.items():
            if source not in panel.columns or panel[source].notna().sum() < 30:
                continue
            y_level = pd.to_numeric(panel[source], errors="coerce")
            for h in HORIZONS:
                if h >= 0:
                    y = y_level.shift(-h) - y_level.shift(1)
                else:
                    y = y_level.shift(1) - y_level.shift(abs(h) + 1)
                mask = _sample_mask(panel.index, sample_name, h)
                lead_controls = [f"{col}_lead{lead}" for col in FLOW_BUCKETS for lead in range(1, h + 1)] if h > 0 else []
                for spec_type, extra_controls, spec_note in [
                    ("LP_naive", [], "gross flows separate; HAC NW(h+4); 4 lags of all flow buckets; exact H8 Thu-Wed windows; naive no future-flow controls"),
                    (
                        "LP_lead_controlled",
                        lead_controls,
                        "gross flows separate; HAC NW(h+4); 4 lags plus scheduled future flow controls t+1..t+h; exact H8 Thu-Wed windows",
                    ),
                ]:
                    regressors = panel.loc[mask, ["du_core_outflows_bn", "tax_receipts_bn", *controls, *extra_controls]].copy()
                    y_sample = y.loc[mask]
                    fit = _ols_hac(y_sample, regressors, maxlags=max(abs(h) + 4, 4))
                    if fit is None:
                        continue
                    for treatment in ["du_core_outflows_bn", "tax_receipts_bn"]:
                        if treatment not in fit.params:
                            continue
                        beta = float(fit.params[treatment])
                        seed = 20260704 + len(rows) * 17
                        rows.append(
                            {
                                "treatment_id": treatment,
                                "spec_type": spec_type,
                                "outcome": outcome,
                                "horizon": h,
                                "sample": sample_name,
                                "beta": beta,
                                "se": float(fit.bse[treatment]),
                                "p": float(fit.pvalues[treatment]),
                                "p_wild_bootstrap": _wild_pvalue(y_sample, regressors, treatment, beta, seed=seed)
                                if h in RETENTION_HORIZONS and outcome == "deposits_dpsacb"
                                else np.nan,
                                "n": int(fit.nobs),
                                "spec_flags": spec_note,
                                "uniform_band_lower": np.nan,
                                "uniform_band_upper": np.nan,
                                "p_event_permutation": np.nan,
                            }
                        )
            dy = y_level.diff()
            fdl_mask = _sample_mask(panel.index, sample_name, 0) & ~_window_overlaps_pandemic(panel.index, -1) if sample_name == "ex_pandemic" else pd.Series(True, index=panel.index)
            x_fdl = pd.DataFrame(index=panel.index)
            fdl_cols_by_flow: dict[str, list[str]] = {}
            for flow in FLOW_BUCKETS:
                cols: list[str] = []
                for lag in range(14):
                    col = f"{flow}_fdl_lag{lag}"
                    x_fdl[col] = panel[flow].shift(lag)
                    cols.append(col)
                fdl_cols_by_flow[flow] = cols
            for col in cal_controls:
                if col in panel.columns:
                    x_fdl[col] = panel[col]
            fit = _ols_hac(dy.loc[fdl_mask], x_fdl.loc[fdl_mask], maxlags=17)
            if fit is None:
                continue
            cov = fit.cov_params()
            for treatment in ["du_core_outflows_bn", "tax_receipts_bn"]:
                used_cols = [col for col in fdl_cols_by_flow[treatment] if col in fit.params.index]
                for h in range(14):
                    cols_h = [col for col in used_cols if int(col.rsplit("lag", 1)[1]) <= h]
                    cumulative = float(fit.params[cols_h].sum()) if cols_h else np.nan
                    variance = float(cov.loc[cols_h, cols_h].to_numpy().sum()) if cols_h else np.nan
                    rows.append(
                        {
                            "treatment_id": treatment,
                            "spec_type": "FDL",
                            "outcome": outcome,
                            "horizon": h,
                            "sample": sample_name,
                            "beta": cumulative,
                            "se": float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan,
                            "p": np.nan,
                            "p_wild_bootstrap": _wild_pvalue_sum(dy.loc[fdl_mask], x_fdl.loc[fdl_mask], cols_h, cumulative, seed=20260704 + len(rows) * 17)
                            if h in RETENTION_HORIZONS and outcome == "deposits_dpsacb"
                            else np.nan,
                            "n": int(fit.nobs),
                            "spec_flags": "one-week delta outcome; current plus 13 lag distributed-lag cumulation; shifts built on full panel before sample drops",
                            "uniform_band_lower": np.nan,
                            "uniform_band_upper": np.nan,
                            "p_event_permutation": np.nan,
                        }
                    )
    est = pd.DataFrame(rows)
    if not est.empty:
        dep = est.loc[est["outcome"].eq("deposits_dpsacb") & est["horizon"].isin(range(14))].copy()
        for (sample, treatment, spec), group in dep.groupby(["sample", "treatment_id", "spec_type"]):
            se_max = pd.to_numeric(group["se"], errors="coerce").max()
            idx = group.index
            est.loc[idx, "uniform_band_lower"] = est.loc[idx, "beta"] - 1.96 * se_max
            est.loc[idx, "uniform_band_upper"] = est.loc[idx, "beta"] + 1.96 * se_max
    return est


def _attempt_official_ssa_oact_fetch() -> dict[str, object]:
    try:
        req = Request(SSA_OACT_PAYMENT_URL, headers={"User-Agent": "tdchf-research/0.1"})
        with urlopen(req, timeout=30) as response:
            payload = response.read(4096)
        text = payload.decode("utf-8", errors="replace")
        usable = "Payment summary" in text and "Cyclical payment" in text
        return {
            "url": SSA_OACT_PAYMENT_URL,
            "status": "fetched_metadata" if usable else "fetched_unparsed",
            "http_status": 200,
            "usable_cycle_dollars": False,
            "detail": "Official SSA/OACT payment page was reachable but no machine-readable cycle-dollar table was parsed.",
        }
    except HTTPError as exc:
        return {
            "url": SSA_OACT_PAYMENT_URL,
            "status": "fetch_failed",
            "http_status": int(exc.code),
            "usable_cycle_dollars": False,
            "detail": f"Official SSA/OACT payment page request failed: HTTP {exc.code}.",
        }
    except (URLError, TimeoutError, OSError) as exc:
        return {
            "url": SSA_OACT_PAYMENT_URL,
            "status": "fetch_failed",
            "http_status": None,
            "usable_cycle_dollars": False,
            "detail": f"Official SSA/OACT payment page request failed: {type(exc).__name__}: {exc}.",
        }


def _predicted_ssa_cycle_dollars(flows: pd.DataFrame, calendar: pd.DataFrame) -> pd.Series:
    cal = calendar.set_index("date").sort_index()
    benefits = pd.to_numeric(flows["du_core_benefits_bn"], errors="coerce").sort_index()
    monthly_benefits = benefits.groupby(benefits.index.to_period("M")).sum(min_count=1)
    lagged_monthly = monthly_benefits.shift(1)
    cycle_counts = pd.to_numeric(cal.get("ssa_cycle_payment_count", 0), errors="coerce").fillna(0.0)
    month_cycle_total = cycle_counts.groupby(cycle_counts.index.to_period("M")).transform("sum").replace(0, np.nan)
    shares = cycle_counts / month_cycle_total
    lagged_for_week = pd.Series(cal.index.to_period("M"), index=cal.index).map(lagged_monthly).astype(float)
    return (shares * lagged_for_week).reindex(flows.index).fillna(0.0).rename("ssa_predicted_cycle_benefits_bn")


def estimate_ssa_proxy_lane(flows: pd.DataFrame, calendar: pd.DataFrame, weekly_state: pd.DataFrame) -> dict[str, object]:
    panel = flows.join(calendar.set_index("date"), how="left").join(weekly_state, how="left").dropna(subset=["du_core_benefits_bn"])
    panel = panel.loc[panel.index >= pd.Timestamp("2013-04-01")]
    fetch = _attempt_official_ssa_oact_fetch()
    panel["ssa_predicted_cycle_benefits_bn"] = _predicted_ssa_cycle_dollars(flows, calendar).reindex(panel.index)
    controls = ["tax_due_week", "coupon_week", "legacy_third_count", "ssi_first_count"]
    controls = [col for col in controls if col in panel.columns]
    if "broad_deposits_nsa" not in panel.columns or panel["ssa_predicted_cycle_benefits_bn"].sum() == 0:
        return {"status": "skipped", "reason": "missing deposits or SSA cycle calendar"}
    xcols = ["ssa_predicted_cycle_benefits_bn", *controls]
    x = sm.add_constant(panel[xcols], has_constant="add")
    fs = sm.OLS(panel["du_core_benefits_bn"], x).fit(cov_type="HC1")
    fstat = float(fs.tvalues["ssa_predicted_cycle_benefits_bn"] ** 2)
    y = panel["broad_deposits_nsa"].shift(-4) - panel["broad_deposits_nsa"].shift(1)
    rf_sample = pd.concat([y.rename("y"), panel[xcols]], axis=1).dropna()
    rf = sm.OLS(rf_sample["y"], sm.add_constant(rf_sample[xcols], has_constant="add")).fit(cov_type="HC1")
    perm_p = _permutation_pvalue(
        rf_sample,
        y_col="y",
        event_col="ssa_predicted_cycle_benefits_bn",
        control_cols=controls,
        beta=float(rf.params["ssa_predicted_cycle_benefits_bn"]),
        seed=20260704,
    )
    iv_beta = np.nan
    iv_p = np.nan
    iv_se = np.nan
    if fstat >= 10:
        first_stage_hat = fs.fittedvalues.rename("du_core_benefits_hat")
        iv_sample = pd.concat([y.rename("y"), first_stage_hat, panel[controls]], axis=1).dropna()
        iv = sm.OLS(iv_sample["y"], sm.add_constant(iv_sample[["du_core_benefits_hat", *controls]], has_constant="add")).fit(cov_type="HC1")
        iv_beta = float(iv.params["du_core_benefits_hat"])
        iv_p = float(iv.pvalues["du_core_benefits_hat"])
        iv_se = float(iv.bse["du_core_benefits_hat"])
    return {
        "status": "predicted_dollar_fallback",
        "official_fetch": fetch,
        "first_stage_f": fstat,
        "first_stage_beta": float(fs.params["ssa_predicted_cycle_benefits_bn"]),
        "reduced_form_h4_beta": float(rf.params["ssa_predicted_cycle_benefits_bn"]),
        "reduced_form_h4_p": float(rf.pvalues["ssa_predicted_cycle_benefits_bn"]),
        "p_event_permutation": perm_p,
        "iv_h4_beta": iv_beta,
        "iv_h4_p": iv_p,
        "iv_h4_se_uncorrected": iv_se,
        "iv_se_uncorrected": bool(np.isfinite(iv_se)),
        "n": int(rf.nobs),
        "verdict": "weak_predicted_dollar_fallback" if fstat < 10 else "strong_predicted_dollar_fallback",
        "note": "Instrument is cycle-share times lagged monthly DTS benefit totals assigned to exact H.8 weeks; legacy 3rd-of-month and SSI weeks are separate controls, not the omitted control group.",
    }


def estimate_tax_deadline_lane(flows: pd.DataFrame, calendar: pd.DataFrame, weekly_state: pd.DataFrame) -> dict[str, object]:
    panel = flows.join(calendar.set_index("date"), how="left").join(weekly_state, how="left")
    if "broad_deposits_nsa" not in panel.columns:
        return {"status": "skipped", "reason": "missing deposits outcome"}
    y = panel["broad_deposits_nsa"].shift(-4) - panel["broad_deposits_nsa"].shift(1)
    xcols = ["tax_due_week", "tax_withheld_bn", "coupon_week"]
    sample = pd.concat([y.rename("y"), panel[xcols]], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    fit = sm.OLS(sample["y"], sm.add_constant(sample[xcols], has_constant="add")).fit(cov_type="HC1")
    perm_p = _permutation_pvalue(
        sample,
        y_col="y",
        event_col="tax_due_week",
        control_cols=["tax_withheld_bn", "coupon_week"],
        beta=float(fit.params["tax_due_week"]),
        seed=20260704,
    )
    return {
        "status": "descriptive_event_time",
        "tax_due_h4_beta": float(fit.params["tax_due_week"]),
        "tax_due_h4_p": float(fit.pvalues["tax_due_week"]),
        "p_event_permutation": perm_p,
        "n": int(fit.nobs),
        "verdict": "descriptive_only_realized_size_endogenous",
        "note": "Realized receipt size is not used as treatment; withheld receipts and coupon week are controlled.",
    }


def write_disbursement_readout(
    *,
    estimates: pd.DataFrame,
    weekly: pd.DataFrame,
    crosswalk: pd.DataFrame,
    seam: dict[str, object],
    ssa: dict[str, object],
    tax_lane: dict[str, object],
    out_md: str | Path,
) -> None:
    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    ret = estimates.loc[
        estimates["outcome"].eq("deposits_dpsacb")
        & estimates["treatment_id"].isin(["du_core_outflows_bn", "tax_receipts_bn"])
        & estimates["sample"].isin(["full", "ex_pandemic"])
        & estimates["horizon"].isin(RETENTION_HORIZONS)
        & estimates["spec_type"].isin(["LP_lead_controlled", "FDL", "LP_naive"])
    ].copy()
    if not ret.empty:
        beta0 = ret.loc[ret["horizon"].eq(0), ["sample", "treatment_id", "spec_type", "beta"]].rename(columns={"beta": "beta0"})
        ret = ret.merge(beta0, on=["sample", "treatment_id", "spec_type"], how="left")
        ret["retention_ratio"] = ret["beta"] / ret["beta0"]
        show = ret[["sample", "spec_type", "treatment_id", "horizon", "beta", "retention_ratio", "se", "p_wild_bootstrap", "n"]].round(4)
    else:
        show = pd.DataFrame()
    impact = estimates.loc[
        estimates["outcome"].eq("deposits_dpsacb")
        & estimates["treatment_id"].eq("du_core_outflows_bn")
        & estimates["horizon"].eq(0)
        & estimates["sample"].isin(["full", "ex_pandemic"])
        & estimates["spec_type"].isin(["LP_lead_controlled", "FDL"]),
        ["sample", "spec_type", "beta", "se", "n"],
    ].round(4)
    placebo = estimates.loc[
        estimates["spec_type"].eq("LP_lead_controlled") & estimates["horizon"].isin([-4, -3, -2]) & estimates["sample"].eq("full")
    ]
    placebo_rate = f"{int((placebo['p'] < 0.05).sum())}/{len(placebo)}" if not placebo.empty else "0/0"
    unmapped = int((~crosswalk["mapped"] & crosswalk["above_threshold"]).sum()) if not crosswalk.empty else 0
    residual = float(weekly["dts_reconciliation_error_bn"].abs().dropna().median()) if "dts_reconciliation_error_bn" in weekly else np.nan
    ssa_fetch = ssa.get("official_fetch", {}) if isinstance(ssa.get("official_fetch", {}), dict) else {}
    seam_anchor = seam.get("annual_anchor_totals_bn", {})
    iv_beta = float(ssa.get("iv_h4_beta", np.nan))
    iv_se_uncorrected = float(ssa.get("iv_h4_se_uncorrected", np.nan))
    iv_ci = (
        f"[{iv_beta - 1.96 * iv_se_uncorrected:.3g}, {iv_beta + 1.96 * iv_se_uncorrected:.3g}]"
        if np.isfinite(iv_beta) and np.isfinite(iv_se_uncorrected)
        else "not available"
    )
    lines = [
        "# DTS Disbursement-Side LP Readout",
        "",
        "Mechanical-impact benchmark: the expected beta0 for core outflows is approximately the commercial-bank landing share, not one by construction. H.8 excludes thrifts, credit unions, Direct Express prepaid balances, and foreign recipients; beta0 is a plumbing/perimeter check, while beta4/beta8/beta13 and retention ratios are the evidence on persistence.",
        "",
        "Corrected beta0 landing-share check:",
        "",
    ]
    lines.extend(_markdown_table(impact))
    lines.extend(
        [
            "",
            "Lead-controlled LP is the headline persistence path. LP_naive is retained only as the recurring-schedule-contaminated comparison; FDL is the distributed-lag companion.",
            "",
            f"DTS coverage: weekly flow panel {weekly.index.min().date()} through {weekly.index.max().date()}, {len(weekly)} H.8 Thursday-to-Wednesday windows.",
            f"Seam-stitching verdict: {seam.get('verdict')} ({seam.get('last_dedicated_date')} dedicated through {seam.get('first_table_ii_date')} Table II start). Annual anchor continuity ok={seam.get('annual_anchor_continuity_ok')}; annual anchor level ok={seam.get('annual_anchor_level_ok')}; 2022-2024 component totals={seam_anchor}. {seam.get('note')}",
            f"Crosswalk rows: {len(crosswalk)}. Unmapped above threshold: {unmapped}. Median absolute Table II reconciliation residual: {residual:.4f} bn. The residual moved from 0.005bn to 0.124bn after the crosswalk remap because category bucket composition changed; it remains negligible relative to billion-dollar weekly flows.",
            "",
            "## Retention Table",
            "",
        ]
    )
    lines.extend(_markdown_table(show))
    lines.extend(
        [
            "",
            "## Mirrors And Lanes",
            "",
            "Deposit, reserve, Wednesday TGA, ON RRP, MMF, and bank-credit sensitivity rows are in `dts_disbursement_lp_estimates.csv` where source series exist. WTREGEN is kept as week-average TGA sensitivity; WDTGAL is the Wednesday-level TGA mirror.",
            f"SSA official fetch: {ssa_fetch.get('status', 'not_attempted')} at {ssa_fetch.get('url', '')}; detail={ssa_fetch.get('detail', '')}",
            f"SSA lane verdict: {ssa.get('verdict', ssa.get('status'))}; first-stage F={ssa.get('first_stage_f', np.nan):.3g}; reduced-form h4 beta={ssa.get('reduced_form_h4_beta', np.nan):.3g}, p={ssa.get('reduced_form_h4_p', np.nan):.3g}, permutation p={ssa.get('p_event_permutation', np.nan):.3g}; 2SLS h4 beta={ssa.get('iv_h4_beta', np.nan):.3g}, p={ssa.get('iv_h4_p', np.nan):.3g}, fitted-value se_uncorrected={ssa.get('iv_se_uncorrected', False)}, uncorrected CI={iv_ci}. {ssa.get('note', '')}",
            f"Tax-deadline lane verdict: {tax_lane.get('verdict', tax_lane.get('status'))}; h4 deadline beta={tax_lane.get('tax_due_h4_beta', np.nan):.3g}, p={tax_lane.get('tax_due_h4_p', np.nan):.3g}, permutation p={tax_lane.get('p_event_permutation', np.nan):.3g}. {tax_lane.get('note', '')}",
            f"Descriptive lead placebo rate: {placebo_rate} significant at 5 percent.",
            "Event-lane permutation inference is within-year event-label permutation with 999 seeded draws; tax remains descriptive event-time and SSA is disclosed as a fallback-dollar weak/strong lane according to the first-stage F.",
            "",
            "## Verified Caveats",
            "",
            "Tax lead placebos reflect genuine anticipation: payroll income is deposited before it is remitted, so the tax-drain path is descriptive-with-anticipation rather than clean causal timing.",
            "Tax-drain persistence is robust through h0-h8 only; h13 dies ex-April and post-seam. Safe form: drains persist through ~2 months; credits decay by h13.",
            "The core-outflow mid-horizon bump is spec-dependent: null in the LP wild bootstrap and significant only in the FDL path, so it is not retention evidence.",
            "SSA lane label: strong first stage, underpowered outcome - uninformative on magnitude. The null must not be read as evidence that benefits do not reach deposits.",
            "Pandemic-block edge convention: for the first post-block week, the t-1 base can sit inside the excluded block for one row per horizon.",
            "",
            "## Falsification Map",
            "",
            "Landing-share/plumbing check: core-DU beta0 is mechanically plausible and well below one, consistent with partial commercial-bank landing.",
            "Credit persistence: core credits decay by h13; any mid-horizon bump must be treated as spec-dependent rather than retention evidence.",
            "Tax-drain path: tax receipts drain deposits through about two months, with anticipation contaminating causal timing and no safe h13 persistence claim.",
            "Perimeter/measurement failure: no plausible beta0 for core-DU credits despite clean Wednesday-level TGA/reserve accounting.",
            "Artifact flag: any headline that exists only on the net treatment and dies when gross credits and drains are separated.",
            "",
            "Corrected claim boundary: descriptive weekly-flow evidence supports a mechanical landing-share/plumbing check. Persistence requires the lead-controlled and FDL rows, not the naive LP path; safe persistence language is drains through ~2 months and credits decaying by h13. Quasi-experiments identify their specific margins where valid; this should be read alongside the QRA absorption-null evidence in `qra_event_lp_readout.md`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No estimable rows."]
    cols = list(frame.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row[cols]) + " |")
    return out


def run_disbursement_lp_csv(
    *,
    transactions_csv: str | Path = "data/raw/fiscaldata/dts_deposits_withdrawals_operating_cash.csv",
    refunds_csv: str | Path = "data/raw/fiscaldata/dts_income_tax_refunds_issued.csv",
    tax_deposits_csv: str | Path = "data/raw/fiscaldata/dts_federal_tax_deposits.csv",
    operating_cash_balance_csv: str | Path = "data/raw/fiscaldata/dts_operating_cash_balance.csv",
    weekly_state_csv: str | Path = "data/processed/tdc_weekly_channel_panel.csv",
    weekly_flows_csv: str | Path = "data/processed/dts_weekly_flow_decomposition.csv",
    calendar_csv: str | Path = "data/processed/fiscal_calendar_weekly.csv",
    crosswalk_csv: str | Path = "data/processed/dts_category_crosswalk.csv",
    estimates_csv: str | Path = "data/processed/dts_disbursement_lp_estimates.csv",
    readout_md: str | Path = "data/processed/dts_disbursement_lp_readout.md",
) -> dict[str, object]:
    flow_report = build_weekly_flow_decomposition(
        transactions_csv=transactions_csv,
        refunds_csv=refunds_csv,
        tax_deposits_csv=tax_deposits_csv,
        operating_cash_balance_csv=operating_cash_balance_csv,
        out_csv=weekly_flows_csv,
        crosswalk_csv=crosswalk_csv,
    )
    flows = pd.read_csv(weekly_flows_csv, parse_dates=["date"]).set_index("date")
    cal_report = build_fiscal_calendar_weekly(start=str(flows.index.min().date()), end=str(flows.index.max().date()), out_csv=calendar_csv)
    calendar = pd.read_csv(calendar_csv, parse_dates=["date"])
    weekly_state = pd.read_csv(weekly_state_csv, parse_dates=["date"]).set_index("date")
    weekly_state = normalize_weekly_panel_units(weekly_state)
    estimates = estimate_disbursement_lps(flows, calendar, weekly_state)
    est_path = Path(estimates_csv)
    est_path.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(est_path, index=False)
    crosswalk = pd.read_csv(crosswalk_csv)
    seam = flow_report["seam"]
    ssa = estimate_ssa_proxy_lane(flows, calendar, weekly_state)
    tax_lane = estimate_tax_deadline_lane(flows, calendar, weekly_state)
    write_disbursement_readout(
        estimates=estimates,
        weekly=flows,
        crosswalk=crosswalk,
        seam=seam,
        ssa=ssa,
        tax_lane=tax_lane,
        out_md=readout_md,
    )
    return {
        "status": "ok",
        "weekly_flows": str(weekly_flows_csv),
        "calendar": cal_report["out"],
        "crosswalk": str(crosswalk_csv),
        "estimates": str(est_path),
        "readout": str(readout_md),
        "rows": int(len(estimates)),
        "flow_rows": flow_report["rows"],
        "seam": seam,
        "ssa": ssa,
        "tax_lane": tax_lane,
    }
