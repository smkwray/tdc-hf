from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from .indicators import aggregate_flows_to_monthly, level_change_to_monthly_flow, positive_only

FISCALDATA_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
DTS_ACCOUNTING_BASE = f"{FISCALDATA_BASE}/v1/accounting/dts"
DTS_OPERATING_CASH_BALANCE = "operating_cash_balance"
DTS_DEPOSITS_WITHDRAWALS = "deposits_withdrawals_operating_cash"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean_number(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text.lower() in {"", "null", "none", "nan", "."}:
        return float("nan")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace("$", "").replace(",", "")
    out = pd.to_numeric(text, errors="coerce")
    if pd.isna(out):
        return float("nan")
    number = float(out)
    return -number if negative else number


def fiscaldata_url(
    endpoint: str,
    *,
    fields: list[str] | None = None,
    filters: list[str] | None = None,
    sort: str | None = None,
    page_number: int = 1,
    page_size: int = 10_000,
) -> str:
    params: dict[str, object] = {"page[number]": page_number, "page[size]": page_size}
    if fields:
        params["fields"] = ",".join(fields)
    if filters:
        params["filter"] = ",".join(filters)
    if sort:
        params["sort"] = sort
    return f"{DTS_ACCOUNTING_BASE}/{endpoint}?{urlencode(params)}"


def fetch_fiscaldata_page(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_fiscaldata_rows(
    endpoint: str,
    *,
    fields: list[str] | None = None,
    filters: list[str] | None = None,
    sort: str | None = None,
    page_size: int = 10_000,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    first_meta: dict[str, object] = {}
    page_number = 1
    while True:
        url = fiscaldata_url(
            endpoint,
            fields=fields,
            filters=filters,
            sort=sort,
            page_number=page_number,
            page_size=page_size,
        )
        payload = fetch_fiscaldata_page(url)
        page_rows = payload.get("data") or []
        if not isinstance(page_rows, list):
            raise ValueError("FiscalData response `data` must be a list")
        if page_number == 1:
            first_meta = dict(payload.get("meta") or {})
            first_meta["first_url"] = url
        rows.extend(page_rows)
        total_pages = int((payload.get("meta") or {}).get("total-pages") or page_number)
        if page_number >= total_pages or not page_rows:
            break
        page_number += 1
    first_meta["pages_downloaded"] = page_number
    return rows, first_meta


def write_fiscaldata_csv(
    endpoint: str,
    *,
    out_csv: str | Path,
    fields: list[str] | None = None,
    filters: list[str] | None = None,
    sort: str | None = "record_date",
    page_size: int = 10_000,
    manifest_json: str | Path | None = None,
) -> dict[str, object]:
    retrieved_at = _utc_now_iso()
    rows, meta = iter_fiscaldata_rows(endpoint, fields=fields, filters=filters, sort=sort, page_size=page_size)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "endpoint": endpoint,
        "url": meta.get("first_url", fiscaldata_url(endpoint, fields=fields, filters=filters, sort=sort, page_size=page_size)),
        "filters": filters or [],
        "fields": fields or [],
        "sort": sort,
        "rows": len(rows),
        "meta": meta,
        "retrieved_at": retrieved_at,
        "out": str(path),
    }
    manifest_path = Path(manifest_json) if manifest_json else path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"status": "ok", "out": str(path), "manifest": str(manifest_path), "rows": len(rows), "endpoint": endpoint}


def download_default_dts_sources(
    *,
    out_dir: str | Path = "data/raw/fiscaldata",
    start_date: str = "2005-01-01",
    page_size: int = 10_000,
) -> dict[str, object]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    ocb_fields = [
        "record_date",
        "account_type",
        "open_today_bal",
        "close_today_bal",
        "open_month_bal",
        "src_line_nbr",
        "record_calendar_year",
        "record_calendar_month",
    ]
    remit_fields = [
        "record_date",
        "account_type",
        "transaction_type",
        "transaction_catg",
        "transaction_today_amt",
        "transaction_mtd_amt",
        "src_line_nbr",
        "record_calendar_year",
        "record_calendar_month",
    ]
    ocb = write_fiscaldata_csv(
        DTS_OPERATING_CASH_BALANCE,
        out_csv=root / "dts_operating_cash_balance.csv",
        fields=ocb_fields,
        filters=[f"record_date:gte:{start_date}"],
        sort="record_date",
        page_size=page_size,
    )
    remit = write_fiscaldata_csv(
        DTS_DEPOSITS_WITHDRAWALS,
        out_csv=root / "dts_federal_reserve_earnings.csv",
        fields=remit_fields,
        filters=[f"record_date:gte:{start_date}", "transaction_catg:eq:Federal Reserve Earnings"],
        sort="record_date",
        page_size=page_size,
    )
    transactions = write_fiscaldata_csv(
        DTS_DEPOSITS_WITHDRAWALS,
        out_csv=root / "dts_deposits_withdrawals_operating_cash.csv",
        fields=remit_fields,
        filters=[f"record_date:gte:{start_date}"],
        sort="record_date",
        page_size=page_size,
    )
    return {
        "status": "ok",
        "out_dir": str(root),
        "start_date": start_date,
        "sources": {"operating_cash_balance": ocb, "federal_reserve_earnings": remit, "deposits_withdrawals": transactions},
    }


def build_dts_fiscal_indicators_csv(
    *,
    operating_cash_balance_csv: str | Path,
    fed_remit_csv: str | Path | None,
    out_csv: str | Path,
    metadata_csv: str | Path | None = None,
) -> dict[str, object]:
    ocb = pd.read_csv(operating_cash_balance_csv, parse_dates=["record_date"])
    close_rows = ocb["account_type"].astype("string").str.contains("TGA\\) Closing Balance", case=False, na=False)
    tga = ocb.loc[close_rows].copy()
    if tga.empty:
        raise ValueError("DTS operating cash balance CSV has no TGA closing-balance rows")
    tga["operating_cash_balance"] = tga["close_today_bal"].map(_clean_number)
    missing_close = tga["operating_cash_balance"].isna()
    tga.loc[missing_close, "operating_cash_balance"] = tga.loc[missing_close, "open_today_bal"].map(_clean_number)
    tga = tga.set_index("record_date").sort_index()

    out = pd.DataFrame()
    out["minus_toc"] = -level_change_to_monthly_flow(tga["operating_cash_balance"].rename("minus_toc"))
    meta: list[dict[str, object]] = [
        {
            "component": "minus_toc",
            "source": "FiscalData DTS operating_cash_balance",
            "source_column": "TGA Closing Balance",
            "transform": "negative_month_end_level_diff",
            "daily_rows": int(len(tga)),
        }
    ]

    deposit_rows = ocb["account_type"].astype("string").str.contains("Total TGA Deposits", case=False, na=False)
    withdrawal_rows = ocb["account_type"].astype("string").str.contains("Total TGA Withdrawals", case=False, na=False)
    deposits = ocb.loc[deposit_rows].copy()
    withdrawals = ocb.loc[withdrawal_rows].copy()
    if not deposits.empty and not withdrawals.empty:
        deposits["tga_deposits"] = deposits["open_today_bal"].map(_clean_number)
        withdrawals["tga_withdrawals"] = withdrawals["open_today_bal"].map(_clean_number).abs()
        deposit_monthly = deposits.set_index("record_date")["tga_deposits"].sort_index().resample("ME").sum(min_count=1)
        withdrawal_monthly = withdrawals.set_index("record_date")["tga_withdrawals"].sort_index().resample("ME").sum(min_count=1)
        out["tga_deposits"] = deposit_monthly
        out["tga_withdrawals"] = withdrawal_monthly
        out["net_tga_withdrawals"] = withdrawal_monthly - deposit_monthly
        meta.extend(
            [
                {
                    "component": "tga_deposits",
                    "source": "FiscalData DTS operating_cash_balance",
                    "source_column": "Total TGA Deposits open_today_bal",
                    "transform": "daily_sum_to_month",
                    "daily_rows": int(len(deposits)),
                },
                {
                    "component": "tga_withdrawals",
                    "source": "FiscalData DTS operating_cash_balance",
                    "source_column": "Total TGA Withdrawals open_today_bal",
                    "transform": "absolute_daily_sum_to_month",
                    "daily_rows": int(len(withdrawals)),
                },
                {
                    "component": "net_tga_withdrawals",
                    "source": "FiscalData DTS operating_cash_balance",
                    "source_column": "Total TGA Withdrawals minus Total TGA Deposits",
                    "transform": "monthly_withdrawals_minus_deposits",
                    "daily_rows": int(min(len(deposits), len(withdrawals))),
                },
            ]
        )

    if fed_remit_csv is not None and Path(fed_remit_csv).exists():
        remit = pd.read_csv(fed_remit_csv, parse_dates=["record_date"])
        remit = remit.loc[
            remit["transaction_catg"].astype("string").str.fullmatch("Federal Reserve Earnings", case=False, na=False)
        ].copy()
        if not remit.empty:
            remit["fed_remit_positive"] = remit["transaction_today_amt"].map(_clean_number)
            remit_series = remit.set_index("record_date")["fed_remit_positive"].sort_index()
            out["fed_remit_positive"] = aggregate_flows_to_monthly(positive_only(remit_series), how="sum")
            meta.append(
                {
                    "component": "fed_remit_positive",
                    "source": "FiscalData DTS deposits_withdrawals_operating_cash",
                    "source_column": "Federal Reserve Earnings transaction_today_amt",
                    "transform": "positive_daily_sum_to_month",
                    "daily_rows": int(len(remit)),
                }
            )

    out.index.name = "date"
    out = out.sort_index()
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index_label="date")

    metadata_path = None
    if metadata_csv is not None:
        metadata_path = Path(metadata_csv)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(meta).to_csv(metadata_path, index=False)

    non_null = {column: int(out[column].notna().sum()) for column in out.columns}
    return {
        "status": "ok",
        "out": str(out_path),
        "metadata": str(metadata_path) if metadata_path else "",
        "columns": list(out.columns),
        "rows": int(len(out)),
        "non_null": non_null,
    }


def _monthly_sum(frame: pd.DataFrame, mask: pd.Series, column: str) -> pd.Series:
    selected = frame.loc[mask, ["record_date", "transaction_today_amt"]].copy()
    if selected.empty:
        return pd.Series(name=column, dtype="float64")
    selected[column] = selected["transaction_today_amt"].map(_clean_number).abs()
    out = selected.set_index("record_date")[column].sort_index().resample("ME").sum(min_count=1)
    out.name = column
    return out


def build_dts_transaction_indicators_csv(
    transactions_csv: str | Path,
    *,
    out_csv: str | Path,
    metadata_csv: str | Path | None = None,
) -> dict[str, object]:
    """Build monthly category-level indicators from DTS deposits/withdrawals.

    The inputs are Daily Treasury Statement table II rows from FiscalData. The
    indicators are not benchmark components; they are timing and shock controls
    for fiscal-flow robustness designs.
    """
    raw = pd.read_csv(transactions_csv, parse_dates=["record_date"], low_memory=False)
    required = {"record_date", "transaction_type", "transaction_catg", "transaction_today_amt"}
    missing = required.difference(raw.columns)
    if missing:
        raise KeyError(f"DTS transaction CSV missing columns: {sorted(missing)}")

    tx_type = raw["transaction_type"].astype("string")
    cat = raw["transaction_catg"].astype("string")
    is_deposit = tx_type.str.fullmatch("Deposits", case=False, na=False)
    is_withdrawal = tx_type.str.fullmatch("Withdrawals", case=False, na=False)

    specs: list[tuple[str, pd.Series, str]] = [
        ("dts_total_deposits", is_deposit, "all DTS deposit transaction_today_amt rows"),
        ("dts_total_withdrawals", is_withdrawal, "all DTS withdrawal transaction_today_amt rows"),
        ("dts_tax_deposits", is_deposit & cat.str.contains("Taxes", case=False, na=False), "deposit rows with Taxes category"),
        (
            "dts_fed_remit_deposits",
            is_deposit & cat.str.fullmatch("Federal Reserve Earnings", case=False, na=False),
            "Federal Reserve Earnings deposit rows",
        ),
        (
            "dts_interest_withdrawals",
            is_withdrawal & cat.str.fullmatch("Interest on Treasury Securities", case=False, na=False),
            "Interest on Treasury Securities withdrawal rows",
        ),
        (
            "dts_federal_salary_withdrawals",
            is_withdrawal & cat.str.contains("Federal Salaries", case=False, na=False),
            "Federal Salaries withdrawal rows",
        ),
        (
            "dts_hhs_withdrawals",
            is_withdrawal & cat.str.contains("^HHS\\s*-", case=False, na=False),
            "HHS withdrawal rows",
        ),
        (
            "dts_dod_withdrawals",
            is_withdrawal & cat.str.contains("^DoD\\s*-", case=False, na=False),
            "DoD withdrawal rows",
        ),
    ]
    out = pd.DataFrame()
    meta: list[dict[str, object]] = []
    for column, mask, description in specs:
        series = _monthly_sum(raw, mask, column)
        out[column] = series
        meta.append(
            {
                "component": column,
                "source": "FiscalData DTS deposits_withdrawals_operating_cash",
                "source_column": "transaction_today_amt",
                "filter": description,
                "daily_rows": int(mask.sum()),
                "transform": "absolute_daily_sum_to_month",
            }
        )

    out["dts_net_withdrawals"] = out["dts_total_withdrawals"] - out["dts_total_deposits"]
    out["dts_core_payment_withdrawals"] = (
        out["dts_interest_withdrawals"].fillna(0.0)
        + out["dts_federal_salary_withdrawals"].fillna(0.0)
        + out["dts_hhs_withdrawals"].fillna(0.0)
        + out["dts_dod_withdrawals"].fillna(0.0)
    )
    meta.extend(
        [
            {
                "component": "dts_net_withdrawals",
                "source": "FiscalData DTS deposits_withdrawals_operating_cash",
                "source_column": "dts_total_withdrawals - dts_total_deposits",
                "filter": "all deposit and withdrawal rows",
                "daily_rows": int((is_deposit | is_withdrawal).sum()),
                "transform": "monthly_total_withdrawals_minus_deposits",
            },
            {
                "component": "dts_core_payment_withdrawals",
                "source": "FiscalData DTS deposits_withdrawals_operating_cash",
                "source_column": "interest + federal salaries + HHS + DoD withdrawals",
                "filter": "selected recurring payment categories",
                "daily_rows": int(
                    (
                        specs[4][1]
                        | specs[5][1]
                        | specs[6][1]
                        | specs[7][1]
                    ).sum()
                ),
                "transform": "monthly_sum_of_selected_withdrawal_categories",
            },
        ]
    )

    out.index.name = "date"
    out = out.sort_index()
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index_label="date")

    metadata_path = None
    if metadata_csv is not None:
        metadata_path = Path(metadata_csv)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(meta).to_csv(metadata_path, index=False)

    return {
        "status": "ok",
        "out": str(out_path),
        "metadata": str(metadata_path) if metadata_path else "",
        "columns": list(out.columns),
        "rows": int(len(out)),
        "non_null": {column: int(out[column].notna().sum()) for column in out.columns},
    }
