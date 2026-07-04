from __future__ import annotations

from pathlib import Path

import pandas as pd

from .indicators import aggregate_flows_to_monthly, level_change_to_monthly_flow, positive_only, read_wide_time_series_csv


def build_tic_row_indicator_csv(
    tic_csv: str | Path,
    *,
    out_csv: str | Path,
    date_column: str = "date",
) -> dict[str, object]:
    """Normalize a local TIC monthly Treasury purchases extract.

    Accepted value columns, in priority order:
    - row_tsy
    - treasury_net_purchases
    - net_purchases
    - official_net_purchases + private_net_purchases
    - bonds_notes_net_purchases + bills_net_purchases
    """
    df = read_wide_time_series_csv(tic_csv, date_column=date_column)
    if "row_tsy" in df.columns:
        row = df["row_tsy"]
        source = "row_tsy"
    elif "treasury_net_purchases" in df.columns:
        row = df["treasury_net_purchases"]
        source = "treasury_net_purchases"
    elif "net_purchases" in df.columns:
        row = df["net_purchases"]
        source = "net_purchases"
    elif {"official_net_purchases", "private_net_purchases"}.issubset(df.columns):
        row = df["official_net_purchases"] + df["private_net_purchases"]
        source = "official_plus_private_net_purchases"
    elif {"bonds_notes_net_purchases", "bills_net_purchases"}.issubset(df.columns):
        row = df["bonds_notes_net_purchases"] + df["bills_net_purchases"]
        source = "bonds_notes_plus_bills_net_purchases"
    else:
        raise KeyError("TIC extract needs row_tsy, treasury_net_purchases, net_purchases, official/private, or bonds/bills columns")

    out = pd.DataFrame({"row_tsy": row})
    out.index.name = "date"
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")
    return {"status": "ok", "out": str(path), "source": source, "rows": int(len(out))}


def build_fiscal_indicator_csv(
    fiscal_csv: str | Path,
    *,
    out_csv: str | Path,
    date_column: str = "date",
) -> dict[str, object]:
    """Normalize local DTS/MTS-style fiscal extracts into monthly indicators.

    Accepted columns:
    - operating_cash_balance or tga_balance: level, converted to minus_toc as
      negative month-end change.
    - fed_remit_positive or fed_remittances or federal_reserve_earnings:
      flow, aggregated monthly and clipped at zero when not already positive.
    """
    df = read_wide_time_series_csv(fiscal_csv, date_column=date_column)
    out = pd.DataFrame()
    meta: list[str] = []

    toc_col = None
    for candidate in ["operating_cash_balance", "tga_balance", "toc_balance"]:
        if candidate in df.columns:
            toc_col = candidate
            break
    if toc_col is not None:
        out["minus_toc"] = -level_change_to_monthly_flow(df[toc_col].rename("minus_toc"))
        meta.append(f"minus_toc:{toc_col}:negative_month_end_level_diff")

    remit_col = None
    for candidate in ["fed_remit_positive", "fed_remittances", "federal_reserve_earnings"]:
        if candidate in df.columns:
            remit_col = candidate
            break
    if remit_col is not None:
        remit = positive_only(df[remit_col].rename("fed_remit_positive"))
        out["fed_remit_positive"] = aggregate_flows_to_monthly(remit, how="sum")
        meta.append(f"fed_remit_positive:{remit_col}:positive_monthly_sum")

    if out.empty:
        raise KeyError("Fiscal extract needs operating_cash_balance/tga_balance/toc_balance and/or Fed remittance columns")

    out.index.name = "date"
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")
    return {"status": "ok", "out": str(path), "columns": list(out.columns), "transforms": meta, "rows": int(len(out))}


def merge_indicator_csvs(inputs: list[str | Path], *, out_csv: str | Path) -> dict[str, object]:
    merged: pd.DataFrame | None = None
    for input_path in inputs:
        frame = read_wide_time_series_csv(input_path)
        if merged is None:
            merged = frame.copy()
            continue
        full_index = merged.index.union(frame.index).sort_values()
        merged = merged.reindex(full_index)
        frame = frame.reindex(full_index)
        for column in frame.columns:
            if column in merged.columns:
                merged[column] = frame[column].combine_first(merged[column])
            else:
                merged[column] = frame[column]
    if merged is None:
        raise ValueError("At least one indicator CSV is required")
    merged = merged.sort_index()
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index_label="date")
    return {"status": "ok", "out": str(path), "inputs": [str(item) for item in inputs], "columns": list(merged.columns), "rows": int(len(merged))}
