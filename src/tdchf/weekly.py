from __future__ import annotations

from pathlib import Path

import pandas as pd

from .indicators import read_wide_time_series_csv

WEEKLY_STATE_COLUMNS = [
    "tga_week_avg",
    "tga_wednesday",
    "fed_treasury_holdings",
    "fed_remittances_due",
    "bank_treasury_agency",
    "onrrp",
    "reserves",
]

FRED_WEEKLY_RENAME = {
    "WTREGEN": "tga_week_avg",
    "WDTGAL": "tga_wednesday",
    "TREAST": "fed_treasury_holdings",
    "RESPPLLOPNWW": "fed_remittances_due",
    "TASACBW027NBOG": "bank_treasury_agency",
    "DPSACBW027SBOG": "broad_deposits",
    "DPSDCBW027SBOG": "domestic_deposits",
    "LTDACBW027SBOG": "large_time_deposits",
    "RRPONTSYD": "onrrp",
    "WRESBAL": "reserves",
    "DGS2": "yield_2y",
    "DGS10": "yield_10y",
    "EFFR": "effr",
    "IORB": "iorb",
    "SOFR": "sofr",
    "WRMFNS": "retail_mmf",
    "WIMFNS": "institutional_mmf",
}


def build_weekly_state_from_fred_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=pd.to_datetime(df.index).sort_values())
    for source, target in FRED_WEEKLY_RENAME.items():
        if source in df.columns:
            out[target] = pd.to_numeric(df[source], errors="coerce")
    out = out.resample("W-WED").last()
    out.index.name = "date"
    return out


def build_weekly_state_csv(raw_fred_csv: str | Path, *, out_csv: str | Path) -> dict[str, object]:
    raw = read_wide_time_series_csv(raw_fred_csv)
    state = build_weekly_state_from_fred_frame(raw)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state.to_csv(out_path, index_label="date")
    return {
        "status": "ok",
        "out": str(out_path),
        "columns": [col for col in state.columns if state[col].notna().any()],
        "rows": int(len(state)),
    }


def build_weekly_channel_panel(
    inputs: list[str | Path],
    *,
    change_columns: list[str] | None = None,
    lag_columns: list[str] | None = None,
) -> pd.DataFrame:
    if not inputs:
        raise ValueError("At least one weekly input CSV is required")
    frames = [read_wide_time_series_csv(path) for path in inputs]
    panel = pd.concat(frames, axis=1, sort=False)
    panel = panel.loc[:, ~panel.columns.duplicated(keep="last")].sort_index()

    if "domestic_deposits" in panel.columns and "large_time_deposits" in panel.columns:
        panel["domestic_non_large_time_deposits"] = panel["domestic_deposits"] - panel["large_time_deposits"]
    if "broad_deposits" in panel.columns and "large_time_deposits" in panel.columns:
        panel["broad_non_large_time_deposits"] = panel["broad_deposits"] - panel["large_time_deposits"]
    if "retail_mmf" in panel.columns or "institutional_mmf" in panel.columns:
        retail = pd.to_numeric(panel["retail_mmf"], errors="coerce") if "retail_mmf" in panel.columns else 0.0
        institutional = pd.to_numeric(panel["institutional_mmf"], errors="coerce") if "institutional_mmf" in panel.columns else 0.0
        panel["total_mmf"] = retail + institutional

    if change_columns is None:
        change_columns = [
            "tga_wednesday",
            "tga_week_avg",
            "fed_treasury_holdings",
            "bank_treasury_agency",
            "onrrp",
            "reserves",
            "broad_deposits",
            "domestic_deposits",
            "large_time_deposits",
            "domestic_non_large_time_deposits",
            "broad_non_large_time_deposits",
            "retail_mmf",
            "institutional_mmf",
            "total_mmf",
        ]
    for column in change_columns:
        if column in panel.columns:
            panel[f"d_{column}"] = pd.to_numeric(panel[column], errors="coerce").diff()

    if lag_columns is None:
        lag_columns = [column for column in panel.columns if column.startswith("d_")]
    for column in lag_columns:
        if column in panel.columns:
            panel[f"lag_{column}"] = pd.to_numeric(panel[column], errors="coerce").shift(1)

    panel.index.name = "date"
    return panel


def build_weekly_channel_panel_csv(
    inputs: list[str | Path],
    *,
    out_csv: str | Path,
    change_columns: list[str] | None = None,
    lag_columns: list[str] | None = None,
) -> dict[str, object]:
    panel = build_weekly_channel_panel(inputs, change_columns=change_columns, lag_columns=lag_columns)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index_label="date")
    return {
        "status": "ok",
        "out": str(path),
        "rows": int(len(panel)),
        "columns": [col for col in panel.columns if panel[col].notna().any()],
    }
