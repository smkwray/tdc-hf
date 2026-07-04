from __future__ import annotations

from pathlib import Path

import pandas as pd

from .indicators import level_change_to_monthly_flow, positive_only, read_wide_time_series_csv
from .proxy import COMPONENT_ORDER

FRED_HF_SERIES = [
    "TREAST",
    "WTREGEN",
    "WDTGAL",
    "RESPPLLOPNWW",
    "WRESBAL",
    "WRBWFRBL",
    "WLRRAL",
    "WALCL",
    "RRPONTSYD",
    "TASACBW027NBOG",
    "DPSACBW027NBOG",
    "TNMACBM027NBOG",
    "TNMACBW027NBOG",
    "USGSEC",
]


def _first_available(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns and df[candidate].notna().any():
            return candidate
    return None


def build_fred_monthly_indicators_from_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.DataFrame()
    meta: list[dict[str, object]] = []

    if "TREAST" in df.columns:
        out["fed_tsy"] = level_change_to_monthly_flow(df["TREAST"].rename("fed_tsy"))
        meta.append({"component": "fed_tsy", "source_series": "TREAST", "transform": "month_end_level_diff"})

    bank_source = _first_available(df, ["TNMACBM027NBOG", "TNMACBW027NBOG", "USGSEC", "TASACBW027NBOG"])
    if bank_source is not None:
        out["banks_tsy"] = level_change_to_monthly_flow(df[bank_source].rename("banks_tsy"))
        meta.append({"component": "banks_tsy", "source_series": bank_source, "transform": "month_end_level_diff"})

    toc_source = _first_available(df, ["WDTGAL", "WTREGEN"])
    if toc_source is not None:
        out["minus_toc"] = -level_change_to_monthly_flow(df[toc_source].rename("minus_toc"))
        meta.append({"component": "minus_toc", "source_series": toc_source, "transform": "negative_month_end_level_diff"})

    if "RESPPLLOPNWW" in df.columns:
        weekly_positive = positive_only(df["RESPPLLOPNWW"].rename("fed_remit_positive"))
        out["fed_remit_positive"] = weekly_positive.resample("ME").sum(min_count=1)
        meta.append({"component": "fed_remit_positive", "source_series": "RESPPLLOPNWW", "transform": "positive_weekly_sum"})

    out.index.name = "date"
    for component in COMPONENT_ORDER:
        if component in out.columns:
            out[component] = pd.to_numeric(out[component], errors="coerce")
    return out.sort_index(), pd.DataFrame(meta)


def build_fred_monthly_indicators_csv(
    raw_fred_csv: str | Path,
    *,
    out_csv: str | Path,
    metadata_csv: str | Path | None = None,
) -> dict[str, object]:
    raw = read_wide_time_series_csv(raw_fred_csv)
    indicators, metadata = build_fred_monthly_indicators_from_frame(raw)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indicators.to_csv(out_path, index_label="date")
    metadata_path = None
    if metadata_csv is not None:
        metadata_path = Path(metadata_csv)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(metadata_path, index=False)
    return {
        "status": "ok",
        "out": str(out_path),
        "metadata": str(metadata_path) if metadata_path else "",
        "columns": [col for col in indicators.columns if indicators[col].notna().any()],
        "rows": int(len(indicators)),
    }
