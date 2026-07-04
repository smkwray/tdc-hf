from __future__ import annotations

from io import StringIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import pandas as pd


def fred_graph_csv_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={quote(series_id)}"


def parse_fred_graph_csv(text: str, *, series_id: str) -> pd.Series:
    df = pd.read_csv(StringIO(text))
    if "observation_date" not in df.columns:
        raise KeyError("FRED graph CSV missing observation_date")
    if series_id not in df.columns:
        raise KeyError(f"FRED graph CSV missing series column: {series_id}")
    values = df[series_id].astype("string")
    values = values.mask(values == ".", pd.NA)
    out = pd.Series(
        pd.to_numeric(values, errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(df["observation_date"]),
        name=series_id,
    ).sort_index()
    return out


def fetch_fred_series(series_id: str) -> pd.Series:
    df = pd.read_csv(fred_graph_csv_url(series_id))
    if "observation_date" not in df.columns or series_id not in df.columns:
        raise KeyError(f"Unexpected FRED graph CSV schema for {series_id}")
    values = df[series_id].astype("string")
    values = values.mask(values == ".", pd.NA)
    return pd.Series(
        pd.to_numeric(values, errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(df["observation_date"]),
        name=series_id,
    ).sort_index()


def fetch_fred_series_many(series_ids: list[str]) -> pd.DataFrame:
    if not series_ids:
        raise ValueError("At least one FRED series id is required")
    max_workers = min(12, len(series_ids))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frames = list(executor.map(fetch_fred_series, series_ids))
    out = pd.concat(frames, axis=1, sort=False).sort_index()
    out.index.name = "date"
    return out


def write_fred_series_csv(series_ids: list[str], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = fetch_fred_series_many(series_ids)
    frame.to_csv(path, index_label="date")
    return path
