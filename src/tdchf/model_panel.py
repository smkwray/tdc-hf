from __future__ import annotations

from pathlib import Path

import pandas as pd

from .indicators import read_wide_time_series_csv


def _read_csv_with_prefix(path: str | Path, *, prefix: str = "") -> pd.DataFrame:
    frame = read_wide_time_series_csv(path)
    if prefix:
        frame = frame.rename(columns={col: f"{prefix}{col}" for col in frame.columns})
    return frame


def assemble_model_panel(
    inputs: list[str | Path],
    *,
    prefixes: list[str] | None = None,
    lags: list[str] | None = None,
    diffs: list[str] | None = None,
    require: list[str] | None = None,
) -> pd.DataFrame:
    if not inputs:
        raise ValueError("At least one input CSV is required")
    if prefixes is None:
        prefixes = [""] * len(inputs)
    if len(prefixes) != len(inputs):
        raise ValueError("prefix count must match input count")

    frames = [_read_csv_with_prefix(path, prefix=prefix) for path, prefix in zip(inputs, prefixes)]
    panel = pd.concat(frames, axis=1, sort=False)
    panel = panel.loc[:, ~panel.columns.duplicated(keep="last")].sort_index()

    for column in lags or []:
        if column not in panel.columns:
            raise KeyError(f"Missing lag column: {column}")
        panel[f"lag_{column}"] = panel[column].shift(1)

    for column in diffs or []:
        if column not in panel.columns:
            raise KeyError(f"Missing diff column: {column}")
        panel[f"d_{column}"] = panel[column].diff()

    if require:
        missing = [column for column in require if column not in panel.columns]
        if missing:
            raise KeyError(f"Missing required complete-case columns: {missing}")
        panel = panel.dropna(subset=require)

    panel.index.name = "date"
    return panel


def assemble_model_panel_csv(
    inputs: list[str | Path],
    *,
    out_csv: str | Path,
    prefixes: list[str] | None = None,
    lags: list[str] | None = None,
    diffs: list[str] | None = None,
    require: list[str] | None = None,
) -> dict[str, object]:
    panel = assemble_model_panel(inputs, prefixes=prefixes, lags=lags, diffs=diffs, require=require)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index_label="date")
    return {"status": "ok", "out": str(path), "rows": int(len(panel)), "columns": list(panel.columns)}
