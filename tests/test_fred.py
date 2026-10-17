from __future__ import annotations

import pandas as pd

from tdchf.fred import fred_graph_csv_url, parse_fred_graph_csv


def test_fred_graph_csv_url_quotes_series_id() -> None:
    assert fred_graph_csv_url("WRESBAL").endswith("id=WRESBAL")


def test_parse_fred_graph_csv_handles_missing_dot() -> None:
    text = "observation_date,TEST\n2024-01-01,1.5\n2024-01-02,.\n"

    out = parse_fred_graph_csv(text, series_id="TEST")

    assert out.loc[pd.Timestamp("2024-01-01")] == 1.5
    assert pd.isna(out.loc[pd.Timestamp("2024-01-02")])


def test_fred_download_drops_observations_after_the_vintage_endpoint(monkeypatch, tmp_path) -> None:
    import json

    from tdchf import fred

    dates = pd.date_range("2026-09-23", "2026-10-13", freq="D")
    frame = pd.DataFrame({"WRESBAL": range(len(dates)), "DPSACBW027NBOG": 1.0}, index=dates)
    frame.loc[dates > "2026-09-30", "DPSACBW027NBOG"] = float("nan")
    frame.index.name = "date"
    monkeypatch.setattr(fred, "fetch_fred_series_many", lambda ids: frame[ids])
    path = fred.write_fred_series_csv(["WRESBAL", "DPSACBW027NBOG"], tmp_path / "raw.csv", end_date="2026-09-30")
    out = pd.read_csv(path, parse_dates=["date"])
    assert out["date"].max() == pd.Timestamp("2026-09-30")
    manifest = json.loads((tmp_path / "raw.csv.manifest.json").read_text())
    assert manifest["end_date"] == "2026-09-30" and manifest["rows"] == len(out)
    assert manifest["last_observation"] == {"WRESBAL": "2026-09-30", "DPSACBW027NBOG": "2026-09-30"}
    assert manifest["retrieved_at"]
