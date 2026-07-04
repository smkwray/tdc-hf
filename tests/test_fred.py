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
