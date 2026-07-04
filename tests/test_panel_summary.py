from __future__ import annotations

import pandas as pd

from tdchf.reporting import summarize_panel_csv


def test_summarize_panel_csv(tmp_path) -> None:
    path = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=2, freq="ME"),
            "x": [1.0, None],
            "y": [2.0, 3.0],
        }
    ).to_csv(path, index=False)

    report = summarize_panel_csv(path, out_csv=tmp_path / "summary.csv")
    out = pd.read_csv(tmp_path / "summary.csv")

    assert report["status"] == "ok"
    assert out[out["column"] == "x"]["nonnull"].iloc[0] == 1
