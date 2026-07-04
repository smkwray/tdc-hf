from __future__ import annotations

import pandas as pd

from tdchf.model_panel import assemble_model_panel, assemble_model_panel_csv


def test_assemble_model_panel_adds_lags_and_diffs(tmp_path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    pd.DataFrame({"date": dates, "x": [1.0, 2.0, 4.0]}).to_csv(left, index=False)
    pd.DataFrame({"date": dates, "y": [3.0, 5.0, 9.0]}).to_csv(right, index=False)

    panel = assemble_model_panel([left, right], lags=["x"], diffs=["y"])

    assert "lag_x" in panel.columns
    assert "d_y" in panel.columns
    assert panel.loc[pd.Timestamp("2024-02-29"), "lag_x"] == 1.0
    assert panel.loc[pd.Timestamp("2024-03-31"), "d_y"] == 4.0


def test_assemble_model_panel_csv(tmp_path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"date": [pd.Timestamp("2024-01-31")], "x": [1.0]}).to_csv(path, index=False)

    report = assemble_model_panel_csv([path], out_csv=tmp_path / "panel.csv")

    assert report["status"] == "ok"
    assert (tmp_path / "panel.csv").exists()


def test_assemble_model_panel_require_complete_cases(tmp_path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=2, freq="ME"),
            "x": [1.0, None],
            "y": [2.0, 3.0],
        }
    ).to_csv(path, index=False)

    panel = assemble_model_panel([path], require=["x", "y"])

    assert len(panel) == 1
