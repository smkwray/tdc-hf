from __future__ import annotations

import pandas as pd

from tdchf.analysis_spec import run_analysis_spec


def test_run_analysis_spec_assemble_and_summarize(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=3, freq="ME"),
            "x": [1.0, 2.0, 4.0],
            "y": [3.0, 5.0, 9.0],
        }
    ).to_csv(raw, index=False)
    spec = tmp_path / "spec.yml"
    spec.write_text(
        """
root: .
report: report.json
steps:
  - id: panel
    action: assemble-panel
    inputs: [raw.csv]
    lags: [x]
    diffs: [y]
    out: panel.csv
  - id: summary
    action: summarize-panel
    data: panel.csv
    out: summary.csv
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = run_analysis_spec(spec)

    panel = pd.read_csv(tmp_path / "panel.csv")
    assert report["status"] == "ok"
    assert (tmp_path / "report.json").exists()
    assert {"lag_x", "d_y"}.issubset(panel.columns)
    assert (tmp_path / "summary.csv").exists()


def test_run_analysis_spec_only_filter(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    pd.DataFrame({"date": [pd.Timestamp("2024-01-31")], "x": [1.0]}).to_csv(raw, index=False)
    spec = tmp_path / "spec.yml"
    spec.write_text(
        """
root: .
steps:
  - id: panel_a
    action: assemble-panel
    inputs: [raw.csv]
    out: panel_a.csv
  - id: panel_b
    action: assemble-panel
    inputs: [raw.csv]
    out: panel_b.csv
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = run_analysis_spec(spec, only=["panel_b"])

    assert not (tmp_path / "panel_a.csv").exists()
    assert (tmp_path / "panel_b.csv").exists()
    assert [step["status"] for step in report["steps"] if step["id"] == "panel_a"] == ["skipped_by_filter"]
