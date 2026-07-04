from __future__ import annotations

import numpy as np
import pandas as pd

from tdchf.first_stage import run_first_stage, run_first_stage_csv


def test_run_first_stage_returns_f_stat() -> None:
    n = 30
    instrument = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "treatment": 2.0 * instrument,
            "instrument": instrument,
            "control": np.ones(n),
        }
    )

    out = run_first_stage(df, treatment="treatment", instruments=["instrument"], controls=["control"])

    assert out.loc[0, "excluded_instrument_f"] > 0
    assert out.loc[0, "n"] == n


def test_run_first_stage_csv(tmp_path) -> None:
    n = 30
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=n, freq="ME"),
            "treatment": np.arange(n, dtype=float),
            "instrument": np.arange(n, dtype=float),
        }
    )
    path = tmp_path / "first.csv"
    df.to_csv(path, index=False)

    report = run_first_stage_csv(path, treatment="treatment", instruments=["instrument"], out_csv=tmp_path / "out.csv")

    assert report["status"] == "ok"
    assert (tmp_path / "out.csv").exists()
