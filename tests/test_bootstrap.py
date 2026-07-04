from __future__ import annotations

import numpy as np
import pandas as pd

from tdchf.bootstrap import bootstrap_lp_iv_csv


def test_bootstrap_lp_iv_csv(tmp_path) -> None:
    n = 48
    instrument = np.linspace(0.0, 5.0, n)
    treatment = 1.5 * instrument + 0.1 * np.sin(instrument)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=n, freq="ME"),
            "instrument": instrument,
            "treatment": treatment,
            "outcome": 2.0 * treatment,
            "control": np.ones(n),
        }
    )
    data = tmp_path / "panel.csv"
    df.to_csv(data, index=False)

    report = bootstrap_lp_iv_csv(
        data,
        treatment="treatment",
        instruments=["instrument"],
        outcomes=["outcome"],
        controls=["control"],
        horizons=[0, 1],
        replications=10,
        block_length=4,
        out_csv=tmp_path / "bootstrap.csv",
    )
    out = pd.read_csv(tmp_path / "bootstrap.csv")

    assert report["status"] == "ok"
    assert report["rows"] == 2
    assert out["draws"].min() > 0
