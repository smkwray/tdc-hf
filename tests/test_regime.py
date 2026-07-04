from __future__ import annotations

import numpy as np
import pandas as pd

from tdchf.regime import run_regime_exclusion_robustness_csv


def test_run_regime_exclusion_robustness_csv(tmp_path) -> None:
    n = 72
    instrument = np.linspace(0.0, 5.0, n)
    treatment = 2.0 * instrument
    data = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2007-01-31", periods=n, freq="ME"),
            "instrument": instrument,
            "tdc_monthly": treatment,
            "deposits": 0.5 * treatment / 1000.0,
            "control": np.ones(n),
        }
    ).to_csv(data, index=False)

    report = run_regime_exclusion_robustness_csv(
        data,
        treatment="tdc_monthly",
        instruments=["instrument"],
        outcomes=["deposits"],
        controls=["control"],
        horizons=[0, 1],
        out_dir=tmp_path / "regime",
    )
    out = pd.read_csv(tmp_path / "regime" / "regime_exclusion_summary.csv")

    assert report["status"] == "ok"
    assert "full_sample" in set(out["sample"])
    assert "same_unit_beta" in out.columns
