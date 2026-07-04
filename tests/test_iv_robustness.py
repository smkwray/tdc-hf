from __future__ import annotations

import numpy as np
import pandas as pd

from tdchf.iv_robustness import run_iv_robustness_csv


def test_run_iv_robustness_csv(tmp_path) -> None:
    n = 48
    z1 = np.linspace(0.0, 5.0, n)
    z2 = np.cos(np.linspace(0.0, 2.0, n))
    treatment = 1.5 * z1 + 0.2 * z2
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=n, freq="ME"),
            "z1": z1,
            "z2": z2,
            "treatment": treatment,
            "outcome": 2.0 * treatment,
            "control": np.ones(n),
        }
    )
    data = tmp_path / "panel.csv"
    df.to_csv(data, index=False)

    report = run_iv_robustness_csv(
        data,
        treatment="treatment",
        instrument_specs=["both=z1+z2", "z1_only=z1"],
        outcomes=["outcome"],
        controls=["control"],
        horizons=[0, 1],
        out_dir=tmp_path / "robustness",
    )
    out = pd.read_csv(tmp_path / "robustness" / "iv_robustness_summary.csv")

    assert report["status"] == "ok"
    assert report["summary_rows"] == 2
    assert set(out["iv_spec"]) == {"both", "z1_only"}
    assert {"weak_iv_label", "weak_iv_flag"}.issubset(out.columns)
