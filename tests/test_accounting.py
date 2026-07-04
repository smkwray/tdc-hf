from __future__ import annotations

import pandas as pd

from tdchf.accounting import export_accounting_decomposition


def test_export_accounting_decomposition(tmp_path) -> None:
    lp = tmp_path / "lp.csv"
    pd.DataFrame(
        {
            "outcome": ["deposits", "onrrp", "bank_credit"],
            "horizon": [12, 12, 12],
            "same_unit_beta": [0.5, 0.2, 1.0],
            "same_unit_lower95": [0.1, -0.1, 0.2],
            "same_unit_upper95": [0.9, 0.5, 1.8],
        }
    ).to_csv(lp, index=False)

    report = export_accounting_decomposition(lp, out_dir=tmp_path / "accounting", horizons=[12])
    summary = pd.read_csv(tmp_path / "accounting" / "accounting_decomposition_summary.csv")

    assert report["status"] == "ok"
    assert report["rows"] == 3
    assert round(summary.loc[0, "simple_sum_same_unit_beta"], 1) == 0.7
