from __future__ import annotations

import pandas as pd

from tdchf.figures import export_thesis_figures


def test_export_thesis_figures(tmp_path) -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    proxy = tmp_path / "proxy.csv"
    components = tmp_path / "components.csv"
    lp = tmp_path / "lp.csv"
    pd.DataFrame({"date": dates, "tdc_monthly": [1.0, 2.0, 1.5, 3.0]}).to_csv(proxy, index=False)
    pd.DataFrame({"date": dates, "fed_tsy": [1, 2, 3, 4], "banks_tsy": [2, 1, 2, 1]}).to_csv(components, index=False)
    pd.DataFrame(
        {
            "outcome": ["deposits", "deposits"],
            "horizon": [0, 1],
            "beta": [0.1, 0.2],
            "lower95": [0.0, 0.1],
            "upper95": [0.2, 0.3],
        }
    ).to_csv(lp, index=False)

    report = export_thesis_figures(proxy_csv=proxy, components_csv=components, lp_csv=lp, out_dir=tmp_path / "figures")

    assert report["status"] == "ok"
    assert report["count"] >= 3
    assert (tmp_path / "figures" / "monthly_tdc_proxy.png").exists()
