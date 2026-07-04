from __future__ import annotations

import pandas as pd

from tdchf.calendar_controls import add_calendar_controls_csv


def test_add_calendar_controls_csv(tmp_path) -> None:
    data = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31", "2020-04-30", "2023-04-30"]),
            "x": [1.0, 2.0, 3.0],
        }
    ).to_csv(data, index=False)

    report = add_calendar_controls_csv(data, out_csv=tmp_path / "controlled.csv")
    out = pd.read_csv(tmp_path / "controlled.csv")

    assert report["status"] == "ok"
    assert "tax_month" in out.columns
    assert "treasury_coupon_interest_month" in out.columns
    assert "major_benefit_payment_month" in out.columns
    assert "month_04" in out.columns
    assert out.loc[out["date"] == "2020-04-30", "tax_month"].iloc[0] == 1
    assert out.loc[out["date"] == "2020-03-31", "corporate_tax_month"].iloc[0] == 1
    assert out.loc[out["date"] == "2020-03-31", "crisis_covid"].iloc[0] == 1
