from __future__ import annotations

from pathlib import Path

import pandas as pd

from .indicators import read_wide_time_series_csv


DEBT_CEILING_WINDOWS = [
    ("1995-11-01", "1996-03-31"),
    ("2011-05-01", "2011-08-31"),
    ("2013-02-01", "2013-10-31"),
    ("2015-03-01", "2015-11-30"),
    ("2017-03-01", "2017-09-30"),
    ("2019-03-01", "2019-08-31"),
    ("2021-08-01", "2021-12-31"),
    ("2023-01-01", "2023-06-30"),
]

CRISIS_WINDOWS = [
    ("2008-09-01", "2009-06-30", "gfc"),
    ("2020-03-01", "2020-12-31", "covid"),
    ("2023-03-01", "2023-05-31", "bank_stress_2023"),
]


def add_calendar_controls(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy().sort_index()
    index = pd.DatetimeIndex(out.index)
    out["quarter_end_month"] = index.month.isin([3, 6, 9, 12]).astype(int)
    out["tax_month"] = index.month.isin([4, 6, 9, 12]).astype(int)
    out["april_tax_month"] = (index.month == 4).astype(int)
    out["corporate_tax_month"] = index.month.isin([3, 6, 9, 12]).astype(int)
    out["refund_season_month"] = index.month.isin([2, 3, 4]).astype(int)
    out["treasury_coupon_interest_month"] = index.month.isin([2, 5, 8, 11]).astype(int)
    out["major_benefit_payment_month"] = index.month.isin([1, 3, 6, 9, 12]).astype(int)
    out["fiscal_quarter_end_month"] = index.month.isin([3, 6, 9, 12]).astype(int)
    out["year_end_month"] = (index.month == 12).astype(int)
    out["post_2008"] = (index >= pd.Timestamp("2008-09-30")).astype(int)
    out["post_2020"] = (index >= pd.Timestamp("2020-03-31")).astype(int)

    for month in range(2, 13):
        out[f"month_{month:02d}"] = (index.month == month).astype(int)

    out["debt_ceiling_window"] = 0
    for start, end in DEBT_CEILING_WINDOWS:
        mask = (index >= pd.Timestamp(start).to_period("M").to_timestamp("M")) & (
            index <= pd.Timestamp(end).to_period("M").to_timestamp("M")
        )
        out.loc[mask, "debt_ceiling_window"] = 1

    out["crisis_window"] = 0
    for start, end, name in CRISIS_WINDOWS:
        col = f"crisis_{name}"
        mask = (index >= pd.Timestamp(start).to_period("M").to_timestamp("M")) & (
            index <= pd.Timestamp(end).to_period("M").to_timestamp("M")
        )
        out[col] = mask.astype(int)
        out.loc[mask, "crisis_window"] = 1

    out.index.name = "date"
    return out


def add_calendar_controls_csv(data_csv: str | Path, *, out_csv: str | Path) -> dict[str, object]:
    panel = read_wide_time_series_csv(data_csv)
    out = add_calendar_controls(panel)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")
    added = [column for column in out.columns if column not in panel.columns]
    return {"status": "ok", "out": str(path), "rows": int(len(out)), "added_columns": added}
