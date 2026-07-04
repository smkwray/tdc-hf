from __future__ import annotations

import pandas as pd

from tdchf.live_indicators import build_fred_monthly_indicators_from_frame


def test_build_fred_monthly_indicators_from_frame() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-03",
            "2024-01-31",
            "2024-02-07",
            "2024-02-29",
        ]
    )
    df = pd.DataFrame(
        {
            "TREAST": [10.0, 13.0, 14.0, 18.0],
            "TNMACBM027NBOG": [20.0, 21.0, 25.0, 27.0],
            "WDTGAL": [100.0, 90.0, 92.0, 95.0],
            "RESPPLLOPNWW": [-1.0, 2.0, 3.0, -2.0],
        },
        index=idx,
    )

    indicators, meta = build_fred_monthly_indicators_from_frame(df)

    assert indicators.loc[pd.Timestamp("2024-02-29"), "fed_tsy"] == 5.0
    assert indicators.loc[pd.Timestamp("2024-02-29"), "banks_tsy"] == 6.0
    assert indicators.loc[pd.Timestamp("2024-02-29"), "minus_toc"] == -5.0
    assert indicators.loc[pd.Timestamp("2024-01-31"), "fed_remit_positive"] == 2.0
    assert set(meta["component"]) == {"fed_tsy", "banks_tsy", "minus_toc", "fed_remit_positive"}
