from __future__ import annotations

import pandas as pd

from tdchf.validation import loqo_month_share_diagnostics


def test_loqo_month_share_diagnostics_returns_rows() -> None:
    idx = pd.date_range("2024-01-31", periods=6, freq="ME")
    benchmarked = pd.DataFrame(
        {"fed_tsy": [1.0, 2.0, 3.0, 2.0, 2.0, 2.0]},
        index=idx,
    )
    raw = {"fed_tsy": pd.Series([1.0, 2.0, 3.0, 1.0, 1.0, 4.0], index=idx)}

    out = loqo_month_share_diagnostics(benchmarked, raw)

    assert not out.empty
    assert set(out["component"]) == {"fed_tsy"}
