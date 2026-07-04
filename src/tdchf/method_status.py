from __future__ import annotations

from pathlib import Path

import pandas as pd


METHOD_STATUS_ROWS = [
    {
        "method": "residual_spread",
        "status": "implemented_placebo",
        "headline": False,
        "notes": "Equal residual spreading is retained as a bootstrap/placebo comparator.",
    },
    {
        "method": "additive_denton",
        "status": "implemented_headline",
        "headline": True,
        "notes": "Current headline temporal-disaggregation method for monthly TDC components.",
    },
    {
        "method": "chow_lin",
        "status": "stub_not_headline",
        "headline": False,
        "notes": "Requires component-specific quarterly calibration and indicator error model before use.",
    },
    {
        "method": "litterman",
        "status": "stub_not_headline",
        "headline": False,
        "notes": "Requires AR(1) residual calibration and component-specific validation before use.",
    },
]


def temporal_disaggregation_method_status() -> pd.DataFrame:
    return pd.DataFrame(METHOD_STATUS_ROWS)


def write_method_status_csv(out_csv: str | Path) -> dict[str, object]:
    status = temporal_disaggregation_method_status()
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(path, index=False)
    return {
        "status": "ok",
        "out": str(path),
        "rows": int(len(status)),
        "headline_method": str(status.loc[status["headline"], "method"].iloc[0]),
    }
