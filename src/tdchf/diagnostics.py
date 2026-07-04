from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .calendar import to_quarter_end
from .proxy import COMPONENT_ORDER


def raw_indicator_quarterly_fit(
    monthly_indicators: Mapping[str, pd.Series],
    quarterly_anchors: Mapping[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component in COMPONENT_ORDER:
        if component not in monthly_indicators or component not in quarterly_anchors:
            continue
        monthly = monthly_indicators[component].dropna().copy()
        monthly.index = pd.to_datetime(monthly.index)
        quarterly_indicator = monthly.groupby(to_quarter_end(monthly.index)).sum(min_count=1).rename("indicator")
        anchor = quarterly_anchors[component].dropna().rename("anchor")
        aligned = pd.concat([quarterly_indicator, anchor], axis=1, sort=False).dropna()
        if aligned.empty:
            rows.append(
                {
                    "component": component,
                    "quarters": 0,
                    "corr": float("nan"),
                    "mean_error": float("nan"),
                    "mean_abs_error": float("nan"),
                    "rmse": float("nan"),
                    "first_quarter": "",
                    "last_quarter": "",
                }
            )
            continue
        error = aligned["indicator"] - aligned["anchor"]
        rows.append(
            {
                "component": component,
                "quarters": int(len(aligned)),
                "corr": float(aligned["indicator"].corr(aligned["anchor"])) if len(aligned) > 1 else float("nan"),
                "mean_error": float(error.mean()),
                "mean_abs_error": float(error.abs().mean()),
                "rmse": float((error.pow(2).mean()) ** 0.5),
                "first_quarter": aligned.index.min().date().isoformat(),
                "last_quarter": aligned.index.max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows)
