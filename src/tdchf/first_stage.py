from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from .indicators import read_wide_time_series_csv


def run_first_stage(
    df: pd.DataFrame,
    *,
    treatment: str,
    instruments: Sequence[str],
    controls: Sequence[str] = (),
) -> pd.DataFrame:
    required = [treatment, *instruments, *controls]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    sample = df[required].dropna()
    if len(sample) < len(required) + 8:
        raise ValueError("Not enough observations for first-stage regression")

    x_cols = [*instruments, *controls]
    x = sm.add_constant(sample[x_cols], has_constant="add")
    fit = sm.OLS(sample[treatment], x).fit()
    restriction = " = 0, ".join(instruments) + " = 0"
    ftest = fit.f_test(restriction)
    rows = [
        {
            "treatment": treatment,
            "instruments": ",".join(instruments),
            "controls": ",".join(controls),
            "n": int(fit.nobs),
            "r2": float(fit.rsquared),
            "excluded_instrument_f": float(ftest.fvalue),
            "excluded_instrument_pvalue": float(ftest.pvalue),
        }
    ]
    for instrument in instruments:
        rows.append(
            {
                "treatment": treatment,
                "instruments": instrument,
                "controls": ",".join(controls),
                "n": int(fit.nobs),
                "r2": float(fit.rsquared),
                "excluded_instrument_f": float("nan"),
                "excluded_instrument_pvalue": float("nan"),
                "coef": float(fit.params[instrument]),
                "se": float(fit.bse[instrument]),
                "t": float(fit.tvalues[instrument]),
                "pvalue": float(fit.pvalues[instrument]),
            }
        )
    return pd.DataFrame(rows)


def run_first_stage_csv(
    data_csv: str | Path,
    *,
    treatment: str,
    instruments: Sequence[str],
    controls: Sequence[str] = (),
    out_csv: str | Path,
) -> dict[str, object]:
    df = read_wide_time_series_csv(data_csv)
    result = run_first_stage(df, treatment=treatment, instruments=instruments, controls=controls)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return {"status": "ok", "out": str(path), "rows": int(len(result))}
