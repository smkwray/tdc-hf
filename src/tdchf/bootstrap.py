from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .indicators import read_wide_time_series_csv
from .lp import cumulative_forward_sum
from .units import add_same_unit_columns


def _sample_lp_iv_frame(
    df: pd.DataFrame,
    *,
    outcome: str,
    horizon: int,
    treatment: str,
    instruments: Sequence[str],
    controls: Sequence[str],
    cumulative: bool,
) -> pd.DataFrame:
    dep = cumulative_forward_sum(df[outcome], horizon) if cumulative else df[outcome].shift(-horizon)
    sample = pd.DataFrame({"dep": dep, treatment: df[treatment]})
    for column in [*instruments, *controls]:
        sample[column] = df[column]
    return sample.dropna().reset_index(drop=True)


def _block_resample(sample: pd.DataFrame, *, block_length: int, rng: np.random.Generator) -> pd.DataFrame:
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    n = len(sample)
    starts = rng.integers(0, max(1, n - block_length + 1), size=int(np.ceil(n / block_length)))
    pieces = [sample.iloc[start : start + block_length] for start in starts]
    return pd.concat(pieces, ignore_index=True).iloc[:n].reset_index(drop=True)


def _lp_iv_beta(
    sample: pd.DataFrame,
    *,
    treatment: str,
    instruments: Sequence[str],
    controls: Sequence[str],
) -> float:
    first_x = sm.add_constant(sample[[*instruments, *controls]], has_constant="add")
    first_fit = sm.OLS(sample[treatment], first_x).fit()
    fitted = f"{treatment}_hat"
    second_sample = sample.copy()
    second_sample[fitted] = first_fit.fittedvalues
    second_x = sm.add_constant(second_sample[[fitted, *controls]], has_constant="add")
    second_fit = sm.OLS(second_sample["dep"], second_x).fit()
    return float(second_fit.params[fitted])


def bootstrap_lp_iv(
    df: pd.DataFrame,
    *,
    treatment: str,
    instruments: Sequence[str],
    outcomes: Sequence[str],
    controls: Sequence[str] = (),
    horizons: Sequence[int] = (0, 1, 2, 3, 6, 12),
    replications: int = 200,
    block_length: int = 6,
    seed: int = 12345,
    cumulative: bool = True,
) -> pd.DataFrame:
    required = [treatment, *instruments, *outcomes, *controls]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if replications <= 0:
        raise ValueError("replications must be positive")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        for horizon in horizons:
            sample = _sample_lp_iv_frame(
                df,
                outcome=outcome,
                horizon=int(horizon),
                treatment=treatment,
                instruments=instruments,
                controls=controls,
                cumulative=cumulative,
            )
            if len(sample) < len(instruments) + len(controls) + 10:
                continue
            baseline = _lp_iv_beta(sample, treatment=treatment, instruments=instruments, controls=controls)
            draws: list[float] = []
            failures = 0
            for _ in range(replications):
                boot_sample = _block_resample(sample, block_length=block_length, rng=rng)
                try:
                    value = _lp_iv_beta(boot_sample, treatment=treatment, instruments=instruments, controls=controls)
                except Exception:
                    failures += 1
                    continue
                if np.isfinite(value):
                    draws.append(float(value))
                else:
                    failures += 1
            if not draws:
                continue
            arr = np.asarray(draws, dtype=float)
            rows.append(
                {
                    "outcome": outcome,
                    "horizon": int(horizon),
                    "baseline_beta": baseline,
                    "bootstrap_mean": float(np.mean(arr)),
                    "bootstrap_se": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
                    "bootstrap_lower95": float(np.quantile(arr, 0.025)),
                    "bootstrap_upper95": float(np.quantile(arr, 0.975)),
                    "draws": int(len(arr)),
                    "failures": int(failures),
                    "block_length": int(block_length),
                    "replications_requested": int(replications),
                    "n": int(len(sample)),
                    "response_type": "cumulative_sum_h0_to_h" if cumulative else "lead_h",
                }
            )
    return add_same_unit_columns(
        pd.DataFrame(rows),
        treatment_col=treatment,
        value_columns=("baseline_beta", "bootstrap_mean", "bootstrap_se", "bootstrap_lower95", "bootstrap_upper95"),
    )


def bootstrap_lp_iv_csv(
    data_csv: str | Path,
    *,
    treatment: str,
    instruments: Sequence[str],
    outcomes: Sequence[str],
    controls: Sequence[str] = (),
    horizons: Sequence[int] = (0, 1, 2, 3, 6, 12),
    out_csv: str | Path,
    replications: int = 200,
    block_length: int = 6,
    seed: int = 12345,
    cumulative: bool = True,
) -> dict[str, object]:
    df = read_wide_time_series_csv(data_csv)
    result = bootstrap_lp_iv(
        df,
        treatment=treatment,
        instruments=instruments,
        outcomes=outcomes,
        controls=controls,
        horizons=horizons,
        replications=replications,
        block_length=block_length,
        seed=seed,
        cumulative=cumulative,
    )
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return {
        "status": "ok",
        "out": str(path),
        "rows": int(len(result)),
        "replications": int(replications),
        "block_length": int(block_length),
    }
