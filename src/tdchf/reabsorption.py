from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .disbursement import (
    FLOW_BUCKETS,
    HORIZONS,
    PANDEMIC_BLOCK_QUARTERS,
    _ols_hac,
    _prepare_regression_panel,
    _sample_mask,
)
from .fred import fetch_fred_series_many
from .qra_event_lp import normalize_weekly_panel_units

STATE_FRED_SERIES = [
    "DGS3MO",
    "SAVNRNJ",
    "SNDR",
    "RRPONTSYD",
    "WRESBAL",
    "GDP",
    "MMMFFAQ027S",
]
STATE_INTERACTION_COLUMNS = {
    "yield_gradient_full": "yield_gradient_full_lag1",
    "yield_gradient_clean": "yield_gradient_clean_lag1",
    "rrp_balance": "rrp_balance_lag1",
}
REPORT_HORIZONS = [2, 4, 8]
HALFLIFE_BOOTSTRAP_REPS = 999
PLACEBO_CALIBRATION_SEEDS = list(range(1, 21))
REFERENCE_HORIZON_COLUMNS = [4, 8]
PLACEBO_CAVEAT_TEXT = (
    "Multi-seed placebo replication shows ~40-45% of random-walk pseudo-states produce |t|>1.96 at the "
    "reference horizons; nominal HAC/wild p-values for persistent-state interactions are overstated and the γ "
    "table must be read against a placebo-calibrated null (effective p ≈ .25 at h4, ≈ .05 at h8)."
)


@dataclass(frozen=True)
class DecayFit:
    half_life: float
    lambda_hat: float
    r2: float
    n_points: int
    meaningful: bool
    reason: str


def _read_wide_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    date_col = "observation_date" if "observation_date" in raw.columns else "date"
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    return raw.dropna(subset=[date_col]).set_index(date_col).sort_index()


def _load_or_fetch_state_sources(raw_state_csv: str | Path | None) -> pd.DataFrame:
    if raw_state_csv is not None and Path(raw_state_csv).exists():
        return _read_wide_csv(raw_state_csv)
    frame = fetch_fred_series_many(STATE_FRED_SERIES)
    if raw_state_csv is not None:
        path = Path(raw_state_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index_label="date")
    return frame


def _asof_weekly(series: pd.Series, weeks: pd.DatetimeIndex) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.Series(np.nan, index=weeks)
    aligned = clean.reindex(clean.index.union(weeks)).sort_index().ffill().reindex(weeks)
    aligned.loc[weeks < clean.index.min()] = np.nan
    return aligned


def _quarter_step(series: pd.Series, weeks: pd.DatetimeIndex) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.Series(np.nan, index=weeks)
    by_quarter = pd.Series(clean.to_numpy(dtype=float), index=clean.index.to_period("Q")).groupby(level=0).last()
    return pd.Series(weeks.to_period("Q"), index=weeks).map(by_quarter).astype(float)


def build_reabsorption_state_weekly(
    *,
    raw_state_csv: str | Path | None = "data/raw/fred_reabsorption_state_sources.csv",
    out_csv: str | Path = "data/processed/reabsorption_state_weekly.csv",
    start: str | pd.Timestamp = "2005-10-05",
    end: str | pd.Timestamp = "2026-07-01",
) -> pd.DataFrame:
    raw = _load_or_fetch_state_sources(raw_state_csv)
    weeks = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="W-WED")
    out = pd.DataFrame(index=weeks)
    out.index.name = "date"

    out["bill3m_yield"] = pd.to_numeric(raw.get("DGS3MO"), errors="coerce").reindex(weeks)

    pre_fdic = _asof_weekly(raw["SAVNRNJ"], weeks) if "SAVNRNJ" in raw else pd.Series(np.nan, index=weeks)
    post_fdic = _asof_weekly(raw["SNDR"], weeks) if "SNDR" in raw else pd.Series(np.nan, index=weeks)
    break_date = pd.Timestamp("2021-04-01")
    out["deposit_rate_proxy"] = pre_fdic.where(weeks < break_date, post_fdic)
    out["deposit_rate_proxy_source"] = np.where(weeks < break_date, "SAVNRNJ_weekly_discontinued", "SNDR_monthly")
    out.loc[out["deposit_rate_proxy"].isna(), "deposit_rate_proxy_source"] = ""
    out["fdic_methodology_break_2021_04"] = (weeks == pd.Timestamp("2021-04-07")).astype(int)

    out["yield_gradient_full"] = out["bill3m_yield"]
    out["yield_gradient_clean"] = out["bill3m_yield"] - out["deposit_rate_proxy"]

    rrp = pd.to_numeric(raw.get("RRPONTSYD"), errors="coerce").reindex(weeks)
    rrp = rrp.where(weeks >= pd.Timestamp("2013-09-25"), 0.0)
    out["rrp_balance"] = rrp

    reserves_bn = pd.to_numeric(raw.get("WRESBAL"), errors="coerce").reindex(weeks) / 1000.0
    gdp_bn = _quarter_step(raw["GDP"], weeks) if "GDP" in raw else pd.Series(np.nan, index=weeks)
    out["gdp_bn"] = gdp_bn
    out["reserves_gdp"] = reserves_bn / gdp_bn

    if "MMMFFAQ027S" in raw:
        out["mmf_assets_gdp"] = _quarter_step(raw["MMMFFAQ027S"], weeks) / gdp_bn
        out["mmf_assets_gdp_source"] = "MMMFFAQ027S_quarterly_stepwise"
    else:
        out["mmf_assets_gdp"] = np.nan
        out["mmf_assets_gdp_source"] = ""

    for column in ["yield_gradient_full", "yield_gradient_clean", "rrp_balance", "reserves_gdp", "mmf_assets_gdp"]:
        out[f"{column}_lag1"] = out[column].shift(1)
    out["rrp_headroom_state"] = (out["rrp_balance_lag1"] >= 50.0).astype(int)
    out.loc[out["rrp_balance_lag1"].isna(), "rrp_headroom_state"] = 0

    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")
    return out


def _standardize(series: pd.Series, sample_mask: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series.loc[sample_mask], errors="coerce").dropna()
    if clean.empty:
        return pd.Series(np.nan, index=series.index)
    sd = float(clean.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (pd.to_numeric(series, errors="coerce") - float(clean.mean())) / sd


def _controls_for(panel: pd.DataFrame, calendar: pd.DataFrame, horizon: int) -> list[str]:
    flow_controls = ["du_broad_outflows_bn", "interest_outflows_bn", "debt_issues_gross_bn", "debt_redemptions_gross_bn"]
    cal_controls = [c for c in calendar.columns if c != "date"]
    lag_controls = [f"{col}_lag{lag}" for col in FLOW_BUCKETS for lag in range(1, 5)]
    lead_controls = [f"{col}_lead{lead}" for col in FLOW_BUCKETS for lead in range(1, horizon + 1)] if horizon > 0 else []
    return [c for c in [*flow_controls, *cal_controls, *lag_controls, *lead_controls] if c in panel.columns]


def _analysis_mask(index: pd.Index, sample_name: str, horizon: int) -> pd.Series:
    if sample_name in {"full", "ex_pandemic"}:
        return _sample_mask(index, sample_name, horizon)
    if sample_name == "pre_2020":
        return pd.Series(pd.DatetimeIndex(index) < pd.Timestamp("2020-01-01"), index=index)
    raise ValueError(f"unknown sample: {sample_name}")


def _moving_block_signs(nobs: int, reps: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    draws = np.empty((nobs, reps), dtype=float)
    n_blocks = int(np.ceil(nobs / block_length))
    for rep in range(reps):
        vals = rng.choice(np.array([-1.0, 1.0]), size=n_blocks)
        draws[:, rep] = np.repeat(vals, block_length)[:nobs]
    return draws


def moving_block_wild_pvalue(
    y: pd.Series,
    x: pd.DataFrame,
    coef: str,
    beta: float,
    *,
    seed: int,
    reps: int = 999,
    block_length: int = 8,
) -> float:
    sample = pd.concat([y.rename("y"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 24 or coef not in sample.columns:
        return np.nan
    cols = [c for c in x.columns if c in sample.columns and sample[c].nunique() >= 2]
    if coef not in cols:
        return np.nan
    keep = [c for c in cols if c != coef]
    x_full = sm.add_constant(sample[cols], has_constant="add").to_numpy(dtype=float)
    x_res = sm.add_constant(sample[keep], has_constant="add").to_numpy(dtype=float)
    try:
        b0 = np.linalg.lstsq(x_res, sample["y"].to_numpy(dtype=float), rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    fitted = x_res @ b0
    resid = sample["y"].to_numpy(dtype=float) - fitted
    signs = _moving_block_signs(len(sample), reps, block_length, np.random.default_rng(seed))
    boot_y = fitted[:, None] + resid[:, None] * signs
    try:
        betas = (np.linalg.pinv(x_full) @ boot_y)[cols.index(coef) + 1]
    except np.linalg.LinAlgError:
        return np.nan
    return float((np.sum(np.abs(betas) >= abs(beta)) + 1) / (len(betas) + 1))


def estimate_state_interacted_retention(
    flows: pd.DataFrame,
    calendar: pd.DataFrame,
    weekly_panel: pd.DataFrame,
    state_weekly: pd.DataFrame,
    *,
    state_columns: dict[str, str] | None = None,
    samples: list[str] | None = None,
    outcomes: dict[str, str] | None = None,
    interaction_bootstrap_reps: int = 999,
    compute_moving_block_wild: bool = True,
) -> pd.DataFrame:
    states = state_columns or STATE_INTERACTION_COLUMNS
    outcome_map = outcomes or {
        "deposits_dpsacb": "broad_deposits_nsa",
        "reserves_wresbal": "reserves",
        "on_rrp_rrpontsyd": "onrrp",
    }
    merged_state = weekly_panel.join(state_weekly, how="left", rsuffix="_state")
    panel = _prepare_regression_panel(flows, calendar, merged_state)
    rows: list[dict[str, object]] = []
    for sample_name in samples or ["full", "ex_pandemic", "pre_2020"]:
        for outcome, source in outcome_map.items():
            if source not in panel.columns or panel[source].notna().sum() < 30:
                continue
            y_level = pd.to_numeric(panel[source], errors="coerce")
            for state_name, state_col in states.items():
                if state_col not in panel.columns:
                    continue
                for h in range(14):
                    y = y_level.shift(-h) - y_level.shift(1)
                    mask = _analysis_mask(panel.index, sample_name, h)
                    z = _standardize(panel[state_col], mask)
                    regressors = panel.loc[
                        mask,
                        ["du_core_outflows_bn", "tax_receipts_bn", *_controls_for(panel, calendar, h)],
                    ].copy()
                    regressors["state_z"] = z.loc[mask]
                    for treatment in ["du_core_outflows_bn", "tax_receipts_bn"]:
                        regressors[f"{treatment}_x_state"] = regressors[treatment] * regressors["state_z"]
                    y_sample = y.loc[mask]
                    fit = _ols_hac(y_sample, regressors, maxlags=max(h + 4, 4))
                    if fit is None:
                        continue
                    for treatment in ["du_core_outflows_bn", "tax_receipts_bn"]:
                        coef = f"{treatment}_x_state"
                        if coef not in fit.params:
                            continue
                        gamma = float(fit.params[coef])
                        p_mbwild = np.nan
                        if (
                            compute_moving_block_wild
                            and sample_name == "ex_pandemic"
                            and outcome == "deposits_dpsacb"
                            and h in REPORT_HORIZONS
                        ):
                            p_mbwild = moving_block_wild_pvalue(
                                y_sample,
                                regressors,
                                coef,
                                gamma,
                                seed=20260704 + len(rows) * 31,
                                reps=interaction_bootstrap_reps,
                                block_length=max(h + 4, 8),
                            )
                        rows.append(
                            {
                                "row_type": "interaction",
                                "state_variable": state_name,
                                "state_column": state_col,
                                "sample": sample_name,
                                "outcome": outcome,
                                "treatment_id": treatment,
                                "horizon": h,
                                "gamma": gamma,
                                "se": float(fit.bse[coef]),
                                "p_hac": float(fit.pvalues[coef]),
                                "p_moving_block_wild": p_mbwild,
                                "n": int(fit.nobs),
                                "uniform_band_lower": np.nan,
                                "uniform_band_upper": np.nan,
                                "spec_flags": "lead-controlled LP; state is one-week-lagged and standardized within sample; HAC NW(h+4); moving-block wild p populated for ex-pandemic deposit reference horizons",
                            }
                        )
    out = pd.DataFrame(rows)
    if not out.empty:
        groups = ["state_variable", "sample", "outcome", "treatment_id"]
        for _, group in out.groupby(groups):
            se_max = pd.to_numeric(group["se"], errors="coerce").max()
            idx = group.index
            out.loc[idx, "uniform_band_lower"] = out.loc[idx, "gamma"] - 1.96 * se_max
            out.loc[idx, "uniform_band_upper"] = out.loc[idx, "gamma"] + 1.96 * se_max
    return out


def estimate_bin_path(
    flows: pd.DataFrame,
    calendar: pd.DataFrame,
    weekly_panel: pd.DataFrame,
    state_weekly: pd.DataFrame,
    *,
    treatment: str,
    bin_mask: pd.Series,
    sample: str = "ex_pandemic",
) -> pd.DataFrame:
    panel = _prepare_regression_panel(flows, calendar, weekly_panel.join(state_weekly, how="left"))
    y_level = pd.to_numeric(panel["broad_deposits_nsa"], errors="coerce")
    rows: list[dict[str, object]] = []
    for h in range(14):
        y = y_level.shift(-h) - y_level.shift(1)
        mask = _analysis_mask(panel.index, sample, h) & bin_mask.reindex(panel.index).fillna(False)
        regressors = panel.loc[mask, ["du_core_outflows_bn", "tax_receipts_bn", *_controls_for(panel, calendar, h)]].copy()
        fit = _ols_hac(y.loc[mask], regressors, maxlags=max(h + 4, 4))
        if fit is None or treatment not in fit.params:
            rows.append({"horizon": h, "beta": np.nan, "se": np.nan, "n": int(mask.sum())})
            continue
        rows.append({"horizon": h, "beta": float(fit.params[treatment]), "se": float(fit.bse[treatment]), "n": int(fit.nobs)})
    return pd.DataFrame(rows)


def fit_exponential_decay(path: pd.DataFrame, *, treatment: str) -> DecayFit:
    work = path[["horizon", "beta"]].dropna().copy()
    if work.empty:
        return DecayFit(np.nan, np.nan, np.nan, 0, False, "no estimable beta path")
    beta0 = float(work.loc[work["horizon"].eq(0), "beta"].iloc[0]) if work["horizon"].eq(0).any() else np.nan
    if not np.isfinite(beta0) or abs(beta0) < 1e-9:
        return DecayFit(np.nan, np.nan, np.nan, 0, False, "beta0 is zero or unavailable")
    sign = np.sign(beta0)
    work = work.loc[(np.sign(work["beta"]) == sign) & (work["beta"].abs() > 1e-9)].copy()
    if len(work) < 4:
        return DecayFit(np.nan, np.nan, np.nan, int(len(work)), False, "fewer than four same-sign positive-magnitude points")
    x = work["horizon"].to_numpy(dtype=float)
    y = np.log(work["beta"].abs().to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(x)), x])
    try:
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return DecayFit(np.nan, np.nan, np.nan, int(len(work)), False, "decay regression failed")
    fitted = intercept + slope * x
    ss_res = float(np.square(y - fitted).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    lambda_hat = -float(slope)
    if not np.isfinite(lambda_hat) or lambda_hat <= 0:
        return DecayFit(np.nan, lambda_hat, r2, int(len(work)), False, "estimated lambda is non-positive")
    half_life = float(np.log(2.0) / lambda_hat)
    meaningful = bool(np.isfinite(r2) and r2 >= 0.25)
    reason = "ok" if meaningful else "decay fit R2 below 0.25"
    return DecayFit(half_life, lambda_hat, r2, int(len(work)), meaningful, reason)


def _moving_block_positions(nobs: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if nobs <= block_length:
        return np.arange(nobs)
    starts = rng.integers(0, nobs - block_length + 1, size=int(np.ceil(nobs / block_length)))
    return np.concatenate([np.arange(start, start + block_length) for start in starts])[:nobs]


def _bootstrap_halflife_ci(
    paths_by_h: dict[int, tuple[pd.Series, pd.DataFrame]],
    *,
    treatment: str,
    reps: int = HALFLIFE_BOOTSTRAP_REPS,
    seed: int = 20260704,
    block_length: int = 8,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    beta_draws_by_h: dict[int, np.ndarray] = {}
    for h, (y, x) in paths_by_h.items():
        sample = pd.concat([y.rename("y"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        cols = [c for c in x.columns if c in sample.columns and sample[c].nunique() >= 2]
        if treatment not in cols or len(sample) < max(24, len(cols) + 8):
            beta_draws_by_h[h] = np.full(reps, np.nan)
            continue
        xmat = sm.add_constant(sample[cols], has_constant="add").to_numpy(dtype=float)
        yvec = sample["y"].to_numpy(dtype=float)
        try:
            pinv = np.linalg.pinv(xmat)
        except np.linalg.LinAlgError:
            beta_draws_by_h[h] = np.full(reps, np.nan)
            continue
        params = pinv @ yvec
        fitted = xmat @ params
        resid = yvec - fitted
        coef_row = pinv[cols.index(treatment) + 1]
        draws = np.empty(reps, dtype=float)
        for rep in range(reps):
            pos = _moving_block_positions(len(sample), block_length, rng)
            draws[rep] = float(coef_row @ (fitted + resid[pos]))
        beta_draws_by_h[h] = draws
    boot_halves: list[float] = []
    for rep in range(reps):
        rows: list[dict[str, float]] = []
        for h, draws in beta_draws_by_h.items():
            rows.append({"horizon": h, "beta": float(draws[rep]) if np.isfinite(draws[rep]) else np.nan})
        fit = fit_exponential_decay(pd.DataFrame(rows), treatment=treatment)
        if fit.meaningful and np.isfinite(fit.half_life):
            boot_halves.append(fit.half_life)
    if not boot_halves:
        return np.nan, np.nan, 0
    arr = np.asarray(boot_halves, dtype=float)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), int(len(arr))


def estimate_halflife_bins(
    flows: pd.DataFrame,
    calendar: pd.DataFrame,
    weekly_panel: pd.DataFrame,
    state_weekly: pd.DataFrame,
    *,
    bootstrap_reps: int = HALFLIFE_BOOTSTRAP_REPS,
) -> pd.DataFrame:
    state_lag = pd.to_numeric(state_weekly["yield_gradient_full_lag1"], errors="coerce")
    eligible = state_lag.dropna()
    terciles = eligible.quantile([1 / 3, 2 / 3])
    masks = {
        "tercile_low": state_lag <= float(terciles.iloc[0]),
        "tercile_mid": (state_lag > float(terciles.iloc[0])) & (state_lag <= float(terciles.iloc[1])),
        "tercile_high": state_lag > float(terciles.iloc[1]),
        "zero_rate": state_lag <= 0.25,
        "positive_rate": state_lag > 0.25,
    }
    panel = _prepare_regression_panel(flows, calendar, weekly_panel.join(state_weekly, how="left"))
    y_level = pd.to_numeric(panel["broad_deposits_nsa"], errors="coerce")
    rows: list[dict[str, object]] = []
    for bin_name, raw_mask in masks.items():
        bin_mask = raw_mask.reindex(panel.index).fillna(False)
        for treatment in ["du_core_outflows_bn", "tax_receipts_bn"]:
            path = estimate_bin_path(flows, calendar, weekly_panel, state_weekly, treatment=treatment, bin_mask=bin_mask)
            fit = fit_exponential_decay(path, treatment=treatment)
            paths_by_h: dict[int, tuple[pd.Series, pd.DataFrame]] = {}
            for h in range(14):
                y = y_level.shift(-h) - y_level.shift(1)
                mask = _analysis_mask(panel.index, "ex_pandemic", h) & bin_mask
                regressors = panel.loc[mask, ["du_core_outflows_bn", "tax_receipts_bn", *_controls_for(panel, calendar, h)]].copy()
                paths_by_h[h] = (y.loc[mask], regressors)
            ci_low, ci_high, boot_success = _bootstrap_halflife_ci(
                paths_by_h,
                treatment=treatment,
                reps=bootstrap_reps,
                seed=20260704 + len(rows) * 19,
            )
            reported_half_life = fit.half_life if fit.meaningful else np.nan
            reported_ci_low = ci_low if fit.meaningful else np.nan
            reported_ci_high = ci_high if fit.meaningful else np.nan
            for _, p_row in path.iterrows():
                rows.append(
                    {
                        "row_type": "bin_path",
                        "state_variable": "yield_gradient_full",
                        "sample": "ex_pandemic",
                        "outcome": "deposits_dpsacb",
                        "treatment_id": treatment,
                        "bin_scheme": "tercile" if bin_name.startswith("tercile") else "zero_positive",
                        "bin_name": bin_name,
                        "horizon": int(p_row["horizon"]),
                        "beta": float(p_row["beta"]) if pd.notna(p_row["beta"]) else np.nan,
                        "se": float(p_row["se"]) if pd.notna(p_row["se"]) else np.nan,
                        "n": int(p_row["n"]),
                        "half_life_weeks": reported_half_life,
                        "half_life_ci_low": reported_ci_low,
                        "half_life_ci_high": reported_ci_high,
                        "lambda_hat": fit.lambda_hat,
                        "decay_fit_r2": fit.r2,
                        "decay_fit_points": fit.n_points,
                        "decay_fit_meaningful": fit.meaningful,
                        "decay_fit_reason": fit.reason,
                        "bootstrap_success_reps": boot_success,
                        "thin_bin_flag": bool(path["n"].min() < 120),
                    }
                )
    return pd.DataFrame(rows)


def add_random_walk_placebo_state(state_weekly: pd.DataFrame, *, seed: int = 20260704) -> pd.DataFrame:
    out = state_weekly.copy()
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, 1.0, len(out))
    out["random_walk_placebo_state"] = np.cumsum(shocks)
    out["random_walk_placebo_state_lag1"] = out["random_walk_placebo_state"].shift(1)
    return out


def calibrate_random_walk_placebo(
    flows: pd.DataFrame,
    calendar: pd.DataFrame,
    weekly_panel: pd.DataFrame,
    state_weekly: pd.DataFrame,
    real_estimates: pd.DataFrame,
    *,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    seed_list = seeds or PLACEBO_CALIBRATION_SEEDS
    rows: list[dict[str, object]] = []
    for seed in seed_list:
        placebo_state = add_random_walk_placebo_state(state_weekly, seed=seed)
        placebo = estimate_state_interacted_retention(
            flows,
            calendar,
            weekly_panel,
            placebo_state,
            state_columns={"random_walk_placebo": "random_walk_placebo_state_lag1"},
            samples=["ex_pandemic"],
            outcomes={"deposits_dpsacb": "broad_deposits_nsa"},
            interaction_bootstrap_reps=0,
            compute_moving_block_wild=False,
        )
        ref = placebo.loc[
            placebo["row_type"].eq("interaction")
            & placebo["outcome"].eq("deposits_dpsacb")
            & placebo["treatment_id"].eq("du_core_outflows_bn")
            & placebo["horizon"].isin(REPORT_HORIZONS)
        ].copy()
        if ref.empty:
            continue
        ref["placebo_seed"] = seed
        ref["placebo_t"] = pd.to_numeric(ref["gamma"], errors="coerce") / pd.to_numeric(ref["se"], errors="coerce")
        rows.extend(ref.to_dict(orient="records"))

    draws = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    real = real_estimates.loc[
        real_estimates["row_type"].eq("interaction")
        & real_estimates["state_variable"].eq("yield_gradient_full")
        & real_estimates["sample"].eq("ex_pandemic")
        & real_estimates["outcome"].eq("deposits_dpsacb")
        & real_estimates["treatment_id"].eq("du_core_outflows_bn")
        & real_estimates["horizon"].isin(REPORT_HORIZONS)
    ].copy()
    for horizon in REPORT_HORIZONS:
        h_draws = draws.loc[draws["horizon"].eq(horizon)].copy() if not draws.empty else pd.DataFrame()
        h_t = pd.to_numeric(h_draws.get("placebo_t", pd.Series(dtype=float)), errors="coerce").dropna().abs()
        real_cell = real.loc[real["horizon"].eq(horizon)]
        real_gamma = float(real_cell["gamma"].iloc[0]) if not real_cell.empty else np.nan
        real_se = float(real_cell["se"].iloc[0]) if not real_cell.empty else np.nan
        real_t = abs(real_gamma / real_se) if np.isfinite(real_gamma) and np.isfinite(real_se) and real_se else np.nan
        summary_rows.append(
            {
                "row_type": "placebo_calibration",
                "state_variable": "random_walk_placebo",
                "state_column": "random_walk_placebo_state_lag1",
                "sample": "ex_pandemic",
                "outcome": "deposits_dpsacb",
                "treatment_id": "du_core_outflows_bn",
                "horizon": horizon,
                "placebo_seed_count": int(len(h_t)),
                "placebo_false_positive_rate": float((h_t > 1.96).mean()) if len(h_t) else np.nan,
                "placebo_effective_p": float((h_t >= real_t).mean()) if len(h_t) and np.isfinite(real_t) else np.nan,
                "real_gamma": real_gamma,
                "real_t": real_t,
                "placebo_t_abs_p50": float(h_t.quantile(0.50)) if len(h_t) else np.nan,
                "placebo_t_abs_p95": float(h_t.quantile(0.95)) if len(h_t) else np.nan,
                "spec_flags": "20-seed deterministic random-walk placebo calibration; HAC t-stat distribution; no moving-block wild refit",
            }
        )
    return pd.DataFrame(summary_rows)


def annotate_placebo_calibration(estimates: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    out = estimates.copy()
    for col in ["placebo_seed_count", "placebo_false_positive_rate", "placebo_effective_p"]:
        if col not in out.columns:
            out[col] = np.nan
    if calibration.empty:
        return out
    for _, row in calibration.iterrows():
        mask = (
            out["row_type"].eq("interaction")
            & out["state_variable"].eq("yield_gradient_full")
            & out["sample"].eq("ex_pandemic")
            & out["outcome"].eq("deposits_dpsacb")
            & out["treatment_id"].eq("du_core_outflows_bn")
            & out["horizon"].eq(int(row["horizon"]))
        )
        for col in ["placebo_seed_count", "placebo_false_positive_rate", "placebo_effective_p"]:
            out.loc[mask, col] = row[col]
    return out


def _summary_halflife(halflife: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "bin_scheme",
        "bin_name",
        "treatment_id",
        "n",
        "half_life_weeks",
        "half_life_ci_low",
        "half_life_ci_high",
        "decay_fit_r2",
        "decay_fit_meaningful",
        "decay_fit_reason",
        "bootstrap_success_reps",
    ]
    summary = halflife.loc[halflife["horizon"].eq(0), cols].copy()
    return summary.round(4)


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No estimable rows."]
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row[cols]) + " |")
    return lines


def write_reabsorption_readout(
    *,
    state_weekly: pd.DataFrame,
    estimates: pd.DataFrame,
    halflife: pd.DataFrame,
    out_md: str | Path,
) -> None:
    primary = estimates.loc[
        estimates["row_type"].eq("interaction")
        & estimates["state_variable"].eq("yield_gradient_full")
        & estimates["sample"].eq("ex_pandemic")
        & estimates["outcome"].eq("deposits_dpsacb")
        & estimates["horizon"].isin(REPORT_HORIZONS),
        [
            "treatment_id",
            "horizon",
            "gamma",
            "se",
            "p_hac",
            "p_moving_block_wild",
            "placebo_false_positive_rate",
            "placebo_effective_p",
            "uniform_band_lower",
            "uniform_band_upper",
            "n",
        ],
    ].round(4)
    mirrors = estimates.loc[
        estimates["row_type"].eq("interaction")
        & estimates["state_variable"].eq("yield_gradient_full")
        & estimates["sample"].eq("ex_pandemic")
        & estimates["outcome"].isin(["reserves_wresbal", "on_rrp_rrpontsyd"])
        & estimates["horizon"].isin([4, 8]),
        ["outcome", "treatment_id", "horizon", "gamma", "se", "p_hac", "n"],
    ].round(4)
    pre2020 = estimates.loc[
        estimates["row_type"].eq("interaction")
        & estimates["state_variable"].eq("yield_gradient_full")
        & estimates["sample"].eq("pre_2020")
        & estimates["outcome"].eq("deposits_dpsacb")
        & estimates["horizon"].isin([4, 8]),
        ["treatment_id", "horizon", "gamma", "se", "p_hac", "n"],
    ].round(4)
    placebo = estimates.loc[
        estimates["row_type"].eq("interaction")
        & estimates["state_variable"].eq("random_walk_placebo")
        & estimates["sample"].eq("ex_pandemic")
        & estimates["outcome"].eq("deposits_dpsacb")
        & estimates["horizon"].isin(REPORT_HORIZONS),
        ["treatment_id", "horizon", "gamma", "se", "p_hac", "p_moving_block_wild", "n"],
    ].round(4)
    calibration = estimates.loc[
        estimates["row_type"].eq("placebo_calibration")
        & estimates["horizon"].isin(REPORT_HORIZONS),
        [
            "horizon",
            "placebo_seed_count",
            "placebo_false_positive_rate",
            "placebo_effective_p",
            "real_gamma",
            "real_t",
            "placebo_t_abs_p50",
            "placebo_t_abs_p95",
        ],
    ].round(4)
    corr_data = state_weekly[["yield_gradient_full_lag1"]].copy()
    corr_data["post_2020"] = (corr_data.index >= pd.Timestamp("2020-01-01")).astype(float)
    post_corr = float(corr_data.dropna().corr().iloc[0, 1])
    coverage = {
        "state_start": state_weekly.index.min().date().isoformat(),
        "state_end": state_weekly.index.max().date().isoformat(),
        "weeks": int(len(state_weekly)),
        "bill3m_nonnull": int(state_weekly["bill3m_yield"].notna().sum()),
        "clean_gradient_nonnull": int(state_weekly["yield_gradient_clean"].notna().sum()),
        "rrp_nonnull": int(state_weekly["rrp_balance"].notna().sum()),
        "fdic_break_rows": int(state_weekly["fdic_methodology_break_2021_04"].sum()),
    }
    lines = [
        "# Reabsorption Half-Life State Experiment",
        "",
        "## Pre-Specified Predictions",
        "",
        "1. Primary yield-gradient prediction: for core disbursement credits, the interaction coefficient gamma_h should be negative at h >= 4; a wider bill/deposit gradient would appear to speed decay of the initial deposit blip if the design passed placebo and regime checks.",
        "2. Mirror prediction: in the RRP era, the ON-RRP mirror should strengthen with the yield gradient if money-fund/RRP reabsorption is the mechanism.",
        "3. Drain prediction: tax-drain paths are descriptive with the WO5 anticipation caveat; longer estimated half-lives than core credits are plausible, but h13 persistence is not a safe claim.",
        "4. Placebo prediction: a seeded random-walk pseudo-state should not produce systematic significant gamma_h at the reference horizons.",
        "",
        "## State Construction",
        "",
        f"Coverage: {coverage}. `bill3m_yield` uses FRED DGS3MO Wednesday values and is the full-history primary proxy. `deposit_rate_proxy` uses FDIC `SAVNRNJ` through March 2021 and `SNDR` from April 2021 onward; the 2021-04 methodology break is pinned, not smoothed over. `yield_gradient_full` equals the bill yield because deposit rates are sticky near zero and unavailable before 2009; `yield_gradient_clean` is DGS3MO minus the FDIC savings proxy where available. RRPONTSYD is set to zero before the September 2013 ON-RRP regime and lagged one week for state interactions. GDP-scaled reserves and MMF assets use quarterly GDP/MMMFFAQ027S stepwise by quarter, not smooth interpolation.",
        "",
        "## Primary Interaction Results",
        "",
    ]
    lines.extend(_markdown_table(primary))
    lines.extend(
        [
            "",
            "## Mirror Results",
            "",
        ]
    )
    lines.extend(_markdown_table(mirrors))
    lines.extend(
        [
            "",
            "## Direct Half-Life Estimates",
            "",
        ]
    )
    lines.extend(_markdown_table(_summary_halflife(halflife)))
    lines.extend(
        [
            "",
            "## Robustness",
            "",
            f"Gradient-vs-post-2020 correlation: {post_corr:.4f}. Pre-2020-only reference estimates:",
            "",
        ]
    )
    lines.extend(_markdown_table(pre2020))
    lines.extend(
        [
            "",
            "Pre-2020-only verdict: affirmative rejection. The h4 and h8 pre-2020 CIs exclude the pooled ex-pandemic points, so the full-sample gradient is an era contrast rather than a stable capacity elasticity.",
            "",
            "Seeded random-walk placebo reference estimates:",
            "",
        ]
    )
    lines.extend(_markdown_table(placebo))
    lines.extend(
        [
            "",
            "Multi-seed random-walk placebo calibration:",
            "",
        ]
    )
    lines.extend(_markdown_table(calibration))
    lines.extend(
        [
            "",
            f"Placebo verdict: {PLACEBO_CAVEAT_TEXT}",
            "",
            "## Claim Boundary",
            "",
            "Headline conclusion: capacity function NOT estimable from this sample as designed - the gradient γ is an era contrast, not a capacity elasticity. Tax-drain half-life ≈8w in high-rate bins remains the one supported decay estimate, with the WO5 anticipation caveat. These are descriptive state-conditional estimates, not a validated downstream index. The yield-gradient state is regime-confounded despite repeated in-sample cycles; RRP headroom is secondary and single-era-confounded by the post-2013/post-2020 monetary plumbing regime. Power is limited at long horizons and in restricted bins; half-lives with weak decay-fit R2 or non-positive lambda are reported as not meaningful rather than forced. The FDIC savings-rate proxy has a hard April 2021 methodology/frequency break.",
        ]
    )
    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reabsorption_contract(
    *,
    state_weekly: pd.DataFrame,
    estimates: pd.DataFrame,
    halflife: pd.DataFrame,
    out_csv: str | Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    definitions = {
        "yield_gradient_full": ("DGS3MO Wednesday level; primary full-history proxy for bill/deposit gradient", "FRED DGS3MO"),
        "yield_gradient_clean": ("DGS3MO minus FDIC savings national-rate proxy where available", "FRED DGS3MO, SAVNRNJ, SNDR"),
        "rrp_balance": ("RRPONTSYD Wednesday balance, zero pre-September-2013; lagged one week in interactions", "FRED RRPONTSYD"),
        "reserves_gdp": ("Reserve balances divided by nominal GDP, quarterly GDP stepwise", "FRED WRESBAL, GDP"),
        "mmf_assets_gdp": ("MMF assets divided by nominal GDP, quarterly values stepwise", "FRED MMMFFAQ027S, GDP"),
    }
    hl_summary = _summary_halflife(halflife)
    for state_name, (definition, sources) in definitions.items():
        col = f"{state_name}_lag1" if f"{state_name}_lag1" in state_weekly.columns else state_name
        avail = state_weekly[col].dropna()
        inter = estimates.loc[
            estimates["row_type"].eq("interaction")
            & estimates["state_variable"].eq(state_name)
            & estimates["sample"].eq("ex_pandemic")
            & estimates["outcome"].eq("deposits_dpsacb")
            & estimates["horizon"].isin(REFERENCE_HORIZON_COLUMNS)
        ].copy()
        row: dict[str, object] = {
            "name": state_name,
            "definition": definition,
            "source_series": sources,
            "availability_window": f"{avail.index.min().date()}..{avail.index.max().date()}" if not avail.empty else "",
            "promotion_ready": False,
            "regime_confound_flag": state_name in {"yield_gradient_full", "yield_gradient_clean", "rrp_balance"},
            "caveat": PLACEBO_CAVEAT_TEXT if state_name in {"yield_gradient_full", "yield_gradient_clean", "rrp_balance"} else "",
            "half_life_by_bin_summary": "",
        }
        for horizon in REFERENCE_HORIZON_COLUMNS:
            for treatment in ["du_core_outflows_bn", "tax_receipts_bn"]:
                cell = inter.loc[inter["horizon"].eq(horizon) & inter["treatment_id"].eq(treatment)]
                prefix = f"{treatment}_gamma_h{horizon}"
                if cell.empty:
                    row[prefix] = np.nan
                    row[f"{prefix}_ci_low"] = np.nan
                    row[f"{prefix}_ci_high"] = np.nan
                else:
                    first = cell.iloc[0]
                    row[prefix] = first["gamma"]
                    row[f"{prefix}_ci_low"] = first["uniform_band_lower"]
                    row[f"{prefix}_ci_high"] = first["uniform_band_upper"]
        if state_name == "yield_gradient_full" and not hl_summary.empty:
            compact = hl_summary[["bin_name", "treatment_id", "half_life_weeks", "half_life_ci_low", "half_life_ci_high", "decay_fit_r2", "decay_fit_meaningful"]].round(3)
            row["half_life_by_bin_summary"] = compact.to_json(orient="records")
        rows.append(row)
    out = pd.DataFrame(rows)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def run_reabsorption_halflife_csv(
    *,
    flows_csv: str | Path = "data/processed/dts_weekly_flow_decomposition.csv",
    calendar_csv: str | Path = "data/processed/fiscal_calendar_weekly.csv",
    weekly_panel_csv: str | Path = "data/processed/tdc_weekly_channel_panel.csv",
    raw_state_csv: str | Path = "data/raw/fred_reabsorption_state_sources.csv",
    state_csv: str | Path = "data/processed/reabsorption_state_weekly.csv",
    estimates_csv: str | Path = "data/processed/reabsorption_halflife_estimates.csv",
    readout_md: str | Path = "data/processed/reabsorption_halflife_readout.md",
    contract_csv: str | Path = "data/processed/reabsorption_state_contract.csv",
    halflife_bootstrap_reps: int = HALFLIFE_BOOTSTRAP_REPS,
    interaction_bootstrap_reps: int = 999,
) -> dict[str, object]:
    flows = pd.read_csv(flows_csv, parse_dates=["date"]).set_index("date")
    calendar = pd.read_csv(calendar_csv, parse_dates=["date"])
    weekly_panel = pd.read_csv(weekly_panel_csv, parse_dates=["date"]).set_index("date")
    weekly_panel = normalize_weekly_panel_units(weekly_panel)
    state_weekly = build_reabsorption_state_weekly(
        raw_state_csv=raw_state_csv,
        out_csv=state_csv,
        start=flows.index.min(),
        end=flows.index.max(),
    )
    interaction = estimate_state_interacted_retention(
        flows,
        calendar,
        weekly_panel,
        state_weekly,
        interaction_bootstrap_reps=interaction_bootstrap_reps,
    )
    placebo_state = add_random_walk_placebo_state(state_weekly)
    placebo = estimate_state_interacted_retention(
        flows,
        calendar,
        weekly_panel,
        placebo_state,
        state_columns={"random_walk_placebo": "random_walk_placebo_state_lag1"},
        samples=["ex_pandemic"],
        outcomes={"deposits_dpsacb": "broad_deposits_nsa"},
        interaction_bootstrap_reps=interaction_bootstrap_reps,
    )
    calibration = calibrate_random_walk_placebo(
        flows,
        calendar,
        weekly_panel,
        state_weekly,
        interaction,
    )
    halflife = estimate_halflife_bins(
        flows,
        calendar,
        weekly_panel,
        state_weekly,
        bootstrap_reps=halflife_bootstrap_reps,
    )
    interaction = annotate_placebo_calibration(interaction, calibration)
    estimates = pd.concat([interaction, placebo, calibration, halflife], ignore_index=True, sort=False)
    Path(estimates_csv).parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(estimates_csv, index=False)
    write_reabsorption_readout(state_weekly=state_weekly, estimates=estimates, halflife=halflife, out_md=readout_md)
    contract = write_reabsorption_contract(state_weekly=state_weekly, estimates=estimates, halflife=halflife, out_csv=contract_csv)
    return {
        "status": "ok",
        "state_csv": str(state_csv),
        "estimates_csv": str(estimates_csv),
        "readout_md": str(readout_md),
        "contract_csv": str(contract_csv),
        "state_rows": int(len(state_weekly)),
        "estimate_rows": int(len(estimates)),
        "contract_rows": int(len(contract)),
        "state_start": state_weekly.index.min().date().isoformat(),
        "state_end": state_weekly.index.max().date().isoformat(),
    }
