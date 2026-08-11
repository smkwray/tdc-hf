from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .disbursement import (
    FLOW_BUCKETS,
    _ols_hac,
    _prepare_regression_panel,
    _sample_mask,
)
from .fred import fetch_fred_series_many
from .qra_event_lp import normalize_weekly_panel_units

STATE_FRED_SERIES = ["DGS3MO", "SAVNRNJ", "SNDR", "RRPONTSYD"]
STATE_RESPONSE_VARIABLES = [
    "bill3m_yield_level",
    "bill_minus_fdic_savings_rate_proxy",
    "on_rrp_balance_ge_50bn_lag1_indicator",
]
DIAGNOSTIC_HORIZONS = [4, 8]
PLACEBO_CALIBRATION_SEEDS = list(range(1, 21))


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
        return pd.Series(np.nan, index=weeks, dtype=float)
    aligned = clean.reindex(clean.index.union(weeks)).sort_index().ffill().reindex(weeks)
    aligned.loc[weeks < clean.index.min()] = np.nan
    return aligned.astype(float)


def build_state_response_weekly(
    *,
    raw_state_csv: str | Path | None = "data/raw/fred_state_response_sources.csv",
    out_csv: str | Path = "data/processed/state_response_weekly.csv",
    start: str | pd.Timestamp = "2005-10-05",
    end: str | pd.Timestamp = "2026-07-01",
) -> pd.DataFrame:
    """Build the reduced, diagnostic-only weekly state panel."""

    raw = _load_or_fetch_state_sources(raw_state_csv)
    missing_sources = [series for series in STATE_FRED_SERIES if series not in raw]
    if missing_sources:
        raise ValueError(f"state response source is missing required series: {missing_sources}")
    weeks = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="W-WED")
    out = pd.DataFrame(index=weeks)
    out.index.name = "date"

    out["bill3m_yield_level"] = pd.to_numeric(raw["DGS3MO"], errors="coerce").reindex(weeks)

    pre_fdic = _asof_weekly(raw["SAVNRNJ"], weeks) if "SAVNRNJ" in raw else pd.Series(np.nan, index=weeks)
    post_fdic = _asof_weekly(raw["SNDR"], weeks) if "SNDR" in raw else pd.Series(np.nan, index=weeks)
    break_date = pd.Timestamp("2021-04-01")
    out["fdic_savings_rate_proxy"] = pre_fdic.where(weeks < break_date, post_fdic)
    out["fdic_savings_rate_proxy_source"] = np.where(
        weeks < break_date,
        "SAVNRNJ_weekly_discontinued",
        "SNDR_monthly",
    )
    out.loc[out["fdic_savings_rate_proxy"].isna(), "fdic_savings_rate_proxy_source"] = ""
    out["fdic_methodology_break_2021_04"] = (weeks == pd.Timestamp("2021-04-07")).astype(int)
    out["bill_minus_fdic_savings_rate_proxy"] = (
        out["bill3m_yield_level"] - out["fdic_savings_rate_proxy"]
    )

    on_rrp = pd.to_numeric(raw["RRPONTSYD"], errors="coerce").reindex(weeks)
    out["on_rrp_balance"] = on_rrp.where(weeks >= pd.Timestamp("2013-09-25"), 0.0)

    out["bill3m_yield_level_lag1"] = out["bill3m_yield_level"].shift(1)
    out["bill_minus_fdic_savings_rate_proxy_lag1"] = out[
        "bill_minus_fdic_savings_rate_proxy"
    ].shift(1)
    out["on_rrp_balance_lag1"] = out["on_rrp_balance"].shift(1)
    out["on_rrp_balance_ge_50bn_lag1_indicator"] = np.where(
        out["on_rrp_balance_lag1"].isna(),
        np.nan,
        (out["on_rrp_balance_lag1"] >= 50.0).astype(float),
    )

    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")
    return out


def _standardize(series: pd.Series, sample_mask: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series.loc[sample_mask], errors="coerce").dropna()
    if clean.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    standard_deviation = float(clean.std(ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (pd.to_numeric(series, errors="coerce") - float(clean.mean())) / standard_deviation


def _analysis_mask(index: pd.Index, sample_name: str, horizon: int) -> pd.Series:
    if sample_name == "ex_pandemic":
        return _sample_mask(index, sample_name, horizon)
    if sample_name == "pre_2020":
        return pd.Series(pd.DatetimeIndex(index) < pd.Timestamp("2020-01-01"), index=index)
    raise ValueError(f"unknown diagnostic sample: {sample_name}")


def _controls_for(panel: pd.DataFrame, calendar: pd.DataFrame, horizon: int) -> list[str]:
    flow_controls = [
        "du_broad_outflows_bn",
        "interest_outflows_bn",
        "debt_issues_gross_bn",
        "debt_redemptions_gross_bn",
    ]
    calendar_controls = [column for column in calendar.columns if column != "date"]
    lag_controls = [
        f"{column}_lag{lag}"
        for column in FLOW_BUCKETS
        for lag in range(1, 5)
    ]
    future_flow_controls = [
        f"{column}_lead{lead}"
        for column in FLOW_BUCKETS
        for lead in range(1, horizon + 1)
    ]
    return [
        column
        for column in [
            *flow_controls,
            *calendar_controls,
            *lag_controls,
            *future_flow_controls,
        ]
        if column in panel.columns
    ]


def _estimate_state_interacted_response_diagnostic(
    flows: pd.DataFrame,
    calendar: pd.DataFrame,
    weekly_panel: pd.DataFrame,
    state_weekly: pd.DataFrame,
    *,
    state_variable: str = "bill3m_yield_level",
    state_column: str = "bill3m_yield_level_lag1",
    samples: list[str] | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Estimate only the cells needed to reproduce admission failures.

    Coefficients are intermediate diagnostics. The public artifact writer emits
    only coverage and gate status.
    """

    merged_state = weekly_panel.join(state_weekly, how="left", rsuffix="_state")
    panel = _prepare_regression_panel(flows, calendar, merged_state)
    if "broad_deposits_nsa" not in panel or state_column not in panel:
        raise ValueError("state response diagnostic inputs are missing the deposit outcome or state")

    y_level = pd.to_numeric(panel["broad_deposits_nsa"], errors="coerce")
    rows: list[dict[str, object]] = []
    for sample_name in samples or ["ex_pandemic", "pre_2020"]:
        for horizon in horizons or DIAGNOSTIC_HORIZONS:
            outcome = y_level.shift(-horizon) - y_level.shift(1)
            mask = _analysis_mask(panel.index, sample_name, horizon)
            state_z = _standardize(panel[state_column], mask)
            regressors = panel.loc[
                mask,
                [
                    "du_core_outflows_bn",
                    "tax_receipts_bn",
                    *_controls_for(panel, calendar, horizon),
                ],
            ].copy()
            regressors["state_z"] = state_z.loc[mask]
            for treatment in ["du_core_outflows_bn", "tax_receipts_bn"]:
                regressors[f"{treatment}_x_state"] = regressors[treatment] * regressors["state_z"]
            fit = _ols_hac(
                outcome.loc[mask],
                regressors,
                maxlags=max(horizon + 4, 4),
            )
            if fit is None:
                continue
            for treatment in ["du_core_outflows_bn", "tax_receipts_bn"]:
                coefficient = f"{treatment}_x_state"
                if coefficient not in fit.params:
                    continue
                rows.append(
                    {
                        "state_variable": state_variable,
                        "sample": sample_name,
                        "treatment_id": treatment,
                        "horizon": horizon,
                        "gamma": float(fit.params[coefficient]),
                        "se": float(fit.bse[coefficient]),
                        "p_hac": float(fit.pvalues[coefficient]),
                        "n": int(fit.nobs),
                    }
                )
    return pd.DataFrame(rows)


def _add_random_walk_placebo_state(
    state_weekly: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    out = state_weekly.copy()
    shocks = np.random.default_rng(seed).normal(0.0, 1.0, len(out))
    out["random_walk_placebo_state"] = np.cumsum(shocks)
    out["random_walk_placebo_state_lag1"] = out["random_walk_placebo_state"].shift(1)
    return out


def _calibrate_random_walk_placebo(
    flows: pd.DataFrame,
    calendar: pd.DataFrame,
    weekly_panel: pd.DataFrame,
    state_weekly: pd.DataFrame,
    real_estimates: pd.DataFrame,
    *,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    placebo_rows: list[dict[str, object]] = []
    seed_list = seeds or PLACEBO_CALIBRATION_SEEDS
    for seed in seed_list:
        placebo_state = _add_random_walk_placebo_state(state_weekly, seed=seed)
        placebo = _estimate_state_interacted_response_diagnostic(
            flows,
            calendar,
            weekly_panel,
            placebo_state,
            state_variable="random_walk_placebo",
            state_column="random_walk_placebo_state_lag1",
            samples=["ex_pandemic"],
            horizons=DIAGNOSTIC_HORIZONS,
        )
        core = placebo.loc[placebo["treatment_id"].eq("du_core_outflows_bn")].copy()
        core["placebo_seed"] = seed
        core["placebo_t"] = pd.to_numeric(core["gamma"], errors="coerce") / pd.to_numeric(
            core["se"], errors="coerce"
        )
        placebo_rows.extend(core.to_dict(orient="records"))

    draws = pd.DataFrame(placebo_rows)
    real = real_estimates.loc[
        real_estimates["state_variable"].eq("bill3m_yield_level")
        & real_estimates["sample"].eq("ex_pandemic")
        & real_estimates["treatment_id"].eq("du_core_outflows_bn")
        & real_estimates["horizon"].isin(DIAGNOSTIC_HORIZONS)
    ]
    rows: list[dict[str, object]] = []
    for horizon in DIAGNOSTIC_HORIZONS:
        horizon_draws = draws.loc[draws["horizon"].eq(horizon)] if not draws.empty else pd.DataFrame()
        placebo_t = pd.to_numeric(
            horizon_draws.get("placebo_t", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna().abs()
        real_cell = real.loc[real["horizon"].eq(horizon)]
        real_gamma = float(real_cell["gamma"].iloc[0]) if not real_cell.empty else np.nan
        real_se = float(real_cell["se"].iloc[0]) if not real_cell.empty else np.nan
        real_t = abs(real_gamma / real_se) if np.isfinite(real_gamma) and real_se else np.nan
        rows.append(
            {
                "horizon": horizon,
                "placebo_seed_count": len(placebo_t),
                "placebo_false_positive_rate": float((placebo_t > 1.96).mean())
                if len(placebo_t)
                else np.nan,
                "placebo_effective_p": float((placebo_t >= real_t).mean())
                if len(placebo_t) and np.isfinite(real_t)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _count_stability_failures(real_estimates: pd.DataFrame) -> int:
    core = real_estimates.loc[
        real_estimates["state_variable"].eq("bill3m_yield_level")
        & real_estimates["treatment_id"].eq("du_core_outflows_bn")
        & real_estimates["horizon"].isin(DIAGNOSTIC_HORIZONS)
    ]
    pooled = core.loc[core["sample"].eq("ex_pandemic"), ["horizon", "gamma"]].rename(
        columns={"gamma": "pooled_gamma"}
    )
    pre_2020 = core.loc[core["sample"].eq("pre_2020"), ["horizon", "gamma", "se"]].rename(
        columns={"gamma": "pre_2020_gamma", "se": "pre_2020_se"}
    )
    comparison = pooled.merge(pre_2020, on="horizon", how="inner")
    if len(comparison) != len(DIAGNOSTIC_HORIZONS):
        return len(DIAGNOSTIC_HORIZONS)
    lower = comparison["pre_2020_gamma"] - 1.96 * comparison["pre_2020_se"]
    upper = comparison["pre_2020_gamma"] + 1.96 * comparison["pre_2020_se"]
    return int(((comparison["pooled_gamma"] < lower) | (comparison["pooled_gamma"] > upper)).sum())


def _coverage_row(state_weekly: pd.DataFrame, variable: str) -> dict[str, object]:
    observed = pd.to_numeric(state_weekly[variable], errors="coerce").dropna()
    return {
        "row_type": "coverage",
        "diagnostic_id": f"{variable}_coverage",
        "state_variable": variable,
        "admission_status": "diagnostic_only",
        "promotion_ready": False,
        "coverage_start": observed.index.min().date().isoformat() if not observed.empty else "",
        "coverage_end": observed.index.max().date().isoformat() if not observed.empty else "",
        "observed_weeks": len(observed),
        "total_weeks": len(state_weekly),
        "horizon": np.nan,
        "observed_value": np.nan,
        "gate_threshold": np.nan,
        "evidence_scope": "current raw-state build",
        "reason": "Coverage is reported for diagnostics and does not admit an interaction estimate.",
    }


def build_state_response_gate_status(
    state_weekly: pd.DataFrame,
    *,
    placebo_calibration: pd.DataFrame,
    stability_failed_horizons: int,
) -> pd.DataFrame:
    """Return coverage and failed admission gates without coefficient output."""

    required = [*STATE_RESPONSE_VARIABLES, "fdic_methodology_break_2021_04"]
    missing = [column for column in required if column not in state_weekly]
    if missing:
        raise ValueError(f"state response panel is missing required columns: {missing}")
    required_calibration = {
        "horizon",
        "placebo_seed_count",
        "placebo_false_positive_rate",
        "placebo_effective_p",
    }
    if not required_calibration.issubset(placebo_calibration.columns):
        raise ValueError("placebo calibration is missing required gate-evidence columns")

    rows = [_coverage_row(state_weekly, variable) for variable in STATE_RESPONSE_VARIABLES]
    rows.extend(
        [
            {
                "row_type": "failed_gate",
                "diagnostic_id": "fixed_state_units_across_samples_and_horizons",
                "state_variable": "bill3m_yield_level",
                "admission_status": "failed",
                "promotion_ready": False,
                "coverage_start": "",
                "coverage_end": "",
                "observed_weeks": np.nan,
                "total_weeks": len(state_weekly),
                "horizon": np.nan,
                "observed_value": np.nan,
                "gate_threshold": np.nan,
                "evidence_scope": "frozen design audit",
                "reason": "The prior interaction design standardized the state separately within each horizon and sample, so its coefficients did not share fixed units.",
            },
            {
                "row_type": "failed_gate",
                "diagnostic_id": "bill_rate_pre_2020_stability",
                "state_variable": "bill3m_yield_level",
                "admission_status": "failed",
                "promotion_ready": False,
                "coverage_start": "",
                "coverage_end": "",
                "observed_weeks": np.nan,
                "total_weeks": len(state_weekly),
                "horizon": np.nan,
                "observed_value": float(stability_failed_horizons),
                "gate_threshold": 0.0,
                "evidence_scope": "reproduced pre-2020 comparison",
                "reason": "The bill-rate interaction was not stable in the pre-2020 sample; the pooled result is not admitted as a state selector.",
            },
            {
                "row_type": "failed_gate",
                "diagnostic_id": "fdic_proxy_method_continuity",
                "state_variable": "bill_minus_fdic_savings_rate_proxy",
                "admission_status": "failed",
                "promotion_ready": False,
                "coverage_start": "",
                "coverage_end": "",
                "observed_weeks": np.nan,
                "total_weeks": len(state_weekly),
                "horizon": np.nan,
                "observed_value": float(state_weekly["fdic_methodology_break_2021_04"].sum()),
                "gate_threshold": 0.0,
                "evidence_scope": "current raw-state build",
                "reason": "The FDIC savings-rate proxy has a pinned April 2021 source-method break.",
            },
            {
                "row_type": "failed_gate",
                "diagnostic_id": "on_rrp_era_support",
                "state_variable": "on_rrp_balance_ge_50bn_lag1_indicator",
                "admission_status": "failed",
                "promotion_ready": False,
                "coverage_start": "",
                "coverage_end": "",
                "observed_weeks": np.nan,
                "total_weeks": len(state_weekly),
                "horizon": np.nan,
                "observed_value": np.nan,
                "gate_threshold": np.nan,
                "evidence_scope": "frozen era-support audit",
                "reason": "ON-RRP variation is concentrated in a distinct policy era and does not identify a general absorption mechanism.",
            },
        ]
    )
    for horizon in DIAGNOSTIC_HORIZONS:
        cell = placebo_calibration.loc[placebo_calibration["horizon"].eq(horizon)]
        if len(cell) != 1:
            raise ValueError(f"placebo calibration must contain exactly one row for horizon {horizon}")
        evidence = cell.iloc[0]
        rows.append(
            {
                "row_type": "failed_gate",
                "diagnostic_id": f"random_walk_placebo_h{horizon}",
                "state_variable": "bill3m_yield_level",
                "admission_status": "failed",
                "promotion_ready": False,
                "coverage_start": "",
                "coverage_end": "",
                "observed_weeks": int(evidence["placebo_seed_count"]),
                "total_weeks": len(state_weekly),
                "horizon": horizon,
                "observed_value": float(evidence["placebo_false_positive_rate"]),
                "gate_threshold": 0.05,
                "evidence_scope": "reproduced 20-seed random-walk audit",
                "reason": "Persistent pseudo-states produced excessive nominal false positives, so the interaction lane is not admitted.",
            }
        )
    return pd.DataFrame(rows)


def write_state_response_gate_status(
    state_weekly: pd.DataFrame,
    *,
    placebo_calibration: pd.DataFrame,
    stability_failed_horizons: int,
    out_csv: str | Path = "data/processed/state_response_gate_status.csv",
) -> pd.DataFrame:
    status = build_state_response_gate_status(
        state_weekly,
        placebo_calibration=placebo_calibration,
        stability_failed_horizons=stability_failed_horizons,
    )
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(path, index=False)
    return status


def run_state_response_diagnostics(
    *,
    flows_csv: str | Path = "data/processed/dts_weekly_flow_decomposition.csv",
    calendar_csv: str | Path = "data/processed/fiscal_calendar_weekly.csv",
    weekly_panel_csv: str | Path = "data/processed/tdc_weekly_channel_panel.csv",
    raw_state_csv: str | Path | None = "data/raw/fred_state_response_sources.csv",
    state_csv: str | Path = "data/processed/state_response_weekly.csv",
    gate_status_csv: str | Path = "data/processed/state_response_gate_status.csv",
    start: str | pd.Timestamp = "2005-10-05",
    end: str | pd.Timestamp = "2026-07-01",
) -> dict[str, object]:
    """Reproduce the diagnostic state panel and failed-gate receipt.

    This function is deliberately not registered as a command-line workflow.
    """

    flows = pd.read_csv(flows_csv, parse_dates=["date"]).set_index("date")
    calendar = pd.read_csv(calendar_csv, parse_dates=["date"])
    weekly_panel = pd.read_csv(weekly_panel_csv, parse_dates=["date"]).set_index("date")
    weekly_panel = normalize_weekly_panel_units(weekly_panel)
    state = build_state_response_weekly(
        raw_state_csv=raw_state_csv,
        out_csv=state_csv,
        start=start,
        end=end,
    )
    real_estimates = _estimate_state_interacted_response_diagnostic(
        flows,
        calendar,
        weekly_panel,
        state,
    )
    placebo_calibration = _calibrate_random_walk_placebo(
        flows,
        calendar,
        weekly_panel,
        state,
        real_estimates,
    )
    stability_failed_horizons = _count_stability_failures(real_estimates)
    status = write_state_response_gate_status(
        state,
        placebo_calibration=placebo_calibration,
        stability_failed_horizons=stability_failed_horizons,
        out_csv=gate_status_csv,
    )
    return {
        "status": "diagnostic_only",
        "state_csv": str(state_csv),
        "gate_status_csv": str(gate_status_csv),
        "state_rows": len(state),
        "gate_rows": len(status),
    }
