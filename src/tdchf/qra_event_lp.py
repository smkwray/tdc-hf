from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import statsmodels.api as sm


MILLION_DOLLAR_LEVEL_COLUMNS = {
    "tga_week_avg",
    "tga_wednesday",
    "fed_treasury_holdings",
    "fed_reverse_repos",
    "fed_liquidity_facilities",
    "fed_total_assets",
    "reserves",
}

BILLION_DOLLAR_LEVEL_COLUMNS = {
    "broad_deposits",
    "domestic_deposits",
    "large_time_deposits",
    "broad_deposits_nsa",
    "bank_credit",
    "bank_credit_nsa",
    "onrrp",
}

OUTCOME_SPECS = {
    "deposits_dpsacb": "broad_deposits_nsa",
    "reserves_wresbal": "reserves",
    "on_rrp_rrpontsyd": "onrrp",
    "on_rrp_total_incl_foreign_wlrral": "fed_liquidity_facilities",
    "tga_wtregen": "tga_week_avg",
    "walcl": "fed_total_assets",
    "bank_credit_totbkcr": "bank_credit",
    "sensitivity_tga_wdts": "tga_wednesday",
}

HEADLINE_OUTCOMES = ["deposits_dpsacb", "reserves_wresbal", "on_rrp_rrpontsyd", "tga_wtregen"]
REPORT_HORIZONS = [1, 4, 8]
PLACEBO_HORIZONS = [-4, -3, -2]
HORIZONS = [*PLACEBO_HORIZONS, *range(1, 9)]
OUTLIER_QUARTERS = {"2020Q2", "2020Q3"}
PANDEMIC_BLOCK_QUARTERS = {"2020Q2", "2020Q3", "2020Q4", "2021Q1"}
FRED_GDP_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP"
WILD_BOOTSTRAP_REPS = 9999
WILD_BOOTSTRAP_SEED = 20260704

OUTCOME_LABELS = {
    "deposits_dpsacb": "Deposits",
    "reserves_wresbal": "Reserves",
    "on_rrp_rrpontsyd": "ON RRP facility (RRPONTSYD)",
    "on_rrp_total_incl_foreign_wlrral": "Total reverse repos incl. foreign pool (WLRRAL)",
    "tga_wtregen": "TGA",
    "walcl": "Fed total assets",
    "bank_credit_totbkcr": "Bank credit",
    "sensitivity_tga_wdts": "TGA Wednesday sensitivity",
}

SAMPLE_LABELS = {
    "full": "(a) full",
    "exclude_2020Q2_Q3": "(b) ex-2020Q2/Q3",
    "gdp_scaled": "(c) GDP-scaled",
    "exclude_pandemic_block": "(d) ex-pandemic-block",
}


@dataclass(frozen=True)
class EventWeek:
    release_date: pd.Timestamp
    week_date: pd.Timestamp
    base_date: pd.Timestamp


def normalize_weekly_panel_units(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for column in MILLION_DOLLAR_LEVEL_COLUMNS.intersection(out.columns):
        out[column] = pd.to_numeric(out[column], errors="coerce") / 1000.0
    for column in BILLION_DOLLAR_LEVEL_COLUMNS.intersection(out.columns):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def find_event_week(release_date: pd.Timestamp, weekly_dates: pd.DatetimeIndex) -> EventWeek:
    prior_weeks = weekly_dates[weekly_dates < release_date.normalize()]
    if len(prior_weeks) < 2:
        raise ValueError(f"Need at least two complete weeks before {release_date.date()}")
    return EventWeek(release_date=release_date, week_date=prior_weeks[-1], base_date=prior_weeks[-2])


def _download_gdp_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(FRED_GDP_URL, timeout=90) as response:
        path.write_bytes(response.read())


def load_gdp_by_quarter(gdp_csv: str | Path | None = None) -> pd.Series:
    path = Path(gdp_csv or "data/raw/fred_gdp.csv")
    if not path.exists():
        _download_gdp_csv(path)
    raw = pd.read_csv(path)
    date_col = "observation_date" if "observation_date" in raw.columns else "date"
    value_col = "GDP" if "GDP" in raw.columns else raw.columns[-1]
    dates = pd.to_datetime(raw[date_col], errors="coerce")
    values = pd.to_numeric(raw[value_col], errors="coerce")
    quarters = dates.dt.to_period("Q").astype(str).str.replace("Q", "Q", regex=False)
    return pd.Series(values.to_numpy(dtype=float), index=quarters).dropna()


def construct_qra_event_panel(
    qra_csv: str | Path,
    weekly_panel_csv: str | Path,
    *,
    gdp_csv: str | Path | None = None,
    outcome_specs: dict[str, str] | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    specs = outcome_specs or OUTCOME_SPECS
    horizon_values = horizons or HORIZONS
    events = pd.read_csv(qra_csv)
    weekly = pd.read_csv(weekly_panel_csv)
    weekly["date"] = pd.to_datetime(weekly["date"], errors="coerce")
    weekly = weekly.dropna(subset=["date"]).set_index("date").sort_index()
    weekly = normalize_weekly_panel_units(weekly)
    gdp = load_gdp_by_quarter(gdp_csv)

    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        if pd.isna(event.get("surprise_bn")) or pd.isna(event.get("release_date")):
            continue
        release_date = pd.Timestamp(event["release_date"])
        try:
            event_week = find_event_week(release_date, weekly.index)
        except ValueError:
            continue
        quarter = str(event["quarter"])
        base_pos = weekly.index.get_loc(event_week.base_date)
        week_pos = weekly.index.get_loc(event_week.week_date)
        prior_estimate = pd.to_numeric(pd.Series([event.get("prior_estimate_bn")]), errors="coerce").iloc[0]
        surprise = float(event["surprise_bn"])
        gdp_bn = float(gdp.get(quarter, np.nan))
        rrp_state_col = "onrrp" if "onrrp" in weekly.columns else "fed_liquidity_facilities"
        rrp_lag_bn = float(weekly.iloc[week_pos][rrp_state_col]) if rrp_state_col in weekly.columns else np.nan
        tga_prior = pd.to_numeric(pd.Series([event.get("tga_assumption_prior_bn")]), errors="coerce").iloc[0]
        tga_announced = pd.to_numeric(pd.Series([event.get("tga_assumption_announced_bn")]), errors="coerce").iloc[0]
        tga_component = float(tga_announced - tga_prior) if pd.notna(tga_announced) and pd.notna(tga_prior) else np.nan

        for outcome_name, source_col in specs.items():
            if source_col not in weekly.columns:
                continue
            base_value = weekly.iloc[base_pos][source_col]
            pre_base_pos = base_pos - 4
            pretrend = np.nan
            if pre_base_pos >= 0:
                pretrend = weekly.iloc[base_pos][source_col] - weekly.iloc[pre_base_pos][source_col]
            for horizon in horizon_values:
                target_pos = week_pos + horizon
                if target_pos < 0 or target_pos >= len(weekly):
                    continue
                target_value = weekly.iloc[target_pos][source_col]
                if pd.isna(base_value) or pd.isna(target_value):
                    continue
                rows.append(
                    {
                        "event_id": event.get("event_id", ""),
                        "quarter": quarter,
                        "release_date": release_date.date().isoformat(),
                        "week_date": event_week.week_date.date().isoformat(),
                        "base_date": event_week.base_date.date().isoformat(),
                        "outcome": outcome_name,
                        "horizon": int(horizon),
                        "y_change_bn": float(target_value - base_value),
                        "surprise_bn": surprise,
                        "surprise_pct_gdp": surprise / gdp_bn * 100.0 if np.isfinite(gdp_bn) and gdp_bn else np.nan,
                        "gdp_bn": gdp_bn,
                        "prior_estimate_bn": prior_estimate,
                        "pretrend_4w_bn": float(pretrend) if pd.notna(pretrend) else np.nan,
                        "exclude_2020_outlier": quarter in OUTLIER_QUARTERS,
                        "exclude_pandemic_block": quarter in PANDEMIC_BLOCK_QUARTERS,
                        "post_2020": release_date >= pd.Timestamp("2020-01-01"),
                        "rrp_active": bool(pd.notna(rrp_lag_bn) and rrp_lag_bn > 50.0),
                        "rrp_lag_bn": rrp_lag_bn,
                        "tga_target_component_bn": tga_component,
                        "deficit_component_bn": surprise - tga_component if pd.notna(tga_component) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return WILD_BOOTSTRAP_SEED + sum((idx + 1) * ord(char) for idx, char in enumerate(text))


def _clean_regression_sample(sample: pd.DataFrame, shock_cols: list[str]) -> pd.DataFrame:
    cols = ["y_change_bn", *shock_cols, "prior_estimate_bn", "pretrend_4w_bn"]
    return sample[cols].replace([np.inf, -np.inf], np.nan).dropna()


def _wild_bootstrap_p(
    clean: pd.DataFrame,
    shock_cols: list[str],
    shock_col: str,
    beta_hat: float,
    *,
    reps: int = WILD_BOOTSTRAP_REPS,
    seed: int = WILD_BOOTSTRAP_SEED,
) -> float:
    y = clean["y_change_bn"].to_numpy(dtype=float)
    x_full = sm.add_constant(clean[[*shock_cols, "prior_estimate_bn", "pretrend_4w_bn"]], has_constant="add").to_numpy(dtype=float)
    x_restricted = sm.add_constant(clean[["prior_estimate_bn", "pretrend_4w_bn"]], has_constant="add").to_numpy(dtype=float)
    try:
        restricted_beta = np.linalg.lstsq(x_restricted, y, rcond=None)[0]
        fitted_null = x_restricted @ restricted_beta
        residuals_null = y - fitted_null
        pinv_full = np.linalg.pinv(x_full)
        shock_idx = [*shock_cols, "prior_estimate_bn", "pretrend_4w_bn"].index(shock_col) + 1
    except np.linalg.LinAlgError:
        return np.nan
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(len(y), reps))
    y_star = fitted_null[:, None] + residuals_null[:, None] * signs
    boot_betas = (pinv_full @ y_star)[shock_idx]
    return float(np.mean(np.abs(boot_betas) >= abs(beta_hat)))


def _fit_ols(sample: pd.DataFrame, shock_cols: list[str]) -> dict[str, dict[str, float]] | None:
    sample = _clean_regression_sample(sample, shock_cols)
    if len(sample) < 6:
        return None
    for shock_col in shock_cols:
        if sample[shock_col].nunique() < 2:
            return None
    xcols = [*shock_cols, "prior_estimate_bn", "pretrend_4w_bn"]
    x = sm.add_constant(sample[xcols], has_constant="add")
    fit = sm.OLS(sample["y_change_bn"], x).fit(cov_type="HC1")
    out: dict[str, dict[str, float]] = {}
    for shock_col in shock_cols:
        out[shock_col] = {
            "beta": float(fit.params[shock_col]),
            "se": float(fit.bse[shock_col]),
            "p": float(fit.pvalues[shock_col]),
            "t": float(fit.tvalues[shock_col]),
            "n": int(fit.nobs),
        }
    return out


def _wild_bootstrap_for_row(
    group: pd.DataFrame,
    *,
    sample_name: str,
    scaling: str,
    shock_cols: list[str],
    shock_col: str,
    outcome: str,
    horizon: int,
    beta: float,
) -> float:
    headline = outcome in HEADLINE_OUTCOMES and horizon in REPORT_HORIZONS
    headline_sample = (sample_name, scaling) in {
        ("full", "bn"),
        ("exclude_2020Q2_Q3", "bn"),
        ("gdp_scaled", "pct_gdp"),
        ("exclude_pandemic_block", "bn"),
    }
    if not (headline and headline_sample):
        return np.nan
    clean = _clean_regression_sample(group, shock_cols)
    if len(clean) < 6:
        return np.nan
    seed = _stable_seed(sample_name, scaling, shock_col, outcome, horizon)
    return _wild_bootstrap_p(clean, shock_cols, shock_col, beta, seed=seed)


def _estimate_group(
    panel: pd.DataFrame,
    *,
    sample_name: str,
    scaling: str,
    shock_cols: list[str],
    thin_cell: bool,
    spec_flags: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mean_gdp = float(panel["gdp_bn"].dropna().mean()) if "gdp_bn" in panel.columns else np.nan
    for (outcome, horizon), group in panel.groupby(["outcome", "horizon"], sort=True):
        fit = _fit_ols(group, shock_cols)
        if fit is None:
            continue
        if scaling == "pct_gdp":
            multiplier = (100.0 / mean_gdp) * 100.0 if np.isfinite(mean_gdp) and mean_gdp else np.nan
        else:
            multiplier = 100.0
        for shock_col, values in fit.items():
            beta = values["beta"]
            se = values["se"]
            p_wild = _wild_bootstrap_for_row(
                group,
                sample_name=sample_name,
                scaling=scaling,
                shock_cols=shock_cols,
                shock_col=shock_col,
                outcome=str(outcome),
                horizon=int(horizon),
                beta=beta,
            )
            rows.append(
                {
                    "outcome": outcome,
                    "horizon": int(horizon),
                    "sample": sample_name,
                    "scaling": scaling,
                    "shock": shock_col,
                    "beta": beta,
                    "se": se,
                    "p": values["p"],
                    "p_wild_bootstrap": p_wild,
                    "t": values["t"],
                    "n": values["n"],
                    "beta_per_100bn": beta * multiplier if np.isfinite(multiplier) else np.nan,
                    "se_per_100bn": se * multiplier if np.isfinite(multiplier) else np.nan,
                    "thin_cell": bool(thin_cell),
                    "spec_flags": spec_flags,
                }
            )
    return rows


def estimate_qra_event_lps(event_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variants = [
        ("full", "bn", ["surprise_bn"], event_panel, "HC1; controls=prior_estimate_bn+pretrend_4w_bn"),
        (
            "exclude_2020Q2_Q3",
            "bn",
            ["surprise_bn"],
            event_panel.loc[~event_panel["exclude_2020_outlier"]],
            "HC1; excludes 2020Q2 and 2020Q3; controls=prior_estimate_bn+pretrend_4w_bn",
        ),
        (
            "gdp_scaled",
            "pct_gdp",
            ["surprise_pct_gdp"],
            event_panel,
            "HC1; surprise scaled by nominal GDP; controls=prior_estimate_bn+pretrend_4w_bn",
        ),
        (
            "exclude_pandemic_block",
            "bn",
            ["surprise_bn"],
            event_panel.loc[~event_panel["exclude_pandemic_block"]],
            "HC1; excludes 2020Q2 through 2021Q1; controls=prior_estimate_bn+pretrend_4w_bn",
        ),
    ]
    for sample_name, scaling, shock_cols, sample, flags in variants:
        rows.extend(_estimate_group(sample, sample_name=sample_name, scaling=scaling, shock_cols=shock_cols, thin_cell=False, spec_flags=flags))

    split_specs = [
        ("pre_2020", event_panel.loc[~event_panel["post_2020"]]),
        ("post_2020", event_panel.loc[event_panel["post_2020"]]),
        ("rrp_inactive", event_panel.loc[~event_panel["rrp_active"]]),
        ("rrp_active", event_panel.loc[event_panel["rrp_active"]]),
    ]
    for sample_name, sample in split_specs:
        event_n = int(sample[["quarter", "release_date"]].drop_duplicates().shape[0])
        rows.extend(
            _estimate_group(
                sample,
                sample_name=sample_name,
                scaling="bn",
                shock_cols=["surprise_bn"],
                thin_cell=event_n < 15,
                spec_flags=f"HC1; split_n={event_n}; controls=prior_estimate_bn+pretrend_4w_bn",
            )
        )

    decomp_base = event_panel.loc[event_panel["outcome"].isin(HEADLINE_OUTCOMES)]
    decomp_base = decomp_base.dropna(subset=["tga_target_component_bn", "deficit_component_bn"])
    for sample_name, sample, flags in [
        ("component_decomp_full", decomp_base, "HC1; sensitivity only; both TGA-target and deficit components in the same regression"),
        (
            "component_decomp_ex_pandemic_block",
            decomp_base.loc[~decomp_base["exclude_pandemic_block"]],
            "HC1; sensitivity only; excludes 2020Q2 through 2021Q1; both components in the same regression",
        ),
    ]:
        event_n = int(sample[["quarter", "release_date"]].drop_duplicates().shape[0])
        rows.extend(
            _estimate_group(
                sample,
                sample_name=sample_name,
                scaling="component_bn",
                shock_cols=["tga_target_component_bn", "deficit_component_bn"],
                thin_cell=event_n < 30,
                spec_flags=f"{flags}; decomp_n={event_n}; controls=prior_estimate_bn+pretrend_4w_bn",
            )
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return ["No estimable rows."]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        out.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    return out


def _display_estimates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["outcome"] = out["outcome"].map(OUTCOME_LABELS).fillna(out["outcome"])
    out["sample"] = out["sample"].map(SAMPLE_LABELS).fillna(out["sample"])
    for col in ["beta_per_100bn", "se_per_100bn"]:
        if col in out.columns:
            out[col] = out[col].round(2)
    for col in ["p", "p_wild_bootstrap"]:
        if col in out.columns:
            out[col] = out[col].round(3)
    return out


def write_qra_event_lp_readout(estimates: pd.DataFrame, event_panel: pd.DataFrame, out_md: str | Path) -> None:
    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    headline = estimates.loc[
        estimates["outcome"].isin(HEADLINE_OUTCOMES)
        & estimates["horizon"].isin(REPORT_HORIZONS)
        & estimates["sample"].isin(["full", "exclude_2020Q2_Q3", "gdp_scaled", "exclude_pandemic_block"])
        & estimates["shock"].isin(["surprise_bn", "surprise_pct_gdp"])
    ].copy()
    headline = _display_estimates(headline)

    placebo = estimates.loc[
        estimates["horizon"].isin(PLACEBO_HORIZONS)
        & estimates["sample"].eq("full")
        & estimates["scaling"].eq("bn")
        & estimates["shock"].eq("surprise_bn")
    ].copy()
    placebo_sig = int((placebo["p"] < 0.05).sum()) if not placebo.empty else 0
    placebo = _display_estimates(placebo)

    event_rows = event_panel[["quarter", "release_date", "post_2020", "rrp_active", "tga_target_component_bn", "deficit_component_bn"]].drop_duplicates()
    event_count = int(event_rows.shape[0])
    tga_coverage = int(event_rows["tga_target_component_bn"].notna().sum())

    split_counts = pd.DataFrame(
        [
            {"sample": "pre_2020", "n": int(event_rows.loc[~event_rows["post_2020"]].shape[0]), "thin_cell": False},
            {"sample": "post_2020", "n": int(event_rows.loc[event_rows["post_2020"]].shape[0]), "thin_cell": False},
            {"sample": "rrp_inactive", "n": int(event_rows.loc[~event_rows["rrp_active"]].shape[0]), "thin_cell": False},
            {"sample": "rrp_active", "n": int(event_rows.loc[event_rows["rrp_active"]].shape[0]), "thin_cell": False},
        ]
    )
    split_counts["thin_cell"] = split_counts["n"] < 15

    split_rows = estimates.loc[
        estimates["sample"].isin(["pre_2020", "post_2020", "rrp_inactive", "rrp_active"])
        & estimates["outcome"].isin(HEADLINE_OUTCOMES)
        & estimates["horizon"].isin(REPORT_HORIZONS)
        & estimates["shock"].eq("surprise_bn")
    ].copy()
    split_rows = _display_estimates(split_rows)

    component = estimates.loc[
        estimates["sample"].isin(["component_decomp_full", "component_decomp_ex_pandemic_block"])
        & estimates["outcome"].isin(HEADLINE_OUTCOMES)
        & estimates["horizon"].isin(REPORT_HORIZONS)
        & estimates["shock"].isin(["tga_target_component_bn", "deficit_component_bn"])
    ].copy()
    component = _display_estimates(component)

    wlrral = estimates.loc[
        estimates["outcome"].eq("on_rrp_total_incl_foreign_wlrral")
        & estimates["horizon"].isin(REPORT_HORIZONS)
        & estimates["sample"].isin(["full", "exclude_2020Q2_Q3", "gdp_scaled", "exclude_pandemic_block"])
        & estimates["shock"].isin(["surprise_bn", "surprise_pct_gdp"])
    ].copy()
    wlrral = _display_estimates(wlrral)

    dep_b = estimates.loc[
        estimates["outcome"].eq("deposits_dpsacb")
        & estimates["horizon"].eq(4)
        & estimates["sample"].eq("exclude_2020Q2_Q3")
        & estimates["shock"].eq("surprise_bn"),
        "beta_per_100bn",
    ]
    dep_c = estimates.loc[
        estimates["outcome"].eq("deposits_dpsacb")
        & estimates["horizon"].eq(4)
        & estimates["sample"].eq("gdp_scaled")
        & estimates["shock"].eq("surprise_pct_gdp"),
        "beta_per_100bn",
    ]
    dep_reversal = ""
    if not dep_b.empty and not dep_c.empty:
        dep_reversal = f" At h=4, sample (b) is {float(dep_b.iloc[0]):+.2f} while sample (c) is {float(dep_c.iloc[0]):+.2f} per $100bn-equivalent surprise."

    lines = [
        "# QRA Announcement-Window LP Readout",
        "",
        "Estimator: event-level local projections with HC1 robust SEs; controls are the prior borrowing estimate and the lagged four-week outcome change. WRESBAL is the reserve outcome. RRPONTSYD is the headline ON-RRP facility outcome, converted to a Wednesday weekly value by the weekly panel; WLRRAL is retained only as a total-reverse-repo sensitivity because it includes the foreign official pool. H.4.1 million-dollar series are normalized to billions.",
        "",
        f"Event coverage: {event_count} QRA borrowing-estimate events. TGA cash-assumption decomposition coverage: {tga_coverage}/{event_count} events have both announced and prior cash assumptions.",
        "",
        "## Headline effects per $100bn surprise",
        "",
    ]
    lines.extend(_markdown_table(headline, ["outcome", "horizon", "sample", "beta_per_100bn", "se_per_100bn", "p", "p_wild_bootstrap", "n"]))
    lines.extend(["", "## Placebos", ""])
    lines.append(f"D1-corrected informative placebo rate: {placebo_sig}/{len(placebo)} significant at 5 percent after dropping mechanically zero h=-1 rows.")
    lines.extend(_markdown_table(placebo, ["outcome", "horizon", "beta_per_100bn", "p", "n"]))
    lines.extend(["", "## Splits", ""])
    lines.extend(_markdown_table(split_counts, ["sample", "n", "thin_cell"]))
    lines.extend(["", "Headline split estimates:", ""])
    lines.extend(_markdown_table(split_rows, ["outcome", "horizon", "sample", "beta_per_100bn", "p", "n", "thin_cell"]))
    lines.extend(["", "## Component Decomposition", ""])
    lines.append("For events with both cash assumptions, `tga_target_component_bn = announced cash assumption - prior cash assumption`; `deficit_component_bn = surprise_bn - tga_target_component_bn`. Both components enter the same regression. Cells with n < 30 are labeled sensitivity/thin.")
    lines.extend(_markdown_table(component, ["outcome", "horizon", "sample", "shock", "beta_per_100bn", "p", "n", "thin_cell"]))
    lines.extend(["", "## WLRRAL Sensitivity", ""])
    lines.append("These rows use WLRRAL, total reverse repos including the foreign official pool. They are not the headline ON-RRP facility response.")
    lines.extend(_markdown_table(wlrral, ["outcome", "horizon", "sample", "beta_per_100bn", "p", "p_wild_bootstrap", "n"]))
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "No signed deposit pass-through claim is warranted. The deposit response sign is sample-contingent." + dep_reversal,
            "",
            "The preferred descriptive row is sample (d), excluding 2020Q2 through 2021Q1. Read any qualitative pattern there as balance-sheet plumbing around reserves, TGA, and ON RRP rather than a clean deposit pass-through estimate.",
            "",
            "Limits: weekly timing is the floor; n is about 66 before sample restrictions; pre/post-2020 and RRP-active splits have a one-transition regime confound; these are descriptive event-window estimates, not structural identification.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_qra_event_lp_csv(
    *,
    qra_csv: str | Path = "data/processed/qra_borrowing_surprise.csv",
    weekly_panel_csv: str | Path = "data/processed/tdc_weekly_channel_panel.csv",
    gdp_csv: str | Path | None = None,
    out_csv: str | Path = "data/processed/qra_event_lp_estimates.csv",
    readout_md: str | Path = "data/processed/qra_event_lp_readout.md",
) -> dict[str, object]:
    event_panel = construct_qra_event_panel(qra_csv, weekly_panel_csv, gdp_csv=gdp_csv)
    estimates = estimate_qra_event_lps(event_panel)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(out_path, index=False)
    write_qra_event_lp_readout(estimates, event_panel, readout_md)
    return {
        "status": "ok",
        "out": str(out_path),
        "readout": str(readout_md),
        "rows": int(len(estimates)),
        "events": int(event_panel[["quarter", "release_date"]].drop_duplicates().shape[0]),
        "outcomes": sorted(estimates["outcome"].dropna().unique().tolist()) if not estimates.empty else [],
    }
