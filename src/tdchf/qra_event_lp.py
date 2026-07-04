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
    "onrrp_wlrral": "fed_liquidity_facilities",
    "tga_wtregen": "tga_week_avg",
    "walcl": "fed_total_assets",
    "bank_credit_totbkcr": "bank_credit",
    "sensitivity_onrrp_rrpontsyd": "onrrp",
    "sensitivity_tga_wdts": "tga_wednesday",
}

HORIZONS = [*range(-4, 0), *range(1, 9)]
OUTLIER_QUARTERS = {"2020Q2", "2020Q3"}
FRED_GDP_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP"


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
                        "post_2020": release_date >= pd.Timestamp("2020-01-01"),
                        "rrp_active": bool(pd.notna(rrp_lag_bn) and rrp_lag_bn > 50.0),
                        "rrp_lag_bn": rrp_lag_bn,
                        "tga_target_surprise_bn": tga_component,
                        "deficit_surprise_bn": surprise - tga_component if pd.notna(tga_component) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _fit_ols(sample: pd.DataFrame, shock_col: str) -> dict[str, float] | None:
    cols = ["y_change_bn", shock_col, "prior_estimate_bn", "pretrend_4w_bn"]
    sample = sample[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 6 or sample[shock_col].nunique() < 2:
        return None
    xcols = [shock_col, "prior_estimate_bn", "pretrend_4w_bn"]
    x = sm.add_constant(sample[xcols], has_constant="add")
    fit = sm.OLS(sample["y_change_bn"], x).fit(cov_type="HC1")
    return {
        "beta": float(fit.params[shock_col]),
        "se": float(fit.bse[shock_col]),
        "p": float(fit.pvalues[shock_col]),
        "t": float(fit.tvalues[shock_col]),
        "n": int(fit.nobs),
    }


def _estimate_group(
    panel: pd.DataFrame,
    *,
    sample_name: str,
    scaling: str,
    shock_col: str,
    thin_cell: bool,
    spec_flags: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mean_gdp = float(panel["gdp_bn"].dropna().mean()) if "gdp_bn" in panel.columns else np.nan
    for (outcome, horizon), group in panel.groupby(["outcome", "horizon"], sort=True):
        fit = _fit_ols(group, shock_col)
        if fit is None:
            continue
        beta = fit["beta"]
        se = fit["se"]
        if scaling == "pct_gdp":
            multiplier = (100.0 / mean_gdp) * 100.0 if np.isfinite(mean_gdp) and mean_gdp else np.nan
        else:
            multiplier = 100.0
        rows.append(
            {
                "outcome": outcome,
                "horizon": int(horizon),
                "sample": sample_name,
                "scaling": scaling,
                "shock": shock_col,
                "beta": beta,
                "se": se,
                "p": fit["p"],
                "t": fit["t"],
                "n": fit["n"],
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
        ("full", "bn", "surprise_bn", event_panel, "HC1; controls=prior_estimate_bn+pretrend_4w_bn"),
        (
            "exclude_2020Q2_Q3",
            "bn",
            "surprise_bn",
            event_panel.loc[~event_panel["exclude_2020_outlier"]],
            "HC1; excludes 2020Q2 and 2020Q3; controls=prior_estimate_bn+pretrend_4w_bn",
        ),
        ("full", "pct_gdp", "surprise_pct_gdp", event_panel, "HC1; surprise scaled by nominal GDP; controls=prior_estimate_bn+pretrend_4w_bn"),
    ]
    for sample_name, scaling, shock_col, sample, flags in variants:
        rows.extend(_estimate_group(sample, sample_name=sample_name, scaling=scaling, shock_col=shock_col, thin_cell=False, spec_flags=flags))

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
                shock_col="surprise_bn",
                thin_cell=event_n < 15,
                spec_flags=f"HC1; split_n={event_n}; controls=prior_estimate_bn+pretrend_4w_bn",
            )
        )

    decomp = event_panel.loc[event_panel["outcome"].isin(["deposits_dpsacb", "onrrp_wlrral"])]
    decomp = decomp.dropna(subset=["tga_target_surprise_bn", "deficit_surprise_bn"])
    for shock_col, scaling in [("tga_target_surprise_bn", "tga_component_bn"), ("deficit_surprise_bn", "deficit_component_bn")]:
        event_n = int(decomp[["quarter", "release_date"]].drop_duplicates().shape[0])
        rows.extend(
            _estimate_group(
                decomp,
                sample_name="tga_decomp",
                scaling=scaling,
                shock_col=shock_col,
                thin_cell=event_n < 15,
                spec_flags=f"HC1; sensitivity only; decomp_n={event_n}; controls=prior_estimate_bn+pretrend_4w_bn",
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


def write_qra_event_lp_readout(estimates: pd.DataFrame, event_panel: pd.DataFrame, out_md: str | Path) -> None:
    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    headline = estimates.loc[
        estimates["outcome"].isin(["deposits_dpsacb", "reserves_wresbal", "onrrp_wlrral", "tga_wtregen"])
        & estimates["horizon"].between(1, 8)
        & estimates["sample"].isin(["full", "exclude_2020Q2_Q3"])
        & estimates["scaling"].eq("bn")
    ].copy()
    headline["beta_per_100bn"] = headline["beta_per_100bn"].round(2)
    headline["se_per_100bn"] = headline["se_per_100bn"].round(2)

    scaled = estimates.loc[
        estimates["outcome"].isin(["deposits_dpsacb", "reserves_wresbal", "onrrp_wlrral", "tga_wtregen"])
        & estimates["horizon"].between(1, 8)
        & estimates["sample"].eq("full")
        & estimates["scaling"].eq("pct_gdp")
    ].copy()
    scaled["beta_per_100bn"] = scaled["beta_per_100bn"].round(2)

    placebo = estimates.loc[estimates["horizon"].lt(0) & estimates["sample"].eq("full") & estimates["scaling"].eq("bn")].copy()
    placebo_sig = int((placebo["p"] < 0.05).sum()) if not placebo.empty else 0
    split_ns = (
        event_panel.groupby(["post_2020", "rrp_active"])[["quarter", "release_date"]]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .reset_index(name="n")
    )
    split_ns["cell"] = split_ns.apply(
        lambda r: ("post_2020" if r["post_2020"] else "pre_2020") + "/" + ("rrp_active" if r["rrp_active"] else "rrp_inactive"),
        axis=1,
    )

    lines = [
        "# QRA Announcement-Window LP Readout",
        "",
        "Estimator: event-level local projections with HC1 robust SEs; controls are the prior borrowing estimate and the lagged four-week outcome change. WRESBAL is used for reserves because it is the direct reserve-balance H.4.1 series; H.4.1 million-dollar series are normalized to billions.",
        "",
        "Claim boundary: descriptive announcement-window evidence only. H.8 timing is weekly, the event count is small, and the pre/post and RRP-active splits contain a one-transition regime confound.",
        "",
        "## Headline effects per $100bn surprise",
        "",
    ]
    lines.extend(_markdown_table(headline, ["outcome", "horizon", "sample", "beta_per_100bn", "se_per_100bn", "p", "n"]))
    lines.extend(["", "## GDP-scaled full-sample sensitivity", ""])
    lines.extend(_markdown_table(scaled, ["outcome", "horizon", "beta_per_100bn", "p", "n"]))
    lines.extend(["", "## Pre-trend placebos", ""])
    lines.append(f"Full-sample negative-horizon placebo significant rows at 5 percent: {placebo_sig} of {len(placebo)}.")
    place = placebo.copy()
    place["beta_per_100bn"] = place["beta_per_100bn"].round(2)
    lines.extend(_markdown_table(place, ["outcome", "horizon", "beta_per_100bn", "p", "n"]))
    lines.extend(["", "## Split cell counts", ""])
    lines.extend(_markdown_table(split_ns[["cell", "n"]], ["cell", "n"]))
    split_rows = estimates.loc[
        estimates["sample"].isin(["pre_2020", "post_2020", "rrp_inactive", "rrp_active"])
        & estimates["outcome"].isin(["deposits_dpsacb", "onrrp_wlrral"])
        & estimates["horizon"].isin([1, 4, 8])
    ].copy()
    split_rows["beta_per_100bn"] = split_rows["beta_per_100bn"].round(2)
    lines.extend(["", "## Deposits and ON RRP splits", ""])
    lines.extend(_markdown_table(split_rows, ["outcome", "horizon", "sample", "beta_per_100bn", "p", "n", "thin_cell"]))
    lines.extend(
        [
            "",
            "## Bounded interpretation",
            "",
            "Use the no-2020-outlier and GDP-scaled rows as the disciplined checks on any full-sample pattern. The estimates are not a structural pass-through claim; they describe weekly balance-sheet movements around Treasury borrowing-estimate announcements.",
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
