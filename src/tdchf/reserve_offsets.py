"""Weekly Treasury cash-leg and H.4.1 reserve-factor offset associations.

Consumes the TDCest Wednesday reserve-factor ledger (USD million) under a
pinned hash. Every signed factor ``f_<row>`` is +d(level) for a supplying row
and -d(level) for an absorbing row, so the factors plus the rounding remainder
``rho`` sum exactly to the change in reserve balances. The exposure is the
Treasury cash leg ``C_w = -d(TGA) - d(SFP)`` between consecutive Wednesdays.

Everything here is a conditional association or a descriptive accounting path.
No coefficient is a landing share, retention ratio, half-life, or causal
response, and no estimate is normalized by an estimated impact.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .disbursement import assign_h8_week
from .federal_calendar import TaxDeadlineCategory, tax_deadlines
from .manifest import _sha256

SCHEMA = "weekly_reserve_offsets_v1"
LEDGER_SCHEMA = "reserve_factor_ledger_v1"
LEDGER_SHA256 = "0409f778d0a2e20fbe3aa0c743b50830f6f5df51e00511d46c5a7a3ea867c894"
LEDGER_RECEIPT_SHA256 = "62d6ac96136dba8b4c76a1ce7e1c7d4f8364f7322fe4160a3f7a55614d3deec3"
UNITS = "USD million"
STATE_UNIT = 100_000.0  # USD million per $100 billion
HORIZONS = (0, 1, 4, 8)
HAC_BANDWIDTHS = (12, 8, 26)  # baseline first
DWB_BANDWIDTH = 12
EVENT_WEEKS = tuple(range(-2, 5))
REBUILD_WEEKS = 13
TOLERANCE = 1e-6
WEDNESDAY_EPOCH = pd.Timestamp("1970-01-07")

# Exposure-week Wednesdays. Contiguous eras; a versioned config may declare a later
# final era. Transition weeks stay in the full sample only.
ERAS = (
    ("era_2003_2008", "2003-01-01", "2008-09-30"),
    ("transition_2008q4", "2008-10-01", "2008-12-31"),
    ("era_2009_2015", "2009-01-01", "2015-09-30"),
    ("transition_2015q4", "2015-10-01", "2015-12-31"),
    ("era_2016_2025", "2016-01-01", "2025-12-31"),
)


def samples_of(eras: tuple[tuple[str, str, str], ...]) -> dict[str, tuple[str, str]]:
    """The full sample spans every era; each non-transition era is also a sample."""
    for (_, _, end), (label, start, _) in zip(eras, eras[1:]):
        if pd.Timestamp(end) + pd.Timedelta(days=1) != pd.Timestamp(start):
            raise ValueError(f"eras must be contiguous; {label} does not follow the previous era")
    return {"full": (eras[0][1], eras[-1][2]), **{label: (start, end) for label, start, end in eras if label.startswith("era_")}}


SAMPLES = samples_of(ERAS)

# One nonoverlapping map from every ledger factor row to a reported group.
FACTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "cash_leg": ("treasury_general_account", "supplementary_financing_account"),
    "rrp_others": ("rrp_others",),
    "foreign_official": ("rrp_foreign_official", "foreign_official_deposits"),
    "currency": ("currency_in_circulation",),
    "treasury_securities": ("treasury_securities",),
    "mbs_agency": ("mortgage_backed_securities", "federal_agency_debt"),
    "lending_facilities": (
        "repurchase_agreements",
        "term_auction_credit",
        "loans",
        "cpff_llc",
        "mmiff_llcs",
        "maiden_lane_llc",
        "maiden_lane_ii_llc",
        "maiden_lane_iii_llc",
        "talf_llc",
        "aia_alico_preferred_interests",
        "cpff_ii_llc",
        "corporate_credit_facilities_llc",
        "municipal_liquidity_facility_llc",
        "ms_facilities_llc",
        "talf_ii_llc",
        "central_bank_liquidity_swaps",
        "treasury_contributions_to_credit_facilities",
    ),
    "all_other": (
        "unamortized_premiums",
        "unamortized_discounts",
        "float",
        "other_federal_reserve_assets",
        "foreign_currency_denominated_assets",
        "gold_stock",
        "sdr_certificate_account",
        "treasury_currency_outstanding",
        "treasury_cash_holdings",
        "term_deposits",
        "service_related_deposits",
        "other_deposits",
        "aig_disposition_funds_held_as_agent",
        "other_liabilities_and_capital",
    ),
}
CONTRIBUTIONS = tuple(f"contrib_{group}" for group in FACTOR_GROUPS) + ("contrib_rho",)
OUTCOMES = ("delta_reserve_balances", "delta_rrp_others", "delta_reserves_plus_rrp_others")
DEFINITIONS = {
    "exposure_c": "Treasury cash leg c_h = -d(TGA) - d(SFP) between consecutive Wednesdays",
    "delta_reserve_balances": "Change in H.4.1 reserve balances (WRBWFRBL)",
    "delta_rrp_others": "Change in the RRP-others balance (WLRRAOL); equals -contrib_rrp_others",
    "delta_reserves_plus_rrp_others": "Memo: change in reserve balances plus the RRP-others balance; not a reserve perimeter",
    **{
        f"contrib_{group}": "Signed contribution to the reserve change (+d supplying row, -d absorbing row): " + ", ".join(rows)
        for group, rows in FACTOR_GROUPS.items()
    },
    "contrib_rho": "Published rounding remainder of the ledger identity",
}
PRIMARY_OUTCOMES = OUTCOMES
PRIMARY_INTERACTION_OUTCOME = "delta_reserve_balances"
TAX_CATEGORIES = tuple(TaxDeadlineCategory)
LEVEL_INPUTS = ("rrp_others", "treasury_general_account")
TAX_VARIANTS = ("all_events", "no_neighboring_deadline_weeks", "no_codated_or_neighboring_deadline_weeks")
TAX_CALENDAR_COVERAGE = (
    "National calendar-year deadlines for individuals and C corporations only; "
    "excludes regional disaster relief, IRS-campus-specific state-holiday relief, and fiscal-year filers"
)
RRP_BOUNDARY = (
    "WLRRAOL (RRP others) mixes counterparties; it is neither an ON RRP measure "
    "nor a money-fund funding-route measure. The state is its lagged balance."
)
TAX_PATH_BASE = (
    "Cumulative paths start from the end of event week -3; the week-zero value "
    "includes changes in weeks -2, -1 and 0"
)
OTHER_GROUPS = ("currency", "treasury_securities", "mbs_agency", "lending_facilities", "all_other")
ACCOUNT_DEFINITIONS = {
    "cash_leg": "sum C_w = sum[-d(TGA) - d(SFP)] over (base_wednesday, end_wednesday]",
    "delta_reserve_balances": "reserve_balances_end - reserve_balances_base = sum signed reserve contributions including rho",
    "delta_rrp_others": "rrp_others_level_end - rrp_others_level_base = -sum contrib_rrp_others; positive is a balance increase",
    "delta_foreign_official_balances": "-sum contrib_foreign_official = d(RRP foreign official) + d(foreign official deposits); positive is a balance increase",
    "negative_other_factors_and_rho": "-sum(contrib_currency + contrib_treasury_securities + contrib_mbs_agency + contrib_lending_facilities + contrib_all_other + contrib_rho); negative of signed reserve contributions, not an attributed flow",
}

# Production inference sizes; tests pass smaller values explicitly.
BOOTSTRAP_REPS = 1999
CALIBRATION_SIMS = 500
CALIBRATION_REPS = 499
PLACEBO_DRAWS = 500
SEED = 20261009
NULL_AR = 0.5
NULL_SCALE_WEEKS = 13
NOMINAL_SIZE = 0.05
OVERLAP_SPLIT_100BN = 1.0


def _factor_rows() -> list[str]:
    return [row for rows in FACTOR_GROUPS.values() for row in rows]


def _week_ordinals(dates: pd.DatetimeIndex) -> np.ndarray:
    days = (dates - WEDNESDAY_EPOCH).days.to_numpy()
    if (days % 7 != 0).any():
        raise ValueError("ledger dates must all be Wednesdays")
    return days // 7


def final_wednesday(day: str | pd.Timestamp) -> pd.Timestamp:
    """Last Wednesday on or before ``day``: the final exposure week a date bound admits."""
    end = pd.Timestamp(day)
    return end - pd.Timedelta(days=(end.weekday() - 2) % 7)


def load_design_config(path: str | Path) -> dict[str, object]:
    """Versioned rerun design: vintage, declared eras and pinned input hashes."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"design config must declare schema {SCHEMA}")
    eras = tuple(tuple(era) for era in payload["eras"])
    if any(len(era) != 3 for era in eras):
        raise ValueError("each declared era needs a label, a start and an end")
    samples_of(eras)
    return {
        "path": str(path),
        "sha256": _sha256(Path(path)),
        "vintage": payload["vintage"],
        "eras": eras,
        "ledger_sha256": payload["ledger_sha256"],
        "ledger_receipt_sha256": payload["ledger_receipt_sha256"],
        "source_receipt_sha256": payload["source_receipt_sha256"],
    }


def era_label(value: pd.Timestamp, eras: tuple = ERAS) -> str | None:
    for label, start, end in eras:
        if pd.Timestamp(start) <= value <= pd.Timestamp(end):
            return label
    return None


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _check_hash(path: Path, expected: str, *, source: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{source}: missing input {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{source}: sha256 mismatch ({actual} != pinned {expected})")
    return actual


def load_ledger(
    ledger_csv: str | Path,
    receipt_json: str | Path,
    *,
    ledger_sha256: str = LEDGER_SHA256,
    receipt_sha256: str = LEDGER_RECEIPT_SHA256,
) -> tuple[pd.DataFrame, dict]:
    """Read the pinned weekly ledger and fail closed on any contract breach."""
    ledger_path, receipt_path = Path(ledger_csv), Path(receipt_json)
    _check_hash(ledger_path, ledger_sha256, source="ledger")
    _check_hash(receipt_path, receipt_sha256, source="ledger_receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != LEDGER_SCHEMA or receipt.get("units") != UNITS:
        raise ValueError("ledger_receipt: unexpected schema or units")
    if receipt["outputs"]["weekly"]["sha256"] != ledger_sha256:
        raise ValueError("ledger_receipt: weekly output hash does not match the pinned ledger")

    frame = pd.read_csv(ledger_path)
    factor_columns = {col[2:] for col in frame.columns if col.startswith("f_")}
    mapped = _factor_rows()
    if len(mapped) != len(set(mapped)):
        raise AssertionError("a ledger factor is mapped to more than one group")
    if factor_columns != set(mapped):
        raise ValueError(
            "ledger factors and group map differ: "
            f"unmapped={sorted(factor_columns - set(mapped))} missing={sorted(set(mapped) - factor_columns)}"
        )
    required = ["date", "reserve_balances", "delta_reserve_balances", "c_h", "rho", "ledger_status"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"ledger: missing columns {missing}")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("ledger: dates must be unique and increasing")
    if not (dates.diff().dropna() == pd.Timedelta(days=7)).all():
        raise ValueError("ledger: consecutive Wednesdays are required; a week is missing")
    _week_ordinals(pd.DatetimeIndex(dates))
    frame = frame.assign(date=dates).set_index("date")
    status = frame["ledger_status"].astype(str)
    if status.iloc[0] != "unavailable_no_prior_wednesday" or (status.iloc[1:] != "closed_within_rounding_bound").any():
        raise ValueError("ledger: every change row must be closed within its rounding bound")
    values = frame.iloc[1:][["reserve_balances", "delta_reserve_balances", "c_h", "rho", *[f"f_{r}" for r in mapped]]]
    if values.isna().any().any():
        raise ValueError("ledger: missing values in change rows")
    body = frame.iloc[1:]
    identity = body["delta_reserve_balances"] - body[[f"f_{r}" for r in mapped]].sum(axis=1) - body["rho"]
    if identity.abs().max() > TOLERANCE:
        raise ValueError("ledger: signed factors plus rho do not sum to the reserve change")
    cash = body["c_h"] - body[[f"f_{r}" for r in FACTOR_GROUPS["cash_leg"]]].sum(axis=1)
    if cash.abs().max() > TOLERANCE:
        raise ValueError("ledger: c_h differs from the TGA and SFP factors")
    return frame, receipt


def load_level(path: str | Path, receipt: dict, key: str, ledger: pd.DataFrame) -> pd.Series:
    """Read one raw Wednesday level pinned by the ledger receipt and tie it to the ledger."""
    expected = receipt["inputs"]["fiscus_inputs"][key]["sha256"]
    _check_hash(Path(path), expected, source=f"level_{key}")
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"], format="%Y-%m-%d", errors="raise")
    if raw["date"].duplicated().any():
        raise ValueError(f"level_{key}: duplicate dates")
    level = raw.set_index("date")["value"].astype(float).reindex(ledger.index)
    if level.isna().any():
        raise ValueError(f"level_{key}: missing Wednesdays in the ledger range")
    gap = (-level.diff() - ledger[f"f_{key}"]).iloc[1:]
    if gap.abs().max() > TOLERANCE:
        raise ValueError(f"level_{key}: level changes disagree with the ledger factor")
    return level


# --------------------------------------------------------------------------
# Weekly panel, calendar, and state
# --------------------------------------------------------------------------


def tax_event_weeks(first_year: int, last_year: int) -> pd.DataFrame:
    """Typed deadlines mapped to the Thursday-Wednesday week that contains them."""
    rows = []
    for year in range(first_year, last_year + 1):
        for item in tax_deadlines(year):
            rows.append(
                {
                    "category": item.category.value,
                    "statutory_date": pd.Timestamp(item.statutory_date),
                    "due_date": pd.Timestamp(item.due_date),
                    "event_week": assign_h8_week(item.due_date),
                }
            )
    return pd.DataFrame(rows)


def calendar_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    deadlines = tax_event_weeks(dates.min().year - 1, dates.max().year + 1)
    out = pd.DataFrame(index=dates)
    for month in range(2, 13):
        out[f"month_{month:02d}"] = (dates.month == month).astype(float)
    start = dates - pd.Timedelta(days=6)
    month_end = start + pd.offsets.MonthEnd(0)
    out["month_end_week"] = (month_end <= dates).astype(float)
    out["quarter_end_week"] = ((month_end <= dates) & month_end.month.isin([3, 6, 9, 12])).astype(float)
    for category in TAX_CATEGORIES:
        weeks = set(deadlines.loc[deadlines["category"].eq(category.value), "event_week"])
        out[f"tax_{category.value}_week"] = dates.isin(weeks).astype(float)
    return out


def build_weekly_panel(ledger: pd.DataFrame, rrp_level: pd.Series, tga_level: pd.Series, eras: tuple = ERAS) -> pd.DataFrame:
    """One row per exposure week; every regressor is dated at or before the week."""
    panel = pd.DataFrame(index=ledger.index)
    panel["exposure_c"] = ledger["c_h"].astype(float)
    panel["reserve_balances"] = ledger["reserve_balances"].astype(float)
    panel["delta_reserve_balances"] = ledger["delta_reserve_balances"].astype(float)
    for group, rows in FACTOR_GROUPS.items():
        panel[f"contrib_{group}"] = ledger[[f"f_{r}" for r in rows]].sum(axis=1, min_count=1)
    panel["contrib_rho"] = ledger["rho"].astype(float)
    for row in _factor_rows():
        panel[f"f_{row}"] = ledger[f"f_{row}"].astype(float)
    panel["delta_rrp_others"] = -panel["contrib_rrp_others"]
    panel["delta_reserves_plus_rrp_others"] = panel["delta_reserve_balances"] + panel["delta_rrp_others"]
    panel["rrp_others_level"] = rrp_level
    panel["tga_level"] = tga_level
    # The ledger is consecutive, so shift(1) is exactly one calendar week.
    panel["lag_exposure_c"] = panel["exposure_c"].shift(1)
    panel["lag_delta_reserve_balances"] = panel["delta_reserve_balances"].shift(1)
    panel["state_rrp_others_100bn"] = rrp_level.shift(1) / STATE_UNIT
    panel["state_date"] = pd.Series(ledger.index, index=ledger.index).shift(1)
    panel["exposure_start"] = ledger.index - pd.Timedelta(days=6)
    timed = panel["state_date"].notna()
    if not (panel.loc[timed, "state_date"] < panel.loc[timed, "exposure_start"]).all():
        raise AssertionError("state must be observed strictly before the exposure week")
    panel["era"] = [era_label(day, eras) for day in ledger.index]
    return panel.join(calendar_flags(pd.DatetimeIndex(ledger.index)))


def cumulative(series: pd.Series, horizon: int) -> pd.Series:
    """Y_{w,h} = level_{w+h} - level_{w-1}; missing when the endpoint is unobserved."""
    return series.rolling(horizon + 1, min_periods=horizon + 1).sum().shift(-horizon)


# --------------------------------------------------------------------------
# Projection and calendar-distance HAC
# --------------------------------------------------------------------------


BASELINE_CONTROLS = (
    *[f"month_{m:02d}" for m in range(2, 13)],
    "month_end_week",
    "quarter_end_week",
    *[f"tax_{c.value}_week" for c in TAX_CATEGORIES],
    "lag_exposure_c",
    "lag_delta_reserve_balances",
)


@dataclass
class Design:
    rows: pd.DatetimeIndex
    names: list[str]
    x: np.ndarray
    a: np.ndarray  # (X'X)^{-1} X'
    weeks: np.ndarray
    dropped: list[str] = field(default_factory=list)
    _pairs: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def pairs(self, lag: int) -> tuple[np.ndarray, np.ndarray]:
        if lag not in self._pairs:
            pos = np.searchsorted(self.weeks, self.weeks + lag)
            ok = pos < len(self.weeks)
            ok[ok] = self.weeks[pos[ok]] == self.weeks[ok] + lag
            self._pairs[lag] = (np.flatnonzero(ok), pos[ok])
        return self._pairs[lag]

    def index(self, name: str) -> int:
        return self.names.index(name)


def make_design(columns: pd.DataFrame) -> Design:
    """Drop structurally all-zero columns, then require full column rank."""
    if columns.isna().any().any():
        raise ValueError("design contains missing values")
    dropped = [c for c in columns.columns if c != "const" and not columns[c].abs().gt(0).any()]
    kept = columns.drop(columns=dropped)
    x = kept.to_numpy(dtype=float)
    scale = np.sqrt((x**2).mean(axis=0))
    q, r = np.linalg.qr(x / scale)
    if np.linalg.matrix_rank(r) < x.shape[1] or abs(np.diag(r)).min() < 1e-10 * abs(np.diag(r)).max():
        raise ValueError(f"design is rank deficient: {list(kept.columns)}")
    a = np.linalg.solve(r, q.T) / scale[:, None]
    rows = pd.DatetimeIndex(kept.index)
    return Design(rows=rows, names=list(kept.columns), x=x, a=a, weeks=_week_ordinals(rows), dropped=dropped)


def design_for(panel: pd.DataFrame, rows: pd.DatetimeIndex, spec: str, *, state: pd.Series | None = None, eras: tuple = ERAS) -> Design:
    frame = panel.loc[rows]
    cols = pd.DataFrame({"const": 1.0, "exposure_c": frame["exposure_c"]}, index=rows)
    if spec == "intercept_only":
        return make_design(cols)
    for name in BASELINE_CONTROLS:
        cols[name] = frame[name]
    if spec == "state":
        level = frame["state_rrp_others_100bn"] if state is None else state.loc[rows]
        cols["state_rrp_others_100bn"] = level
        cols["exposure_x_state"] = frame["exposure_c"] * level
        for label, _, _ in eras[1:]:
            in_era = frame["era"].eq(label).astype(float)
            cols[f"era_{label}"] = in_era
            cols[f"exposure_x_{label}"] = frame["exposure_c"] * in_era
    elif spec != "baseline":
        raise ValueError(f"unknown specification: {spec}")
    return make_design(cols)


def _quad_hac(design: Design, z: np.ndarray, bandwidth: int) -> np.ndarray:
    """sum_ij k(|week_i - week_j|) z_i z_j with Bartlett weights 1 - d/(L+1)."""
    total = np.einsum("i...,i...->...", z, z)
    for lag in range(1, bandwidth + 1):
        i, j = design.pairs(lag)
        if len(i):
            total = total + 2.0 * (1.0 - lag / (bandwidth + 1)) * np.einsum("i...,i...->...", z[i], z[j])
    return total


def fit(design: Design, y: np.ndarray, coef: str) -> dict[str, np.ndarray]:
    """OLS coefficient and calendar-distance HAC SEs (no small-sample scaling)."""
    beta_all = design.a @ y
    resid = y - design.x @ beta_all
    j = design.index(coef)
    z = design.a[j][:, None] * resid if resid.ndim == 2 else design.a[j] * resid
    out = {"beta": beta_all[j]}
    for bandwidth in HAC_BANDWIDTHS:
        out[f"se_hac{bandwidth}"] = np.sqrt(_quad_hac(design, z, bandwidth))
    return out


def dwb_weights(weeks: np.ndarray, reps: int, rng: np.random.Generator, bandwidth: int = DWB_BANDWIDTH) -> np.ndarray:
    """Dependent wild bootstrap multipliers indexed by calendar week.

    A moving sum of b = L+1 iid normals gives Cov(eta_t, eta_s) = 1 - |t-s|/(L+1),
    the same Bartlett kernel as the baseline HAC. Every outcome and horizon that
    shares a calendar week receives the same multiplier.
    """
    width = bandwidth + 1
    origin = weeks.min() - width + 1
    shocks = rng.standard_normal((weeks.max() - origin + 1, reps))
    csum = np.vstack([np.zeros((1, reps)), np.cumsum(shocks, axis=0)])
    pos = weeks - origin
    return (csum[pos + 1] - csum[pos + 1 - width]) / np.sqrt(width)


@dataclass
class Member:
    name: str
    design: Design
    y: np.ndarray
    coef: str


def joint_max_t(
    members: list[Member], eta: np.ndarray, *, hac_bandwidth: int = HAC_BANDWIDTHS[0], chunk: int = 250
) -> dict[str, object]:
    """Simultaneous max-t band and joint zero test with common rows and weights.

    Each replicate perturbs every member's residuals with the same calendar-week
    multipliers, refits, and studentizes with the same calendar HAC.
    """
    rows = members[0].design.rows
    for member in members[1:]:
        if not member.design.rows.equals(rows):
            raise ValueError(f"joint family rows differ for {member.name}")
    if eta.shape[0] != len(rows):
        raise ValueError("bootstrap weights do not match the family rows")
    groups: dict[int, list[int]] = {}
    for pos, member in enumerate(members):
        groups.setdefault(id(member.design), []).append(pos)
    stats = [(0.0, 0.0)] * len(members)
    boot = np.empty((len(members), eta.shape[1]))
    for positions in groups.values():
        d = members[positions[0]].design
        js = [d.index(members[p].coef) for p in positions]
        y = np.column_stack([members[p].y for p in positions])
        beta = d.a @ y
        resid = y - d.x @ beta
        a_rows = d.a[js].T  # n x m
        se = np.sqrt(_quad_hac(d, a_rows * resid, hac_bandwidth))
        for m, p in enumerate(positions):
            stats[p] = (float(beta[js[m], m]), float(se[m]))
        cols = np.arange(len(positions))
        for lo in range(0, eta.shape[1], chunk):
            u_star = eta[:, lo : lo + chunk, None] * resid[:, None, :]
            delta = np.tensordot(d.a, u_star, axes=(1, 0))
            resid_star = u_star - np.tensordot(d.x, delta, axes=(1, 0))
            se_star = np.sqrt(_quad_hac(d, a_rows[:, None, :] * resid_star, hac_bandwidth))
            boot[positions, lo : lo + chunk] = (delta[js, :, cols] / se_star.T)
    max_star = np.abs(boot).max(axis=0)
    crit = float(np.quantile(max_star, 1 - NOMINAL_SIZE))
    observed = max(abs(b / s) for b, s in stats)
    return {
        "members": [
            {"name": m.name, "beta": b, "se": s, "t": b / s, "band_lower": b - crit * s, "band_upper": b + crit * s}
            for m, (b, s) in zip(members, stats)
        ],
        "critical_value": crit,
        "max_abs_t": observed,
        "joint_p": float((1 + np.sum(max_star >= observed)) / (len(max_star) + 1)),
        "reps": int(eta.shape[1]),
    }


# --------------------------------------------------------------------------
# Item 3: estimates, decomposition, horizons, state, family, calibration
# --------------------------------------------------------------------------


COEFFICIENT_UNITS = {
    "exposure_c": "USD per USD of cash leg",
    "exposure_x_state": "USD per USD of cash leg, per $100bn of pre-exposure RRP-others balance",
    "state_rrp_others_100bn": "USD million per $100bn of pre-exposure RRP-others balance",
}


def outcome_kind(name: str) -> str:
    if name.startswith("contrib_"):
        return "factor_contribution"
    return "memo" if name == "delta_reserves_plus_rrp_others" else "outcome"


def sample_rows(panel: pd.DataFrame, sample: str, eras: tuple = ERAS) -> tuple[pd.DatetimeIndex, dict]:
    """Rows with lags, state, and every horizon endpoint observed (common rows)."""
    start, end = (pd.Timestamp(v) for v in samples_of(eras)[sample])
    window = panel.loc[(panel.index >= start) & (panel.index <= end)]
    lagged = window.dropna(subset=["exposure_c", "lag_exposure_c", "lag_delta_reserve_balances", "state_rrp_others_100bn"])
    endpoint = cumulative(panel["delta_reserve_balances"], max(HORIZONS)).reindex(lagged.index)
    rows = pd.DatetimeIndex(lagged.index[endpoint.notna().to_numpy()])
    return rows, {
        "n_in_window": int(len(window)),
        "n_dropped_missing_lag": int(len(window) - len(lagged)),
        "n_dropped_unobserved_endpoint": int(len(lagged) - len(rows)),
    }


def outcome_matrix(panel: pd.DataFrame, names: list[str], horizon: int, rows: pd.DatetimeIndex) -> np.ndarray:
    return np.column_stack([cumulative(panel[name], horizon).loc[rows].to_numpy(dtype=float) for name in names])


def sample_dates(rows: pd.DatetimeIndex, horizon: int) -> dict[str, str]:
    """Era membership follows the exposure date; endpoints may cross an era boundary."""
    return {
        "sample_date_basis": "exposure_week_end_date",
        "first_week": rows.min().date().isoformat(),
        "last_week": rows.max().date().isoformat(),
        "first_outcome_base_week": (rows.min() - pd.Timedelta(weeks=1)).date().isoformat(),
        "last_outcome_base_week": (rows.max() - pd.Timedelta(weeks=1)).date().isoformat(),
        "first_outcome_endpoint_week": (rows.min() + pd.Timedelta(weeks=horizon)).date().isoformat(),
        "last_outcome_endpoint_week": (rows.max() + pd.Timedelta(weeks=horizon)).date().isoformat(),
    }


def estimate_all(panel: pd.DataFrame, eras: tuple = ERAS) -> pd.DataFrame:
    names = [*OUTCOMES, *CONTRIBUTIONS]
    records = []
    plans = [("baseline", h) for h in HORIZONS] + [("intercept_only", 0)]
    for sample in samples_of(eras):
        rows, counts = sample_rows(panel, sample, eras)
        designs = {spec: design_for(panel, rows, spec, eras=eras) for spec in ("baseline", "intercept_only")}
        for spec, horizon in plans:
            y = outcome_matrix(panel, names, horizon, rows)
            result = fit(designs[spec], y, "exposure_c")
            for k, name in enumerate(names):
                records.append(_record(spec, sample, name, horizon, "exposure_c", result, k, designs[spec], counts))
    rows, counts = sample_rows(panel, "full", eras)
    design = design_for(panel, rows, "state", eras=eras)
    y = outcome_matrix(panel, names, 0, rows)
    for coef in ("exposure_x_state", "state_rrp_others_100bn", "exposure_c"):
        result = fit(design, y, coef)
        for k, name in enumerate(names):
            records.append(_record("state", "full", name, 0, coef, result, k, design, counts))
    out = pd.DataFrame(records)
    _assert_decomposition(out)
    return out


def _record(spec, sample, name, horizon, coef, result, k, design, counts) -> dict[str, object]:
    beta = float(result["beta"][k])
    se = float(result["se_hac12"][k])
    return {
        "schema_version": SCHEMA,
        "specification": spec,
        "sample": sample,
        "outcome": name,
        "outcome_kind": outcome_kind(name),
        "horizon": horizon,
        "coefficient": coef,
        "beta": beta,
        "se_hac12": se,
        "se_hac8": float(result["se_hac8"][k]),
        "se_hac26": float(result["se_hac26"][k]),
        "t_hac12": beta / se if se > 0 else np.nan,
        "ci95_lower_hac12": beta - 1.959964 * se,
        "ci95_upper_hac12": beta + 1.959964 * se,
        "n": len(design.rows),
        "k": len(design.names),
        **sample_dates(design.rows, horizon),
        "dropped_all_zero_columns": ";".join(design.dropped),
        **counts,
        "units": COEFFICIENT_UNITS[coef],
    }


def _assert_decomposition(estimates: pd.DataFrame) -> None:
    keys = ["specification", "sample", "horizon", "coefficient"]
    for key, cell in estimates.groupby(keys):
        total = cell.loc[cell["outcome"].eq("delta_reserve_balances"), "beta"].iloc[0]
        parts = cell.loc[cell["outcome"].isin(CONTRIBUTIONS), "beta"].sum()
        if abs(total - parts) > TOLERANCE * max(1.0, abs(total)):
            raise AssertionError(f"factor slopes do not sum to the reserve slope for {key}")
        if key[0] != "state" and key[2] == 0 and key[3] == "exposure_c":
            cash = cell.loc[cell["outcome"].eq("contrib_cash_leg"), "beta"].iloc[0]
            if abs(cash - 1.0) > TOLERANCE:
                raise AssertionError("the same-week cash-leg contribution must have slope one")


def primary_family(
    panel: pd.DataFrame, *, state: pd.Series | None = None, outcomes: dict[str, pd.Series] | None = None, eras: tuple = ERAS
) -> list[Member]:
    """Declared family: reserve, RRP-others and memo outcomes at h = 0, 1, 4, 8 and one interaction."""
    source = panel if outcomes is None else panel.assign(**outcomes)
    rows, _ = sample_rows(panel, "full", eras)
    base = design_for(panel, rows, "baseline", eras=eras)
    members = [
        Member(f"{name}|h{h}", base, cumulative(source[name], h).loc[rows].to_numpy(dtype=float), "exposure_c")
        for name in PRIMARY_OUTCOMES
        for h in HORIZONS
    ]
    inter = design_for(panel, rows, "state", state=state, eras=eras)
    y = source[PRIMARY_INTERACTION_OUTCOME].loc[rows].to_numpy(dtype=float)
    members.append(Member(f"{PRIMARY_INTERACTION_OUTCOME}|h0|exposure_x_state", inter, y, "exposure_x_state"))
    return members


def random_walk_state(panel: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Pseudo-state: a Gaussian random walk observed with the same one-week lag."""
    walk = pd.Series(np.cumsum(rng.standard_normal(len(panel))), index=panel.index)
    return walk.shift(1)


def _rolling_rms(values: pd.Series) -> pd.Series:
    rms = np.sqrt((values**2).rolling(NULL_SCALE_WEEKS, center=True, min_periods=1).mean())
    return rms.ffill().bfill()


def null_outcomes(panel: pd.DataFrame, residuals: pd.DataFrame, rng: np.random.Generator) -> dict[str, pd.Series]:
    """Serially correlated, heteroskedastic null flows with zero cash-leg coefficients.

    Scale follows the 13-week RMS of the observed baseline residuals and rises
    in large-|C| weeks; AR(1) persistence 0.5; cross-outcome innovation
    correlation equals the observed residual correlation.
    """
    index = panel.index
    exposure = panel["exposure_c"].fillna(0.0)
    z = exposure / _rolling_rms(exposure).replace(0.0, np.nan).fillna(1.0)
    exposure_scale = np.sqrt(0.5 + 0.5 * z.to_numpy() ** 2)
    corr = float(np.clip(residuals.corr().iloc[0, 1], -0.95, 0.95))
    chol = np.linalg.cholesky(np.array([[1.0, corr], [corr, 1.0]]))
    shocks = rng.standard_normal((len(index), 2)) @ chol.T
    ar = np.empty_like(shocks)
    ar[0] = shocks[0]
    for t in range(1, len(index)):
        ar[t] = NULL_AR * ar[t - 1] + np.sqrt(1 - NULL_AR**2) * shocks[t]
    out = {}
    for k, name in enumerate(residuals.columns):
        scale = _rolling_rms(residuals[name].reindex(index)).to_numpy()
        out[name] = pd.Series(scale * exposure_scale * ar[:, k], index=index)
    reserves, rrp_contrib = out["delta_reserve_balances"], out["contrib_rrp_others"]
    return {
        "delta_reserve_balances": reserves,
        "delta_rrp_others": -rrp_contrib,
        "delta_reserves_plus_rrp_others": reserves - rrp_contrib,
    }


def calibrate(
    panel: pd.DataFrame,
    *,
    sims: int,
    reps: int,
    placebo_draws: int,
    seed: int,
    real_interaction_t: float,
    eras: tuple = ERAS,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    rng = np.random.default_rng(seed + 1)
    rows, _ = sample_rows(panel, "full", eras)
    base = design_for(panel, rows, "baseline", eras=eras)
    names = ["delta_reserve_balances", "contrib_rrp_others"]
    y = panel.loc[rows, names].to_numpy(dtype=float)
    residuals = pd.DataFrame(y - base.x @ (base.a @ y), index=rows, columns=names)
    rejections = {"observed_state": [], "random_walk_state": []}
    pointwise = {"observed_state": [], "random_walk_state": []}
    for _ in range(sims):
        simulated = null_outcomes(panel, residuals, rng)
        for variant in rejections:
            state = None if variant == "observed_state" else random_walk_state(panel, rng)
            members = primary_family(panel, state=state, outcomes=simulated, eras=eras)
            eta = dwb_weights(members[0].design.weeks, reps, rng)
            result = joint_max_t(members, eta)
            rejections[variant].append(result["joint_p"] <= NOMINAL_SIZE)
            pointwise[variant].append(abs(result["members"][-1]["t"]) > 1.959964)
    threshold = NOMINAL_SIZE + 2 * np.sqrt(NOMINAL_SIZE * (1 - NOMINAL_SIZE) / max(sims, 1))
    records = []
    for variant in rejections:
        for check, draws in (("joint_max_t_size", rejections[variant]), ("interaction_hac12_size", pointwise[variant])):
            rate = float(np.mean(draws))
            records.append(_calibration_row(f"{check}|{variant}", "simulated_null", sims, rate, threshold, reps))
    placebo = []
    design_rows = rows
    y_real = panel.loc[design_rows, PRIMARY_INTERACTION_OUTCOME].to_numpy(dtype=float)
    for _ in range(placebo_draws):
        design = design_for(panel, design_rows, "state", state=random_walk_state(panel, rng), eras=eras)
        result = fit(design, y_real, "exposure_x_state")
        placebo.append(float(result["beta"] / result["se_hac12"]))
    placebo = np.abs(np.asarray(placebo))
    p_threshold = NOMINAL_SIZE + 2 * np.sqrt(NOMINAL_SIZE * (1 - NOMINAL_SIZE) / max(placebo_draws, 1))
    row = _calibration_row(
        "interaction_hac12_false_positive|random_walk_state|real_outcome",
        "real_outcome_placebo",
        placebo_draws,
        float(np.mean(placebo > 1.959964)),
        p_threshold,
        0,
    )
    row["real_abs_t"] = abs(real_interaction_t)
    row["share_placebo_abs_t_ge_real"] = float(np.mean(placebo >= abs(real_interaction_t)))
    records.append(row)
    table = pd.DataFrame(records)
    passed = dict(zip(table["check"], table["pass"]))
    family_ok = all(passed[f"joint_max_t_size|{v}"] for v in rejections)
    pointwise_ok = all(passed[f"interaction_hac12_size|{v}"] for v in rejections)
    return table, {
        "family": family_ok,
        "interaction_pointwise": pointwise_ok,
        "interaction_real_placebo": bool(passed[row["check"]]),
    }


def _calibration_row(check, kind, draws, rate, threshold, reps) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "check": check,
        "check_kind": kind,
        "draws": draws,
        "bootstrap_reps_per_draw": reps,
        "nominal_size": NOMINAL_SIZE,
        "rejection_rate": rate,
        "monte_carlo_se": float(np.sqrt(rate * (1 - rate) / max(draws, 1))),
        "pass_threshold": float(threshold),
        "pass": bool(rate <= threshold),
    }


def overlap_table(panel: pd.DataFrame, eras: tuple = ERAS) -> pd.DataFrame:
    """Support of C in low and high pre-exposure state (high: RRP others above $100bn)."""
    rows, _ = sample_rows(panel, "full", eras)
    frame = panel.loc[rows, ["exposure_c", "state_rrp_others_100bn", "era"]].copy()
    split = OVERLAP_SPLIT_100BN
    frame["state_group"] = np.where(frame["state_rrp_others_100bn"] > split, "high", "low")
    lo, hi = frame.loc[frame["state_group"].eq("low"), "exposure_c"].quantile([0.05, 0.95])
    cells = [(g, "all", c) for g, c in frame.groupby("state_group")]
    cells += [(g, e, c) for (g, e), c in frame.groupby(["state_group", "era"])]
    records = []
    for group, era, cell in cells:
        c = cell["exposure_c"]
        records.append(
            {
                "schema_version": SCHEMA,
                "state_group": group,
                "era": era,
                "state_split_100bn": split,
                "n": int(len(c)),
                "state_min_100bn": float(cell["state_rrp_others_100bn"].min()),
                "state_max_100bn": float(cell["state_rrp_others_100bn"].max()),
                "c_sd": float(c.std()),
                "c_p05": float(c.quantile(0.05)),
                "c_p50": float(c.quantile(0.5)),
                "c_p95": float(c.quantile(0.95)),
                "share_c_inside_low_state_p05_p95": float(c.between(lo, hi).mean()),
            }
        )
    return pd.DataFrame(records)


def influence_table(
    panel: pd.DataFrame, inference_status: str = "descriptive_outside_declared_family", eras: tuple = ERAS
) -> pd.DataFrame:
    """Partial exposure variation, leverage concentration and leave-one-week-out ranges.

    No observation is winsorized or reweighted; these rows only show how much
    each slope depends on individual weeks.
    """
    records = []
    for sample in samples_of(eras):
        rows, _ = sample_rows(panel, sample, eras)
        design = design_for(panel, rows, "baseline", eras=eras)
        j = design.index("exposure_c")
        exposure = panel.loc[rows, "exposure_c"].to_numpy(dtype=float)
        partial = design.a[j] / (design.a[j] @ design.a[j])  # residualized exposure
        shares = partial**2 / (partial**2).sum()
        order = np.argsort(shares)[::-1]
        hat = np.einsum("ik,ki->i", design.x, design.a)
        for name in OUTCOMES:
            y = panel.loc[rows, name].to_numpy(dtype=float)
            beta_all = design.a @ y
            resid = y - design.x @ beta_all
            loo = beta_all[j] - design.a[j] * resid / (1.0 - hat)
            move = np.abs(loo - beta_all[j])
            worst = int(np.argmax(move))
            records.append(
                {
                    "schema_version": SCHEMA,
                    "sample": sample,
                    "outcome": name,
                    "n": len(rows),
                    **sample_dates(rows, 0),
                    "inference_status": inference_status,
                    "uncertainty_status": "diagnostic",
                    "beta": float(beta_all[j]),
                    "exposure_sd": float(exposure.std(ddof=1)),
                    "partial_exposure_sd": float(partial.std(ddof=1)),
                    "partial_to_total_variation": float((partial**2).sum() / ((exposure - exposure.mean()) ** 2).sum()),
                    "top_week": rows[order[0]].date().isoformat(),
                    "top_week_variation_share": float(shares[order[0]]),
                    "top5_weeks_variation_share": float(shares[order[:5]].sum()),
                    "top5_weeks": ";".join(rows[order[:5]].strftime("%Y-%m-%d")),
                    "loo_beta_min": float(loo.min()),
                    "loo_beta_max": float(loo.max()),
                    "largest_move_week": rows[worst].date().isoformat(),
                    "beta_without_largest_move_week": float(loo[worst]),
                }
            )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Item 4: descriptive tax-deadline event paths
# --------------------------------------------------------------------------

PATH_SERIES = (
    "exposure_c",
    *OUTCOMES,
    *CONTRIBUTIONS,
    *[f"f_{row}" for row in _factor_rows()],
)


def _series_kind(name: str) -> str:
    if name == "exposure_c":
        return "cash_leg_exposure"
    if name.startswith("f_"):
        return "ledger_factor"
    return outcome_kind(name)


def tax_event_windows(panel: pd.DataFrame, eras: tuple = ERAS) -> pd.DataFrame:
    """One event per category and H.4.1 week; overlaps with other deadline weeks are listed."""
    start, end = (pd.Timestamp(v) for v in samples_of(eras)["full"])
    deadlines = tax_event_weeks(start.year - 1, end.year + 1)
    by_week: dict[pd.Timestamp, set[str]] = {}
    for item in deadlines.itertuples():
        by_week.setdefault(item.event_week, set()).add(item.category)
    records = []
    for (category, week), cell in deadlines.groupby(["category", "event_week"], sort=True):
        if not start <= week <= end:
            continue
        window = [week + pd.Timedelta(weeks=k) for k in EVENT_WEEKS]
        base = week + pd.Timedelta(weeks=min(EVENT_WEEKS) - 1)
        complete = base in panel.index and all(day in panel.index for day in window)
        if complete:
            complete = bool(panel.loc[window, list(PATH_SERIES)].notna().all().all())
        codated = sorted(by_week.get(week, set()) - {category})
        others = [f"{k:+d}:{other}" for k, day in zip(EVENT_WEEKS, window) if k != 0 for other in sorted(by_week.get(day, ()))]
        records.append(
            {
                "schema_version": SCHEMA,
                "category": category,
                "event_week": week.date().isoformat(),
                "era": era_label(week, eras),
                "due_dates": ";".join(sorted({d.date().isoformat() for d in cell["due_date"]})),
                "statutory_dates": ";".join(sorted(d.date().isoformat() for d in cell["statutory_date"])),
                "n_deadlines_in_event": int(len(cell)),
                "window_base_week": base.date().isoformat(),
                "window_complete": complete,
                "codated_other_categories": ";".join(codated),
                "other_deadline_weeks_in_window": ";".join(others),
                "no_neighboring_deadline_weeks": not others,
                "no_codated_or_neighboring_deadline_weeks": not codated and not others,
            }
        )
    return pd.DataFrame(records)


def tax_event_counts(events: pd.DataFrame, eras: tuple = ERAS) -> pd.DataFrame:
    """Category events and distinct calendar weeks, including empty isolation cells."""
    records = []
    for category in ("all_categories", *[c.value for c in TAX_CATEGORIES]):
        for era in ("full", *[label for label, _, _ in eras]):
            for variant in TAX_VARIANTS:
                cell = events if category == "all_categories" else events.loc[events["category"].eq(category)]
                if era != "full":
                    cell = cell.loc[cell["era"].eq(era)]
                if variant != "all_events":
                    cell = cell.loc[cell[variant]]
                complete = cell.loc[cell["window_complete"]]
                records.append({
                    "category": category, "era": era, "variant": variant,
                    "n_category_events": len(cell),
                    "n_distinct_calendar_weeks": cell["event_week"].nunique(),
                    "n_category_events_complete": len(complete),
                    "n_distinct_complete_calendar_weeks": complete["event_week"].nunique(),
                })
    return pd.DataFrame(records)


def tax_event_paths(panel: pd.DataFrame, events: pd.DataFrame, eras: tuple = ERAS) -> pd.DataFrame:
    """Mean weekly and cumulative (from the window base) paths, unnormalized, by category and era.

    Every event keeps all of its window weeks. When windows overlap, a shared
    calendar week enters each event's path; the per-week counts of other
    deadline weeks make that reuse visible rather than silent.
    """
    usable = events.loc[events["window_complete"]].copy()
    series = list(PATH_SERIES)
    per_event = {}
    for item in usable.itertuples():
        week = pd.Timestamp(item.event_week)
        weekly = panel.loc[[week + pd.Timedelta(weeks=k) for k in EVENT_WEEKS], series].to_numpy(dtype=float)
        per_event[(item.category, item.event_week)] = (weekly, np.cumsum(weekly, axis=0))
    records = []
    era_names = ["full", *[label for label, _, _ in eras]]
    for category in sorted(usable["category"].unique()):
        for era in era_names:
            for variant in TAX_VARIANTS:
                cell = usable.loc[usable["category"].eq(category)]
                if era != "full":
                    cell = cell.loc[cell["era"].eq(era)]
                if variant != "all_events":
                    cell = cell.loc[cell[variant]]
                if cell.empty:
                    continue
                stacks = [per_event[(category, week)] for week in cell["event_week"]]
                weekly = np.stack([s[0] for s in stacks])
                cumul = np.stack([s[1] for s in stacks])
                others = cell["other_deadline_weeks_in_window"].fillna("")
                for k_pos, k in enumerate(EVENT_WEEKS):
                    if k == 0:
                        at_k = int(cell["codated_other_categories"].fillna("").ne("").sum())
                    else:
                        at_k = int(others.map(lambda text: any(p.split(":")[0] == f"{k:+d}" for p in text.split(";") if p)).sum())
                    for s_pos, name in enumerate(series):
                        records.append(
                            {
                                "schema_version": SCHEMA,
                                "category": category,
                                "era": era,
                                "variant": variant,
                                "event_week": k,
                                "series": name,
                                "series_kind": _series_kind(name),
                                "n_events": int(len(cell)),
                                "n_distinct_calendar_weeks": int(cell["event_week"].nunique()),
                                "n_events_with_other_deadline_at_week": at_k,
                                "n_events_with_any_other_deadline_week": int(others.ne("").sum()),
                                "mean_weekly": float(weekly[:, k_pos, s_pos].mean()),
                                "mean_cumulative": float(cumul[:, k_pos, s_pos].mean()),
                                "sd_cumulative": float(cumul[:, k_pos, s_pos].std(ddof=1)) if len(cell) > 1 else np.nan,
                                "units": UNITS,
                                "inference_status": "descriptive_accounting_only",
                                "uncertainty_status": "descriptive_standard_deviation",
                            }
                        )
    paths = pd.DataFrame(records)
    _assert_path_sums(paths)
    return paths


def _assert_path_sums(paths: pd.DataFrame) -> None:
    keys = ["category", "era", "variant", "event_week"]
    for column in ("mean_weekly", "mean_cumulative"):
        wide = paths.pivot_table(index=keys, columns="series", values=column, aggfunc="first")
        groups = wide[list(CONTRIBUTIONS)].sum(axis=1)
        factors = wide[[f"f_{r}" for r in _factor_rows()]].sum(axis=1) + wide["contrib_rho"]
        for parts in (groups, factors):
            if (parts - wide["delta_reserve_balances"]).abs().max() > TOLERANCE:
                raise AssertionError(f"event-path factors do not sum to the reserve path ({column})")
        if (wide["exposure_c"] - wide["contrib_cash_leg"]).abs().max() > TOLERANCE:
            raise AssertionError("event-path cash leg differs from the cash-leg factors")


# --------------------------------------------------------------------------
# Item 4: debt-limit extraordinary-measures and rebuild accounts
# --------------------------------------------------------------------------


def _as_timestamp(value: object, *, field_name: str) -> pd.Timestamp:
    if not isinstance(value, date):
        raise ValueError(f"debt-limit registry: {field_name} must be an ISO date")
    return pd.Timestamp(value)


DEBT_LIMIT_ROLES = {
    # role: (YAML date key, registry label)
    "extraordinary_measures": ("extraordinary_measures_start", "extraordinary_measures_start"),
    "interim_increase": ("interim_increase", "limit_increase_enacted_while_measures_continued"),
    "resolution": ("resolution", "limit_{resolution_type}_enacted"),
}


def load_debt_limit_episodes(path: str | Path) -> list[dict[str, object]]:
    """Read dated episodes; every date needs a primary-source title, issue date and URL."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != "debt_limit_episodes_v1":
        raise ValueError("debt-limit registry: unexpected schema_version")
    episodes = []
    for raw in payload.get("episodes") or []:
        episode = {"episode_id": str(raw["episode_id"]), "resolution_type": str(raw["resolution_type"])}
        for role, (date_key, _) in DEBT_LIMIT_ROLES.items():
            if role == "interim_increase" and date_key not in raw:
                continue
            source = raw.get(f"{role}_source") or {}
            for key in ("title", "issued", "url"):
                if not source.get(key):
                    raise ValueError(f"debt-limit registry: {episode['episode_id']} {role} source lacks {key}")
            episode[role] = _as_timestamp(raw[date_key], field_name=role)
            episode[f"{role}_source"] = {
                "title": str(source["title"]),
                "issued": _as_timestamp(source["issued"], field_name=f"{role} source issued").date().isoformat(),
                "url": str(source["url"]),
                "evidence": str(source.get("evidence", "")),
            }
        if episode["resolution_type"] not in {"suspension", "increase"}:
            raise ValueError(f"debt-limit registry: {episode['episode_id']} resolution_type")
        dated = [episode[r] for r in DEBT_LIMIT_ROLES if r in episode]
        if dated != sorted(dated) or len(set(dated)) != len(dated):
            raise ValueError(f"debt-limit registry: {episode['episode_id']} dates must increase from measures to resolution")
        episode["notes"] = str(raw.get("notes", ""))
        episodes.append(episode)
    if not episodes:
        raise ValueError("debt-limit registry: no episodes")
    for prev, nxt in zip(episodes, episodes[1:]):
        if not prev["resolution"] <= nxt["extraordinary_measures"]:
            raise ValueError(f"debt-limit registry: {nxt['episode_id']} overlaps {prev['episode_id']} or is out of order")
    return episodes


def last_wednesday_before(value: pd.Timestamp) -> pd.Timestamp:
    offset = (value.weekday() - 2) % 7 or 7
    return (value - pd.Timedelta(days=offset)).normalize()


def debt_limit_registry(episodes: list[dict[str, object]]) -> pd.DataFrame:
    records = []
    for ep in episodes:
        for role, (_, label) in DEBT_LIMIT_ROLES.items():
            if role not in ep:
                continue
            source = ep[f"{role}_source"]
            records.append(
                {
                    "schema_version": SCHEMA,
                    "episode_id": ep["episode_id"],
                    "date_role": label.format(resolution_type=ep["resolution_type"]),
                    "date": ep[role].date().isoformat(),
                    "source_title": source["title"],
                    "source_issued": source["issued"],
                    "source_url": source["url"],
                    "source_evidence": source["evidence"],
                    "notes": ep["notes"],
                }
            )
    return pd.DataFrame(records)


def debt_limit_accounts(panel: pd.DataFrame, episodes: list[dict[str, object]]) -> pd.DataFrame:
    """Cumulative cash leg and the ledger factors that accounted for it, by window.

    Measures window: weeks after the last Wednesday before measures began, through
    the last Wednesday before enactment; an interim limit increase that did not
    end the measures splits it at the last Wednesday before that enactment.
    Rebuild window: the following 13 weeks, ending earlier at the last Wednesday
    before the next episode's measures. Windows are contiguous and never share a week.
    """
    records = []
    for pos, ep in enumerate(episodes):
        em_base = last_wednesday_before(ep["extraordinary_measures"])
        em_end = last_wednesday_before(ep["resolution"])
        rebuild_end = em_end + pd.Timedelta(weeks=REBUILD_WEEKS)
        capped = False
        if pos + 1 < len(episodes):
            next_base = last_wednesday_before(episodes[pos + 1]["extraordinary_measures"])
            if next_base < rebuild_end:
                rebuild_end, capped = next_base, True
        phases = [("extraordinary_measures", em_base, em_end), ("rebuild", em_end, rebuild_end)]
        if "interim_increase" in ep:
            split = last_wednesday_before(ep["interim_increase"])
            phases[:1] = [
                ("extraordinary_measures", em_base, split),
                ("extraordinary_measures_after_interim_increase", split, em_end),
            ]
        for phase, base, end in phases:
            weeks = int((end - base).days // 7)
            cell = panel.loc[(panel.index > base) & (panel.index <= end)]
            if weeks < 0 or len(cell) != weeks or base not in panel.index:
                raise ValueError(f"debt-limit window {ep['episode_id']} {phase} is not fully observed")
            sums = cell[["exposure_c", "delta_reserve_balances", *CONTRIBUTIONS]].sum()
            row = {
                "schema_version": SCHEMA,
                "episode_id": ep["episode_id"],
                "phase": phase,
                "extraordinary_measures_start": ep["extraordinary_measures"].date().isoformat(),
                "interim_increase_date": ep["interim_increase"].date().isoformat() if "interim_increase" in ep else "",
                "resolution_date": ep["resolution"].date().isoformat(),
                "resolution_type": ep["resolution_type"],
                "resolution_source_title": ep["resolution_source"]["title"],
                "base_wednesday": base.date().isoformat(),
                "end_wednesday": end.date().isoformat(),
                "weeks": weeks,
                "rebuild_capped_by_next_episode": capped if phase == "rebuild" else False,
                "tga_level_base": float(panel.loc[base, "tga_level"]),
                "tga_level_end": float(panel.loc[end, "tga_level"]),
                "reserve_balances_base": float(panel.loc[base, "reserve_balances"]),
                "reserve_balances_end": float(panel.loc[end, "reserve_balances"]),
                "rrp_others_level_base": float(panel.loc[base, "rrp_others_level"]),
                "rrp_others_level_end": float(panel.loc[end, "rrp_others_level"]),
                "cash_leg": float(sums["exposure_c"]),
                "delta_reserve_balances": float(sums["delta_reserve_balances"]),
                **{name: float(sums[name]) for name in CONTRIBUTIONS},
                "delta_rrp_others": float(-sums["contrib_rrp_others"]),
                "delta_foreign_official_balances": float(-sums["contrib_foreign_official"]),
                "negative_other_factors_and_rho": float(-(sums[[f"contrib_{g}" for g in OTHER_GROUPS]].sum() + sums["contrib_rho"])),
                "units": UNITS,
            }
            account_sum = sum(row[name] for name in ACCOUNT_DEFINITIONS if name != "cash_leg")
            if abs(account_sum - row["cash_leg"]) > TOLERANCE or abs(sum(row[n] for n in CONTRIBUTIONS) - row["delta_reserve_balances"]) > TOLERANCE:
                raise AssertionError(f"debt-limit account {ep['episode_id']} {phase} does not close")
            records.append(row)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Assembly, thin tables, receipt
# --------------------------------------------------------------------------


def _family_columns(frame: pd.DataFrame, joint: dict[str, object], status: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["primary_family"] = False
    frame["joint_critical_value"] = np.nan
    frame["simband_lower"] = np.nan
    frame["simband_upper"] = np.nan
    frame["inference_status"] = outside_family_status(status)
    frame["uncertainty_status"] = "diagnostic"
    for member in joint["members"]:
        parts = member["name"].split("|")
        coef = parts[2] if len(parts) == 3 else "exposure_c"
        mask = (
            frame["specification"].eq("state" if len(parts) == 3 else "baseline")
            & frame["sample"].eq("full")
            & frame["outcome"].eq(parts[0])
            & frame["horizon"].eq(int(parts[1][1:]))
            & frame["coefficient"].eq(coef)
        )
        if mask.sum() != 1 or abs(frame.loc[mask, "beta"].iloc[0] - member["beta"]) > TOLERANCE * max(1.0, abs(member["beta"])):
            raise AssertionError(f"family member {member['name']} does not match its estimate row")
        frame.loc[mask, "primary_family"] = True
        frame.loc[mask, "joint_critical_value"] = joint["critical_value"]
        frame.loc[mask, "simband_lower"] = member["band_lower"]
        frame.loc[mask, "simband_upper"] = member["band_upper"]
        frame.loc[mask, "inference_status"] = status
        if status == "calibrated_joint_band":
            frame.loc[mask, "uncertainty_status"] = "calibrated_joint_band_only"
    return frame


def outside_family_status(status: str) -> str:
    # Calibration of the declared family never validates era/factor comparisons.
    return "descriptive_calibration_failed" if status == "descriptive_calibration_failed" else "descriptive_outside_declared_family"


def factor_decomposition(estimates: pd.DataFrame) -> pd.DataFrame:
    base = estimates.loc[
        estimates["specification"].eq("baseline") & estimates["horizon"].eq(0) & estimates["coefficient"].eq("exposure_c")
    ]
    records = []
    for sample, cell in base.groupby("sample", sort=False):
        total = float(cell.loc[cell["outcome"].eq("delta_reserve_balances"), "beta"].iloc[0])
        parts = cell.loc[cell["outcome"].isin(CONTRIBUTIONS)].set_index("outcome").loc[list(CONTRIBUTIONS)]
        for name, row in parts.iterrows():
            records.append(
                {
                    "schema_version": SCHEMA,
                    "sample": sample,
                    "component": name,
                    "factors": ";".join(FACTOR_GROUPS.get(name.removeprefix("contrib_"), ("rho",))),
                    "beta": row["beta"],
                    "se_hac12": row["se_hac12"],
                    "se_hac8": row["se_hac8"],
                    "se_hac26": row["se_hac26"],
                    "n": row["n"],
                    **sample_dates(pd.DatetimeIndex([row["first_week"], row["last_week"]]), 0),
                    "inference_status": row["inference_status"],
                    "uncertainty_status": "diagnostic",
                    "reserve_slope": total,
                    "sum_of_components": float(parts["beta"].sum()),
                    "closure_gap": float(parts["beta"].sum() - total),
                    "reserve_slope_minus_one": total - 1.0,
                }
            )
    return pd.DataFrame(records)


def thin_factor_table(decomposition: pd.DataFrame, estimates: pd.DataFrame, eras: tuple = ERAS) -> pd.DataFrame:
    rows = []
    memo = estimates.loc[
        estimates["specification"].eq("baseline")
        & estimates["horizon"].eq(0)
        & estimates["coefficient"].eq("exposure_c")
        & estimates["outcome"].isin(["delta_reserve_balances", "delta_rrp_others", "delta_reserves_plus_rrp_others"])
    ]
    order = [*CONTRIBUTIONS, "delta_reserve_balances", "delta_rrp_others", "delta_reserves_plus_rrp_others"]
    source = pd.concat(
        [decomposition.rename(columns={"component": "row"})[["sample", "row", "beta", "se_hac12", "n"]],
         memo.rename(columns={"outcome": "row"})[["sample", "row", "beta", "se_hac12", "n"]]]
    )
    for row in order:
        out = {
            "row": row, "definition": DEFINITIONS[row],
            "inference_status": decomposition["inference_status"].iloc[0],
            "uncertainty_status": "diagnostic",
            "sample_date_basis": "exposure_week_end_date",
        }
        for sample in samples_of(eras):
            cell = source.loc[source["sample"].eq(sample) & source["row"].eq(row)].iloc[0]
            out[f"{sample}_beta"] = round(float(cell["beta"]), 3)
            out[f"{sample}_se_hac12"] = round(float(cell["se_hac12"]), 3)
            out[f"{sample}_n"] = int(cell["n"])
            dates = estimates.loc[estimates["sample"].eq(sample) & estimates["horizon"].eq(0)].iloc[0]
            for key in ("first_week", "last_week", "first_outcome_endpoint_week", "last_outcome_endpoint_week"):
                out[f"{sample}_{key}"] = dates[key]
        rows.append(out)
    return pd.DataFrame(rows)


def thin_debt_limit_table(accounts: pd.DataFrame) -> pd.DataFrame:
    out = accounts[["episode_id", "phase", "extraordinary_measures_start", "resolution_date", "base_wednesday", "end_wednesday", "weeks"]].copy()
    out["tga_change_bn"] = ((accounts["tga_level_end"] - accounts["tga_level_base"]) / 1000).round(1)
    for name in ACCOUNT_DEFINITIONS:
        out[f"{name}_bn"] = (accounts[name] / 1000).round(1)
    out["inference_status"] = "descriptive_accounting_only"
    out["uncertainty_status"] = "not_applicable"
    return out


def thin_tax_table(paths: pd.DataFrame) -> pd.DataFrame:
    cell = paths.loc[paths["era"].eq("full") & paths["variant"].eq("all_events")]
    wide = cell.pivot_table(index=["category", "event_week"], columns="series", values="mean_cumulative", aggfunc="first")
    counts = cell.groupby(["category", "event_week"])[["n_events", "n_distinct_calendar_weeks", "n_events_with_other_deadline_at_week"]].first()
    other = [f"contrib_{g}" for g in OTHER_GROUPS]
    out = pd.DataFrame(index=wide.index)
    out["n_events"] = counts["n_events"]
    out["n_distinct_calendar_weeks"] = counts["n_distinct_calendar_weeks"]
    out["n_events_with_other_deadline_at_week"] = counts["n_events_with_other_deadline_at_week"]
    out["cash_leg_bn"] = wide["exposure_c"] / 1000
    out["rrp_others_reserve_contribution_bn"] = wide["contrib_rrp_others"] / 1000
    out["foreign_official_reserve_contribution_bn"] = wide["contrib_foreign_official"] / 1000
    out["other_factors_and_rho_bn"] = (wide[other].sum(axis=1) + wide["contrib_rho"]) / 1000
    out["delta_reserve_balances_bn"] = wide["delta_reserve_balances"] / 1000
    out["inference_status"] = "descriptive_accounting_only"
    out["uncertainty_status"] = "not_applicable"
    return out.reset_index().round(2)


def thin_table_definitions(eras: tuple = ERAS) -> dict[str, dict[str, str]]:
    """One companion definition per exported thin-table column, also in the receipt."""
    status = {
        "inference_status": "Descriptive qualification; neither era nor factor comparisons are validated by calibration of the declared family",
        "uncertainty_status": "Diagnostic SEs and intervals only for coefficients; not applicable to accounting paths",
    }
    factor = {
        "row": "Ledger contribution, reserve outcome, or memo outcome named in definitions",
        "definition": "Formula and reserve sign for this row",
        **status,
        "sample_date_basis": "Era/sample membership is determined by the exposure-week end date",
    }
    for sample in samples_of(eras):
        factor.update({
            f"{sample}_beta": "Same-week conditional slope on C_w, USD per USD; rounded to 3 decimals",
            f"{sample}_se_hac12": "Diagnostic calendar-distance HAC(12) standard error; rounded to 3 decimals",
            f"{sample}_n": "Common exposure-week count with lags and all horizon endpoints observed",
            f"{sample}_first_week": "First exposure-week Wednesday",
            f"{sample}_last_week": "Last exposure-week Wednesday",
            f"{sample}_first_outcome_endpoint_week": "First outcome-endpoint Wednesday (h=0)",
            f"{sample}_last_outcome_endpoint_week": "Last outcome-endpoint Wednesday (h=0)",
        })
    debt = {
        "episode_id": "Sourced debt-limit episode identifier",
        "phase": "Extraordinary measures, measures after interim increase, or rebuild window",
        "extraordinary_measures_start": "Sourced calendar date measures began",
        "resolution_date": "Sourced calendar date the resolution was enacted",
        "base_wednesday": "Last Wednesday before the phase starts; balances at this date are subtracted",
        "end_wednesday": "Last included Wednesday; endpoint balances at this date",
        "weeks": "(end_wednesday - base_wednesday) / 7 days",
        "tga_change_bn": "(tga_level_end - tga_level_base) / 1000; positive is a TGA balance increase, USD billion",
        **{f"{name}_bn": definition + "; divided by 1000, USD billion, rounded to 1 decimal" for name, definition in ACCOUNT_DEFINITIONS.items()},
        **status,
    }
    tax = {
        "category": "National calendar-year tax payment category; all complete category events are included",
        "event_week": "Week relative to the deadline's assigned Wednesday; " + TAX_PATH_BASE,
        "n_events": "Number of complete category events entering each path mean (not the aggregate calendar-week count)",
        "n_distinct_calendar_weeks": "Distinct event-week Wednesdays entering this category's path mean",
        "n_events_with_other_deadline_at_week": "Events with another deadline category at week zero or any deadline category at the stated nonzero relative week",
        "cash_leg_bn": "mean sum[-d(TGA) - d(SFP)] from event week -2 through event_week / 1000, USD billion",
        "rrp_others_reserve_contribution_bn": "mean sum[-d(WLRRAOL)] from event week -2 through event_week / 1000; positive adds to reserves, USD billion",
        "foreign_official_reserve_contribution_bn": "mean sum[-d(RRP foreign official) - d(foreign official deposits)] from event week -2 through event_week / 1000; positive adds to reserves, USD billion",
        "other_factors_and_rho_bn": "mean sum(contrib_currency + contrib_treasury_securities + contrib_mbs_agency + contrib_lending_facilities + contrib_all_other + rho) from event week -2 through event_week / 1000; signed reserve contribution, USD billion",
        "delta_reserve_balances_bn": "mean sum[d(reserve balances)] from event week -2 through event_week / 1000, USD billion",
        **status,
    }
    return {
        "thin_factor_decomposition_by_era_v1": factor,
        "thin_debt_limit_episodes_v1": debt,
        "thin_tax_event_paths_v1": tax,
    }


def _git_state() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"producer_commit": "unknown", "producer_tracked_changes": True}
    return {"producer_commit": commit, "producer_tracked_changes": bool(dirty)}


def _require_unpublished(out_dir: Path) -> None:
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"output bundle already exists; move it aside before publishing: {out_dir}")


def _rename_unpublished_directory(staged: Path, out_dir: Path) -> None:
    """One atomic, no-replace rename, including when another publisher races us."""
    if os.name == "nt":
        # Windows rename already refuses every existing destination.
        staged.rename(out_dir)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    source, target = os.fsencode(staged), os.fsencode(out_dir)
    if platform.system() == "Darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (source, target, 0x00000004)  # RENAME_EXCL
    elif platform.system() == "Linux":
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = (-100, source, -100, target, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise OSError("atomic no-replace directory publication is unsupported on this platform")
    rename.restype = ctypes.c_int
    if rename(*args) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(out_dir))


def _write_atomic(out_dir: Path, tables: dict[str, pd.DataFrame], receipt: dict[str, object], *,
                  receipt_name: str | None = None, text_files: dict[str, str] | None = None) -> dict[str, str]:
    """Publish a complete sibling staging directory with one directory rename."""
    _require_unpublished(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    hashes = {}
    try:
        for name, frame in tables.items():
            temp = staged / f"{name}.csv"
            frame.to_csv(temp, index=False, encoding="utf-8", lineterminator="\n")
            hashes[f"{name}.csv"] = _sha256(temp)
        for name, content in (text_files or {}).items():
            if Path(name).name != name:
                raise ValueError("bundle text files must have simple names")
            temp = staged / name
            temp.write_text(content, encoding="utf-8")
            hashes[name] = _sha256(temp)
        receipt["outputs"] = hashes
        temp = staged / (receipt_name or f"{SCHEMA}_receipt.json")
        temp.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        _require_unpublished(out_dir)
        _rename_unpublished_directory(staged, out_dir)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return hashes


def sample_coverage(panel: pd.DataFrame, eras: tuple = ERAS) -> list[dict[str, object]]:
    """Declared versus used exposure weeks; the only admitted shortfall is the unobserved h=8 endpoint."""
    last_ledger = panel.index.max()
    records = []
    for sample, (start, end) in samples_of(eras).items():
        rows, counts = sample_rows(panel, sample, eras)
        declared_final = final_wednesday(end)
        if last_ledger < declared_final:
            raise ValueError(f"{sample}: the ledger ends {last_ledger.date()}, before the declared final week {declared_final.date()}")
        expected = min(declared_final, last_ledger - pd.Timedelta(weeks=max(HORIZONS)))
        if rows.max() != expected:
            raise ValueError(f"{sample}: sample ends {rows.max().date()}, not {expected.date()}; an input ends early or has gaps")
        records.append({
            "sample": sample, "declared_start": start, "declared_end": end,
            "declared_final_week": declared_final.date().isoformat(),
            "last_ledger_week": last_ledger.date().isoformat(),
            "first_week": rows.min().date().isoformat(), "last_week": rows.max().date().isoformat(),
            "weeks_after_last_week_without_h8_endpoint": int((declared_final - rows.max()).days // 7),
            **counts,
        })
    return records


def run_weekly_reserve_offsets(
    *,
    ledger_csv: str | Path,
    ledger_receipt_json: str | Path,
    rrp_others_level_csv: str | Path,
    tga_level_csv: str | Path,
    debt_limit_yml: str | Path,
    out_dir: str | Path,
    ledger_sha256: str = LEDGER_SHA256,
    ledger_receipt_sha256: str = LEDGER_RECEIPT_SHA256,
    bootstrap_reps: int = BOOTSTRAP_REPS,
    calibration_sims: int = CALIBRATION_SIMS,
    calibration_reps: int = CALIBRATION_REPS,
    placebo_draws: int = PLACEBO_DRAWS,
    seed: int = SEED,
    design_config: dict[str, object] | None = None,
    source_receipt_json: str | Path | None = None,
) -> dict[str, object]:
    """Build every item-3 and item-4 table; nothing is written unless all succeed.

    A versioned design config supplies the vintage, the declared eras and the
    pinned ledger hashes, and requires the upstream refresh receipt.
    """
    _require_unpublished(Path(out_dir))
    producer = _git_state()
    eras, vintage, source_receipt = ERAS, None, None
    if design_config is not None:
        eras, vintage = design_config["eras"], design_config["vintage"]
        ledger_sha256, ledger_receipt_sha256 = design_config["ledger_sha256"], design_config["ledger_receipt_sha256"]
        if source_receipt_json is None:
            raise ValueError("a versioned design config requires the upstream refresh receipt")
        source_receipt = {
            "path": str(source_receipt_json),
            "sha256": _check_hash(Path(source_receipt_json), design_config["source_receipt_sha256"], source="source_receipt"),
        }
    ledger, ledger_receipt = load_ledger(
        ledger_csv, ledger_receipt_json, ledger_sha256=ledger_sha256, receipt_sha256=ledger_receipt_sha256
    )
    rrp = load_level(rrp_others_level_csv, ledger_receipt, "rrp_others", ledger)
    tga = load_level(tga_level_csv, ledger_receipt, "treasury_general_account", ledger)
    episodes = load_debt_limit_episodes(debt_limit_yml)
    panel = build_weekly_panel(ledger, rrp, tga, eras)
    coverage = sample_coverage(panel, eras)

    estimates = estimate_all(panel, eras)
    family = primary_family(panel, eras=eras)
    joint = joint_max_t(family, dwb_weights(family[0].design.weeks, bootstrap_reps, np.random.default_rng(seed)))
    calibration, passed = calibrate(
        panel,
        sims=calibration_sims,
        reps=calibration_reps,
        placebo_draws=placebo_draws,
        seed=seed,
        real_interaction_t=joint["members"][-1]["t"],
        eras=eras,
    )
    family_status = "calibrated_joint_band" if passed["family"] else "descriptive_calibration_failed"
    calibration = pd.concat(
        [
            calibration,
            pd.DataFrame(
                [
                    {
                        "schema_version": SCHEMA,
                        "check": "joint_max_t_test|observed_data",
                        "check_kind": "declared_family_result",
                        "draws": 1,
                        "bootstrap_reps_per_draw": joint["reps"],
                        "nominal_size": NOMINAL_SIZE,
                        "joint_critical_value": joint["critical_value"],
                        "max_abs_t": joint["max_abs_t"],
                        "joint_p": joint["joint_p"],
                        "inference_status": family_status,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    estimates = _family_columns(estimates, joint, family_status)
    interaction_rows = estimates["specification"].eq("state") & estimates["coefficient"].eq("exposure_x_state") & estimates["primary_family"]
    if passed["family"] and not passed["interaction_pointwise"]:
        estimates.loc[interaction_rows, "inference_status"] = "descriptive_interaction_size_failed"
    elif passed["family"] and not passed["interaction_real_placebo"]:
        estimates.loc[interaction_rows, "inference_status"] = "descriptive_random_walk_placebo_failed"
    estimates.loc[estimates["inference_status"].str.startswith("descriptive"), "uncertainty_status"] = "diagnostic"

    events = tax_event_windows(panel, eras)
    event_counts = tax_event_counts(events, eras)
    paths = tax_event_paths(panel, events, eras)
    accounts = debt_limit_accounts(panel, episodes)
    decomposition = factor_decomposition(estimates)
    baseline = estimates["specification"].eq("baseline") & estimates["coefficient"].eq("exposure_c")
    tables = {
        "weekly_estimates_v1": estimates.loc[estimates["horizon"].eq(0) & estimates["specification"].isin(["baseline", "intercept_only"])],
        "factor_decomposition_v1": decomposition,
        "cumulative_horizons_v1": estimates.loc[baseline],
        "state_interaction_v1": estimates.loc[estimates["specification"].eq("state")],
        "state_overlap_v1": overlap_table(panel, eras),
        "influence_v1": influence_table(panel, outside_family_status(family_status), eras),
        "calibration_v1": calibration,
        "tax_event_windows_v1": events,
        "tax_event_counts_v1": event_counts,
        "tax_event_paths_v1": paths,
        "debt_limit_episode_accounts_v1": accounts,
        "debt_limit_date_registry_v1": debt_limit_registry(episodes),
        "thin_factor_decomposition_by_era_v1": thin_factor_table(decomposition, estimates, eras),
        "thin_debt_limit_episodes_v1": thin_debt_limit_table(accounts),
        "thin_tax_event_paths_v1": thin_tax_table(paths),
    }
    thin_definitions = thin_table_definitions(eras)
    tables["thin_table_definitions_v1"] = pd.DataFrame([
        {"table": table, "column": column, "definition": definition,
         "inference_status": "descriptive_definitions_only",
         "boundary": RRP_BOUNDARY if table != "thin_tax_event_paths_v1" else RRP_BOUNDARY + " " + TAX_CALENDAR_COVERAGE + ". " + TAX_PATH_BASE}
        for table, columns in thin_definitions.items() for column, definition in columns.items()
    ])
    receipt = {
        "schema_version": SCHEMA,
        "vintage": vintage,
        **producer,
        "command": "tdchf run-weekly-reserve-offsets",
        "design_config": None if design_config is None else {"path": design_config["path"], "sha256": design_config["sha256"]},
        "units": UNITS,
        "inputs": {
            "source_receipt": source_receipt,
            "ledger": {"path": str(ledger_csv), "sha256": ledger_sha256, "producer_commit": ledger_receipt.get("producer_commit")},
            "ledger_receipt": {"path": str(ledger_receipt_json), "sha256": ledger_receipt_sha256},
            "rrp_others_level": {"path": str(rrp_others_level_csv), "sha256": ledger_receipt["inputs"]["fiscus_inputs"]["rrp_others"]["sha256"]},
            "tga_level": {"path": str(tga_level_csv), "sha256": ledger_receipt["inputs"]["fiscus_inputs"]["treasury_general_account"]["sha256"]},
            "debt_limit_episodes": {"path": str(debt_limit_yml), "sha256": _sha256(Path(debt_limit_yml))},
        },
        "design": {
            "exposure": "C_w = c_h = -d(TGA) - d(SFP) between consecutive H.4.1 Wednesdays",
            "samples": samples_of(eras),
            "sample_coverage": coverage,
            "sample_date_basis": "Exposure-date eras; first_week and last_week are exposure-week Wednesdays, not outcome endpoints",
            "sample_horizon_ranges": estimates.loc[baseline & estimates["outcome"].eq("delta_reserve_balances"), ["sample", "horizon", *sample_dates(panel.index, 0)]].to_dict(orient="records"),
            "eras": [list(e) for e in eras],
            "controls": ["const", *BASELINE_CONTROLS],
            "omitted_unqualified_controls": ["coupon_week", "redemption_week", "auction_settlement_week"],
            "state": "Lagged RRP-others (WLRRAOL) Wednesday level at w-1 (strictly before the Thursday that opens exposure week w), USD 100bn",
            "state_specification_extra_terms": "state, C x state, era intercepts and C x era for transition and later eras",
            "overlap_split_100bn": OVERLAP_SPLIT_100BN,
            "horizons": list(HORIZONS),
            "cumulative_outcome": "Y_{w,h} = level_{w+h} - level_{w-1}; rows need every horizon endpoint observed",
            "hac": {"kernel": "Bartlett on calendar-week distance", "bandwidths": list(HAC_BANDWIDTHS), "baseline": HAC_BANDWIDTHS[0], "small_sample_scaling": "none", "critical_values": "normal"},
            "bootstrap": {"method": "dependent wild bootstrap, Gaussian moving-sum multipliers on the calendar-week grid", "kernel_bandwidth": DWB_BANDWIDTH, "reps": bootstrap_reps, "seed": seed},
            "primary_family": [m.name for m in family],
            "calibration": {"sims": calibration_sims, "reps_per_sim": calibration_reps, "placebo_draws": placebo_draws, "null_ar": NULL_AR, "null_scale_weeks": NULL_SCALE_WEEKS, "pass_rule": "rejection rate <= 0.05 + 2 * sqrt(0.05 * 0.95 / draws)"},
            "factor_groups": {k: list(v) for k, v in FACTOR_GROUPS.items()},
            "definitions": DEFINITIONS,
            "debt_account_definitions": ACCOUNT_DEFINITIONS,
            "thin_table_definitions": thin_definitions,
            "uncertainty_labels": "SEs, t statistics, pointwise intervals and uncalibrated joint bands are diagnostic; calibration never validates outside-family comparisons",
            "tax_categories": [c.value for c in TAX_CATEGORIES],
            "tax_calendar_coverage": TAX_CALENDAR_COVERAGE,
            "tax_path_base": TAX_PATH_BASE,
            "tax_event_variants": {
                "all_events": "All complete category events; overlapping and co-dated categories remain",
                "no_neighboring_deadline_weeks": "No deadline in another week of the -2 through +4 window; co-dated categories remain",
                "no_codated_or_neighboring_deadline_weeks": "No co-dated other category and no deadline in another week of the -2 through +4 window",
            },
            "event_weeks": list(EVENT_WEEKS),
            "rebuild_weeks": REBUILD_WEEKS,
        },
        "results": {
            "joint": {k: v for k, v in joint.items()},
            "family_inference_status": family_status,
            "interaction_inference_status": str(estimates.loc[interaction_rows, "inference_status"].iloc[0]),
            "calibration_passed": passed,
            "n_tax_events": int(len(events)),
            "n_tax_events_complete": int(events["window_complete"].sum()),
            "n_tax_distinct_calendar_weeks": int(events["event_week"].nunique()),
            "tax_event_counts": event_counts.to_dict(orient="records"),
            "n_debt_limit_episodes": len(episodes),
        },
        "boundaries": [
            "Conditional associations and descriptive accounting paths; no causal, landing, retention, or half-life interpretation",
            "Factor slopes are an exact additive decomposition of the reserve slope under one design and one row set",
            "The memo outcome adds the RRP-others balance to reserve balances; it does not redefine reserves",
            "The state interaction is a conditional association with a persistent, policy-dependent state",
            "Cumulative coefficients include covariance with later Treasury flows and other factors; they have no unit benchmark",
            RRP_BOUNDARY,
            TAX_CALENDAR_COVERAGE,
            TAX_PATH_BASE,
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    hashes = _write_atomic(Path(out_dir), tables, receipt)
    return {"status": "ok", "out_dir": str(out_dir), "outputs": hashes, "family_inference_status": family_status, "joint_p": joint["joint_p"]}
