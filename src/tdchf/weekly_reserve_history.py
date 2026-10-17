"""Weekly reserve period coefficients on core TRC with recomputed restricted-score inference.

Each declared period is a same-week projection of the H.4.1 reserve change on
cash-valued core TRC with calendar controls: a weekly core-TRC association, not
the quarterly preferred-TRC coefficient. Inference is a scalar null-imposed
dependent wild bootstrap whose restricted control fit and HC3/Bartlett score
scale are recomputed in every draw. Because the draw's slope deviation and
restricted residual are linear in the tested value b, every statistic is
T*(b) = (c_e - t c_x) / sqrt(Q*(t)); tests of any b, confidence-set
inversion and equivalence tests are exact functions of a few per-draw numbers.
Everything is centred at the estimate (t = b - beta) and built from the full
regression residual, so near-exact fits keep relative accuracy.
Nothing here identifies landing, retention, a fund route or a causal response.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from . import reserve_offsets as ro
from . import weekly_trc_projection as w
from .manifest import _sha256

SCHEMA = 'weekly_reserve_history_v1'
CONFIG = 'config/weekly_reserve_history_2026q2.json'
MIN_RESIDUAL_DF = 20


# ---------------------------------------------------------------------------
# Periods and design
# ---------------------------------------------------------------------------


def periods(config: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Declared main partition plus nested sensitivities; the main periods tile the pooled span."""
    main = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in config['eras'].items()}
    nested = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in config['nested'].items()}
    if next(iter(main)) != 'pooled' or set(main) & set(nested):
        raise ValueError('pooled must come first and nested names must be distinct')
    parts = list(main.items())[1:]
    if parts[0][1][0] != main['pooled'][0] or parts[-1][1][1] != main['pooled'][1]:
        raise ValueError('main periods must span the pooled sample')
    for (_, (_, end)), (_, (start, _)) in pairwise(parts):
        if start != end + pd.Timedelta(days=7):
            raise ValueError('main periods must be consecutive Wednesday blocks')
    for name, (start, end) in {**main, **nested}.items():
        if start.weekday() != 2 or end.weekday() != 2 or end <= start:
            raise ValueError(f'{name}: bounds must be ordered Wednesdays')
    for name, (start, end) in nested.items():
        if not any(s <= start and end <= e for k, (s, e) in parts):
            raise ValueError(f'{name}: nested sensitivity must lie inside one main period')
    declared = set(config['inferential_periods']) | set(config['descriptive_periods'])
    if declared != set(main) | set(nested) or set(config['inferential_periods']) & set(config['descriptive_periods']):
        raise ValueError('every period must be either inferential or descriptive')
    return {**main, **nested}


@dataclass(frozen=True)
class Design:
    """One period's fixed design: controls basis, partialled exposure and leverage."""
    dates: pd.DatetimeIndex
    weeks: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    q: np.ndarray
    rank: int
    xp: np.ndarray
    a: np.ndarray
    inv: np.ndarray
    full: np.ndarray          # orthonormal basis of [controls, exposure]

    @property
    def n(self) -> int:
        return len(self.x)


def make_design(dates: pd.DatetimeIndex, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                min_df: int = MIN_RESIDUAL_DF, ordinals: np.ndarray | None = None) -> Design:
    """Inferential designs need MIN_RESIDUAL_DF residual degrees of freedom; descriptive ones only one.

    `ordinals` are calendar positions for HAC distances (default: weeks of Wednesday dates; months for monthly data).
    """
    q, rank = w.control_basis(z)
    xp = w.residual(x, q)
    den = float(xp @ xp)
    if den <= 1e-12 * max(1.0, float(x @ x)):
        raise ValueError('no partial exposure variation after controls')
    leverage = (q * q).sum(axis=1)
    if len(x) - rank - 1 < min_df:
        raise ValueError(f'saturated design: {len(x)} weeks for {rank + 1} coefficients')
    positions = ro._week_ordinals(dates) if ordinals is None else np.asarray(ordinals, int)
    return Design(dates, positions, x, y, z, q, rank, xp, xp / den,
                  1.0 / np.maximum(1.0 - leverage, 1e-8), w.control_basis(np.c_[z, x])[0])


def period_design(v: pd.DataFrame, rows: pd.DatetimeIndex, min_df: int = MIN_RESIDUAL_DF) -> Design:
    return make_design(rows, v.loc[rows, 'x'].to_numpy(float), v.loc[rows, 'delta_reserve_balances'].to_numpy(float),
                       w.controls(v.loc[rows], 'calendar').to_numpy(float), min_df)


@dataclass(frozen=True)
class Layout:
    """Observed weeks placed on the full weekly calendar; Bartlett width L (weights 1 - lag/(L+1))."""
    offsets: np.ndarray
    span: int
    width: int


def layout(weeks: np.ndarray, width: int) -> Layout:
    offsets = np.asarray(weeks - weeks[0], int)
    return Layout(offsets, int(offsets[-1]) + 1, width)


def bartlett_factor(scores: np.ndarray, lay: Layout) -> np.ndarray:
    """F'v with K = FF' the calendar Bartlett kernel: zero-filled moving sums of L+1 weeks over sqrt(L+1).

    sum_j (F'v)_j^2 = sum_{t,s} v_t v_s max(1 - |t-s|/(L+1), 0), so every long-run variance below is a
    sum of squares and cannot cancel to a negative number.
    """
    calendar = np.zeros((*scores.shape[:-1], lay.span + 2 * lay.width))
    calendar[..., lay.width + lay.offsets] = scores
    cumulative = np.concatenate([np.zeros((*scores.shape[:-1], 1)), np.cumsum(calendar, axis=-1)], axis=-1)
    return (cumulative[..., lay.width + 1:] - cumulative[..., :-(lay.width + 1)]) / math.sqrt(lay.width + 1)


def scale_factor(f_e: np.ndarray, f_x: np.ndarray) -> np.ndarray:
    """2x2 upper-triangular R with ||F'(s_e - t s_x)||^2 = (R00 - t R01)^2 + (t R11)^2 (QR of [F's_e, F's_x])."""
    return np.linalg.qr(np.stack([f_e, f_x], axis=-1), mode='r')


def quadratic(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (r[..., 0, 0] - t * r[..., 0, 1]) ** 2 + (t * r[..., 1, 1]) ** 2


def resolved(r: np.ndarray) -> np.ndarray:
    """Positive residual scale, and the exposure column not collinear with it to working precision.

    Invariant to rescaling either factored column: R00 only scales with the first, R01 and R11 with the second.
    """
    return (np.abs(r[..., 0, 0]) > 0) & (np.abs(r[..., 1, 1]) > 1e3 * np.finfo(float).eps * np.hypot(r[..., 0, 1], r[..., 1, 1]))


def full_residual(d: Design, y: np.ndarray) -> np.ndarray:
    """Residual of y on [controls, exposure], reorthogonalized once so near-exact fits keep relative accuracy."""
    e = y - d.full @ (d.full.T @ y)
    return e - d.full @ (d.full.T @ e)


@dataclass(frozen=True)
class Parts:
    """Everything is centred at the estimate: t = b - beta.

    T(b) = -t / sqrt(Q_R(t)) observed and T*(b) = (c_e - t c_x) / sqrt(Q_D(t)) per draw, built from the full
    regression residual e (the restricted residual at b is e - t x_perp), so no coefficient comes from a
    difference of nearly equal quantities.
    """
    beta: float
    observed: np.ndarray     # (2, 2)
    c_e: np.ndarray          # per draw
    c_x: np.ndarray
    draws: np.ndarray        # (draws, 2, 2)


def parts(d: Design, y: np.ndarray, multipliers: np.ndarray, lay: Layout) -> Parts:
    """y may differ from d.y (simulation); the design (x, controls, dates) is fixed."""
    e = full_residual(d, y)
    if np.linalg.norm(e) <= 1e3 * np.finfo(float).eps * np.linalg.norm(y):
        raise ValueError('restricted score scale is unresolved (exact fit); no p-value is reported')
    g_e, g_x = e * d.inv, d.xp * d.inv
    s_e, s_x = d.a * g_e, d.a * g_x
    observed = scale_factor(bartlett_factor(s_e, lay), bartlett_factor(s_x, lay))
    w_e, w_x = multipliers * g_e, multipliers * g_x
    u_e = w_e - (w_e @ d.q) @ d.q.T
    u_x = w_x - (w_x @ d.q) @ d.q.T
    draws = scale_factor(bartlett_factor(d.a * u_e * d.inv, lay), bartlett_factor(d.a * u_x * d.inv, lay))
    if not resolved(observed) or not resolved(draws).all():
        raise ValueError('restricted score scale is unresolved (near-exact fit); no p-value is reported')
    return Parts(float(d.a @ y), observed, multipliers @ s_e, multipliers @ s_x, draws)


def statistics(pt: Parts, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observed T and draw T* (last axis = draws) at centred values t = b - beta."""
    t = np.asarray(t, float)
    t_obs = -t / np.sqrt(quadratic(pt.observed, t))
    tt = t[..., None]
    t_draw = (pt.c_e - tt * pt.c_x) / np.sqrt(quadratic(pt.draws, tt))
    if not (np.isfinite(t_obs).all() and np.isfinite(t_draw).all()):
        raise ValueError('nonfinite test statistic')
    return t_obs, t_draw


def p_values(pt: Parts, bs: np.ndarray, chunk: int = 256, *, centred: bool = False) -> np.ndarray:
    """(3, len(bs)): two-sided, upper-tail and lower-tail bootstrap p-values at each tested b (t if centred)."""
    ts = np.atleast_1d(np.asarray(bs, float)) - (0.0 if centred else pt.beta)
    draws = len(pt.c_e)
    out = np.empty((3, len(ts)))
    for start in range(0, len(ts), chunk):
        t_obs, t_draw = statistics(pt, ts[start:start + chunk])
        out[0, start:start + chunk] = (1 + (np.abs(t_draw) >= np.abs(t_obs)[:, None]).sum(axis=1)) / (draws + 1)
        out[1, start:start + chunk] = (1 + (t_draw >= t_obs[:, None]).sum(axis=1)) / (draws + 1)
        out[2, start:start + chunk] = (1 + (t_draw <= t_obs[:, None]).sum(axis=1)) / (draws + 1)
    return out


BOUNDARY_TOLERANCE = 1e-10   # boundaries located to this relative precision in standard-error units
COMPONENT_RESOLUTION = 1e-8  # absolute SE width below which components (incl. isolated points) are not reported
NEAR_REAL = 1e-3             # complex root pairs this close to the real line may be perturbed real pairs


def crossings(pt: Parts) -> tuple[np.ndarray, float]:
    """Candidate crossings u = t/s (s = |R00|) of |T*(t)| = |T(t)| for every draw, from O(1) quartics.

    In u: (c_e/s - u c_x)^2 q_R(u) - u^2 q_D(u) = 0, q_R(u) = (R00/s - u R01)^2 + (u R11)^2,
    q_D(u) = (D00/s - u D01)^2 + (u D11)^2. Root finding is backward stable, so a true real root is near a
    computed root; a near-double real pair can come back as a complex pair a +/- bi, so near-real pairs
    contribute a - b, a and a + b. Every candidate is only a place to look: acceptance is decided by the direct
    test and boundaries by bisection on it.
    """
    s = abs(float(pt.observed[0, 0]))  # QR may return a negative diagonal; the scale must be positive
    r = pt.observed
    obs = np.array([r[0, 1] ** 2 + r[1, 1] ** 2, -2 * (r[0, 0] / s) * r[0, 1], (r[0, 0] / s) ** 2])
    d00 = pt.draws[:, 0, 0] / s
    drw = np.stack([pt.draws[:, 0, 1] ** 2 + pt.draws[:, 1, 1] ** 2, -2 * d00 * pt.draws[:, 0, 1], d00 ** 2], axis=-1)
    ce = pt.c_e / s
    num_draw = np.stack([pt.c_x ** 2, -2 * ce * pt.c_x, ce ** 2], axis=-1)
    num_obs = np.array([1.0, 0.0, 0.0])
    quartic = np.zeros((len(ce), 5))
    for i in range(3):
        for j in range(3):
            quartic[:, i + j] += num_draw[:, i] * obs[j] - num_obs[i] * drw[:, j]
    found = []
    for row in quartic:
        nonzero = np.flatnonzero(row != 0)
        if len(nonzero) and nonzero[0] < 4:
            values = np.roots(row[nonzero[0]:])
            size = np.maximum(1.0, np.abs(values.real))
            real = np.abs(values.imag) <= 1e-12 * size
            pair = ~real & (np.abs(values.imag) <= NEAR_REAL * size)
            found += [values.real[real], values.real[pair], values.real[pair] - np.abs(values.imag[pair]),
                      values.real[pair] + np.abs(values.imag[pair])]
    return (np.unique(np.concatenate(found)) if found else np.array([])), s


def _accepted(pt: Parts, u: np.ndarray, s: float, alpha: float) -> np.ndarray:
    return p_values(pt, s * np.atleast_1d(u), centred=True)[0] > alpha


def acceptance_set(pt: Parts, alpha: float, verify: bool = True) -> dict:
    """Inverted two-sided set to the declared resolution: every component wider than COMPONENT_RESOLUTION SE."""
    u, s = crossings(pt)
    span = max(1.0, float(np.abs(u).max())) if len(u) else 1.0
    middles = (u[:-1] + u[1:]) / 2 if len(u) > 1 else np.array([])
    points = np.unique(np.concatenate([u, middles, [0.0], [(u[0] if len(u) else 0.0) - span, (u[-1] if len(u) else 0.0) + span]]))
    ok = _accepted(pt, points, s, alpha)

    def boundary(inside: float, outside: float) -> float:
        while abs(outside - inside) > BOUNDARY_TOLERANCE * max(1.0, abs(inside)):
            middle = (inside + outside) / 2
            inside, outside = (middle, outside) if _accepted(pt, middle, s, alpha)[0] else (inside, middle)
        return inside

    segments, i = [], 0
    while i < len(points):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(points) and ok[j + 1]:
            j += 1
        low = -math.inf if i == 0 else boundary(points[i], points[i - 1])
        high = math.inf if j == len(points) - 1 else boundary(points[j], points[j + 1])
        # Absolute resolution in SE units, relaxed only where float spacing exceeds it (beyond ~1e7 SE);
        # the estimate's own component is always reported.
        floor = max(COMPONENT_RESOLUTION, 4 * np.finfo(float).eps * max(abs(low), abs(high)))
        if not (math.isfinite(low) and math.isfinite(high)) or low <= 0.0 <= high or high - low > floor:
            segments.append((low, high))
        i = j + 1
    if not any(a <= 0.0 <= b for a, b in segments):
        raise ValueError('inversion failed: the set excludes the estimate')
    if verify:
        verify_set(pt, segments, u, s, alpha)
    return {'segments': [(pt.beta + s * a, pt.beta + s * b) for a, b in segments], 'crossings': len(u),
            'unbounded_below': bool(segments and segments[0][0] == -math.inf),
            'unbounded_above': bool(segments and segments[-1][1] == math.inf)}


def verify_set(pt: Parts, segments: list, u: np.ndarray, s: float, alpha: float) -> None:
    """Independent dense check in standard-error units: direct acceptance must equal set membership."""
    check = np.concatenate([np.linspace(-60, 60, 2401), np.logspace(1.8, 9, 120), -np.logspace(1.8, 9, 120)])
    edges = np.array([e for seg in segments for e in seg if math.isfinite(e)] + list(u))
    if len(edges):
        near = np.abs(check[:, None] - edges[None, :]).min(axis=1) <= 1e-7 * np.maximum(1.0, np.abs(check))
        check = check[~near]
    direct = _accepted(pt, check, s, alpha)
    member = np.array([any(a <= c <= b for a, b in segments) for c in check])
    if (direct != member).any():
        bad = check[direct != member][0]
        raise ValueError(f'inversion failed verification at b = {pt.beta + s * bad!r}')


def describe_set(cs: dict) -> dict:
    bounded = not (cs['unbounded_below'] or cs['unbounded_above'])
    return {'set_bounded': bounded, 'set_components': len(cs['segments']),
            'set_length': float(sum(b - a for a, b in cs['segments'])) if bounded else math.inf}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def cells(config: dict, period: str) -> list[dict]:
    spec = config['calibration']['cells']
    out = [{'family': 'baseline', 'ar1': ar, 'innovation': inn, 'variance': var, 'mean': 'none'}
           for ar in spec['baseline']['ar1'] for inn in spec['baseline']['innovations'] for var in spec['baseline']['variance']]
    out += [{'family': 'misspecification', 'ar1': 'empirical', 'innovation': inn, 'variance': var, 'mean': mean}
            for mean in spec['misspecification']['means'] for inn in spec['misspecification']['innovations']
            for var in spec['misspecification']['variance']]
    if config['calibration']['crisis_episodes'][period]:
        out += [{'family': 'crisis', 'ar1': ar, 'innovation': inn, 'variance': 'crisis', 'mean': 'none'}
                for ar in spec['crisis']['ar1'] for inn in spec['crisis']['innovations']]
    out += [{'family': 'differenced', 'ma1': spec['differenced']['ma1'], 'innovation': inn, 'variance': var, 'mean': 'none'}
            for inn in spec['differenced']['innovations'] for var in spec['differenced']['variance']]
    for index, cell in enumerate(out):
        cell['cell'] = index
    return out


def unrestricted_residuals(d: Design) -> np.ndarray:
    return w.residual(d.y, w.control_basis(np.c_[d.z, d.x])[0])


def calendar_ar1(residuals: np.ndarray, weeks: np.ndarray) -> float:
    position = {int(week): i for i, week in enumerate(weeks)}
    pairs = [(i, position[int(week) - 1]) for i, week in enumerate(weeks) if int(week) - 1 in position]
    now, before = residuals[[i for i, _ in pairs]], residuals[[j for _, j in pairs]]
    rho = float(now @ before / (before @ before))
    if not (math.isfinite(rho) and abs(rho) < 1):
        raise ValueError(f'empirical AR(1) {rho} is not a stationary coefficient')
    return rho


def cell_mean(d: Design, cell: dict, config: dict) -> np.ndarray:
    position = (d.weeks - d.weeks[0]) / (d.weeks[-1] - d.weeks[0])
    if cell['mean'] == 'none':
        return np.zeros(d.n)
    if cell['mean'] == 'slope_drift':
        return (position - 0.5) * d.x
    if cell['mean'] == 'midpoint_step':
        return 0.5 * (position >= 0.5) * d.x
    if cell['mean'] == 'nuisance_drift':
        month_end = d.z[:, 1 + w.CALENDAR.index('month_end_week')]
        standardized = (month_end - month_end.mean()) / month_end.std()
        return 0.25 * d.y.std() * (2 * position - 1) * standardized
    raise ValueError(f"unknown mean {cell['mean']}")


def cell_scale(d: Design, cell: dict, sd: float, episodes: list) -> np.ndarray:
    if cell['variance'] == 'constant':
        return np.full(d.n, sd)
    if cell['variance'] == 'concentrated':
        return sd * np.sqrt(0.2 + 4.8 * d.xp ** 2 / (d.xp ** 2).mean())
    if cell['variance'] == 'crisis':
        inside = np.zeros(d.n, bool)
        for start, end in episodes:
            inside |= (d.dates >= pd.Timestamp(start)) & (d.dates <= pd.Timestamp(end))
        return sd * np.where(inside, 5.0, 0.6)
    raise ValueError(f"unknown variance {cell['variance']}")


def innovations(rng, shape, kind: str) -> np.ndarray:
    if kind == 'gaussian':
        return rng.standard_normal(shape)
    if kind == 't4':
        return rng.standard_t(4, size=shape) / math.sqrt(2.0)
    raise ValueError(f'unknown innovation {kind}')


def cell_errors(rng, d: Design, cell: dict, rho_empirical: float, simulations: int, burn_in: int = 300) -> np.ndarray:
    """Unit-variance errors on the full weekly calendar, returned at the period's observed weeks: (n, simulations)."""
    offsets = d.weeks - d.weeks[0]
    span = int(offsets[-1]) + 1
    if 'ma1' in cell:
        u = innovations(rng, (span + 1, simulations), cell['innovation'])
        e = (u[1:] + cell['ma1'] * u[:-1]) / math.sqrt(1 + cell['ma1'] ** 2)
        return e[offsets]
    rho = rho_empirical if cell['ar1'] == 'empirical' else float(cell['ar1'])
    u = innovations(rng, (span + burn_in, simulations), cell['innovation'])
    e = np.empty_like(u)
    e[0] = u[0]
    scale = math.sqrt(1 - rho * rho)
    for t in range(1, len(u)):
        e[t] = rho * e[t - 1] + scale * u[t]
    return e[burn_in:][offsets]


def wilson(successes: int, n: int, z: float) -> tuple[float, float]:
    return w.wilson(successes, n, z)


def run_cell(task: dict) -> dict:
    """One calibration cell: coverage and one-sided rejection at the exact target, plus a precision subset."""
    d, cell, cfg, width = task['design'], task['cell'], task['calibration'], task['width']
    alpha = task['alpha']
    root = np.random.SeedSequence([cfg['seed'], task['period_index'], width, cell['cell']])
    error_ss, multiplier_ss = root.spawn(2)
    error_rng, multiplier_rng = np.random.default_rng(error_ss), np.random.default_rng(multiplier_ss)
    residuals = unrestricted_residuals(d)
    sd = float(residuals.std())
    mu = cell_mean(d, cell, cfg)
    target = float(d.a @ mu)
    scale = cell_scale(d, cell, sd, task['episodes'])
    simulations = cfg['outer']
    errors = cell_errors(error_rng, d, cell, calendar_ar1(residuals, d.weeks), simulations)
    rho = calendar_ar1(residuals, d.weeks)
    if not sd > 0:
        raise ValueError('residual scale must be positive')
    errors = cell_errors(error_rng, d, cell, rho, simulations)
    lay = layout(d.weeks, width)
    precision = cfg['precision_subset']
    offsets = np.asarray(precision['null_offsets'], float)
    covered = upper = lower = 0
    lengths, unbounded, disconnected, rejections = [], 0, 0, np.zeros(len(offsets))
    measure_precision = width == precision['width']
    for s in range(simulations):
        y = mu + scale * errors[:, s]
        bank = w.multiplier_bank(d.dates, width, cfg['inner'], multiplier_rng)
        pt = parts(d, y, bank, lay)
        p_two, p_up, p_lo = p_values(pt, np.array([target]))[:, 0]
        covered += p_two > alpha
        upper += p_up <= alpha
        lower += p_lo <= alpha
        if measure_precision and s < precision['simulations']:
            shape = describe_set(acceptance_set(pt, alpha))
            if shape['set_bounded']:
                lengths.append(shape['set_length'])
            else:
                unbounded += 1
            disconnected += shape['set_components'] > 1
            rejections += p_values(pt, target + offsets)[0] <= alpha
    z = cfg['standard']['wilson_z']
    coverage_ci = wilson(covered, simulations, z)
    upper_ci, lower_ci = wilson(upper, simulations, z), wilson(lower, simulations, z)
    standard = cfg['standard']
    subset = precision['simulations'] if measure_precision else 0
    row = {'period': task['period'], 'width': width, **{k: cell.get(k, '') for k in ('cell', 'family', 'mean', 'variance', 'innovation')},
           'ar1': cell.get('ar1', ''), 'ma1': cell.get('ma1', ''), 'rho_used': rho if cell.get('ar1') == 'empirical' else cell.get('ar1', ''),
           'residual_sd_used': sd, 'n': d.n, 'simulations': simulations, 'inner_draws': cfg['inner'],
           'target': target, 'coverage': covered / simulations, 'coverage_wilson_lower': coverage_ci[0],
           'upper_rejection': upper / simulations, 'upper_wilson_upper': upper_ci[1],
           'lower_rejection': lower / simulations, 'lower_wilson_upper': lower_ci[1],
           'two_sided_pass': bool(coverage_ci[0] >= standard['coverage_floor']),
           'one_sided_pass': bool(upper_ci[1] <= standard['one_sided_ceiling'] and lower_ci[1] <= standard['one_sided_ceiling']),
           'precision_simulations': subset}
    if subset:
        row |= {'median_length_bounded': float(np.median(lengths)) if lengths else math.nan,
                'share_unbounded': unbounded / subset, 'share_disconnected': disconnected / subset,
                **{f'rejection_null_offset_{o:+.2f}': float(r / subset) for o, r in zip(offsets, rejections)}}
    return row


# ---------------------------------------------------------------------------
# Observed results
# ---------------------------------------------------------------------------


def run_observed(task: dict) -> dict:
    d, cfg, width = task['design'], task['inference'], task['width']
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], task['period_index'], width]))
    bank = w.multiplier_bank(d.dates, width, cfg['draws'], rng)
    pt = parts(d, d.y, bank, layout(d.weeks, width))
    margin = cfg['equivalence_margin']
    tested = p_values(pt, np.array([0.0, 1.0, -margin, margin]))
    cs = acceptance_set(pt, cfg['alpha'])
    return {'period': task['period'], 'width': width, 'beta': pt.beta, 'draws': cfg['draws'],
            'p_two_sided_at_0': tested[0, 0], 'p_two_sided_at_1': tested[0, 1],
            'tost_p_lower_boundary': tested[1, 2], 'tost_p_upper_boundary': tested[2, 3],
            'equivalence_rejections': bool(tested[1, 2] <= cfg['alpha'] and tested[2, 3] <= cfg['alpha']),
            'segments': cs['segments'], 'crossings': cs['crossings'], **describe_set(cs)}


def descriptive_rows(p: pd.DataFrame, rows: pd.DatetimeIndex, name: str, role: str) -> list[dict]:
    out = []
    for exposure in w.EXPOSURES:
        v = w.variant(p, exposure)
        row, _ = w.project(v, rows, 'calendar', 0)
        d = period_design(v, rows, min_df=1)
        shares = d.xp ** 2 / (d.xp @ d.xp)
        top = int(np.argmax(shares))
        out.append({'period': name, 'role': role, 'exposure': exposure, 'n': row['n'], 'control_rank': row['control_rank'],
                    'residual_df': row['residual_df'], 'first_week': row['first_week'], 'last_week': row['last_week'],
                    'beta_R': row['beta_R'], 'theta_RRP': row['theta_RRP'], 'n_F': float(1 / (shares ** 2).sum()),
                    'top_week': str(d.dates[top].date()), 'top_week_share': float(shares[top]),
                    'max_control_leverage': float((d.q * d.q).sum(axis=1).max())})
    return out


def produce(p: pd.DataFrame, config: dict, workers: int, progress=None) -> dict[str, pd.DataFrame]:
    spans = periods(config)
    rows_all = w.common_mask([w.variant(p, e) for e in w.EXPOSURES], 'calendar', 0)
    expected = ro.final_wednesday(config['eras']['pooled'][1])
    if rows_all.max() != expected:
        raise ValueError(f'pooled sample ends {rows_all.max().date()}, not the declared final week {expected.date()}')
    core = w.variant(p, 'trc_core')
    rows = {name: rows_all[(rows_all >= start) & (rows_all <= end)] for name, (start, end) in spans.items()}
    descriptive = [row for name in spans for row in descriptive_rows(
        p, rows[name], name, 'inferential' if name in config['inferential_periods'] else 'descriptive')]
    inference, calibration = config['inference'], config['calibration']
    widths = [inference['primary_width'], *inference['sensitivity_widths']]
    observed_tasks, cell_tasks = [], []
    for index, name in enumerate(config['inferential_periods']):
        d = period_design(core, rows[name])
        for width in widths:
            observed_tasks.append({'design': d, 'inference': inference, 'width': width, 'period': name, 'period_index': index})
            for cell in cells(config, name):
                cell_tasks.append({'design': d, 'cell': cell, 'calibration': calibration, 'width': width, 'period': name,
                                   'period_index': index, 'alpha': inference['alpha'],
                                   'episodes': calibration['crisis_episodes'][name]})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        observed = list(pool.map(run_observed, observed_tasks))
        calibrated = []
        for row in pool.map(run_cell, cell_tasks):
            calibrated.append(row)
            if progress:
                progress(row)
    cal = pd.DataFrame(calibrated)
    status = calibration_status(cal, {(t['period'], t['width'], t['cell']['cell']) for t in cell_tasks})
    results, sets = [], []
    for row in observed:
        flags = status.set_index(['period', 'width']).loc[(row['period'], row['width'])]
        segments = row.pop('segments')
        label = 'calibrated' if flags['interval_calibrated'] else 'failed_calibration_descriptive_only'
        results.append(row | {'segments': len(segments), 'interval_calibrated': bool(flags['interval_calibrated']),
                              'one_sided_calibrated': bool(flags['one_sided_calibrated']),
                              'equivalence_established': bool(row['equivalence_rejections'] and flags['one_sided_calibrated']),
                              'interval_status': label})
        sets += [{'period': row['period'], 'width': row['width'], 'segment': i, 'lower': a, 'upper': b,
                  'interval_status': label} for i, (a, b) in enumerate(segments)]
    return {'period_coefficients': pd.DataFrame(descriptive), 'inference': pd.DataFrame(results),
            'confidence_sets': pd.DataFrame(sets, columns=['period', 'width', 'segment', 'lower', 'upper', 'interval_status']),
            'calibration': cal, 'calibration_status': status}


def calibration_status(cal: pd.DataFrame, declared: set) -> pd.DataFrame:
    """Per period and width; every declared cell exactly once, both tails checked."""
    found = list(zip(cal['period'], cal['width'], cal['cell']))
    if len(found) != len(set(found)) or set(found) != declared:
        raise ValueError('calibration cells differ from the declared period x width x cell product')
    status = (cal.groupby(['period', 'width'])
              .agg(cells=('cell', 'size'), two_sided_failures=('two_sided_pass', lambda s: int((~s).sum())),
                   one_sided_failures=('one_sided_pass', lambda s: int((~s).sum())),
                   min_coverage_lower=('coverage_wilson_lower', 'min'),
                   max_upper_tail_upper=('upper_wilson_upper', 'max'), max_lower_tail_upper=('lower_wilson_upper', 'max'))
              .reset_index())
    status['interval_calibrated'] = status['two_sided_failures'].eq(0)
    status['one_sided_calibrated'] = status['one_sided_failures'].eq(0)
    return status


def run(source_root: Path, out: Path, config_path: Path = Path(CONFIG), *, workers: int | None = None, progress=None) -> dict:
    """Build thin artifacts only from an entirely committed producer and pre-declared config."""
    repository = Path(__file__).resolve().parents[2]
    config_name = config_path.resolve().relative_to(repository).as_posix()
    producer = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=repository, text=True)
    dirty = '\n'.join(line for line in dirty.splitlines() if line != '?? uv.lock')
    if dirty.strip():
        raise ValueError('commit the producer before generating retained evidence')
    config = json.loads(config_path.read_text())
    if config.get('schema') != SCHEMA:
        raise ValueError(f'config must declare schema {SCHEMA}')
    if subprocess.check_output(['git', 'show', f'{producer}:{config_name}'], cwd=repository) != config_path.read_bytes():
        raise ValueError('configuration differs from the committed declaration')
    if out.name != SCHEMA:
        raise ValueError('output directory must be named for the schema')
    ro._require_unpublished(out)
    p, source_receipt = w.load_inputs(source_root, config)
    tables = produce(p, config, workers or os.cpu_count() or 1, progress)
    receipt = {'schema_version': SCHEMA, 'vintage': config['vintage'], 'producer_commit': producer,
               'source_main_commit': config['source_main'], 'source_producer_commit': source_receipt['producer_commit'],
               'config': config_name, 'config_sha256': _sha256(config_path), 'input_root': str(source_root.resolve()),
               'inputs': config['inputs'], 'periods': {k: [str(a.date()), str(b.date())] for k, (a, b) in periods(config).items()},
               'calibration_status': tables['calibration_status'].to_dict(orient='records'),
               'environment': {'numpy': np.__version__, 'pandas': pd.__version__}, 'workers': workers or os.cpu_count()}
    ro._write_atomic(out, {name + '_v1': table for name, table in tables.items()}, receipt, receipt_name='receipt_v1.json')
    return {'producer_commit': producer, 'receipt_sha256': _sha256(out / 'receipt_v1.json'), 'out': str(out)}


# ---------------------------------------------------------------------------
# Rolling weekly windows (simple descriptive path over time)
# ---------------------------------------------------------------------------

ROLLING_SCHEMA = 'weekly_reserve_rolling_v1'
ROLLING_CONFIG = 'config/weekly_reserve_rolling_2026q2.json'


def newey_west(d: Design, width: int) -> float:
    """Conventional Newey-West SE of the exposure slope: Bartlett calendar-distance weights, scale n/(n-k)."""
    scores = d.a * full_residual(d, d.y)
    variance = float((bartlett_factor(scores, layout(d.weeks, width)) ** 2).sum())
    k = d.rank + 1
    return math.sqrt(variance * d.n / (d.n - k))


def rolling(p: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows_all = w.common_mask([w.variant(p, e) for e in w.EXPOSURES], 'calendar', 0)
    core = w.variant(p, 'trc_core')
    weeks = ro._week_ordinals(rows_all)
    span, width = int(config['window_weeks']), int(config['hac_width'])
    out = []
    for end in range(len(rows_all)):
        first = weeks[end] - span + 1
        if first < weeks[0]:
            continue
        inside = (weeks >= first) & (weeks <= weeks[end])
        rows = rows_all[inside]
        start_date = rows_all[end] - pd.Timedelta(weeks=span - 1)
        row = {'window_start': str(start_date.date()), 'window_end': str(rows_all[end].date()),
               'midpoint': str((start_date + (rows_all[end] - start_date) / 2).date()), 'observed_weeks': len(rows)}
        if len(rows) < config['minimum_observed_weeks']:
            out.append(row | {'status': 'too_few_observed_weeks'})
            continue
        d = period_design(core, rows, min_df=int(config['minimum_residual_df']))
        beta, se = float(d.a @ d.y), newey_west(d, width)
        shares = d.xp ** 2 / (d.xp @ d.xp)
        top = int(np.argmax(shares))
        out.append(row | {'status': 'reported', 'beta': beta, 'se': se, 'lower95': beta - 1.959963984540054 * se,
                          'upper95': beta + 1.959963984540054 * se, 'k': d.rank + 1, 'n_F': float(1 / (shares ** 2).sum()),
                          'top_week': str(d.dates[top].date()), 'top_week_share': float(shares[top])})
    return pd.DataFrame(out)


def run_rolling(source_root: Path, out: Path, config_path: Path = Path(ROLLING_CONFIG)) -> dict:
    repository = Path(__file__).resolve().parents[2]
    config_name = config_path.resolve().relative_to(repository).as_posix()
    producer = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=repository, text=True)
    if '\n'.join(line for line in dirty.splitlines() if line != '?? uv.lock').strip():
        raise ValueError('commit the producer before generating retained evidence')
    config = json.loads(config_path.read_text())
    if config.get('schema') != ROLLING_SCHEMA:
        raise ValueError(f'config must declare schema {ROLLING_SCHEMA}')
    if subprocess.check_output(['git', 'show', f'{producer}:{config_name}'], cwd=repository) != config_path.read_bytes():
        raise ValueError('configuration differs from the committed declaration')
    if out.name != ROLLING_SCHEMA:
        raise ValueError('output directory must be named for the schema')
    ro._require_unpublished(out)
    p, source_receipt = w.load_inputs(source_root, config)
    table = rolling(p, config)
    receipt = {'schema_version': ROLLING_SCHEMA, 'vintage': config['vintage'], 'producer_commit': producer,
               'source_main_commit': config['source_main'], 'source_producer_commit': source_receipt['producer_commit'],
               'config': config_name, 'config_sha256': _sha256(config_path), 'input_root': str(source_root.resolve()),
               'inputs': config['inputs'], 'windows': len(table), 'reported': int(table['status'].eq('reported').sum()),
               'environment': {'numpy': np.__version__, 'pandas': pd.__version__}}
    ro._write_atomic(out, {'rolling_v1': table}, receipt, receipt_name='receipt_v1.json')
    return {'producer_commit': producer, 'receipt_sha256': _sha256(out / 'receipt_v1.json'), 'out': str(out)}


# ---------------------------------------------------------------------------
# Monthly clock with the money-fund route term: one-month and three-month horizons
# ---------------------------------------------------------------------------

MONTHLY_SCHEMA = 'monthly_reserve_rolling_v1'
MONTHLY_CONFIG = 'config/monthly_reserve_rolling_2026q2.json'


def load_monthly(source_root: Path, config: dict) -> pd.DataFrame:
    source = config['source']
    for key in ('table', 'receipt'):
        ro._check_hash(Path(source_root) / source[key]['name'], source[key]['sha256'], source=f"monthly {key}")
    frame = pd.read_csv(Path(source_root) / source['table']['name'])
    frame['period'] = pd.PeriodIndex(frame['month'], freq='M')
    if frame['period'].duplicated().any() or not frame['period'].is_monotonic_increasing:
        raise ValueError('monthly table must have unique ordered months')
    if ((frame['period'] - frame['period'].iloc[0]).apply(lambda d: d.n) != np.arange(len(frame))).any():
        raise ValueError('monthly table must have consecutive months')
    return frame.set_index('period')


def horizon_sample(frame: pd.DataFrame, exposure: str, months: int) -> pd.DataFrame:
    """One observation per start month: sums of the outcome and exposure over months t..t+h-1, all present."""
    y = frame['delta_reserve_balances'].rolling(months, min_periods=months).sum().shift(-(months - 1))
    x = frame[exposure].rolling(months, min_periods=months).sum().shift(-(months - 1))
    out = pd.DataFrame({'x': x, 'y': y}).dropna()
    out['start_month'] = out.index
    out['end_month'] = out.index + (months - 1)
    return out


def monthly_design(sample: pd.DataFrame, min_df: int) -> Design:
    starts = sample['start_month']
    z = np.column_stack([np.ones(len(sample)), *[(starts.dt.month == m).to_numpy(float) for m in range(2, 13)]])
    dates = pd.DatetimeIndex(starts.dt.to_timestamp())
    ordinals = (starts.dt.year * 12 + starts.dt.month).to_numpy()
    return make_design(dates, sample['x'].to_numpy(float), sample['y'].to_numpy(float), z, min_df, ordinals)


def common_months(frame: pd.DataFrame, exposures) -> pd.DataFrame:
    """Blank every exposure in months where any declared exposure is missing, so all exposures share one sample."""
    out = frame.copy()
    present = out[list(exposures)].notna().all(axis=1)
    out.loc[~present, list(exposures)] = np.nan
    return out


def monthly_rolling(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pooled = [], []
    span = int(config['window_months'])
    frame = common_months(frame, config['exposures'])
    for exposure in config['exposures']:
        for spec in config['horizons'].values():
            months, width = int(spec['months']), int(spec['hac_width'])
            sample = horizon_sample(frame, exposure, months)
            d = monthly_design(sample, min_df=MIN_RESIDUAL_DF)
            beta, se = float(d.a @ d.y), newey_west(d, width)
            pooled.append({'exposure': exposure, 'horizon_months': months, 'starts': 'every month',
                           'first_start': str(sample.index[0]), 'last_end': str(sample['end_month'].iloc[-1]), 'n': d.n,
                           'beta': beta, 'se': se, 'lower95': beta - 1.959963984540054 * se,
                           'upper95': beta + 1.959963984540054 * se})
            if months == 3:
                quarters = sample[sample['start_month'].dt.month.isin([1, 4, 7, 10])]
                dq = monthly_design(quarters, min_df=MIN_RESIDUAL_DF)
                bq, eq = float(dq.a @ dq.y), newey_west(dq, width)
                pooled.append({'exposure': exposure, 'horizon_months': months, 'starts': 'calendar quarters',
                               'first_start': str(quarters.index[0]), 'last_end': str(quarters['end_month'].iloc[-1]),
                               'n': dq.n, 'beta': bq, 'se': eq, 'lower95': bq - 1.959963984540054 * eq,
                               'upper95': bq + 1.959963984540054 * eq})
            for end in pd.period_range(sample.index[0] + (span - 1), sample['end_month'].iloc[-1], freq='M'):
                first = end - (span - 1)
                inside = sample[(sample.index >= first) & (sample['end_month'] <= end)]
                midpoint = first.to_timestamp() + (end.to_timestamp(how='end') - first.to_timestamp()) / 2
                row = {'exposure': exposure, 'horizon_months': months, 'window_start': str(first), 'window_end': str(end),
                       'midpoint': str(midpoint.date()), 'n': len(inside)}
                if len(inside) < spec['minimum_observations']:
                    rows.append(row | {'status': 'too_few_observations'})
                    continue
                dw = monthly_design(inside, min_df=MIN_RESIDUAL_DF)
                b, e = float(dw.a @ dw.y), newey_west(dw, width)
                shares = dw.xp ** 2 / (dw.xp @ dw.xp)
                top = int(np.argmax(shares))
                rows.append(row | {'status': 'reported', 'beta': b, 'se': e, 'lower95': b - 1.959963984540054 * e,
                                   'upper95': b + 1.959963984540054 * e, 'n_F': float(1 / (shares ** 2).sum()),
                                   'top_start_month': str(inside.index[top]), 'top_share': float(shares[top])})
    return pd.DataFrame(rows), pd.DataFrame(pooled)


def run_monthly(source_root: Path, out: Path, config_path: Path = Path(MONTHLY_CONFIG)) -> dict:
    repository = Path(__file__).resolve().parents[2]
    config_name = config_path.resolve().relative_to(repository).as_posix()
    producer = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=repository, text=True)
    if '\n'.join(line for line in dirty.splitlines() if line != '?? uv.lock').strip():
        raise ValueError('commit the producer before generating retained evidence')
    config = json.loads(config_path.read_text())
    if config.get('schema') != MONTHLY_SCHEMA:
        raise ValueError(f'config must declare schema {MONTHLY_SCHEMA}')
    if subprocess.check_output(['git', 'show', f'{producer}:{config_name}'], cwd=repository) != config_path.read_bytes():
        raise ValueError('configuration differs from the committed declaration')
    if out.name != MONTHLY_SCHEMA:
        raise ValueError('output directory must be named for the schema')
    ro._require_unpublished(out)
    rolling_table, pooled = monthly_rolling(load_monthly(source_root, config), config)
    receipt = {'schema_version': MONTHLY_SCHEMA, 'vintage': config['vintage'], 'producer_commit': producer,
               'config': config_name, 'config_sha256': _sha256(config_path), 'source_root': str(Path(source_root).resolve()),
               'source': config['source'], 'pooled': pooled.to_dict(orient='records'),
               'environment': {'numpy': np.__version__, 'pandas': pd.__version__}}
    ro._write_atomic(out, {'rolling_v1': rolling_table, 'pooled_v1': pooled}, receipt, receipt_name='receipt_v1.json')
    return {'producer_commit': producer, 'receipt_sha256': _sha256(out / 'receipt_v1.json'), 'out': str(out)}
