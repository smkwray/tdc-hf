"""Pinned Wednesday TRC conditional projections and accounting diagnostics.

Route paths are assumptions. Calibration concerns only modern h=0 zero-route
calendar projections; it never licenses a causal interpretation.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import reserve_offsets as ro
from .manifest import _sha256

SCHEMA = 'weekly_trc_projection_v1'
CONFIG = 'config/weekly_trc_projection_v1.json'
ERAS = {
    'pooled': ('2005-10-12', '2025-12-31'),
    '2005-10-12_2008Q3': ('2005-10-12', '2008-09-30'),
    '2008Q4_transition': ('2008-10-01', '2008-12-31'),
    '2009Q1_2015Q3': ('2009-01-01', '2015-09-30'),
    '2015Q4_transition': ('2015-10-01', '2015-12-31'),
    '2016Q1_2025Q4': ('2016-01-01', '2025-12-31'),
}
MODERN = '2016Q1_2025Q4'
CALENDAR = ro.BASELINE_CONTROLS[:-2]
LAGGED = ro.BASELINE_CONTROLS
EXPOSURES = ('trc_core', 'trc_par_sensitivity', 'c_h')
BRIDGES = (
    'bridge_all_securities_premium_discount',
    'bridge_treasury_securities_source_inflation',
    'bridge_maturity_payment_clock', 'bridge_primary_cash_price',
    'bridge_redemption_inflation', 'bridge_internal_postings',
    'bridge_coupon_remittance_payments', 'bridge_published_rounding',
)
# Disjoint ledger rows: cash, securities, RRP and valuation enter elsewhere.
EXCLUDED = {'treasury_general_account', 'supplementary_financing_account',
            'treasury_securities', 'rrp_others', 'unamortized_premiums',
            'unamortized_discounts'}
FACTORS = tuple('f_' + r for r in ro._factor_rows() if r not in EXCLUDED)
PARTS = (*FACTORS, 'rrp_net_route', *BRIDGES)


def design_eras(config: dict) -> tuple[dict[str, tuple[str, str]], str]:
    """Declared eras and modern era; the frozen v1 config predates both fields."""
    if 'eras' not in config:
        return ERAS, MODERN
    eras = {name: tuple(bounds) for name, bounds in config['eras'].items()}
    if next(iter(eras)) != 'pooled' or config['modern'] not in eras:
        raise ValueError('declared eras must start with pooled and contain the modern era')
    return eras, config['modern']


def read_csv(path: Path, keys=('date',)) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame['date'] = pd.to_datetime(frame['date'], format='%Y-%m-%d', errors='raise')
    if frame.duplicated(list(keys)).any() or not frame['date'].is_monotonic_increasing:
        raise ValueError(f'{path.name}: duplicate or unordered dates')
    if not np.isfinite(frame.select_dtypes(include='number').replace(np.nan, 0)).all().all():
        raise ValueError(f'{path.name}: nonfinite numeric values')
    return frame


def load_inputs(root: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    """Pin every consumed file; check independent ledger and daily aggregation."""
    eras, _ = design_eras(config)
    for name, expected in config['inputs'].items():
        ro._check_hash(root / name, expected, source=name)
    receipt = json.loads((root / 'data/processed/trc_weekly_v1_receipt.json').read_text())
    if receipt['units'] != ro.UNITS:
        raise ValueError('TRC units must be USD million')
    for key in ('weekly', 'identity', 'daily_route'):
        entry = receipt['outputs'][key]
        if config['inputs'].get(entry['path']) != entry['sha256']:
            raise ValueError('TRC output receipt pin mismatch')
    ledger, lr = ro.load_ledger(
        root / 'data/processed/reserve_factor_ledger_weekly_v1.csv',
        root / 'data/processed/reserve_factor_ledger_v1_receipt.json',
        ledger_sha256=config['inputs']['data/processed/reserve_factor_ledger_weekly_v1.csv'],
        receipt_sha256=config['inputs']['data/processed/reserve_factor_ledger_v1_receipt.json'])
    rrp = ro.load_level(root / 'data/raw/fiscus/h41_ledger/rrp_others.csv', lr, 'rrp_others', ledger)
    core = read_csv(root / 'data/processed/trc_weekly_components_v1.csv').set_index('date')
    identity = read_csv(root / 'data/processed/trc_weekly_identity_v1.csv', ('date', 'variant'))
    identity = identity.loc[identity['variant'].eq('core')].set_index('date')
    daily = read_csv(root / 'data/processed/trc_weekly_route_daily_v1.csv')
    daily['week'] = pd.to_datetime(daily['week'], errors='raise')
    if not (daily['week'] == pd.to_datetime([ro.assign_h8_week(d) for d in daily['date']])).all():
        raise ValueError('daily route window mismatch')
    for name in ('route_forward', 'route_reverse'):
        grouped = daily.groupby('week')[name]
        sums = grouped.sum(min_count=7).reindex(core.index)
        if not sums.isna().equals(core[name].isna()) or not np.allclose(sums.dropna(), core.loc[sums.notna(), name], atol=1e-5):
            raise ValueError('daily route capacity does not match complete weekly sums')
    if (core['route_forward'].dropna() < 0).any() or (core['route_reverse'].dropna() > 0).any():
        raise ValueError('route capacities have incorrect signs')
    p = ledger.join(core[list(EXPOSURES[0:2]) + ['route_forward', 'route_reverse']], how='left')
    for name in BRIDGES:
        p[name] = identity[name].reindex(p.index)
    p['rrp_level'] = rrp
    p['state'] = rrp.shift(1) / ro.STATE_UNIT
    p['state_date'] = pd.Series(p.index, index=p.index).shift(1)
    p['lag_exposure_c'] = p['c_h'].shift(1)
    p['lag_delta_reserve_balances'] = p['delta_reserve_balances'].shift(1)
    p = p.join(ro.calendar_flags(p.index))
    p['era'] = 'outside'
    for era, (start, end) in list(eras.items())[1:]:
        p.loc[(p.index >= start) & (p.index <= end), 'era'] = era
    valid = p['trc_core'].notna()
    if not np.allclose(p.loc[valid, 'c_h'], core.loc[p.index[valid], 'c_h'], atol=1e-6):
        raise ValueError('cash legs differ across producers')
    if not np.allclose(p.loc[valid, FACTORS].sum(axis=1), identity.loc[p.index[valid], 'non_treasury_signed_factors'], atol=1e-6):
        raise ValueError('disjoint ledger factors differ from identity')
    base = variant(p, 'trc_core')
    if (base.loc[valid, 'delta_reserve_balances'] - base.loc[valid, 'x'] - base.loc[valid, PARTS].sum(axis=1)).abs().max() > 1e-5:
        raise ValueError('TRC factor/bridge identity fails')
    return p, receipt


def variant(p: pd.DataFrame, exposure: str, shares=(0., 0.)) -> pd.DataFrame:
    v = p.copy()
    # An inactive channel is definitionally absent; an active missing channel remains missing.
    gamma = pd.Series(0., index=p.index)
    for share, name in zip(shares, ('route_forward', 'route_reverse')):
        if share:
            gamma = gamma + share * p[name]
    v['gamma'] = gamma
    v['x'] = p[exposure] + gamma
    v['rrp_net_route'] = p['f_rrp_others'] - gamma
    v['raw_rrp'] = -p['f_rrp_others']
    # Cash and par comparators move the exposure difference once into a named bridge.
    v['bridge_exposure_comparator'] = p['trc_core'] - p[exposure]
    return v


def controls(p: pd.DataFrame, spec: str) -> pd.DataFrame:
    names = CALENDAR if spec == 'calendar' else LAGGED
    return pd.DataFrame({'intercept': 1., **{c: p[c] for c in names}}, index=p.index)


def control_basis(z: np.ndarray) -> tuple[np.ndarray, int]:
    """SVD basis handles calendar dependencies without exposure-specific selection."""
    if len(z) == 0:
        raise ValueError("empty control sample")
    norms = np.linalg.norm(z, axis=0)
    normalized = z[:, norms > 0] / norms[norms > 0]
    u, s, _ = np.linalg.svd(normalized, full_matrices=False)
    rank = int((s > s[0] * 1e-10).sum())
    return u[:, :rank], rank


def residual(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    return y - q @ (q.T @ y)


def slope(x: np.ndarray, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    xp = residual(x, q)
    den = xp @ xp
    if den <= 1e-12 * max(1., x @ x):
        raise ValueError('no partial exposure variation after controls')
    return xp @ residual(y, q) / den


def common_mask(variants: list[pd.DataFrame], spec: str, horizon: int) -> pd.DatetimeIndex:
    valid = pd.Series(True, index=variants[0].index)
    for v in variants:
        required = ['x', 'raw_rrp', 'delta_reserve_balances', *PARTS, 'bridge_exposure_comparator']
        for c in required:
            valid &= ro.cumulative(v[c], horizon).notna()
        valid &= controls(v, spec).notna().all(axis=1)
    return valid.index[valid]


def era_rows(rows: pd.DatetimeIndex, era: str, eras: dict = ERAS) -> pd.DatetimeIndex:
    start, end = eras[era]
    return rows[(rows >= start) & (rows <= end)]


def project(v: pd.DataFrame, rows: pd.DatetimeIndex, spec: str, h: int) -> tuple[dict, list[dict]]:
    z = controls(v.loc[rows], spec)
    q, rank = control_basis(z.to_numpy())
    x = v.loc[rows, 'x'].to_numpy()
    names = ['delta_reserve_balances', 'rrp_net_route', 'raw_rrp', *FACTORS, *BRIDGES, 'bridge_exposure_comparator']
    y = np.column_stack([ro.cumulative(v[c], h).loc[rows] for c in names])
    b = slope(x, y, q)
    future = ro.cumulative(v['x'], h) - v['x']
    subsequent = float(slope(x, future.loc[rows].to_numpy(), q))
    contributions = b[1] + b[3:].sum()
    gap = b[0] - 1. - subsequent - contributions
    if abs(gap) > 1e-8:
        raise AssertionError('same-mask decomposition does not close')
    row = {'n': len(rows), 'control_rank': rank, 'residual_df': len(rows) - rank - 1,
           'first_week': str(rows.min().date()), 'last_week': str(rows.max().date()),
           'last_endpoint': str((rows.max() + pd.Timedelta(weeks=h)).date()),
           'h': h, 'weekly_flows': h+1, 'beta_R': float(b[0]), 'theta_RRP': float(b[1]),
           'raw_rrp_balance_slope': float(b[2]), 'Omega': float(-b[1]),
           'subsequent_TRC': subsequent, 'remainder_sum': float(contributions), 'closure_gap': float(gap),
           'inference_status': 'descriptive'}
    parts = [{'component': c, 'contribution': float(value)} for c, value in zip(names[1:2]+names[3:], np.r_[b[1], b[3:]])]
    return row, parts


def coefficient_tables(p: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    tables = {name: [] for name in ('coefficients', 'decomposition_h0', 'horizon_profile', 'route_paths', 'influence')}
    eras, _ = design_eras(config)
    for spec in ('calendar', 'lagged'):
        variants = [variant(p, name) for name in EXPOSURES]
        mask0 = common_mask(variants, spec, 0)
        maskh = common_mask(variants, spec, 8)
        for era in eras:
            for exposure, v in zip(EXPOSURES, variants):
                rows0 = era_rows(mask0, era, eras)
                metadata = {'era': era, 'exposure': exposure, 'controls': spec, 'sample_mask': 'matched_h0'}
                try:
                    row, parts = project(v, rows0, spec, 0)
                except ValueError as exc:
                    tables['coefficients'].append({**metadata, 'n': len(rows0), 'inference_status': 'unavailable', 'reason': str(exc)})
                    continue
                tables['coefficients'].append({**metadata, **row})
                tables['decomposition_h0'].extend({**metadata, 'n': len(rows0), **part} for part in parts)
                tables['influence'].append({**metadata, **influence(v, rows0, spec)})
                for h in config['horizons']:
                    rows = era_rows(maskh, era, eras)
                    for trimmed in (False, True):
                        if trimmed and era == 'pooled':
                            continue
                        crosses = (rows + pd.Timedelta(weeks=h) > pd.Timestamp(eras[era][1]))
                        boundary_complete = rows + pd.Timedelta(weeks=8) <= pd.Timestamp(eras[era][1])
                        used = rows[boundary_complete] if trimmed else rows
                        try:
                            profile, _ = project(v, used, spec, h)
                        except ValueError:
                            continue
                        tables['horizon_profile'].append({**metadata, **profile,
                            'sample_mask': 'matched_profile_h8_boundary_trimmed' if trimmed else 'matched_profile_h8',
                            'endpoints_cross_era': int(0 if trimmed else crosses.sum())})
        routed = [variant(p, 'trc_core', shares) for shares in config['route_paths'].values()]
        maskr = common_mask(routed, spec, 0)
        for era in eras:
            rows = era_rows(maskr, era, eras)
            for (name, shares), v in zip(config['route_paths'].items(), routed):
                meta = {'era': era, 'exposure': name, 'controls': spec, 'sample_mask': 'matched_routes_h0'}
                try:
                    row, _ = project(v, rows, spec, 0)
                    tables['route_paths'].append({**meta, **row, 'forward_share': shares[0], 'reverse_share': shares[1], 'route_status': 'declared_assumption'})
                    tables['influence'].append({**meta, **influence(v, rows, spec)})
                except ValueError as exc:
                    tables['route_paths'].append({**meta, 'n': len(rows), 'inference_status': 'unavailable', 'reason': str(exc)})
    return {k: pd.DataFrame(v) for k, v in tables.items()}


def influence(v: pd.DataFrame, rows: pd.DatetimeIndex, spec: str) -> dict:
    x = v.loc[rows, 'x'].to_numpy()
    z = controls(v.loc[rows], spec).to_numpy()
    q, _ = control_basis(z)
    xp = residual(x, q)
    shares = xp**2 / (xp @ xp)
    order = np.argsort(-shares)
    y = v.loc[rows, ['delta_reserve_balances', 'rrp_net_route']].to_numpy()
    def deleted(keep):
        try:
            basis, _ = control_basis(z[keep])
            return slope(x[keep], y[keep], basis)
        except ValueError:
            return np.array([np.nan, np.nan])
    largest = np.ones(len(rows), dtype=bool)
    largest[order[0]] = False
    without = deleted(largest)
    # Nonoverlapping prespecified full-calendar blocks, including partial edges.
    block_ids = ro._week_ordinals(rows) // 13
    block = np.array([deleted(block_ids != b) for b in np.unique(block_ids)])
    crisis = rows == pd.Timestamp('2008-09-24')
    after_crisis = deleted(~crisis) if crisis.any() else [np.nan, np.nan]
    return {'top_week': str(rows[order[0]].date()), 'top_week_share': float(shares[order[0]]),
            'top5_share': float(shares[order[:5]].sum()), 'top5_weeks': ';'.join(rows[order[:5]].strftime('%Y-%m-%d')),
            'beta_without_top_week': float(without[0]), 'theta_without_top_week': float(without[1]),
            'block_length': 13, 'block_anchor': '1970-01-07', 'block_count': len(block),
            'block_unavailable_count': int(np.isnan(block[:, 0]).sum()),
            'block_beta_min': float(np.nanmin(block[:, 0])), 'block_beta_max': float(np.nanmax(block[:, 0])),
            'block_theta_min': float(np.nanmin(block[:, 1])), 'block_theta_max': float(np.nanmax(block[:, 1])),
            'sep24_2008_in_sample': bool(crisis.any()), 'sep24_2008_share': float(shares[crisis].sum()) if crisis.any() else np.nan,
            'beta_without_sep24_2008': float(after_crisis[0]), 'theta_without_sep24_2008': float(after_crisis[1])}


def state_design(p: pd.DataFrame, rows: pd.DatetimeIndex, eras: dict = ERAS) -> tuple[pd.DataFrame, pd.Series]:
    frame = p.loc[rows]
    if not (frame['state_date'] < frame.index - pd.Timedelta(days=6)).all():
        raise ValueError('state clock must predate the exposure window')
    centered = frame['state'] - frame.groupby('era')['state'].transform('mean')
    z = controls(frame, 'calendar')
    z['state_main_effect'] = centered
    for era in list(eras)[1:]:
        indicator = frame['era'].eq(era).astype(float)
        z['era_intercept_' + era] = indicator
        z['era_slope_' + era] = frame['trc_core'] * indicator
    return z, frame['trc_core'] * centered


def state_table(p: pd.DataFrame, eras: dict = ERAS) -> pd.DataFrame:
    rows = era_rows(common_mask([variant(p, 'trc_core')], 'calendar', 0), 'pooled', eras)
    rows = rows[p.loc[rows, 'state'].notna()]
    z, interaction = state_design(p, rows, eras)
    q, rank = control_basis(z.to_numpy())
    ip = residual(interaction.to_numpy(), q)
    b = slope(interaction.to_numpy(), p.loc[rows, ['delta_reserve_balances', 'f_rrp_others']].to_numpy(), q)
    weight = ip**2 / (ip @ ip)
    out = []
    for era in eras:
        sub = era_rows(rows, era, eras)
        positions = rows.get_indexer(sub)
        state = p.loc[sub, 'state']
        leverage = weight[positions]
        out.append({'era': era, 'n': len(sub), 'control_rank': rank,
                    'state_unit': 'prior_Wednesday_RRP_others_per_100bn',
                    'state_min': state.min(), 'state_p10': state.quantile(.1), 'state_median': state.median(),
                    'state_p90': state.quantile(.9), 'state_max': state.max(), 'state_center': state.mean(),
                    'interaction_beta_R': b[0], 'interaction_theta_RRP': b[1],
                    'interaction_partial_variation_share': leverage.sum(),
                    'interaction_top_week_global_share': leverage.max(),
                    'interaction_top_week_within_era_share': leverage.max()/leverage.sum() if leverage.sum() else np.nan,
                    'interaction_top5_within_era_share': np.sort(leverage)[-5:].sum()/leverage.sum() if leverage.sum() else np.nan,
                    'interaction_top_week': str(sub[np.argmax(leverage)].date()),
                    'inference_status': 'descriptive_association_pooled_common_interaction',
                    'support_status': 'support_differs_by_era_no_route_identification'})
    return pd.DataFrame(out)


def rebuild_weekly(x: np.ndarray, factors: np.ndarray, initial_reserve=0.) -> dict:
    """Primitive factor simulation reconstructs stocks, outcomes and reserve lags."""
    delta = x + factors.sum(axis=-1)
    levels = initial_reserve + np.cumsum(delta, axis=0)
    lag_delta = np.r_[np.nan, delta[:-1]]
    return {'delta': delta, 'reserve': levels, 'lag_delta_reserve_balances': lag_delta,
            'cumulative': {h: pd.Series(delta).rolling(h+1, min_periods=h+1).sum().shift(-h).to_numpy() for h in (0,1,4,8)}}


def multiplier_bank(dates: pd.DatetimeIndex, bandwidth: int, reps: int, rng) -> np.ndarray:
    """Gaussian moving-average multipliers: Cov(v_t,v_s)=1-d/(L+1)."""
    weeks = ro._week_ordinals(dates)
    positions = weeks - weeks.min()
    white = rng.normal(size=(reps, positions.max() + bandwidth + 1))
    cs = np.cumsum(np.c_[np.zeros(reps), white], axis=1)
    all_weeks = (cs[:, bandwidth+1:] - cs[:, :-bandwidth-1]) / np.sqrt(bandwidth+1)
    return all_weeks[:, positions]


def restricted_test(x: np.ndarray, z: np.ndarray, contributions: np.ndarray, target: np.ndarray,
                    dates: pd.DatetimeIndex, bandwidth: int, reps: int, rng,
                    multipliers: np.ndarray | None = None) -> dict:
    """Test (reserve departure, signed RRP slope) by jointly constrained factors.

    Null slopes for (RRP,remainder) are (target_theta,target_beta-target_theta).
    Scales are restricted-fit Bartlett score scales, fixed over bootstrap draws.
    Nuisance-fit HC3 corrects residuals; there is no replicate studentization.
    """
    q, _ = control_basis(z)
    xp = residual(x, q)
    a = xp / (xp @ xp)
    slopes_null = np.array([target[1], target[0]-target[1]])
    nuisance = q @ (q.T @ (contributions-x[:, None]*slopes_null))
    errors = contributions-x[:, None]*slopes_null-nuisance
    leverage = (q*q).sum(axis=1)
    adjusted = errors / np.maximum(1-leverage[:, None], 1e-8)
    # Transform scores to the tested reserve departure and signed RRP pair.
    primitive_score = a[:, None] * adjusted
    scores = np.c_[primitive_score.sum(axis=1), primitive_score[:, 0]]
    weeks = ro._week_ordinals(dates)
    variance = (scores*scores).sum(axis=0)
    for lag in range(1, bandwidth+1):
        right = np.searchsorted(weeks, weeks+lag)
        ok = right < len(weeks)
        ok[ok] &= weeks[right[ok]] == weeks[ok]+lag
        variance += 2*(1-lag/(bandwidth+1))*(scores[ok]*scores[right[ok]]).sum(axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-24))
    bank = multiplier_bank(dates, bandwidth, reps, rng) if multipliers is None else multipliers
    # Shared calendar multiplier perturbs primitive vectors; reserve draws are
    # reconstructed as X + RRP_draw + remainder_draw, before projecting.
    simulated = nuisance[None, :, :] + x[None, :, None]*slopes_null + bank[:, :, None]*adjusted[None, :, :]
    reserve_draws = x[None, :] + simulated.sum(axis=2)
    draw_departure = reserve_draws @ a - 1. - target[0]
    draw_theta = simulated[:, :, 0] @ a - target[1]
    statistic_draw = np.max(np.abs(np.c_[draw_departure, draw_theta])/scale, axis=1)
    primitive_slopes = a @ contributions
    estimate = np.array([primitive_slopes.sum(), primitive_slopes[0]])
    observed = float(np.max(np.abs(estimate-target)/scale))
    critical = float(np.quantile(statistic_draw, .95, method='higher'))
    return {'accepted': observed <= critical, 'statistic': observed, 'critical': critical, 'scale': scale,
            'estimate': estimate, 'identity_error': float(np.max(np.abs(reserve_draws-x[None, :]-simulated.sum(axis=2))))}


def invert_tests(x, z, contributions, dates, bandwidth, reps, rng, grid) -> pd.DataFrame:
    """The joint set is exactly the targets accepted by the same restricted test.

    A finite grid is an evaluated cross-section, not an interval/envelope.
    """
    bank = multiplier_bank(dates, bandwidth, reps, rng)
    return pd.DataFrame([{'beta_departure': b, 'theta_RRP': t,
                         'accepted': restricted_test(x,z,contributions,np.array([b,t]),dates,bandwidth,reps,rng,bank)['accepted']}
                        for b,t in grid])


def wilson(successes: int, n: int, z=1.96) -> tuple[float, float]:
    rate = successes/n
    denominator = 1+z*z/n
    center = (rate+z*z/(2*n))/denominator
    half = z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/denominator
    return center-half, center+half


def calibration(p: pd.DataFrame, cfg: dict, progress=None, *, eras: dict = ERAS, modern: str = MODERN) -> pd.DataFrame:
    """Independent size and nonzero-parameter inverted-set inclusion experiments."""
    v = variant(p, 'trc_core')
    rows = era_rows(common_mask([v, variant(p, 'trc_par_sensitivity'), variant(p, 'c_h')], 'calendar', 0), modern, eras)
    x = v.loc[rows, 'x'].to_numpy()
    z = controls(v.loc[rows], 'calendar').to_numpy()
    q, _ = control_basis(z)
    xp = residual(x, q)
    contributions = np.c_[v.loc[rows,'rrp_net_route'], v.loc[rows, [*FACTORS,*BRIDGES]].sum(axis=1)]
    observed_errors = residual(contributions, control_basis(np.c_[z,x])[0])
    sd = observed_errors.std(axis=0)
    correlation = np.corrcoef(observed_errors.T)[0,1]
    chol = np.linalg.cholesky(np.array([[1.,correlation],[correlation,1.]]))
    n = len(rows)
    geometrical = np.sqrt(.2 + 4.8 * xp**2/(xp*xp).mean())
    crisis = ((rows >= '2020-03-01') & (rows <= '2020-06-30')) | ((rows >= '2023-03-01') & (rows <= '2023-05-31'))
    out=[]
    for d, dgp in enumerate(cfg['dgps']):
        for bandwidth in cfg['bandwidths']:
            counts = {'size':0, 'coverage':0}
            for experiment in counts:
                rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'],d,bandwidth,experiment=='coverage']))
                target = np.zeros(2) if experiment=='size' else np.array(cfg['coverage_target'])
                slopes = np.array([target[1], target[0]-target[1]])
                for _ in range(cfg['calibration_simulations']):
                    innovations = rng.standard_t(4, size=(n+300,2))/np.sqrt(2) if dgp=='t4_heavy_tails' else rng.normal(size=(n+300,2))
                    errors = innovations @ chol.T
                    ar = .85 if dgp=='persistent_ar1' else .3
                    for t in range(1,len(errors)):
                        errors[t] = ar*errors[t-1] + np.sqrt(1-ar*ar)*errors[t]
                    errors = errors[300:] * sd
                    if dgp=='concentrated_heteroskedasticity':
                        errors *= geometrical[:,None]
                    if dgp=='crisis_blocks':
                        errors *= np.where(crisis, 5., .6)[:,None]
                    y = x[:,None]*slopes + errors
                    result = restricted_test(x,z,y,target,rows,bandwidth,cfg['calibration_replications'],rng)
                    counts[experiment] += (not result['accepted']) if experiment=='size' else result['accepted']
            simulations=cfg['calibration_simulations']
            size=counts['size']/simulations
            coverage=counts['coverage']/simulations
            size_ci=wilson(counts['size'],simulations,cfg['mc_z'])
            coverage_ci=wilson(counts['coverage'],simulations,cfg['mc_z'])
            out.append({'dgp':dgp,'bandwidth':bandwidth,'n':n,'simulations':simulations,'bootstrap_reps':cfg['calibration_replications'],
                        'size':size,'size_mc_se':np.sqrt(size*(1-size)/simulations),'size_mc_lower':size_ci[0],'size_mc_upper':size_ci[1],
                        'joint_set_coverage':coverage,'coverage_mc_se':np.sqrt(coverage*(1-coverage)/simulations),
                        'coverage_mc_lower':coverage_ci[0],'coverage_mc_upper':coverage_ci[1],
                        'promotion_cell_pass':bool(size_ci[1]<=cfg['size_limit'] and coverage_ci[0]>=cfg['coverage_floor']),
                        'coverage_beta_departure':cfg['coverage_target'][0],'coverage_theta':cfg['coverage_target'][1],
                        'inference_scope':'modern_h0_core_zero_route_calendar_only','errors_ar1':ar,'innovation':'t4_variance_normalized' if dgp=='t4_heavy_tails' else 'gaussian'})
            if progress:
                progress(out[-1])
    return pd.DataFrame(out)


def require_declared_end(coefficients: pd.DataFrame, eras: dict, modern: str) -> str:
    """The modern same-week samples must reach the era's final Wednesday; an earlier source end fails closed."""
    expected = str(ro.final_wednesday(eras[modern][1]).date())
    cell = coefficients.loc[coefficients['era'].eq(modern) & coefficients['controls'].eq('calendar')]
    observed = set(cell['last_week'].astype(str))
    if observed != {expected}:
        raise ValueError(f'{modern} same-week samples end at {sorted(observed)}, not the declared final week {expected}')
    return expected


def run(source_root: Path, out: Path, config_path: Path=Path(CONFIG), *, progress=None) -> dict:
    """Build thin artifacts only from an entirely committed producer/config."""
    repository = Path(__file__).resolve().parents[2]
    config_name = config_path.resolve().relative_to(repository).as_posix()
    producer = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=repository,text=True)
    # uv may create an untracked lockfile; tracked edits and source additions must be committed.
    dirty = '\n'.join(line for line in dirty.splitlines() if line != '?? uv.lock')
    if dirty.strip():
        raise ValueError('commit the producer before generating retained evidence')
    config = json.loads(config_path.read_text())
    committed = subprocess.check_output(['git','show',f'{producer}:{config_name}'],cwd=repository)
    if committed != config_path.read_bytes():
        raise ValueError('configuration differs from the committed pre-fit design')
    if out.name != SCHEMA or out.resolve() == (repository.parents[1]/'output/tables/weekly_reserve_offsets_v1').resolve():
        raise ValueError('output directory must name the separate versioned TRC projection')
    ro._require_unpublished(out)
    eras, modern = design_eras(config)
    p, source_receipt = load_inputs(source_root,config)
    tables = coefficient_tables(p,config)
    final_week = require_declared_end(tables['coefficients'], eras, modern)
    tables['state_interaction'] = state_table(p, eras)
    tables['calibration'] = calibration(p,config['bootstrap'],progress,eras=eras,modern=modern)
    gate = bool(tables['calibration']['promotion_cell_pass'].all())
    definitions = {
        'clock':'Thursday through Wednesday; prior Wednesday state precedes Thursday',
        'units':'USD million inputs; USD per USD slopes; state per USD 100bn',
        'beta_R':'slope_X(R[t+h]-R[t-1]); exact same-clock flow accounting coefficient is one',
        'theta_RRP':'slope_X(sum(-delta RRPothers-Gamma)); signed reserve contribution',
        'raw_rrp_balance_slope':'slope_X(sum(delta RRPothers)); differs from -theta when Gamma is nonzero',
        'Omega':'-theta_RRP; positive-absorption sign convention, conditional association',
        'subsequent_TRC':'slope_X(sum X[t+1:t+h]); explicitly separated from cumulative remainder',
        'routes':'Gamma=a_forward*forward+a_reverse*reverse; reverse already negative; declared paths, never estimates',
        'route_missingness':'Any active missing channel remains missing; route table masks intersect all paths',
        'calendar_controls':';'.join(['intercept',*CALENDAR]),
        'lagged_controls':';'.join(['intercept',*LAGGED]),
        'control_rank':'SVD rank of identically masked control columns; dependencies handled without exposure selection',
        'profile_mask':'Common h8 mask across core/par/cash for each control specification; h8 contains nine flows',
        'eras':'Post-pilot descriptive eras; transitions included in pooled; endpoint crossing counts and trimmed sensitivity reported',
        'influence':'Partial exposure sum of squares; largest-week deletion refits controls; contiguous 13-calendar-week blocks anchored at Wednesday 1970-01-07',
        'state':'Prior-Wednesday RRPothers/100bn, centered within era; main effect plus era intercepts and exposure slopes; pooled common interaction',
        'bootstrap':'Null-imposed vector RRP/remainder, shared full-calendar Gaussian Bartlett multipliers, HC3 nuisance-fit correction; restricted Bartlett scales fixed across draws',
        'bootstrap_bandwidth':'12 baseline; 4 and 8 sensitivities; covariance kernel 1-distance/(bandwidth+1)',
        'inversion':'Joint set defined by acceptance of the same restricted test at each target; coverage checks true nonzero target membership with independent size/coverage streams',
        'calibration':'Observed exposure and calendar geometry; Gaussian AR(.3), concentrated scale sqrt(.2+4.8*partial_X^2/mean_partial_X^2), crisis scale 5 vs .6, persistent AR(.85), variance-normalized t4 with AR(.3); observed residual SD/correlation',
        'mc_gate':'All predeclared DGP/bandwidth cells require 95% Wilson MC upper size <=.075 and lower joint coverage >=.90; MCSE also reported',
        'interpretation':'Departure from one is covariance with other reserve factors and measurement bridges; no causal, landing, retention or fund-route interpretation',
        'source_uncertainty':'Cash core remains a proxy: secondary cash prices, coupon entitlements and unmatched internal postings are not observed; closure is accounting only',
        'promotion':'passed' if gate else 'failed_everything_descriptive',
    }
    if gate:
        v=variant(p,'trc_core')
        rows=era_rows(common_mask([variant(p,e) for e in EXPOSURES],'calendar',0),modern,eras)
        x=v.loc[rows,'x'].to_numpy(); z=controls(v.loc[rows],'calendar').to_numpy()
        y=np.c_[v.loc[rows,'rrp_net_route'],v.loc[rows,[*FACTORS,*BRIDGES]].sum(axis=1)]
        center=slope(x,y,control_basis(z)[0]); pair=np.array([center.sum(),center[0]])
        grid=[(b,t) for b in np.linspace(pair[0]-1,pair[0]+1,41) for t in np.linspace(pair[1]-1,pair[1]+1,41)]
        tables['joint_set_grid']=invert_tests(x,z,y,rows,12,config['bootstrap']['replications'],np.random.default_rng(82911),grid)
    start, end = eras[modern]
    receipt={'schema_version':SCHEMA,'vintage':config.get('vintage'),'producer_commit':producer,'source_main_commit':config['source_main'],
             'source_producer_commit':source_receipt['producer_commit'],'config':config_name,'config_sha256':_sha256(config_path),
             'input_root':str(source_root.resolve()),'inputs':config['inputs'],
             'eras':{name:list(bounds) for name,bounds in eras.items()},'modern_era':modern,
             'modern_final_week':final_week,'core_last_week':str(p['trc_core'].last_valid_index().date()),
             'core_available_weeks':int(p['trc_core'].notna().sum()),'promotion_passed':gate,
             'candidate_scope':f'{start[:4]}-{end[:4]} h0 core zero-route calendar only',
             'definitions':definitions,'environment':{'numpy':np.__version__,'pandas':pd.__version__}}
    readout = render_readout(tables, producer, gate, modern, config_name)
    ro._write_atomic(out, {name+'_v1':table for name,table in tables.items()}, receipt,
                     receipt_name='receipt_v1.json', text_files={
                         'definitions_v1.json':json.dumps(definitions,indent=2)+'\n',
                         'readout_v1.md':readout})
    vintage = config.get('vintage')
    doc = repository / 'docs' / (f'weekly_trc_projection_{vintage}.md' if vintage else 'weekly_trc_projection.md')
    doc.parent.mkdir(exist_ok=True)
    doc.write_text(readout)
    path=out/'receipt_v1.json'
    return {'producer_commit':producer,'receipt_sha256':_sha256(path),'out':str(out),'promotion_passed':gate}


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        cells = [f'{v:.6f}' if isinstance(v, (float,np.floating)) else str(v) for v in row]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def render_readout(tables: dict, producer: str, gate: bool, modern: str = MODERN, config_name: str = CONFIG) -> str:
    c=tables['coefficients']; d=tables['decomposition_h0']; h=tables['horizon_profile']
    r=tables['route_paths']; i=tables['influence']; s=tables['state_interaction']; cal=tables['calibration']
    text = f'# Weekly TRC projection readout\n\nCommitted producer: `{producer}`.\n\n'
    text += ('The calibration promotion rule passed for the narrow modern h=0 candidate.\n\n' if gate else
             'The calibration promotion rule failed. Every coefficient is descriptive; no inferential p-values, stars or confidence claims are exported.\n\n')
    text += ('The accounting coefficient on exact same-clock TRC is one. The measured exposure is a cash-valued proxy. '
             'Projection departures reflect covariance with other reserve factors and measurement bridges. '
             'They do not identify landing, retention, a fund route or a causal response. '
             'Era comparisons were declared after the pilot; non-rejection does not establish equality. '
             'The 2005-10-12–2008Q3 era is source-truncated relative to the declared 2003Q1–2008Q3 era: '
             'the weekly core starts on 2005-10-12 because DTS remittance coverage begins on 2005-10-03.\n\n')
    text += ('Calendar-only controls are primary. Core, par and cash share each h=0 sample; lagged cash and reserve controls are a separate sensitivity. '
             'Transitions stay in the pooled estimate. Short transition fits have little residual variation and are especially fragile; residual degrees of freedom are in the coefficient table. '
             'Missing required inputs are never filled. Secondary prices, coupon entitlement and unmatched internal accounts remain measurement uncertainties.\n\n')
    text += '## Same-week comparisons\n\n' + markdown_table(c.loc[c.controls.eq('calendar'),['era','exposure','n','residual_df','beta_R','theta_RRP']])
    text += '\n## Modern core departure from one\n\n' + markdown_table(d.loc[d.era.eq(modern)&d.exposure.eq('trc_core')&d.controls.eq('calendar'),['component','contribution']])
    text += '\n## Modern cumulative profile\n\nThe common h=8 mask is used at every horizon. h=8 includes nine flows. The subsequent-TRC term is part of the identity.\n\n'
    text += markdown_table(h.loc[h.era.eq(modern)&h.exposure.eq('trc_core')&h.controls.eq('calendar'),['sample_mask','h','n','beta_R','theta_RRP','subsequent_TRC','remainder_sum','endpoints_cross_era']])
    text += '\n## Modern route-path sensitivity\n\nAll paths, including zero, use one route-complete sample. Capacities are not observed routes; participation shares are frozen assumptions. Endpoint fits are not coefficient bounds. Raw RRP slope and Omega differ once the route is nonzero.\n\n'
    text += markdown_table(r.loc[r.era.eq(modern)&r.controls.eq('calendar'),['exposure','n','beta_R','theta_RRP','raw_rrp_balance_slope','Omega']])
    text += '\n## Influence\n\nAll era/exposure/control combinations are in the thin influence table, including route paths. Blocks are contiguous 13-calendar-week deletions with a fixed anchor.\n\n'
    text += markdown_table(i.loc[i.exposure.isin(EXPOSURES)&i.controls.eq('calendar'),['era','exposure','top_week','top_week_share','top5_share','beta_without_top_week','block_beta_min','block_beta_max','sep24_2008_share']])
    text += '\n## Pooled state interaction\n\nPrior-Wednesday RRP-others per $100bn is centered within era. Its main effect, calendar controls and era intercepts and TRC slopes are present. The common pooled interaction is associational. It is not a financing-route estimate.\n\n'
    text += markdown_table(s[['era','n','state_min','state_p10','state_median','state_p90','state_max','interaction_beta_R','interaction_theta_RRP','interaction_partial_variation_share','interaction_top_week_within_era_share']])
    text += '\n## Calibration\n\nNull-imposed full-calendar Gaussian dependent wild bootstrap, Bartlett width 12 with 4 and 8 sensitivities; shared RRP/remainder vector disturbances reconstruct reserves in every draw. Scales are fixed within each restricted test. Joint-set membership uses the same restricted test at each target. '
    text += 'Size and nonzero-parameter coverage use independent simulation streams. The gate requires every 95% Wilson Monte Carlo upper size limit to be at most 7.5% and every lower coverage limit at least 90%. MC standard errors below quantify simulation uncertainty only.\n\n'
    text += markdown_table(cal[['dgp','bandwidth','simulations','size','size_mc_se','size_mc_upper','joint_set_coverage','coverage_mc_se','coverage_mc_lower','promotion_cell_pass']])
    text += f'\nReproduce with `uv run tdchf run-weekly-trc-projection --source-root SOURCE_ROOT --config {config_name}`; run `uv run pytest`. '
    text += 'The receipt pins all consumed input files, the committed configuration, thin outputs and this readout. '
    text += 'No cash-leg output is replaced. Dynamic inference and lagged-control inference are not attempted; their coefficients remain descriptive. '
    text += ('The optional finite inversion grid is a cross-section rather than interval endpoints.\n' if gate else 'No observed confidence set is published after calibration failure.\n')
    return text
