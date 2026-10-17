from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdchf import reserve_offsets as ro
from tdchf import weekly_reserve_history as h
from tdchf import weekly_trc_projection as w

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / h.CONFIG).read_text())


def synthetic_design(n=120, start='2016-01-06', drop=(), seed=3, beta=0.7, tail=1.0) -> h.Design:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq='W-WED').delete(list(drop))
    z = w.controls(ro.calendar_flags(dates).assign(lag_exposure_c=0.0, lag_delta_reserve_balances=0.0), 'calendar').to_numpy(float)
    x = rng.normal(size=len(dates)) * 100
    y = beta * x + z @ rng.normal(size=z.shape[1]) * 10 + rng.standard_t(4, size=len(dates)) * 30 * tail
    return h.make_design(dates, x, y, z)


def explicit_draw_statistics(d: h.Design, y: np.ndarray, bank: np.ndarray, b: float, width: int):
    """Brute force: refit the restricted control regression and its HC3/Bartlett scale in every draw."""
    weeks = d.weeks
    leverage = (d.q * d.q).sum(axis=1)

    def restricted_scale(outcome):
        e = w.residual(outcome - b * d.x, d.q) / np.maximum(1 - leverage, 1e-8)
        score = d.a * e
        total = score @ score
        for i in range(len(weeks)):
            for j in range(len(weeks)):
                lag = int(weeks[j] - weeks[i])
                if 1 <= lag <= width:
                    total += 2 * (1 - lag / (width + 1)) * score[i] * score[j]
        return math.sqrt(total)

    e_b = w.residual(y - b * d.x, d.q)
    nuisance = (y - b * d.x) - e_b
    adjusted = e_b / np.maximum(1 - leverage, 1e-8)
    observed = (w.slope(d.x, y, d.q) - b) / restricted_scale(y)
    draws = []
    for v in bank:
        y_star = b * d.x + nuisance + v * adjusted
        draws.append((w.slope(d.x, y_star, d.q) - b) / restricted_scale(y_star))
    return observed, np.array(draws)


@pytest.mark.parametrize('width', [3, 12])
def test_closed_form_equals_an_explicit_per_draw_restricted_refit(width):
    """The bootstrap must recompute the restricted fit and scale in every draw; the closed form is only a fast path."""
    d = synthetic_design(n=70, drop=(9, 10, 40))
    rng = np.random.default_rng(5)
    bank = w.multiplier_bank(d.dates, width, 25, rng)
    pt = h.parts(d, d.y, bank, h.layout(d.weeks, width))
    for b in (-0.4, 0.0, 0.7, 2.5):
        observed, draws = explicit_draw_statistics(d, d.y, bank, b, width)
        t_obs, t_draw = h.statistics(pt, np.array(b - pt.beta))
        assert t_obs == pytest.approx(observed, rel=1e-9)
        assert t_draw == pytest.approx(draws, rel=1e-8, abs=1e-10)


def test_tests_depend_on_y_only_through_y_minus_bx():
    """Location invariance justifies evaluating calibration at each cell's exact target."""
    d = synthetic_design()
    bank = w.multiplier_bank(d.dates, 12, 199, np.random.default_rng(1))
    lay = h.layout(d.weeks, 12)
    base = h.p_values(h.parts(d, d.y, bank, lay), np.array([0.2, 0.7]))
    shifted = h.p_values(h.parts(d, d.y + 0.9 * d.x, bank, lay), np.array([1.1, 1.6]))
    assert shifted == pytest.approx(base)


def constant_parts(beta: float, observed: list, draw_stats) -> h.Parts:
    """Draws with constant T* = draw_stats (Q* = 1, c_x = 0); T(b) = -(b - beta)/sqrt(Q_observed(b - beta))."""
    draw_stats = np.asarray(draw_stats, float)
    unit = np.tile(np.array([[1.0, 0.0], [0.0, 0.0]]), (len(draw_stats), 1, 1))
    return h.Parts(beta=beta, observed=np.array(observed, float), c_e=draw_stats, c_x=np.zeros(len(draw_stats)), draws=unit)


def test_p_values_count_draws_with_the_plus_one_convention():
    pt = constant_parts(1.0, [[1.0, 0.0], [0.0, 0.0]], [-3.0, -0.5, 0.5, 2.0])
    two, up, lo = h.p_values(pt, np.array([0.0]))[:, 0]
    # T = 1; |T*| >= 1 for -3 and 2; T* >= 1 for 2; T* <= 1 for -3, -0.5, 0.5.
    assert (two, up, lo) == pytest.approx((3 / 5, 2 / 5, 4 / 5))


def test_exact_inversion_matches_the_test_everywhere_and_ends_at_crossings():
    d = synthetic_design(n=200)
    pt = h.parts(d, d.y, w.multiplier_bank(d.dates, 12, 999, np.random.default_rng(2)), h.layout(d.weeks, 12))
    cs = h.acceptance_set(pt, 0.05)
    assert len(cs['segments']) == 1 and not cs['unbounded_below'] and not cs['unbounded_above']
    low, high = cs['segments'][0]
    assert low < pt.beta < high
    probes = np.random.default_rng(0).uniform(low - 1, high + 1, 400)
    inside = np.array([low <= b <= high for b in probes])
    assert ((h.p_values(pt, probes)[0] > 0.05) == inside).all()
    for edge, inward in ((low, 1), (high, -1)):
        assert h.p_values(pt, np.array([edge + inward * 1e-7]))[0, 0] > 0.05
        assert h.p_values(pt, np.array([edge - inward * 1e-7]))[0, 0] <= 0.05


def test_components_far_away_narrow_disconnected_or_huge_are_all_recovered():
    signs = np.where(np.arange(50) % 2, 1.0, -1.0)
    wide = h.acceptance_set(constant_parts(0.0, [[1, 0], [0, 0]], 3 * signs), 0.05)
    assert wide['segments'] == [pytest.approx((-3.0, 3.0))]
    flipped = h.acceptance_set(constant_parts(0.0, [[-1, 0], [0, 0]], 3 * signs), 0.05)
    assert flipped['segments'] == [pytest.approx((-3.0, 3.0))]  # QR may return a negative diagonal
    narrow = h.acceptance_set(constant_parts(20.0, [[1, 0], [0, 0]], 0.0005 * signs), 0.05)
    assert narrow['segments'] == [pytest.approx((19.9995, 20.0005), abs=1e-9)]
    split = h.acceptance_set(constant_parts(0.0, [[1, 1.2], [0, 0.1]], signs), 0.05)
    root = math.sqrt(5.76 - 1.8)
    assert split['segments'] == [pytest.approx((-math.inf, (2.4 - root) / 0.9)), pytest.approx(((2.4 + root) / 0.9, math.inf))]
    assert split['unbounded_below'] and split['unbounded_above'] and h.describe_set(split)['set_components'] == 2
    k = 2e9 * math.sqrt(1 / 1.4)
    huge = h.acceptance_set(constant_parts(0.0, [[1, 0], [0, math.sqrt(1e-19)]], k * signs), 0.05)
    assert huge['segments'] == [pytest.approx((-2e9, 2e9), rel=1e-6)] and h.describe_set(huge)['set_bounded']
    everything = h.acceptance_set(constant_parts(0.0, [[1, 0], [0, 1]], 5 * signs), 0.05)
    assert everything['segments'] == [(-math.inf, math.inf)]


def test_isolated_points_are_not_reported_but_narrow_components_are():
    """Tangency at b = 1/2 is a probability-zero point below the declared resolution; [-1/8, 1/4] is kept."""
    d00 = math.sqrt(45 / 128)
    d01 = (9 / 16) / d00
    draws = np.tile(np.array([[d00, d01], [0.0, math.sqrt(1 - d01 ** 2)]]), (99, 1, 1))
    pt = h.Parts(beta=0.0, observed=np.array([[1.0, 0.0], [0.0, 1.0]]), c_e=np.full(99, 1 / math.sqrt(128)),
                 c_x=np.zeros(99), draws=draws)
    assert h.acceptance_set(pt, 0.05)['segments'] == [pytest.approx((-0.125, 0.25))]


def test_near_double_crossings_returned_as_complex_pairs_are_still_found():
    """Round-3 review case: np.roots returns the two real crossings near 0.513 as a complex pair."""
    draws = np.tile(np.array([[2.5501786772118358, 13.722010390045629], [0.0, 3.109777300054965]]), (99, 1, 1))
    pt = h.Parts(beta=0.0, observed=np.diag([1.0, math.sqrt(10)]), c_e=np.ones(99), c_x=np.full(99, 4.449317745457804),
                 draws=draws)
    cs = h.acceptance_set(pt, 0.05)
    narrow = [seg for seg in cs['segments'] if seg[0] > 0.4]
    assert len(narrow) == 1 and narrow[0][0] < 0.513 < narrow[0][1] and narrow[0][1] - narrow[0][0] < 1e-3
    for low, high in cs['segments']:
        # Leading terms cancel exactly in this case, so acceptance beyond ~1e6 SE is decided by rounding.
        low, high = (v if abs(v) < 1e6 else math.inf * np.sign(v) for v in (low, high))
        if math.isfinite(low):
            assert h.p_values(pt, np.array([low + 1e-9 * max(1, abs(low))]))[0, 0] > 0.05
            assert h.p_values(pt, np.array([low - 1e-8 * max(1, abs(low))]))[0, 0] <= 0.05
        if math.isfinite(high):
            assert h.p_values(pt, np.array([high - 1e-9 * max(1, abs(high))]))[0, 0] > 0.05
            assert h.p_values(pt, np.array([high + 1e-8 * max(1, abs(high))]))[0, 0] <= 0.05


def test_resolution_is_absolute_in_standard_errors_away_from_the_estimate():
    """Round-4 review: a 5e-8 SE component at 10 SE is above the declared 1e-8 SE resolution and must be kept."""
    eps, delta = 2.5e-9, 1e-14
    draws = np.tile(np.array([[1.0, 0.1], [0.0, delta]]), (999, 1, 1))
    pt = h.Parts(beta=0.0, observed=np.eye(2), c_e=np.full(999, eps), c_x=np.zeros(999), draws=draws)
    cs = h.acceptance_set(pt, 0.05)
    far = [seg for seg in cs['segments'] if seg[0] > 1]
    assert len(far) == 1 and far[0][0] < 10 < far[0][1] and 1e-8 < far[0][1] - far[0][0] < 1e-7
    assert any(a <= 0 <= b for a, b in cs['segments'])  # the estimate's own sub-resolution component is kept
    low, high = far[0]
    assert h.p_values(pt, np.array([(low + high) / 2]))[0, 0] > 0.05
    assert (h.p_values(pt, np.array([low - 1e-9, high + 1e-9]))[0] <= 0.05).all()


def test_nearly_tied_draws_do_not_create_false_components():
    """Round-3 review case: a draw at 1 - 5e-10 must not be promoted to a tie at the crossing b = 1."""
    c_e = np.zeros(99)
    c_e[:5] = [2, 2, 2, 1, 1 - 5e-10]
    pt = h.Parts(beta=0.0, observed=np.eye(2), c_e=c_e, c_x=np.zeros(99), draws=np.tile(np.eye(2), (99, 1, 1)))
    (low, high), = h.acceptance_set(pt, 0.05)['segments']
    assert h.p_values(pt, np.array([1.0]))[0, 0] == pytest.approx(0.05) and low < 0 < high


@pytest.mark.parametrize('noise', [1e-2, 1e-8])
def test_near_exact_fits_invert_and_test_exactly(noise):
    """Near-exact fits: p-values equal an explicit refit and the inverted set equals the test everywhere."""
    rng = np.random.default_rng(7)
    d = synthetic_design(n=80)
    y = d.x + noise * rng.standard_normal(d.n)
    bank = w.multiplier_bank(d.dates, 12, 199, np.random.default_rng(8))
    pt = h.parts(d, y, bank, h.layout(d.weeks, 12))
    scale = float(pt.observed[0, 0])
    for b in (pt.beta, pt.beta + 0.7 * scale, pt.beta - 3 * scale):
        observed, draws = explicit_draw_statistics(d, y, bank, b, 12)
        direct = (1 + (np.abs(draws) >= abs(observed)).sum()) / (len(draws) + 1)
        assert h.p_values(pt, np.array([b]))[0, 0] == pytest.approx(direct)
    cs = h.acceptance_set(pt, 0.05)
    probes = pt.beta + scale * np.random.default_rng(1).uniform(-40, 40, 2000)
    member = np.array([any(a <= b <= c for a, c in cs['segments']) for b in probes])
    assert ((h.p_values(pt, probes)[0] > 0.05) == member).all()


def test_exact_fits_are_refused():
    d = synthetic_design(n=80)
    bank = w.multiplier_bank(d.dates, 12, 19, np.random.default_rng(8))
    with pytest.raises(ValueError, match='unresolved'):
        h.parts(d, 0.5 * d.x, bank, h.layout(d.weeks, 12))


def test_saturated_designs_are_refused():
    with pytest.raises(ValueError, match='saturated'):
        synthetic_design(n=30)


def test_calibration_status_requires_every_declared_cell_and_both_tails():
    rows = [{'period': 'p', 'width': 12, 'cell': c, 'two_sided_pass': True, 'one_sided_pass': c != 1,
             'coverage_wilson_lower': 0.95, 'upper_wilson_upper': 0.05, 'lower_wilson_upper': 0.07 if c == 1 else 0.05}
            for c in range(3)]
    status = h.calibration_status(pd.DataFrame(rows), {('p', 12, c) for c in range(3)}).iloc[0]
    assert status['interval_calibrated'] and not status['one_sided_calibrated'] and status['max_lower_tail_upper'] == 0.07
    with pytest.raises(ValueError, match='declared'):
        h.calibration_status(pd.DataFrame(rows[:2]), {('p', 12, c) for c in range(3)})
    with pytest.raises(ValueError, match='declared'):
        h.calibration_status(pd.DataFrame([*rows, rows[1]]), {('p', 12, c) for c in range(3)})


def test_declared_cells_and_periods():
    families = pd.Series([c['family'] for c in h.cells(CONFIG, 'post_liftoff')]).value_counts().to_dict()
    assert families == {'baseline': 16, 'misspecification': 12, 'crisis': 8, 'differenced': 4}
    assert len(h.cells(CONFIG, 'pre_strains')) == 32
    spans = h.periods(CONFIG)
    assert spans['scarce_reserves'][1] == pd.Timestamp('2008-09-10') and spans['transition_2008'][0] == pd.Timestamp('2008-09-17')
    broken = copy.deepcopy(CONFIG)
    broken['eras']['floor_2008_2015'][0] = '2008-12-31'
    with pytest.raises(ValueError, match='consecutive'):
        h.periods(broken)
    broken = copy.deepcopy(CONFIG)
    broken['nested']['pre_strains'] = ['2007-08-08', '2009-01-07']
    with pytest.raises(ValueError, match='inside one main period'):
        h.periods(broken)


def test_error_processes_follow_their_declared_laws():
    d = synthetic_design(n=60, drop=(20,))
    rng_a, rng_b = np.random.default_rng(9), np.random.default_rng(9)
    cell = {'ar1': 0.85, 'innovation': 'gaussian'}
    e = h.cell_errors(rng_a, d, cell, 0.0, 4, burn_in=5)
    u = rng_b.standard_normal((int(d.weeks[-1] - d.weeks[0]) + 1 + 5, 4))
    manual = np.empty_like(u)
    manual[0] = u[0]
    for t in range(1, len(u)):
        manual[t] = 0.85 * manual[t - 1] + math.sqrt(1 - 0.85 ** 2) * u[t]
    assert e == pytest.approx(manual[5:][d.weeks - d.weeks[0]])
    big = synthetic_design(n=400)
    for law in ({'ma1': -0.8, 'innovation': 't4'}, {'ar1': -0.8, 'innovation': 't4'}):
        draws = h.cell_errors(np.random.default_rng(4), big, law, 0.0, 400)
        assert draws.var() == pytest.approx(1.0, abs=0.08)


def test_cell_means_scales_and_exact_targets():
    d = synthetic_design(n=104)
    drift = h.cell_mean(d, {'mean': 'slope_drift'}, CONFIG) / d.x
    assert (drift[0], drift[-1]) == pytest.approx((-0.5, 0.5))
    step = h.cell_mean(d, {'mean': 'midpoint_step'}, CONFIG) / d.x
    assert set(np.round(step, 12)) == {0.0, 0.5}
    crisis = h.cell_scale(d, {'variance': 'crisis'}, 2.0, [['2016-03-01', '2016-03-31']])
    assert set(crisis) == {10.0, 1.2}
    concentrated = h.cell_scale(d, {'variance': 'concentrated'}, 1.0, [])
    assert (concentrated ** 2).mean() == pytest.approx(5.0)


def test_calibration_cell_covers_near_nominal_under_iid_errors_and_is_reproducible():
    d = synthetic_design(n=260, seed=11)
    cfg = copy.deepcopy(CONFIG['calibration']) | {'outer': 300, 'inner': 199}
    cfg['precision_subset'] = cfg['precision_subset'] | {'simulations': 20}
    task = {'design': d, 'cell': {'cell': 0, 'family': 'baseline', 'ar1': 0.0, 'innovation': 'gaussian', 'variance': 'constant',
                                  'mean': 'none'}, 'calibration': cfg, 'width': 12, 'period': 'synthetic', 'period_index': 0,
            'alpha': 0.05, 'episodes': []}
    row = h.run_cell(task)
    assert 0.9 <= row['coverage'] <= 0.99 and row['upper_rejection'] <= 0.1 and row['lower_rejection'] <= 0.1
    assert row['precision_simulations'] == 20 and row['median_length_bounded'] > 0 and row['share_unbounded'] < 0.5
    assert row['rejection_null_offset_+0.50'] >= row['rejection_null_offset_+0.25'] - 0.1
    assert h.run_cell(task) == row


def fixture_panel() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows = pd.date_range('2005-10-12', '2026-09-30', freq='W-WED')
    n = len(rows)
    p = pd.DataFrame(index=rows)
    x = rng.normal(size=n)
    p['trc_core'] = x
    p['trc_par_sensitivity'] = x + .1 * rng.normal(size=n)
    p['c_h'] = x - .2 * rng.normal(size=n)
    p['route_forward'] = np.abs(x) + 1
    p['route_reverse'] = -np.abs(x) - .3
    for c in w.FACTORS:
        p[c] = rng.normal(size=n) * .01
    for c in w.BRIDGES:
        p[c] = rng.normal(size=n) * .02
    p['f_rrp_others'] = -.4 * x + rng.normal(size=n) * .3
    p['delta_reserve_balances'] = x + p[[*w.FACTORS, *w.BRIDGES, 'f_rrp_others']].sum(axis=1)
    p['lag_exposure_c'] = p['c_h'].shift(1)
    p['lag_delta_reserve_balances'] = p['delta_reserve_balances'].shift(1)
    return p.join(ro.calendar_flags(rows))


def test_produce_reports_every_declared_period_width_and_cell(tmp_path):
    config = copy.deepcopy(CONFIG)
    config['calibration'] |= {'outer': 8, 'inner': 19}
    config['calibration']['precision_subset'] |= {'simulations': 2}
    config['inference'] |= {'draws': 99}
    tables = h.produce(fixture_panel(), config, workers=2)
    single = h.produce(fixture_panel(), config, workers=1)
    for name, table in tables.items():
        pd.testing.assert_frame_equal(table, single[name])
    widths = {12, 26}
    assert set(zip(tables['inference']['period'], tables['inference']['width'])) == {
        (p, wd) for p in config['inferential_periods'] for wd in widths}
    status = tables['calibration_status'].set_index(['period', 'width'])['cells'].to_dict()
    assert status == {(p, wd): (32 if p == 'pre_strains' else 40) for p in config['inferential_periods'] for wd in widths}
    coefficients = tables['period_coefficients']
    assert set(coefficients['period']) == set(h.periods(config)) and set(coefficients['exposure']) == set(w.EXPOSURES)
    pooled = coefficients.loc[coefficients['period'].eq('pooled') & coefficients['exposure'].eq('trc_core')].iloc[0]
    assert pooled['first_week'] == '2005-10-12' and pooled['last_week'] == '2026-09-30'
    # Small calibration cannot qualify an interval; uncalibrated sets are never labelled calibrated.
    for _, row in tables['inference'].iterrows():
        assert row['interval_status'] == ('calibrated' if row['interval_calibrated'] else 'failed_calibration_descriptive_only')
        assert not row['equivalence_established'] or row['one_sided_calibrated']
        segments = tables['confidence_sets'].query('period == @row.period and width == @row.width')
        assert len(segments) == row['segments'] and set(segments['interval_status']) == {row['interval_status']}
        assert any(a <= row['beta'] <= b for a, b in zip(segments['lower'], segments['upper']))


def test_newey_west_is_the_explicit_bartlett_sandwich_with_calendar_gaps():
    d = synthetic_design(n=90, drop=(10, 11, 50))
    width = 12
    scores = d.a * h.full_residual(d, d.y)
    total = 0.0
    for i, wi in enumerate(d.weeks):
        for j, wj in enumerate(d.weeks):
            total += max(1 - abs(int(wi - wj)) / (width + 1), 0) * scores[i] * scores[j]
    assert h.newey_west(d, width) == pytest.approx(math.sqrt(total * d.n / (d.n - d.rank - 1)), rel=1e-10)


def test_rolling_windows_are_calendar_spans_and_the_full_span_is_the_pooled_regression():
    p = fixture_panel()
    config = json.loads((ROOT / h.ROLLING_CONFIG).read_text())
    table = h.rolling(p, config)
    reported = table[table['status'].eq('reported')]
    assert (pd.to_datetime(table['window_end']) - pd.to_datetime(table['window_start'])).dt.days.eq(7 * 103).all()
    assert reported['observed_weeks'].eq(104).all() and len(table) == len(p) - 103
    full = h.rolling(p, config | {'window_weeks': len(p)}).iloc[0]
    row, _ = w.project(w.variant(p, 'trc_core'), w.common_mask([w.variant(p, e) for e in w.EXPOSURES], 'calendar', 0), 'calendar', 0)
    assert full['beta'] == pytest.approx(row['beta_R'], rel=1e-10)
    gappy = p.drop(p.index[200:220])
    statuses = h.rolling(gappy, config).set_index('window_end')['status']
    assert (statuses == 'too_few_observed_weeks').any()


def monthly_frame(n=120, start='2010-01', missing=()):
    rng = np.random.default_rng(21)
    periods = pd.period_range(start, periods=n, freq='M')
    x = rng.normal(0, 50, n)
    frame = pd.DataFrame({'month': periods.astype(str), 'trc_core': x, 'trc_preferred': x + rng.normal(0, 5, n),
                          'delta_reserve_balances': 0.8 * x + rng.normal(0, 20, n)}, index=periods)
    frame.loc[periods[list(missing)], ['trc_core', 'trc_preferred']] = np.nan
    return frame


def test_horizon_sums_cover_consecutive_present_months_only():
    frame = monthly_frame(missing=(10,))
    three = h.horizon_sample(frame, 'trc_preferred', 3)
    assert frame.index[8] not in three.index and frame.index[10] not in three.index and frame.index[11] in three.index
    t = frame.index[20]
    assert three.loc[t, 'x'] == pytest.approx(frame['trc_preferred'].iloc[20:23].sum())
    assert three.loc[t, 'y'] == pytest.approx(frame['delta_reserve_balances'].iloc[20:23].sum())
    assert (three['end_month'] - three['start_month']).apply(lambda d: d.n).eq(2).all()


def test_monthly_design_is_ols_with_month_dummies_and_month_distance_hac():
    frame = monthly_frame(missing=(30, 31))
    sample = h.horizon_sample(frame, 'trc_core', 1)
    d = h.monthly_design(sample, min_df=20)
    months = sample['start_month'].dt.month.to_numpy()
    z = np.column_stack([np.ones(len(sample)), *[(months == m).astype(float) for m in range(2, 13)], sample['x']])
    assert float(d.a @ d.y) == pytest.approx(np.linalg.lstsq(z, sample['y'].to_numpy(), rcond=None)[0][-1], rel=1e-10)
    width, scores = 3, d.a * h.full_residual(d, d.y)
    ordinals = (sample['start_month'].dt.year * 12 + sample['start_month'].dt.month).to_numpy()
    total = sum(max(1 - abs(int(a - b)) / (width + 1), 0) * scores[i] * scores[j]
                for i, a in enumerate(ordinals) for j, b in enumerate(ordinals))
    assert h.newey_west(d, width) == pytest.approx(math.sqrt(total * d.n / (d.n - d.rank - 1)), rel=1e-10)


def test_monthly_windows_keep_every_summed_month_inside_the_window():
    config = json.loads((ROOT / h.MONTHLY_CONFIG).read_text())
    table, pooled = h.monthly_rolling(monthly_frame(), config)
    for _, row in table[table['status'].eq('reported')].iterrows():
        first, last = pd.Period(row['window_start'], 'M'), pd.Period(row['window_end'], 'M')
        assert (last - first).n == 47
        assert row['n'] == 48 - (row['horizon_months'] - 1)
    assert set(zip(pooled['exposure'], pooled['horizon_months'])) == {(e, m) for e in ('trc_preferred', 'trc_core') for m in (1, 3)}


def test_monthly_inputs_are_pinned_and_consecutive(tmp_path):
    frame = monthly_frame().reset_index(drop=True)
    frame.to_csv(tmp_path / 'trc_monthly_v1.csv', index=False)
    (tmp_path / 'trc_monthly_v1_receipt.json').write_text('{}')
    import hashlib
    pin = {name: {'name': name, 'sha256': hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
           for name in ('trc_monthly_v1.csv', 'trc_monthly_v1_receipt.json')}
    config = {'source': {'table': pin['trc_monthly_v1.csv'], 'receipt': pin['trc_monthly_v1_receipt.json']}}
    assert len(h.load_monthly(tmp_path, config)) == len(frame)
    with pytest.raises(ValueError, match='sha256'):
        h.load_monthly(tmp_path, {'source': {**config['source'], 'table': {**pin['trc_monthly_v1.csv'], 'sha256': '0' * 64}}})
    frame.drop(index=5).to_csv(tmp_path / 'trc_monthly_v1.csv', index=False)
    config['source']['table']['sha256'] = hashlib.sha256((tmp_path / 'trc_monthly_v1.csv').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='consecutive'):
        h.load_monthly(tmp_path, config)


def test_exposures_share_one_sample_and_calendar_quarter_rows_bridge_to_quarters():
    """Review round 1: corrected and uncorrected coefficients must cover the same months."""
    frame = monthly_frame()
    frame.loc[frame.index[-3:], 'trc_preferred'] = np.nan
    config = json.loads((ROOT / h.MONTHLY_CONFIG).read_text())
    _, pooled = h.monthly_rolling(frame, config)
    ends = pooled[pooled['starts'].eq('every month')].groupby('horizon_months')['last_end'].nunique()
    assert ends.eq(1).all()
    quarters = pooled[pooled['starts'].eq('calendar quarters')]
    assert set(quarters['exposure']) == {'trc_preferred', 'trc_core'} and quarters['horizon_months'].eq(3).all()
    sample = h.horizon_sample(h.common_months(frame, config['exposures']), 'trc_core', 3)
    q = sample[sample['start_month'].dt.month.isin([1, 4, 7, 10])]
    months = q['start_month'].dt.month.to_numpy()
    z = np.column_stack([np.ones(len(q)), *[(months == m).astype(float) for m in (4, 7, 10)], q['x']])
    expected = np.linalg.lstsq(z, q['y'].to_numpy(), rcond=None)[0][-1]
    assert quarters.set_index('exposure').loc['trc_core', 'beta'] == pytest.approx(expected, rel=1e-10)
