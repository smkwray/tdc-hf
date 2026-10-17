from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from tdchf import weekly_trc_projection as w
from tdchf import reserve_offsets as ro


def fixture_panel(n=180):
    rng = np.random.default_rng(17)
    rows = pd.date_range('2016-01-06', periods=n, freq='W-WED')
    p = pd.DataFrame(index=rows)
    x = rng.normal(size=n)
    for i in range(1,n):
        x[i] += .8*x[i-1]
    p['trc_core'] = x
    p['trc_par_sensitivity'] = x + .1*rng.normal(size=n)
    p['c_h'] = x - .2*rng.normal(size=n)
    p['route_forward'] = np.abs(x) + 1
    p['route_reverse'] = -np.abs(x)-.3
    for c in w.FACTORS:
        p[c] = rng.normal(size=n)*.01
    for c in w.BRIDGES:
        p[c] = rng.normal(size=n)*.02
    p['f_rrp_others'] = -.4*x + rng.normal(size=n)*.1
    p['delta_reserve_balances'] = x+p[[*w.FACTORS,*w.BRIDGES,'f_rrp_others']].sum(axis=1)
    p['rrp_level'] = 2e5-p['f_rrp_others'].cumsum()
    p['state'] = p['rrp_level'].shift(1)/ro.STATE_UNIT
    p['state_date'] = pd.Series(rows,index=rows).shift(1)
    p['lag_exposure_c'] = p['c_h'].shift(1)
    p['lag_delta_reserve_balances'] = p['delta_reserve_balances'].shift(1)
    p['era'] = w.MODERN
    return p.join(ro.calendar_flags(rows))


def test_same_mask_decomposition():
    p=fixture_panel()
    p.loc[p.index[12],'trc_par_sensitivity']=np.nan
    variants=[w.variant(p,e) for e in w.EXPOSURES]
    rows=w.common_mask(variants,'calendar',0)
    assert p.index[12] not in rows
    for v in variants:
        row,parts=w.project(v,rows,'calendar',0)
        assert row['n']==len(rows)
        assert sum(d['contribution'] for d in parts)==pytest.approx(row['beta_R']-1,abs=1e-12)
    broken=variants[0].copy()
    broken['bridge_primary_cash_price']+=.4*broken['x']
    with pytest.raises(AssertionError,match='decomposition'):
        w.project(broken,rows,'calendar',0)


def test_horizon_own_flow():
    p=fixture_panel(400); v=w.variant(p,'trc_core')
    rows=w.common_mask([v],'calendar',8)
    r0,_=w.project(v,rows,'calendar',0)
    r,parts=w.project(v,rows,'calendar',8)
    assert r['weekly_flows']==9
    assert r['subsequent_TRC']>1
    assert abs(sum(d['contribution'] for d in parts)-(r['beta_R']-1))>1
    assert r['subsequent_TRC']+sum(d['contribution'] for d in parts)==pytest.approx(r['beta_R']-1)
    assert r0['n']==r['n']


def test_rrp_sign():
    p=fixture_panel(); v=w.variant(p,'trc_core',(.25,.1))
    rows=w.common_mask([v],'calendar',0)
    r,_=w.project(v,rows,'calendar',0)
    q,_=w.control_basis(w.controls(v.loc[rows],'calendar').to_numpy())
    g=float(w.slope(v.loc[rows,'x'].to_numpy(),v.loc[rows,'gamma'].to_numpy(),q))
    assert abs(g)>1e-3
    assert r['theta_RRP']==pytest.approx(-r['raw_rrp_balance_slope']-g)
    assert r['Omega']==pytest.approx(r['raw_rrp_balance_slope']+g)
    assert r['Omega']!=pytest.approx(r['raw_rrp_balance_slope'])


def test_state_clock():
    p=fixture_panel(); rows=p.index[1:]
    z,i=w.state_design(p,rows)
    assert 'state_main_effect' in z
    assert z['state_main_effect'].mean()==pytest.approx(0,abs=1e-12)
    assert (p.loc[rows,'state_date'] < rows-pd.Timedelta(days=6)).all()
    changed=p.copy(); changed.loc[rows[10],'state_date']=rows[10]
    with pytest.raises(ValueError,match='clock'):
        w.state_design(changed,rows)
    shocked=p.copy(); shocked.loc[rows[10],'rrp_level']+=1e6
    shocked['state']=shocked['rrp_level'].shift(1)/ro.STATE_UNIT
    assert shocked.loc[rows[10],'state']==p.loc[rows[10],'state']
    assert shocked.loc[rows[11],'state']!=p.loc[rows[11],'state']


def test_endpoint_scenarios_not_bounds():
    y=np.array([0.,0.,1.]);q=np.ones((3,1))/np.sqrt(3)
    assert w.slope(np.array([0.,1.,2.]),y,q)==pytest.approx(.5)
    assert w.slope(np.array([1.,2.,3.]),y,q)==pytest.approx(.5)
    assert w.slope(np.array([1.,1.,2.]),y,q)==pytest.approx(1)
    source=Path(w.__file__).read_text()
    assert 'coefficient_bound' not in source
    config=json.loads(Path(w.CONFIG).read_text())
    assert config['route_paths']['reverse_100_stress']==[0,1]
    assert len(config['route_paths'])==9


def test_bootstrap_primitive_identity():
    rng=np.random.default_rng(71)
    x=rng.normal(size=100); factors=rng.normal(size=(100,2))
    b=w.rebuild_weekly(x,factors,35)
    np.testing.assert_allclose(b['delta'],x+factors.sum(axis=1))
    for h in [0,1,4,8]:
        for t in range(100-h):
            previous=35 if t==0 else b['reserve'][t-1]
            assert b['cumulative'][h][t]==pytest.approx(b['reserve'][t+h]-previous)
    dates=pd.date_range('2016-01-06',periods=100,freq='W-WED')
    test=w.restricted_test(x,np.ones((100,1)),factors,np.zeros(2),dates,12,99,rng)
    assert test['identity_error']<1e-12
    # Applying origin multipliers to already cumulative outcomes loses primitive overlap.
    bank=w.multiplier_bank(dates,4,1,rng)[0]
    primitive=w.rebuild_weekly(x,factors*bank[:,None])['cumulative'][4]
    wrong=w.rebuild_weekly(x,factors)['cumulative'][4]*bank
    assert not np.allclose(primitive[:-4],wrong[:-4])


def test_bootstrap_lag_consistency():
    rng=np.random.default_rng(18); x=rng.normal(size=70)
    original=w.rebuild_weekly(x,rng.normal(size=(70,2)))
    simulated=w.rebuild_weekly(x,rng.normal(size=(70,2)))
    np.testing.assert_allclose(simulated['lag_delta_reserve_balances'][1:],simulated['delta'][:-1])
    assert not np.allclose(original['lag_delta_reserve_balances'][1:],simulated['lag_delta_reserve_balances'][1:])


def test_missing_route_and_common_profile():
    p=fixture_panel();p.loc[p.index[30],'route_forward']=np.nan
    a=w.variant(p,'trc_core',(0,.1));b=w.variant(p,'trc_core',(.1,0))
    assert a.loc[p.index[30],'x']==a.loc[p.index[30],'x']
    assert pd.isna(b.loc[p.index[30],'x'])
    rows=w.common_mask([a,b],'calendar',8)
    assert not any(d in rows for d in p.index[22:31])


def test_inversion_is_same_restricted_test():
    rng=np.random.default_rng(1);n=70
    x=rng.normal(size=n);z=np.ones((n,1));y=rng.normal(size=(n,2))
    dates=pd.date_range('2018-01-03',periods=n,freq='W-WED')
    grid=[(0,0),(.3,-.2),(10,10)]
    bank=w.multiplier_bank(dates,8,99,np.random.default_rng(9))
    expected=[w.restricted_test(x,z,y,np.array(t),dates,8,99,rng,bank)['accepted'] for t in grid]
    actual=w.invert_tests(x,z,y,dates,8,99,np.random.default_rng(9),grid)
    assert actual['accepted'].tolist()==expected
    assert not actual.iloc[-1]['accepted']


def test_full_calendar_multiplier_does_not_compress_missing_weeks():
    dates=pd.date_range('2016-01-06',periods=100,freq='W-WED')
    full=w.multiplier_bank(dates,4,10,np.random.default_rng(4))
    keep=np.arange(100)%3!=1
    subset=w.multiplier_bank(dates[keep],4,10,np.random.default_rng(4))
    np.testing.assert_allclose(subset,full[:,keep])


def test_calibration_gate_requires_mc_uncertainty():
    lo,hi=w.wilson(70,1000)
    assert .07<.075<hi
    lo,hi=w.wilson(910,1000)
    assert lo<.9<.91


def materialized_projection(tmp_path, rrp_shock=None, ledger_end='2026-03-25', core_end=None, design=None):
    # Existing independent H.4.1 fixture, extended with separately accounted TRC bridges.
    from test_reserve_offsets import _materialize, _synthetic_ledger
    base=tmp_path/'source';base.mkdir(parents=True)
    ledger=_synthetic_ledger(end=ledger_end)
    if rrp_shock is not None:
        week, amount=rrp_shock
        levels=ledger.attrs['rrp_level'].copy()
        levels[ledger.index.get_loc(week)]+=amount
        factor=pd.Series(-np.diff(levels,prepend=0.),index=ledger.index)
        factor.iloc[0]=np.nan
        ledger['delta_reserve_balances']+=factor-ledger['f_rrp_others']
        ledger['f_rrp_others']=factor
        ledger['reserve_balances']=20_000.+ledger['delta_reserve_balances'].cumsum()
        ledger.attrs['rrp_level']=levels
    old=_materialize(base,ledger)
    paths={
        'data/processed/reserve_factor_ledger_weekly_v1.csv':old['ledger'],
        'data/processed/reserve_factor_ledger_v1_receipt.json':old['receipt'],
        'data/raw/fiscus/h41_ledger/rrp_others.csv':old['rrp'],
    }
    for name, original in paths.items():
        target=base/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(original.read_bytes())
    ledger=pd.read_csv(old['ledger'])
    cash=ledger['c_h']
    core=pd.DataFrame({'date':ledger.date,'c_h':cash,'trc_core':cash,'trc_par_sensitivity':cash,
                       'route_forward':100.,'route_reverse':-80.})
    if core_end is not None:
        core=core.loc[pd.to_datetime(core.date)<=core_end]
    _excluded=['f_'+r for r in w.EXCLUDED]
    non_treasury=ledger[list(w.FACTORS)].sum(axis=1)
    identity=pd.DataFrame({'date':ledger.date,'variant':'core','non_treasury_signed_factors':non_treasury})
    for b in w.BRIDGES:
        identity[b]=0.
    identity['bridge_all_securities_premium_discount']=ledger['f_unamortized_premiums']+ledger['f_unamortized_discounts']
    identity['bridge_treasury_securities_source_inflation']=ledger['f_treasury_securities']
    identity['bridge_published_rounding']=ledger['rho']
    daily=[]
    for d in pd.to_datetime(core.date):
        for day in pd.date_range(d-pd.Timedelta(days=6),d):
            daily.append({'date':day.date(),'week':d.date(),'route_forward':100/7,'route_reverse':-80/7})
    paths={
        'data/processed/trc_weekly_components_v1.csv':core,
        'data/processed/trc_weekly_identity_v1.csv':identity,
        'data/processed/trc_weekly_route_daily_v1.csv':pd.DataFrame(daily),
    }
    for name, frame in paths.items():
        frame.to_csv(base/name,index=False)
    cfg=json.loads(Path(w.CONFIG).read_text())
    receipt={'units':ro.UNITS,'producer_commit':'fixture-source','outputs':{
        key:{'path':name,'sha256':w._sha256(base/name)} for key,name in zip(
            ['weekly','identity','daily_route'],paths)}}
    (base/'data/processed/trc_weekly_v1_receipt.json').write_text(json.dumps(receipt))
    cfg['inputs']={name:w._sha256(base/name) for name in cfg['inputs']}
    cfg['bootstrap']['calibration_simulations']=2
    cfg['bootstrap']['calibration_replications']=19
    cfg['bootstrap']['dgps']=['gaussian']
    cfg['bootstrap']['bandwidths']=[12]
    cfg.update(design or {})
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    return base, config


def mock_projection_producer(tmp_path,monkeypatch,config):
    config_bytes=config.read_bytes()
    def git(command,**kwargs):
        if command[1:3]==['rev-parse','HEAD']:
            return 'fixture-producer'
        if command[1]=='status':
            return ''
        if command[1]=='show':
            return config_bytes
        raise AssertionError(command)
    monkeypatch.setattr(w.subprocess,'check_output',git)
    # Keep the integration fixture's readout in its own isolated project, which contains the config.
    monkeypatch.setattr(w,'__file__',str(tmp_path/'src/tdchf/weekly_trc_projection.py'))


def test_loader_state_shock_reaches_export_next_week(tmp_path,monkeypatch):
    week=pd.Timestamp('2018-03-14')
    following=week+pd.Timedelta(days=7)
    amount=500_000.
    source,config=materialized_projection(tmp_path/'baseline')
    shocked_source,shocked_config=materialized_projection(tmp_path/'shocked',(week,amount))
    baseline,_=w.load_inputs(source,json.loads(config.read_text()))
    shocked,_=w.load_inputs(shocked_source,json.loads(shocked_config.read_text()))
    for root,panel in ((source,baseline),(shocked_source,shocked)):
        pinned=pd.read_csv(root/'data/raw/fiscus/h41_ledger/rrp_others.csv',parse_dates=['date']).set_index('date')['value']
        rows=panel.index[panel['state_date'].notna()]
        np.testing.assert_allclose(panel.loc[rows,'state'],pinned.reindex(panel.loc[rows,'state_date']).to_numpy()/ro.STATE_UNIT)
        assert (panel.loc[rows,'state_date']==rows-pd.Timedelta(days=7)).all()
        assert (panel.loc[rows,'state_date']<rows-pd.Timedelta(days=6)).all()
    assert shocked.loc[week,'state']==baseline.loc[week,'state']
    assert shocked.loc[following,'state']==pytest.approx(baseline.loc[following,'state']+amount/ro.STATE_UNIT)
    unchanged=baseline.index.difference([following])
    pd.testing.assert_series_equal(shocked.loc[unchanged,'state'],baseline.loc[unchanged,'state'])

    # An independently placed state shock must reach the exported interaction,
    # including its support and fitted slopes, through the production entry point.
    expected=shocked.copy()
    expected['state']=baseline['state']
    expected.loc[following,'state']+=amount/ro.STATE_UNIT
    exports=[]
    for label,root,cfg in (('baseline',source,config),('shocked',shocked_source,shocked_config)):
        mock_projection_producer(tmp_path,monkeypatch,cfg)
        out=tmp_path/label/w.SCHEMA
        w.run(root,out,cfg)
        exports.append(pd.read_csv(out/'state_interaction_v1.csv'))
    pd.testing.assert_frame_equal(exports[1],w.state_table(expected),check_dtype=False)
    before,after=(table.set_index('era').loc[w.MODERN] for table in exports)
    assert after['state_max']==pytest.approx(expected.loc[expected.era.eq(w.MODERN),'state'].max())
    assert after['state_max']>before['state_max']
    for name in ('interaction_beta_R','interaction_theta_RRP'):
        assert after[name]!=pytest.approx(before[name])


def test_production_entrypoint_and_input_mutation(tmp_path,monkeypatch):
    source, config=materialized_projection(tmp_path)
    mock_projection_producer(tmp_path,monkeypatch,config)
    out=tmp_path/w.SCHEMA
    report=w.run(source,out,config,progress=lambda row: json.dumps(row))
    receipt=json.loads((out/'receipt_v1.json').read_text())
    assert report['producer_commit']=='fixture-producer'
    assert not report['promotion_passed']
    for name,sha in receipt['outputs'].items():
        assert w._sha256(out/name)==sha
    for name in ('coefficients','decomposition_h0','horizon_profile','route_paths','influence','state_interaction','calibration'):
        frame=pd.read_csv(out/(name+'_v1.csv'))
        assert not any('bound' in col.lower() for col in frame.columns if col!='endpoints_cross_era')
    before=(out/'receipt_v1.json').read_bytes()
    with pytest.raises(FileExistsError):
        w.run(source,out,config)
    assert (out/'receipt_v1.json').read_bytes()==before
    with (source/'data/processed/trc_weekly_components_v1.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError,match='sha256 mismatch'):
        w.run(source,tmp_path/'second'/w.SCHEMA,config)
    assert not (tmp_path/'second'/w.SCHEMA).exists()


def test_atomic_projection_text_failure_preserves_destination(tmp_path,monkeypatch):
    out=tmp_path/w.SCHEMA
    original=Path.write_text
    def fail(path,*args,**kwargs):
        if path.name=='definitions_v1.json':
            raise OSError('injected definitions failure')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'write_text',fail)
    with pytest.raises(OSError,match='injected'):
        ro._write_atomic(out,{'coefficients_v1':pd.DataFrame({'beta_R':[1.]})},{},
                         receipt_name='receipt_v1.json',text_files={'definitions_v1.json':'{}'})
    assert not out.exists()
    assert not list(tmp_path.iterdir())


def test_refresh_config_reruns_the_frozen_design_and_moves_only_endpoints():
    # A rerun on new data is not a new chance to pass: the route grid, horizons,
    # bandwidths, DGPs, seeds and promotion thresholds must equal the pre-fit freeze.
    v1=json.loads(Path(w.CONFIG).read_text())
    refresh=json.loads(Path('config/weekly_trc_projection_2026q2.json').read_text())
    for key in ('schema','route_paths','horizons','bootstrap','interpretation'):
        assert refresh[key]==v1[key]
    eras,modern=w.design_eras(refresh)
    assert w.design_eras(v1)==(w.ERAS,w.MODERN)
    assert eras[modern]==('2016-01-01','2026-09-30') and eras['pooled']==('2005-10-12','2026-09-30')
    unchanged=[name for name in w.ERAS if name not in ('pooled',w.MODERN)]
    assert {name:eras[name] for name in unchanged}=={name:w.ERAS[name] for name in unchanged}
    assert list(eras)[:-1]==list(w.ERAS)[:-1]
    assert 'RECEIPT.json' in refresh['inputs']


REFRESH_DESIGN={'vintage':'test','modern':'2016Q1_2026Q3','eras':{
    'pooled':['2005-10-12','2026-09-30'],'2005-10-12_2008Q3':['2005-10-12','2008-09-30'],
    '2008Q4_transition':['2008-10-01','2008-12-31'],'2009Q1_2015Q3':['2009-01-01','2015-09-30'],
    '2015Q4_transition':['2015-10-01','2015-12-31'],'2016Q1_2026Q3':['2016-01-01','2026-09-30']}}


def test_refresh_run_ends_at_the_declared_week_or_fails_closed(tmp_path,monkeypatch):
    # TRC ends 2026-09-30 while the ledger runs to 2026-10-07: the same-week sample
    # must end exactly at the declared final week, and h=8 profiles stop where
    # nine observed flows end. A source ending a week early must not pass as the refresh.
    source,config=materialized_projection(tmp_path/'full',ledger_end='2026-10-07',core_end='2026-09-30',design=REFRESH_DESIGN)
    mock_projection_producer(tmp_path,monkeypatch,config)
    out=tmp_path/'refresh'/w.SCHEMA
    w.run(source,out,config)
    receipt=json.loads((out/'receipt_v1.json').read_text())
    assert (receipt['vintage'],receipt['modern_era'],receipt['modern_final_week'])==('test','2016Q1_2026Q3','2026-09-30')
    assert receipt['candidate_scope'].startswith('2016-2026')
    coefficients=pd.read_csv(out/'coefficients_v1.csv')
    modern=coefficients.loc[coefficients.era.eq('2016Q1_2026Q3')]
    assert set(modern['last_week'])=={'2026-09-30'}
    assert '2016Q1_2025Q4' not in set(coefficients.era)
    profile=pd.read_csv(out/'horizon_profile_v1.csv')
    h8=profile.loc[profile.era.eq('2016Q1_2026Q3')&profile.h.eq(8)&profile.sample_mask.eq('matched_profile_h8')]
    assert set(h8['last_endpoint'])=={'2026-09-30'}
    assert (tmp_path/'docs/weekly_trc_projection_test.md').read_text()==(out/'readout_v1.md').read_text()

    early,early_config=materialized_projection(tmp_path/'early',ledger_end='2026-10-07',core_end='2026-09-23',design=REFRESH_DESIGN)
    mock_projection_producer(tmp_path,monkeypatch,early_config)
    with pytest.raises(ValueError,match='declared final week 2026-09-30'):
        w.run(early,tmp_path/'early_out'/w.SCHEMA,early_config)
    assert not (tmp_path/'early_out').exists()
