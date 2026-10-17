from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tdchf.reserve_offsets as ro
from tdchf.cli import main as cli_main

FACTORS = [row for rows in ro.FACTOR_GROUPS.values() for row in rows]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_ledger(end: str = "2026-03-25", seed: int = 7) -> pd.DataFrame:
    """Signed H.4.1-style ledger whose factors and rho sum to the reserve change."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2002-12-18", end, freq="W-WED")
    n = len(dates)
    cash = rng.normal(0.0, 1.0, n) * np.where(dates.year >= 2009, 40_000.0, 5_000.0)
    rrp_level = np.where(dates.year >= 2014, 150_000.0 + np.cumsum(rng.normal(0, 20_000, n)).clip(-150_000), 0.0)
    rrp_level = np.maximum(rrp_level, 0.0)
    frame = pd.DataFrame(index=dates)
    for row in FACTORS:
        frame[f"f_{row}"] = 0.0
    frame["f_treasury_general_account"] = cash
    frame["f_rrp_others"] = -np.diff(rrp_level, prepend=0.0)
    frame["f_currency_in_circulation"] = rng.normal(-500, 800, n)
    frame["f_treasury_securities"] = rng.normal(1_000, 3_000, n) - 0.1 * cash
    frame["f_loans"] = rng.normal(0, 400, n)
    frame["f_foreign_official_deposits"] = rng.normal(0, 300, n)
    frame["rho"] = rng.integers(-2, 3, n).astype(float)
    frame["delta_reserve_balances"] = frame[[f"f_{r}" for r in FACTORS]].sum(axis=1) + frame["rho"]
    frame["c_h"] = frame["f_treasury_general_account"] + frame["f_supplementary_financing_account"]
    frame["reserve_balances"] = 20_000.0 + frame["delta_reserve_balances"].cumsum()
    frame["memo_f_treasury_inflation_compensation"] = 0.0
    frame["ledger_status"] = "closed_within_rounding_bound"
    change_cols = ["delta_reserve_balances", "c_h", "rho", "memo_f_treasury_inflation_compensation", *[f"f_{r}" for r in FACTORS]]
    frame.iloc[0, [frame.columns.get_loc(c) for c in change_cols]] = np.nan
    frame.iloc[0, frame.columns.get_loc("ledger_status")] = "unavailable_no_prior_wednesday"
    frame.attrs["rrp_level"] = rrp_level
    frame.attrs["tga_level"] = 300_000.0 - np.nan_to_num(frame["f_treasury_general_account"].to_numpy()).cumsum()
    return frame


EPISODES_YML = """schema_version: debt_limit_episodes_v1
episodes:
  - episode_id: "2013a"
    extraordinary_measures_start: 2012-12-31
    extraordinary_measures_source: {title: "Letter A", issued: 2012-12-26, url: "https://example.org/a"}
    resolution: 2013-02-04
    resolution_type: suspension
    resolution_source: {title: "Act A", issued: 2013-02-04, url: "https://example.org/b"}
  - episode_id: "2013b"
    extraordinary_measures_start: 2013-03-20
    extraordinary_measures_source: {title: "Letter B", issued: 2013-03-20, url: "https://example.org/c"}
    resolution: 2013-10-17
    resolution_type: suspension
    resolution_source: {title: "Act B", issued: 2013-10-17, url: "https://example.org/d"}
"""


def _materialize(tmp_path: Path, ledger: pd.DataFrame | None = None) -> dict[str, object]:
    ledger = _synthetic_ledger() if ledger is None else ledger
    paths = {
        "ledger": tmp_path / "ledger.csv",
        "receipt": tmp_path / "ledger_receipt.json",
        "rrp": tmp_path / "rrp_others.csv",
        "tga": tmp_path / "treasury_general_account.csv",
        "episodes": tmp_path / "episodes.yml",
        "out": tmp_path / "out",
    }
    out = ledger.reset_index(names="date")
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(paths["ledger"], index=False)
    for key, level in (("rrp", ledger.attrs["rrp_level"]), ("tga", ledger.attrs["tga_level"])):
        pd.DataFrame({"date": out["date"], "value": level}).to_csv(paths[key], index=False)
    paths["episodes"].write_text(EPISODES_YML, encoding="utf-8")
    receipt = {
        "schema_version": ro.LEDGER_SCHEMA,
        "units": ro.UNITS,
        "producer_commit": "fixture",
        "outputs": {"weekly": {"path": "ledger.csv", "sha256": _sha(paths["ledger"])}},
        "inputs": {
            "fiscus_inputs": {
                "rrp_others": {"sha256": _sha(paths["rrp"])},
                "treasury_general_account": {"sha256": _sha(paths["tga"])},
            }
        },
    }
    paths["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    return paths


def _panel(paths: dict[str, Path]) -> pd.DataFrame:
    ledger, receipt = ro.load_ledger(
        paths["ledger"], paths["receipt"], ledger_sha256=_sha(paths["ledger"]), receipt_sha256=_sha(paths["receipt"])
    )
    rrp = ro.load_level(paths["rrp"], receipt, "rrp_others", ledger)
    tga = ro.load_level(paths["tga"], receipt, "treasury_general_account", ledger)
    return ro.build_weekly_panel(ledger, rrp, tga)


@pytest.fixture(scope="module")
def fixture_panel(tmp_path_factory) -> pd.DataFrame:
    return _panel(_materialize(tmp_path_factory.mktemp("ledger")))


def test_state_is_observed_strictly_before_the_exposure_week(fixture_panel) -> None:
    # The buffer state must not contain the exposure week's own RRP movement:
    # a same-week balance would condition on part of the outcome.
    panel = fixture_panel
    timed = panel["state_date"].notna()
    assert (panel.loc[timed, "state_date"] < panel.loc[timed, "exposure_start"]).all()
    assert (panel.loc[timed, "exposure_start"] - panel.loc[timed, "state_date"] == pd.Timedelta(days=1)).all()
    week = pd.Timestamp("2021-06-02")
    assert panel.loc[week, "state_rrp_others_100bn"] == panel.loc[week - pd.Timedelta(days=7), "rrp_others_level"] / 1e5

    shocked = panel["rrp_others_level"].copy()
    shocked.loc[shocked.index >= week] += 5e5
    ledger = pd.DataFrame(index=panel.index)
    for col in ("c_h", "reserve_balances", "delta_reserve_balances", "rho"):
        ledger[col] = panel["exposure_c"] if col == "c_h" else panel["contrib_rho"] if col == "rho" else panel[col]
    for row in FACTORS:
        ledger[f"f_{row}"] = panel[f"f_{row}"]
    rebuilt = ro.build_weekly_panel(ledger, shocked, panel["tga_level"])
    assert rebuilt.loc[week, "state_rrp_others_100bn"] == panel.loc[week, "state_rrp_others_100bn"]
    assert rebuilt.loc[week + pd.Timedelta(days=7), "state_rrp_others_100bn"] == pytest.approx(
        panel.loc[week + pd.Timedelta(days=7), "state_rrp_others_100bn"] + 5.0
    )


def test_tax_weeks_come_from_typed_categories_not_a_generic_date_set(fixture_panel) -> None:
    # IR-2021-59 moved only the individual annual payment to May 17, 2021;
    # first-quarter estimates (individual and corporate) stayed due April 15.
    flags = fixture_panel
    april, may = pd.Timestamp("2021-04-21"), pd.Timestamp("2021-05-19")
    assert flags.loc[april, "tax_individual_estimated_week"] == 1.0
    assert flags.loc[april, "tax_individual_annual_week"] == 0.0
    assert flags.loc[april, "tax_corporate_estimated_week"] == 1.0
    assert flags.loc[may, "tax_individual_annual_week"] == 1.0
    assert flags.loc[may, "tax_individual_estimated_week"] == 0.0
    # Pub. L. 114-41: corporate annual payments move from March (tax year 2015) to April (tax year 2016).
    assert flags.loc[pd.Timestamp("2016-03-16"), "tax_corporate_annual_week"] == 1.0
    assert flags.loc[pd.Timestamp("2017-03-15"), "tax_corporate_annual_week"] == 0.0
    assert flags.loc[pd.Timestamp("2017-04-19"), "tax_corporate_annual_week"] == 1.0

    events = ro.tax_event_windows(fixture_panel)
    annual_2021 = events.loc[events["category"].eq("individual_annual") & events["event_week"].str.startswith("2021")]
    assert annual_2021["event_week"].tolist() == ["2021-05-19"]
    # Notice 2020-23 moved two individual installments to July 15, 2020: one event, two deadlines.
    july = events.loc[events["category"].eq("individual_estimated") & events["event_week"].eq("2020-07-15")]
    assert july["n_deadlines_in_event"].tolist() == [2]
    assert not events.duplicated(["category", "event_week"]).any()


def test_cumulative_outcomes_use_only_observed_endpoints(tmp_path) -> None:
    # A horizon whose endpoint is unpublished cannot enter any horizon's row set.
    panel = _panel(_materialize(tmp_path, _synthetic_ledger(end="2025-11-26")))
    week = pd.Timestamp("2024-03-06")
    for h in ro.HORIZONS:
        expected = panel.loc[week + pd.Timedelta(weeks=h), "reserve_balances"] - panel.loc[week - pd.Timedelta(weeks=1), "reserve_balances"]
        assert ro.cumulative(panel["delta_reserve_balances"], h).loc[week] == pytest.approx(expected)
    assert np.isnan(ro.cumulative(panel["delta_reserve_balances"], 8).loc[pd.Timestamp("2025-10-08")])
    rows, counts = ro.sample_rows(panel, "full")
    assert counts["n_dropped_unobserved_endpoint"] == 8
    assert rows.max() == pd.Timestamp("2025-10-01")
    estimates = ro.estimate_all(panel)
    full = estimates.loc[estimates["sample"].eq("full") & estimates["specification"].eq("baseline")]
    assert set(full["n"]) == {len(rows)}


def test_overlapping_event_windows_are_explicit_and_counted(fixture_panel) -> None:
    events = ro.tax_event_windows(fixture_panel)
    april = events.loc[events["category"].eq("individual_estimated") & events["event_week"].eq("2021-04-21")].iloc[0]
    assert april["other_deadline_weeks_in_window"] == "+4:individual_annual"
    assert april["codated_other_categories"] == "corporate_annual;corporate_estimated"
    assert not april["no_neighboring_deadline_weeks"]

    paths = ro.tax_event_paths(fixture_panel, events)
    cell = paths.loc[
        paths["category"].eq("individual_estimated") & paths["era"].eq("full") & paths["series"].eq("delta_reserve_balances")
    ]
    est = events.loc[events["category"].eq("individual_estimated") & events["window_complete"]]
    all_events = cell.loc[cell["variant"].eq("all_events")].set_index("event_week")
    isolated = cell.loc[cell["variant"].eq("no_neighboring_deadline_weeks")].set_index("event_week")
    assert set(all_events["n_events"]) == {len(est)}
    assert set(isolated["n_events"]) == {int(est["no_neighboring_deadline_weeks"].sum())}
    assert set(all_events["n_events_with_any_other_deadline_week"]) == {int((~est["no_neighboring_deadline_weeks"]).sum())}
    plus_four = sum("+4:" in text for text in est["other_deadline_weeks_in_window"])
    assert all_events.loc[4, "n_events_with_other_deadline_at_week"] == plus_four
    # Each event enters its own mean once: the cumulative mean is the plain event average.
    by_event = [
        fixture_panel.loc[pd.Timestamp(w) - pd.Timedelta(weeks=2) : pd.Timestamp(w) + pd.Timedelta(weeks=4), "delta_reserve_balances"].sum()
        for w in est["event_week"]
    ]
    assert all_events.loc[4, "mean_cumulative"] == pytest.approx(np.mean(by_event))


def test_tax_isolation_variants_keep_codated_events_explicit(fixture_panel) -> None:
    events = ro.tax_event_windows(fixture_panel)
    loose = events["no_neighboring_deadline_weeks"]
    strict = events["no_codated_or_neighboring_deadline_weeks"]
    # The old sample rule is retained exactly, but no longer called isolated.
    assert loose.equals(events["other_deadline_weeks_in_window"].eq(""))
    assert strict.equals(loose & events["codated_other_categories"].eq(""))
    assert (loose & ~strict).any()
    assert not strict.loc[events["codated_other_categories"].ne("")].any()
    counts = ro.tax_event_counts(events)
    full = counts.loc[counts["category"].eq("all_categories") & counts["era"].eq("full")].set_index("variant")
    assert full.loc["all_events", "n_category_events"] == 228
    assert full.loc["all_events", "n_distinct_calendar_weeks"] == 129
    assert full.loc["no_neighboring_deadline_weeks", "n_category_events"] == 203
    assert full.loc["no_codated_or_neighboring_deadline_weeks", "n_category_events"] == 39
    for row in counts.itertuples():
        cell = events if row.category == "all_categories" else events.loc[events["category"].eq(row.category)]
        if row.era != "full":
            cell = cell.loc[cell["era"].eq(row.era)]
        if row.variant != "all_events":
            cell = cell.loc[cell[row.variant]]
        assert row.n_category_events == len(cell)
        assert row.n_distinct_calendar_weeks == cell["event_week"].nunique()
        complete = cell.loc[cell["window_complete"]]
        assert row.n_category_events_complete == len(complete)
        assert row.n_distinct_complete_calendar_weeks == complete["event_week"].nunique()
    paths = ro.tax_event_paths(fixture_panel, events)
    for variant in ro.TAX_VARIANTS:
        expected = events if variant == "all_events" else events.loc[events[variant]]
        for category, cell in expected.groupby("category"):
            path = paths.loc[paths["category"].eq(category) & paths["era"].eq("full") & paths["variant"].eq(variant)]
            assert set(path["n_events"]) == {len(cell)}
            assert set(path["n_distinct_calendar_weeks"]) == {cell["event_week"].nunique()}


def test_tax_week_zero_cumulative_path_and_rrp_sign(fixture_panel) -> None:
    events = ro.tax_event_windows(fixture_panel)
    paths = ro.tax_event_paths(fixture_panel, events)
    thin = ro.thin_tax_table(paths).set_index(["category", "event_week"])
    for category, cell in events.loc[events["window_complete"]].groupby("category"):
        sums = [fixture_panel.loc[pd.Timestamp(w) - pd.Timedelta(weeks=2):pd.Timestamp(w)] for w in cell["event_week"]]
        reserve = np.mean([c["delta_reserve_balances"].sum() for c in sums]) / 1000
        # Tax table has the opposite sign to the RRP balance change.
        rrp = -np.mean([c["delta_rrp_others"].sum() for c in sums]) / 1000
        assert thin.loc[(category, 0), "delta_reserve_balances_bn"] == pytest.approx(round(reserve, 2))
        assert thin.loc[(category, 0), "rrp_others_reserve_contribution_bn"] == pytest.approx(round(rrp, 2))


def test_exposure_date_eras_export_separate_outcome_endpoint_ranges(fixture_panel) -> None:
    estimates = ro.estimate_all(fixture_panel)
    for (sample, horizon), cell in estimates.groupby(["sample", "horizon"]):
        rows, _ = ro.sample_rows(fixture_panel, sample)
        for column, expected in ro.sample_dates(rows, horizon).items():
            assert set(cell[column]) == {expected}
        assert pd.Timestamp(cell["first_outcome_endpoint_week"].iloc[0]) == rows.min() + pd.Timedelta(weeks=horizon)
        assert pd.Timestamp(cell["last_outcome_endpoint_week"].iloc[0]) == rows.max() + pd.Timedelta(weeks=horizon)
    long = estimates.loc[estimates["sample"].eq("era_2003_2008") & estimates["horizon"].eq(8)].iloc[0]
    assert pd.Timestamp(long["last_week"]) <= pd.Timestamp(ro.SAMPLES["era_2003_2008"][1])
    assert pd.Timestamp(long["last_outcome_endpoint_week"]) > pd.Timestamp(ro.SAMPLES["era_2003_2008"][1])


def test_factor_slopes_and_event_paths_add_up_to_the_reserve_slope(fixture_panel) -> None:
    estimates = ro.estimate_all(fixture_panel)
    keys = ["specification", "sample", "horizon", "coefficient"]
    for _, cell in estimates.groupby(keys):
        parts = cell.loc[cell["outcome"].isin(ro.CONTRIBUTIONS), "beta"].sum()
        total = cell.loc[cell["outcome"].eq("delta_reserve_balances"), "beta"].iloc[0]
        assert parts == pytest.approx(total, abs=1e-9)
    # Independent least-squares check of one reserve slope and the unit cash leg.
    rows, _ = ro.sample_rows(fixture_panel, "era_2016_2025")
    design = ro.design_for(fixture_panel, rows, "baseline")
    direct = np.linalg.lstsq(design.x, fixture_panel.loc[rows, "delta_reserve_balances"].to_numpy(), rcond=None)[0]
    row = estimates.loc[
        estimates["specification"].eq("baseline") & estimates["sample"].eq("era_2016_2025") & estimates["horizon"].eq(0)
    ].set_index("outcome")
    assert row.loc["delta_reserve_balances", "beta"] == pytest.approx(direct[design.index("exposure_c")])
    assert row.loc["contrib_cash_leg", "beta"] == pytest.approx(1.0)
    memo = row.loc["delta_reserve_balances", "beta"] + row.loc["delta_rrp_others", "beta"]
    assert row.loc["delta_reserves_plus_rrp_others", "beta"] == pytest.approx(memo)

    thin = ro.thin_tax_table(ro.tax_event_paths(fixture_panel, ro.tax_event_windows(fixture_panel)))
    parts = thin[["cash_leg_bn", "rrp_others_reserve_contribution_bn", "foreign_official_reserve_contribution_bn", "other_factors_and_rho_bn"]].sum(axis=1)
    assert (parts - thin["delta_reserve_balances_bn"]).abs().max() < 0.02


def test_main_specification_uses_no_future_fiscal_flow(tmp_path) -> None:
    # Regressors for week w must not move when anything after week w changes,
    # including later Treasury cash flows.
    ledger = _synthetic_ledger()
    (tmp_path / "a").mkdir()
    base_panel = _panel(_materialize(tmp_path / "a", ledger))
    cut = pd.Timestamp("2020-01-01")
    changed = ledger.copy()
    later = changed.index > cut
    changed.loc[later, "f_treasury_general_account"] *= 3.0
    changed.loc[later, "c_h"] = changed.loc[later, "f_treasury_general_account"]
    changed.loc[later, "delta_reserve_balances"] = changed.loc[later, [f"f_{r}" for r in FACTORS]].sum(axis=1) + changed.loc[later, "rho"]
    changed.attrs = dict(ledger.attrs)
    changed.attrs["tga_level"] = 300_000.0 - np.nan_to_num(changed["f_treasury_general_account"].to_numpy()).cumsum()
    (tmp_path / "b").mkdir()
    new_panel = _panel(_materialize(tmp_path / "b", changed))
    rows, _ = ro.sample_rows(base_panel, "full")
    early = rows[rows <= cut]
    for spec in ("baseline", "state"):
        old = ro.design_for(base_panel, rows, spec)
        new = ro.design_for(new_panel, rows, spec)
        assert old.names == new.names
        assert not any("lead" in name for name in old.names)
        np.testing.assert_array_equal(old.x[: len(early)], new.x[: len(early)])
    assert not np.array_equal(old.x[len(early) :], new.x[len(early) :])


def test_joint_band_shares_rows_and_weights_across_members(fixture_panel) -> None:
    family = ro.primary_family(fixture_panel)
    assert len(family) == 3 * len(ro.HORIZONS) + 1
    assert all(m.design.rows.equals(family[0].design.rows) for m in family)
    eta = ro.dwb_weights(family[0].design.weeks, 199, np.random.default_rng(3))
    # Identical members under common multipliers have identical bootstrap t
    # draws, so the family critical value equals the single-member value; with
    # member-specific multipliers it would be larger.
    single = ro.joint_max_t(family[:1], eta)
    tripled = ro.joint_max_t([family[0], family[0], family[0]], eta)
    assert tripled["critical_value"] == pytest.approx(single["critical_value"])
    full = ro.joint_max_t(family, eta)
    assert full["critical_value"] > single["critical_value"]
    shifted = ro.Member("shifted", ro.design_for(fixture_panel, family[0].design.rows[1:], "baseline"), family[0].y[1:], "exposure_c")
    with pytest.raises(ValueError, match="rows differ"):
        ro.joint_max_t([family[0], shifted], eta)


def test_random_walk_pseudo_state_does_not_routinely_reject(fixture_panel) -> None:
    rng = np.random.default_rng(11)
    rows, _ = ro.sample_rows(fixture_panel, "full")
    base = ro.design_for(fixture_panel, rows, "baseline")
    y = fixture_panel.loc[rows, ["delta_reserve_balances", "contrib_rrp_others"]].to_numpy(dtype=float)
    residuals = pd.DataFrame(y - base.x @ (base.a @ y), index=rows, columns=["delta_reserve_balances", "contrib_rrp_others"])
    null_hits, power_hits, sims = 0, 0, 30
    for _ in range(sims):
        state = ro.random_walk_state(fixture_panel, rng)
        simulated = ro.null_outcomes(fixture_panel, residuals, rng)
        members = ro.primary_family(fixture_panel, state=state, outcomes=simulated)
        eta = ro.dwb_weights(members[0].design.weeks, 99, rng)
        null_hits += abs(ro.joint_max_t(members, eta)["members"][-1]["t"]) > 1.96
        # The same machinery must detect a genuine slope-by-state dependence.
        scaled = (state - state.loc[rows].mean()) / state.loc[rows].std()
        effect = 0.3 * fixture_panel["exposure_c"] * scaled
        alt = dict(simulated, delta_reserve_balances=simulated["delta_reserve_balances"] + effect.fillna(0.0))
        members = ro.primary_family(fixture_panel, state=state, outcomes=alt)
        power_hits += ro.joint_max_t(members, eta)["joint_p"] <= 0.05
    assert null_hits / sims <= 0.2
    assert power_hits / sims >= 0.8


def _cli_argv(paths) -> list[str]:
    return [
        "run-weekly-reserve-offsets",
        "--ledger", str(paths["ledger"]),
        "--ledger-receipt", str(paths["receipt"]),
        "--rrp-others-level", str(paths["rrp"]),
        "--tga-level", str(paths["tga"]),
        "--debt-limit-episodes", str(paths["episodes"]),
        "--out-dir", str(paths["out"]),
        "--ledger-sha256", _sha(paths["ledger"]),
        "--ledger-receipt-sha256", _sha(paths["receipt"]),
        "--bootstrap-reps", "99",
        "--calibration-sims", "3",
        "--calibration-reps", "49",
        "--placebo-draws", "5",
    ]


def test_cli_writes_hashed_outputs_and_fails_closed(tmp_path) -> None:
    paths = _materialize(tmp_path)
    argv = _cli_argv(paths)
    assert cli_main(argv) == 0
    receipt = json.loads((paths["out"] / f"{ro.SCHEMA}_receipt.json").read_text())
    assert receipt["inputs"]["ledger"]["sha256"] == _sha(paths["ledger"])
    for name, digest in receipt["outputs"].items():
        assert _sha(paths["out"] / name) == digest
    assert {"thin_factor_decomposition_by_era_v1.csv", "thin_debt_limit_episodes_v1.csv", "thin_tax_event_paths_v1.csv"} <= set(receipt["outputs"])
    definitions = pd.read_csv(paths["out"] / "thin_table_definitions_v1.csv")
    for table, columns in receipt["design"]["thin_table_definitions"].items():
        frame = pd.read_csv(paths["out"] / f"{table}.csv")
        companion = definitions.loc[definitions["table"].eq(table)].set_index("column")["definition"].to_dict()
        assert set(columns) == set(frame.columns)
        assert companion == columns
    assert "-sum contrib_rrp_others" in receipt["design"]["debt_account_definitions"]["delta_rrp_others"]
    assert "week -3" in receipt["design"]["tax_path_base"]
    assert "fiscal-year filers" in receipt["design"]["tax_calendar_coverage"]
    assert "regional disaster relief" in receipt["design"]["tax_calendar_coverage"]
    assert any("mixes counterparties" in text and "ON RRP" in text for text in receipt["boundaries"])
    assert "WLRRAOL" in receipt["design"]["state"]
    assert "ON RRP" not in receipt["design"]["state"]
    assert len(receipt["design"]["sample_horizon_ranges"]) == len(ro.SAMPLES) * len(ro.HORIZONS)
    accounts = pd.read_csv(paths["out"] / "debt_limit_episode_accounts_v1.csv")
    account_sum = accounts[[name for name in ro.ACCOUNT_DEFINITIONS if name != "cash_leg"]].sum(axis=1)
    assert (account_sum - accounts["cash_leg"]).abs().max() < 1e-6
    thin = pd.read_csv(paths["out"] / "thin_factor_decomposition_by_era_v1.csv").set_index("row")
    for sample in ro.SAMPLES:
        parts = thin.loc[list(ro.CONTRIBUTIONS), f"{sample}_beta"].sum()
        assert parts == pytest.approx(thin.loc["delta_reserve_balances", f"{sample}_beta"], abs=0.01)
    snapshot = {p.name: p.read_bytes() for p in paths["out"].iterdir()}

    with pytest.raises(FileExistsError, match="already exists"):
        cli_main(argv)
    assert {p.name: p.read_bytes() for p in paths["out"].iterdir()} == snapshot

    paths["out"] = tmp_path / "invalid-out"
    argv[argv.index("--out-dir") + 1] = str(paths["out"])
    ledger = pd.read_csv(paths["ledger"])
    ledger.loc[5, "c_h"] += 1.0
    ledger.to_csv(paths["ledger"], index=False)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        cli_main(argv)
    assert not paths["out"].exists()
    assert {p.name: p.read_bytes() for p in (tmp_path / "out").iterdir()} == snapshot


def _failed_calibration(*args, **kwargs):
    return pd.DataFrame(), {"family": False, "interaction_pointwise": False, "interaction_real_placebo": False}


def test_failed_calibration_qualifies_every_coefficient_and_thin_export(tmp_path, monkeypatch) -> None:
    paths = _materialize(tmp_path)
    monkeypatch.setattr(ro, "calibrate", _failed_calibration)
    assert cli_main(_cli_argv(paths)) == 0
    for name in ("weekly_estimates", "factor_decomposition", "cumulative_horizons", "state_interaction", "influence", "thin_factor_decomposition_by_era"):
        frame = pd.read_csv(paths["out"] / f"{name}_v1.csv")
        assert set(frame["inference_status"]) == {"descriptive_calibration_failed"}
        assert set(frame["uncertainty_status"]) == {"diagnostic"}
    for name in ("thin_debt_limit_episodes", "thin_tax_event_paths"):
        frame = pd.read_csv(paths["out"] / f"{name}_v1.csv")
        assert set(frame["inference_status"]) == {"descriptive_accounting_only"}
    estimates = pd.read_csv(paths["out"] / "cumulative_horizons_v1.csv")
    # A hypothetical successful family calibration still does not admit eras or factors.
    members = ro.primary_family(_panel(paths))
    joint = ro.joint_max_t(members, ro.dwb_weights(members[0].design.weeks, 19, np.random.default_rng(3)))
    all_estimates = ro.estimate_all(_panel(paths))
    qualified = ro._family_columns(all_estimates, joint, "calibrated_joint_band")
    outside = qualified.loc[~qualified["primary_family"]]
    assert set(outside["inference_status"]) == {"descriptive_outside_declared_family"}
    assert set(outside["uncertainty_status"]) == {"diagnostic"}
    assert set(estimates["horizon"]) == set(ro.HORIZONS)


@pytest.mark.parametrize("failure", ["table_write", "receipt_write", "directory_rename"])
def test_cli_publication_failure_leaves_no_partial_bundle(tmp_path, monkeypatch, failure) -> None:
    paths = _materialize(tmp_path)
    monkeypatch.setattr(ro, "calibrate", _failed_calibration)
    original_csv, original_text = pd.DataFrame.to_csv, Path.write_text

    def fail_csv(frame, path, *args, **kwargs):
        if Path(path).name == "factor_decomposition_v1.csv":
            raise OSError("injected table publication failure")
        return original_csv(frame, path, *args, **kwargs)

    def fail_text(path, text, *args, **kwargs):
        if path.name == f"{ro.SCHEMA}_receipt.json":
            raise OSError("injected receipt publication failure")
        return original_text(path, text, *args, **kwargs)

    def fail_rename(source, destination):
        if destination == paths["out"]:
            # Failure after every final file and the matching receipt have been staged.
            staged_receipt = json.loads((source / f"{ro.SCHEMA}_receipt.json").read_text())
            assert len(staged_receipt["outputs"]) == 16
            assert all(_sha(source / name) == digest for name, digest in staged_receipt["outputs"].items())
            assert not destination.exists()
            raise OSError("injected directory publication failure")
        raise AssertionError("unexpected publication destination")

    if failure == "table_write":
        monkeypatch.setattr(pd.DataFrame, "to_csv", fail_csv)
    elif failure == "receipt_write":
        monkeypatch.setattr(Path, "write_text", fail_text)
    else:
        monkeypatch.setattr(ro, "_rename_unpublished_directory", fail_rename)
    with pytest.raises(OSError, match="injected"):
        cli_main(_cli_argv(paths))
    assert not paths["out"].exists()
    assert not list(tmp_path.glob(".out-*"))


def test_publication_refuses_even_an_empty_existing_destination(tmp_path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        ro._write_atomic(out, {"table": pd.DataFrame({"value": [1]})}, {})
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("existing", ["empty_directory", "complete_bundle", "dangling_symlink"])
def test_publication_does_not_clobber_a_racing_destination(tmp_path, monkeypatch, existing) -> None:
    paths = _materialize(tmp_path)
    monkeypatch.setattr(ro, "calibrate", _failed_calibration)
    original = ro._rename_unpublished_directory

    def racing_destination(source, destination):
        # The destination appears after the last existence check.
        if existing == "dangling_symlink":
            destination.symlink_to("missing-bundle", target_is_directory=True)
        else:
            destination.mkdir()
            if existing == "complete_bundle":
                (destination / "previous_receipt.json").write_text('{"producer": "previous"}')
        original(source, destination)

    monkeypatch.setattr(ro, "_rename_unpublished_directory", racing_destination)
    with pytest.raises(FileExistsError):
        cli_main(_cli_argv(paths))
    if existing == "dangling_symlink":
        assert paths["out"].is_symlink()
        assert paths["out"].readlink() == Path("missing-bundle")
    elif existing == "empty_directory":
        assert list(paths["out"].iterdir()) == []
    else:
        assert [p.name for p in paths["out"].iterdir()] == ["previous_receipt.json"]
        assert (paths["out"] / "previous_receipt.json").read_text() == '{"producer": "previous"}'
    assert not list(tmp_path.glob(".out-*"))


def test_ledger_contract_breaches_fail_closed(tmp_path) -> None:
    ledger = _synthetic_ledger()
    breaches = {
        "unmapped": lambda f: f.assign(f_new_facility=0.0),
        "gap": lambda f: f.drop(index=f.index[100]),
        "status": lambda f: f.assign(ledger_status=["unavailable_no_prior_wednesday", "rounding_exceeded", *["closed_within_rounding_bound"] * (len(f) - 2)]),
        "identity": lambda f: f.assign(rho=f["rho"] + 10.0),
    }
    for name, breach in breaches.items():
        broken = breach(ledger.copy())
        broken.attrs = dict(ledger.attrs)
        if name == "gap":
            broken.attrs["rrp_level"] = np.delete(ledger.attrs["rrp_level"], 100)
            broken.attrs["tga_level"] = np.delete(ledger.attrs["tga_level"], 100)
        (tmp_path / name).mkdir()
        paths = _materialize(tmp_path / name, broken)
        with pytest.raises(ValueError):
            _panel(paths)


def test_debt_limit_windows_are_contiguous_capped_and_sourced(fixture_panel, tmp_path) -> None:
    path = tmp_path / "episodes.yml"
    text = EPISODES_YML.replace("2013-03-20", "2013-04-03")
    path.write_text(text, encoding="utf-8")
    episodes = ro.load_debt_limit_episodes(path)
    accounts = ro.debt_limit_accounts(fixture_panel, episodes).set_index(["episode_id", "phase"])
    first_em, first_rebuild = accounts.loc[("2013a", "extraordinary_measures")], accounts.loc[("2013a", "rebuild")]
    # Measures window: after the last Wednesday before measures, through the last Wednesday before enactment.
    assert (first_em["base_wednesday"], first_em["end_wednesday"]) == ("2012-12-26", "2013-01-30")
    assert first_rebuild["base_wednesday"] == first_em["end_wednesday"]
    # The 13-week rebuild would reach 2013-05-01; the next measures begin 2013-04-03.
    assert first_rebuild["end_wednesday"] == "2013-03-27"
    assert bool(first_rebuild["rebuild_capped_by_next_episode"])
    assert accounts.loc[("2013b", "extraordinary_measures"), "base_wednesday"] == first_rebuild["end_wednesday"]
    window = fixture_panel.loc[(fixture_panel.index > pd.Timestamp("2012-12-26")) & (fixture_panel.index <= pd.Timestamp("2013-01-30"))]
    assert first_em["cash_leg"] == pytest.approx(window["exposure_c"].sum())
    assert first_em["delta_rrp_others"] == pytest.approx(first_em["rrp_others_level_end"] - first_em["rrp_others_level_base"])
    assert first_em["delta_rrp_others"] == pytest.approx(-first_em["contrib_rrp_others"])
    assert first_em["delta_foreign_official_balances"] == pytest.approx(-first_em["contrib_foreign_official"])
    thin = ro.thin_debt_limit_table(accounts.reset_index()).set_index(["episode_id", "phase"])
    assert thin.loc[("2013a", "extraordinary_measures"), "delta_rrp_others_bn"] == pytest.approx(round(first_em["delta_rrp_others"] / 1000, 1))

    registry = ro.debt_limit_registry(episodes)
    assert set(registry["date_role"]) == {"extraordinary_measures_start", "limit_suspension_enacted"}
    assert registry["source_title"].ne("").all()
    path.write_text(text.replace('title: "Letter A", ', ""), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks title"):
        ro.load_debt_limit_episodes(path)
    path.write_text(EPISODES_YML.replace("2013-03-20", "2013-01-20"), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        ro.load_debt_limit_episodes(path)


def test_interim_increase_splits_the_measures_window_without_gaps(fixture_panel, tmp_path) -> None:
    # Pub. L. 117-50 raised the limit while the measures continued; the window
    # splits at that enactment and the pieces still tile the episode exactly.
    text = EPISODES_YML.replace(
        "    resolution: 2013-10-17\n",
        "    interim_increase: 2013-07-10\n"
        "    interim_increase_source: {title: \"Interim act\", issued: 2013-07-10, url: \"https://example.org/e\"}\n"
        "    resolution: 2013-10-17\n",
    )
    path = tmp_path / "episodes.yml"
    path.write_text(text, encoding="utf-8")
    episodes = ro.load_debt_limit_episodes(path)
    accounts = ro.debt_limit_accounts(fixture_panel, episodes)
    second = accounts.loc[accounts["episode_id"].eq("2013b")].set_index("phase")
    assert list(second.index) == ["extraordinary_measures", "extraordinary_measures_after_interim_increase", "rebuild"]
    assert second.loc["extraordinary_measures", "end_wednesday"] == "2013-07-03"
    assert second.loc["extraordinary_measures_after_interim_increase", "base_wednesday"] == "2013-07-03"
    assert second.loc["extraordinary_measures_after_interim_increase", "end_wednesday"] == "2013-10-16"
    assert "limit_increase_enacted_while_measures_continued" in set(ro.debt_limit_registry(episodes)["date_role"])
    path.write_text(text.replace("interim_increase: 2013-07-10", "interim_increase: 2013-11-10"), encoding="utf-8")
    with pytest.raises(ValueError, match="dates must increase"):
        ro.load_debt_limit_episodes(path)


def test_hac_uses_calendar_distance_not_row_adjacency(fixture_panel) -> None:
    # After a deletion, rows on either side of the gap are not neighbours; a
    # positional kernel would weight them as if they were one week apart.
    rows, _ = ro.sample_rows(fixture_panel, "era_2016_2025")
    kept = rows.delete(slice(100, 110))
    design = ro.design_for(fixture_panel, kept, "baseline")
    y = fixture_panel.loc[kept, "delta_reserve_balances"].to_numpy(dtype=float)
    result = ro.fit(design, y, "exposure_c")
    resid = y - design.x @ (design.a @ y)
    z = design.a[design.index("exposure_c")] * resid
    distance = np.abs(design.weeks[:, None] - design.weeks[None, :])
    for bandwidth in ro.HAC_BANDWIDTHS:
        kernel = np.clip(1.0 - distance / (bandwidth + 1), 0.0, None)
        assert result[f"se_hac{bandwidth}"] == pytest.approx(np.sqrt(z @ kernel @ z))
    positional = np.clip(1.0 - np.abs(np.subtract.outer(np.arange(len(z)), np.arange(len(z)))) / 13, 0.0, None)
    assert np.sqrt(z @ positional @ z) != pytest.approx(result["se_hac12"])


REFRESH_ERAS = (
    ("era_2003_2008", "2003-01-01", "2008-09-30"),
    ("transition_2008q4", "2008-10-01", "2008-12-31"),
    ("era_2009_2015", "2009-01-01", "2015-09-30"),
    ("transition_2015q4", "2015-10-01", "2015-12-31"),
    ("era_2016_2026q3", "2016-01-01", "2026-09-30"),
)


def _refresh_panel(tmp_path: Path, ledger_end: str = "2026-10-07") -> pd.DataFrame:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = _materialize(tmp_path, _synthetic_ledger(end=ledger_end))
    ledger, receipt = ro.load_ledger(
        paths["ledger"], paths["receipt"], ledger_sha256=_sha(paths["ledger"]), receipt_sha256=_sha(paths["receipt"])
    )
    rrp = ro.load_level(paths["rrp"], receipt, "rrp_others", ledger)
    tga = ro.load_level(paths["tga"], receipt, "treasury_general_account", ledger)
    return ro.build_weekly_panel(ledger, rrp, tga, REFRESH_ERAS)


def test_refresh_eras_reach_the_declared_week_and_report_the_endpoint_shortfall(tmp_path) -> None:
    # The refresh declares exposure weeks through 2026-09-30, but the ledger ends
    # 2026-10-07: rows need the h=8 endpoint, so the last seven declared weeks
    # cannot enter and must be reported, not dropped silently.
    panel = _refresh_panel(tmp_path)
    assert ro.samples_of(REFRESH_ERAS)["full"] == ("2003-01-01", "2026-09-30")
    coverage = {row["sample"]: row for row in ro.sample_coverage(panel, REFRESH_ERAS)}
    for sample in ("full", "era_2016_2026q3"):
        assert coverage[sample]["declared_final_week"] == "2026-09-30"
        assert coverage[sample]["last_week"] == "2026-08-12"
        assert coverage[sample]["weeks_after_last_week_without_h8_endpoint"] == 7
        assert coverage[sample]["n_dropped_unobserved_endpoint"] == 7
    assert coverage["era_2009_2015"]["weeks_after_last_week_without_h8_endpoint"] == 0

    estimates = ro.estimate_all(panel, REFRESH_ERAS)
    assert set(estimates["sample"]) == set(ro.samples_of(REFRESH_ERAS))
    modern = estimates.loc[estimates["sample"].eq("era_2016_2026q3") & estimates["horizon"].eq(8)]
    assert set(modern["last_outcome_endpoint_week"]) == {"2026-10-07"}
    # The same-week reserve slope in the new era is fitted on exactly the reported rows.
    rows, _ = ro.sample_rows(panel, "era_2016_2026q3", REFRESH_ERAS)
    assert set(modern["n"]) == {len(rows)}
    thin = ro.thin_factor_table(ro.factor_decomposition(ro._family_columns(
        estimates, {"members": []}, "descriptive_calibration_failed")), estimates, REFRESH_ERAS)
    assert "era_2016_2026q3_beta" in thin.columns
    assert not any("2016_2025" in str(value) for value in [*thin.columns, *estimates["sample"]])

    # A ledger that stops before the declared final week fails closed.
    short = _refresh_panel(tmp_path / "short", ledger_end="2026-09-23")
    with pytest.raises(ValueError, match="before the declared final week"):
        ro.sample_coverage(short, REFRESH_ERAS)


def test_refresh_tax_events_come_only_from_the_typed_calendar_rule(tmp_path) -> None:
    # 2026 deadlines enter by rule (no national postponement is encoded for 2026).
    # September events need weeks through 2026-10-14, after the ledger ends, so
    # they are listed as incomplete and excluded from every path mean.
    panel = _refresh_panel(tmp_path)
    events = ro.tax_event_windows(panel, REFRESH_ERAS)
    recent = events.loc[events["event_week"] >= "2026-01-01"].set_index(["category", "event_week"])
    assert set(recent.index) == {
        ("individual_estimated", "2026-01-21"),
        ("individual_estimated", "2026-04-15"), ("individual_annual", "2026-04-15"),
        ("corporate_estimated", "2026-04-15"), ("corporate_annual", "2026-04-15"),
        ("individual_estimated", "2026-06-17"), ("corporate_estimated", "2026-06-17"),
        ("individual_estimated", "2026-09-16"), ("corporate_estimated", "2026-09-16"),
    }
    assert set(recent["era"]) == {"era_2016_2026q3"}
    complete = recent["window_complete"]
    assert not complete.loc[[("individual_estimated", "2026-09-16"), ("corporate_estimated", "2026-09-16")]].any()
    assert complete.drop([("individual_estimated", "2026-09-16"), ("corporate_estimated", "2026-09-16")]).all()
    assert recent.loc[("individual_estimated", "2026-01-21"), "due_dates"] == "2026-01-15"
    counts = ro.tax_event_counts(events, REFRESH_ERAS).set_index(["category", "era", "variant"])
    cell = counts.loc[("corporate_estimated", "era_2016_2026q3", "all_events")]
    assert cell["n_category_events"] - cell["n_category_events_complete"] == 1
    v1 = ro.tax_event_windows(panel)
    assert (v1["event_week"] <= "2025-12-31").all()


def test_cli_refresh_config_pins_inputs_and_binds_the_source_receipt(tmp_path) -> None:
    paths = _materialize(tmp_path, _synthetic_ledger(end="2026-10-07"))
    source = tmp_path / "RECEIPT.json"
    source.write_text('{"vintage": "test"}\n', encoding="utf-8")
    config = tmp_path / "design.json"
    design = {
        "schema": ro.SCHEMA, "vintage": "test",
        "source_receipt_sha256": _sha(source),
        "ledger_sha256": _sha(paths["ledger"]), "ledger_receipt_sha256": _sha(paths["receipt"]),
        "eras": [list(era) for era in REFRESH_ERAS],
    }
    config.write_text(json.dumps(design), encoding="utf-8")
    argv = [arg for arg in _cli_argv(paths)]
    for flag in ("--ledger-sha256", "--ledger-receipt-sha256"):
        position = argv.index(flag)
        del argv[position:position + 2]
    argv += ["--config", str(config), "--source-receipt", str(source)]

    with pytest.raises(ValueError, match="requires the upstream refresh receipt"):
        cli_main(argv[:-2])
    source.write_text('{"vintage": "other"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source_receipt: sha256 mismatch"):
        cli_main(argv)
    assert not paths["out"].exists()
    source.write_text('{"vintage": "test"}\n', encoding="utf-8")
    assert cli_main(argv) == 0
    receipt = json.loads((paths["out"] / f"{ro.SCHEMA}_receipt.json").read_text())
    assert receipt["vintage"] == "test"
    assert receipt["design_config"]["sha256"] == _sha(config)
    assert receipt["inputs"]["source_receipt"]["sha256"] == _sha(source)
    assert receipt["design"]["eras"][-1] == ["era_2016_2026q3", "2016-01-01", "2026-09-30"]
    full = next(row for row in receipt["design"]["sample_coverage"] if row["sample"] == "full")
    assert (full["last_week"], full["weeks_after_last_week_without_h8_endpoint"]) == ("2026-08-12", 7)
    thin = pd.read_csv(paths["out"] / "thin_factor_decomposition_by_era_v1.csv").set_index("row")
    assert thin.loc["delta_reserve_balances", "era_2016_2026q3_last_week"] == "2026-08-12"
