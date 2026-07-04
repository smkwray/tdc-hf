from __future__ import annotations

import pandas as pd

from tdchf.closeout import export_anchor_contract_audit, export_noniv_tdc_lp_closeout


def test_export_anchor_contract_audit_identifies_matching_anchor(tmp_path) -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    proxy = pd.DataFrame({"date": dates, "tdc_monthly": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    proxy_path = tmp_path / "proxy.csv"
    proxy.to_csv(proxy_path, index=False)
    estimates = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31", "2020-06-30"]),
            "tdc_base_bank_only_ru_flow": [6.0, 15.0],
            "tdc_tier1_fed_corrected_bank_only_ru_flow": [7.0, 16.0],
        }
    )
    estimates_path = tmp_path / "estimates.csv"
    estimates.to_csv(estimates_path, index=False)

    report = export_anchor_contract_audit(
        monthly_proxy_csv=proxy_path,
        tdcest_estimates_csv=estimates_path,
        out_csv=tmp_path / "audit.csv",
    )

    out = pd.read_csv(report["out"])
    base = out.loc[out["candidate_anchor"].eq("tdc_base_bank_only_ru_flow")].iloc[0]
    assert base["interpretation"] == "near_exact_default_anchor_match"


def test_export_noniv_tdc_lp_closeout_writes_descriptive_table(tmp_path) -> None:
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    tdc = pd.Series(range(36), dtype=float)
    panel = pd.DataFrame(
        {
            "date": dates,
            "tdc_monthly": tdc,
            "deposits": 0.1 * tdc,
            "lag_deposits": tdc.shift(1).fillna(0.0),
        }
    )
    pre = tmp_path / "pre.csv"
    cal = tmp_path / "cal.csv"
    panel.to_csv(pre, index=False)
    panel.to_csv(cal, index=False)

    report = export_noniv_tdc_lp_closeout(
        pretrend_panel_csv=pre,
        calendar_panel_csv=cal,
        out_csv=tmp_path / "noniv.csv",
        outcomes=["deposits"],
        horizons=[0, 1],
        pretrend_controls=["lag_deposits"],
        calendar_controls=["lag_deposits"],
    )

    out = pd.read_csv(report["out"])
    assert set(out["causal_interpretation"]) == {"descriptive_not_causal"}
    assert set(out["model"]) == {"noniv_tdc_pretrend", "noniv_tdc_calendar"}
