from __future__ import annotations

import pandas as pd

from tdchf.anchor_variants import AnchorVariantSpec, run_anchor_variant_robustness_csv


def test_run_anchor_variant_robustness_csv_builds_du_outputs(tmp_path) -> None:
    dates = pd.date_range("2020-01-31", periods=48, freq="ME")
    z = pd.Series(range(1, 49), dtype=float)
    treatment = z * 2.0
    panel = pd.DataFrame(
        {
            "date": dates,
            "tdc_monthly": treatment,
            "tga_long_surprise_z": z,
            "deposits": 0.2 * treatment,
            "bank_credit": 0.5 * treatment,
            "lag_deposits": z.shift(1).fillna(0.0),
            "lag_bank_credit": z.shift(1).fillna(0.0),
        }
    )
    panel_path = tmp_path / "panel.csv"
    panel.to_csv(panel_path, index=False)

    qdates = pd.date_range("2020-03-31", periods=16, freq="QE")
    estimates = pd.DataFrame(
        {
            "date": qdates,
            "tdc_du_fiscal_flow_first_pass_narrow": [30.0 + i for i in range(16)],
        }
    )
    estimates_path = tmp_path / "estimates.csv"
    estimates.to_csv(estimates_path, index=False)

    indicators = pd.DataFrame(
        {
            "date": dates,
            "dts_core_payment_withdrawals": [12.0] * len(dates),
            "dts_tax_deposits": [2.0] * len(dates),
        }
    )
    indicators_path = tmp_path / "indicators.csv"
    indicators.to_csv(indicators_path, index=False)

    report = run_anchor_variant_robustness_csv(
        tdcest_estimates_csv=estimates_path,
        monthly_indicators_csv=indicators_path,
        base_panel_csv=panel_path,
        out_dir=tmp_path / "out",
        variant_specs=[
            "du_gr_narrow:tdc_du_fiscal_flow_first_pass_narrow:dts_core_less_tax:du_g_minus_r_proxy:test"
        ],
        outcomes=["deposits", "bank_credit"],
        bootstrap_outcomes=["deposits"],
        controls=["lag_deposits", "lag_bank_credit"],
        horizons=[0, 1],
        bootstrap_replications=5,
    )

    assert report["status"] == "ok"
    assert (tmp_path / "out" / "anchor_variant_h12_dashboard.csv").exists()
    assert (tmp_path / "out" / "anchor_variant_short_dashboard.csv").exists()
    summary = pd.read_csv(tmp_path / "out" / "anchor_variant_summary.csv")
    assert summary.loc[0, "anchor_variant"] == "du_gr_narrow"
    assert summary.loc[0, "max_abs_quarterly_error"] < 1e-8
