from __future__ import annotations

import pandas as pd

from tdchf.thesis_outputs import build_claim_status_table, export_anchor_variant_forest_plot, export_short_run_profile_plot


def test_build_claim_status_table_labels_core_and_mechanism() -> None:
    dash = pd.DataFrame(
        {
            "anchor_variant": ["tier1", "tier1"],
            "anchor_column": ["a", "a"],
            "outcome": ["bank_credit", "onrrp"],
            "horizon": [12, 12],
            "same_unit_beta": [1.0, 0.8],
            "same_unit_lower95": [0.2, 0.1],
            "same_unit_upper95": [1.8, 1.5],
            "same_unit_bootstrap_lower95": [0.1, 0.2],
            "same_unit_bootstrap_upper95": [2.0, 1.6],
            "first_stage_f": [30.0, 30.0],
            "full_sample_first_stage_f": [35.0, 35.0],
            "placebo_rows": [5, 5],
            "placebo_sig_rows": [0, 2],
            "placebo_clean": [True, False],
            "hac_sig_95": [True, True],
            "bootstrap_sig_95": [True, True],
        }
    )
    regime = pd.DataFrame(
        {
            "sample": ["exclude_gfc", "exclude_covid"],
            "outcome": ["bank_credit", "bank_credit"],
            "hac_sig_95": [True, True],
            "same_unit_beta": [0.7, 1.2],
        }
    )

    out = build_claim_status_table(anchor_dashboard=dash, regime_summary=regime)

    assert out.loc[out["outcome"].eq("bank_credit"), "claim_status"].iloc[0] == "core_evidence"
    assert out.loc[out["outcome"].eq("onrrp"), "claim_status"].iloc[0] == "mechanism_evidence_placebo_contaminated"


def test_build_claim_status_table_labels_ci_as_regime_sensitive() -> None:
    dash = pd.DataFrame(
        {
            "anchor_variant": ["tier1"],
            "anchor_column": ["a"],
            "outcome": ["commercial_industrial_loans"],
            "horizon": [3],
            "same_unit_beta": [0.5],
            "same_unit_lower95": [0.1],
            "same_unit_upper95": [0.9],
            "first_stage_f": [30.0],
            "placebo_rows": [5],
            "placebo_sig_rows": [0],
            "placebo_clean": [True],
            "hac_sig_95": [True],
            "bootstrap_sig_95": [False],
        }
    )
    regime = pd.DataFrame(
        {
            "sample": ["exclude_gfc", "exclude_covid", "exclude_debt_ceiling"],
            "outcome": ["commercial_industrial_loans"] * 3,
            "hac_sig_95": [True, False, False],
            "same_unit_beta": [0.6, 0.1, 0.2],
        }
    )

    out = build_claim_status_table(anchor_dashboard=dash, regime_summary=regime)

    assert out["claim_status"].iloc[0] == "supportive_core_regime_sensitive"


def test_export_anchor_variant_forest_plot(tmp_path) -> None:
    dash = pd.DataFrame(
        {
            "anchor_variant": ["tier1", "du"],
            "outcome": ["deposits", "deposits"],
            "horizon": [12, 12],
            "same_unit_beta": [0.5, 0.7],
            "same_unit_lower95": [-0.2, -0.1],
            "same_unit_upper95": [1.2, 1.5],
            "same_unit_bootstrap_lower95": [-0.3, -0.2],
            "same_unit_bootstrap_upper95": [1.3, 1.6],
            "placebo_sig_rows": [0, 0],
        }
    )
    path = tmp_path / "dashboard.csv"
    dash.to_csv(path, index=False)

    report = export_anchor_variant_forest_plot(anchor_dashboard_csv=path, out_dir=tmp_path / "figs", outcomes=["deposits"])

    assert report["status"] == "ok"
    assert (tmp_path / "figs" / "anchor_variant_h12_forest.png").exists()


def test_export_short_run_profile_plot(tmp_path) -> None:
    dash = pd.DataFrame(
        {
            "anchor_variant": ["tier1", "tier1", "du", "du"],
            "outcome": ["deposits", "deposits", "deposits", "deposits"],
            "horizon": [0, 1, 0, 1],
            "same_unit_beta": [0.2, 0.4, 0.3, 0.5],
            "same_unit_lower95": [-0.1, 0.0, -0.2, 0.1],
            "same_unit_upper95": [0.5, 0.8, 0.8, 0.9],
        }
    )
    path = tmp_path / "dashboard.csv"
    dash.to_csv(path, index=False)

    report = export_short_run_profile_plot(
        anchor_dashboard_csv=path,
        out_dir=tmp_path / "short",
        outcomes=["deposits"],
        horizons=[0, 1],
    )

    assert report["status"] == "ok"
    assert (tmp_path / "short" / "anchor_variant_short_run_profile.png").exists()
