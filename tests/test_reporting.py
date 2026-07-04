from __future__ import annotations

from tdchf.pipeline import run_monthly_proxy_pipeline
import pandas as pd

from tdchf.reporting import (
    compare_iv_robustness_summaries,
    compare_lp_results,
    compare_monthly_proxies,
    export_publication_tables,
    export_thesis_status_report,
    summarize_estimates_bundle,
    summarize_proxy_run,
)


def test_summarize_proxy_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_monthly_proxy_pipeline(out_dir=run_dir)

    summary = summarize_proxy_run(run_dir, out_json=tmp_path / "summary.json", out_md=tmp_path / "summary.md")

    assert summary["validation_status"] == "ok"
    assert summary["proxy_rows"] > 0
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()


def test_summarize_estimates_bundle(tmp_path) -> None:
    lp = tmp_path / "lp.csv"
    pd.DataFrame(
        {
            "outcome": ["deposits", "deposits", "bank_credit"],
            "horizon": [0, 1, 0],
            "beta": [1.0, 3.0, -2.0],
            "se": [0.2, 0.5, 1.0],
            "lower95": [0.6, 2.0, -4.0],
            "upper95": [1.4, 4.0, 0.0],
            "n": [20, 19, 20],
        }
    ).to_csv(lp, index=False)
    first = tmp_path / "first.csv"
    pd.DataFrame(
        {
            "n": [20],
            "excluded_instrument_f": [12.3],
            "excluded_instrument_pvalue": [0.01],
        }
    ).to_csv(first, index=False)

    report = summarize_estimates_bundle(lp, first_stage_csv=first, out_dir=tmp_path / "tables")
    peak = pd.read_csv(tmp_path / "tables" / "lp_peak_responses.csv")

    assert report["status"] == "ok"
    assert set(peak["outcome"]) == {"deposits", "bank_credit"}
    assert peak.loc[peak["outcome"] == "deposits", "horizon"].iloc[0] == 1
    assert (tmp_path / "tables" / "estimate_summary.md").exists()


def test_compare_monthly_proxies(tmp_path) -> None:
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame({"date": dates, "tdc_monthly": [1.0, 2.0, 3.0]}).to_csv(left, index=False)
    pd.DataFrame({"date": dates, "tdc_monthly": [1.0, 1.5, 3.5]}).to_csv(right, index=False)

    report = compare_monthly_proxies(left, right, out_dir=tmp_path / "compare", left_label="a", right_label="b")

    assert report["status"] == "ok"
    assert report["rows"] == 3
    assert report["max_abs_difference"] == 0.5
    assert (tmp_path / "compare" / "proxy_comparison.md").exists()


def test_export_publication_tables(tmp_path) -> None:
    lp = tmp_path / "lp.csv"
    pd.DataFrame(
        {
            "outcome": ["deposits", "deposits"],
            "horizon": [0, 1],
            "beta": [0.1, 0.2],
            "se": [0.01, 0.02],
            "lower95": [0.08, 0.16],
            "upper95": [0.12, 0.24],
            "n": [50, 49],
        }
    ).to_csv(lp, index=False)
    boot = tmp_path / "boot.csv"
    pd.DataFrame(
        {
            "outcome": ["deposits", "deposits"],
            "horizon": [0, 1],
            "bootstrap_se": [0.02, 0.03],
            "bootstrap_lower95": [0.05, 0.10],
            "bootstrap_upper95": [0.15, 0.30],
            "draws": [100, 100],
            "block_length": [6, 6],
        }
    ).to_csv(boot, index=False)
    first = tmp_path / "first.csv"
    pd.DataFrame({"n": [50], "excluded_instrument_f": [18.0], "excluded_instrument_pvalue": [0.001]}).to_csv(first, index=False)
    validation = tmp_path / "validation.csv"
    pd.DataFrame(
        {
            "component": ["fed_tsy"],
            "quarters_checked": [4],
            "first_quarter": ["2024Q1"],
            "last_quarter": ["2024Q4"],
            "max_abs_quarterly_error": [1e-10],
            "method": ["additive_denton"],
            "status": ["ok"],
        }
    ).to_csv(validation, index=False)
    comparison = tmp_path / "comparison.csv"
    pd.DataFrame({"difference_sum": [0.0], "mean_abs_difference": [1.0], "max_abs_difference": [2.0]}).to_csv(comparison, index=False)

    report = export_publication_tables(
        lp_csv=lp,
        bootstrap_csv=boot,
        first_stage_csv=first,
        validation_csv=validation,
        proxy_comparison_csv=comparison,
        out_dir=tmp_path / "publication",
    )
    table = pd.read_csv(tmp_path / "publication" / "table_lp_iv_with_bootstrap.csv")

    assert report["status"] == "ok"
    assert report["lp_rows"] == 2
    assert "bootstrap_lower95" in table.columns
    assert (tmp_path / "publication" / "publication_tables.md").exists()


def test_compare_iv_robustness_summaries(tmp_path) -> None:
    baseline = tmp_path / "baseline.csv"
    controlled = tmp_path / "controlled.csv"
    pd.DataFrame(
        {
            "iv_spec": ["both"],
            "outcome": ["deposits"],
            "peak_horizon": [6],
            "peak_beta": [0.1],
            "first_stage_f": [20.0],
            "peak_hac_sig_95": [True],
        }
    ).to_csv(baseline, index=False)
    pd.DataFrame(
        {
            "iv_spec": ["both"],
            "outcome": ["deposits"],
            "peak_horizon": [3],
            "peak_beta": [0.08],
            "first_stage_f": [18.0],
            "peak_hac_sig_95": [False],
        }
    ).to_csv(controlled, index=False)

    report = compare_iv_robustness_summaries(baseline, controlled, out_dir=tmp_path / "compare")
    out = pd.read_csv(tmp_path / "compare" / "iv_robustness_control_comparison.csv")

    assert report["status"] == "ok"
    assert out.loc[0, "delta_peak_beta"] == -0.02


def test_compare_lp_results(tmp_path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame(
        {
            "outcome": ["deposits"],
            "horizon": [6],
            "beta": [0.2],
            "lower95": [0.1],
            "upper95": [0.3],
        }
    ).to_csv(left, index=False)
    pd.DataFrame(
        {
            "outcome": ["deposits"],
            "horizon": [6],
            "beta": [0.15],
            "lower95": [0.0],
            "upper95": [0.3],
        }
    ).to_csv(right, index=False)

    report = compare_lp_results(left, right, out_dir=tmp_path / "lp_compare", left_label="a", right_label="b")
    out = pd.read_csv(tmp_path / "lp_compare" / "lp_result_comparison.csv")

    assert report["status"] == "ok"
    assert round(out.loc[0, "delta_beta"], 2) == 0.05


def test_export_thesis_status_report(tmp_path) -> None:
    section = tmp_path / "section.md"
    section.write_text("# Section\n\nBody\n", encoding="utf-8")

    report = export_thesis_status_report(out_md=tmp_path / "report.md", publication_md=section)

    assert report["status"] == "ok"
    assert report["sections"] == 1
    assert "TDC-HF Thesis Status Report" in (tmp_path / "report.md").read_text(encoding="utf-8")
