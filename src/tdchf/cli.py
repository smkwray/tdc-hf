from __future__ import annotations

import argparse
import json

from .analysis_spec import run_analysis_spec
from .auction_shocks import build_auction_size_shock, build_shock_bundle_csv, build_tga_rebuild_shock_csv
from .bootstrap import bootstrap_lp_iv_csv
from .calendar_controls import add_calendar_controls_csv
from .demo import run_demo
from .envelope import build_method_envelope
from .figures import export_thesis_figures
from .first_stage import run_first_stage_csv
from .fiscaldata import build_dts_fiscal_indicators_csv, build_dts_transaction_indicators_csv, download_default_dts_sources, write_fiscaldata_csv
from .fred import write_fred_series_csv
from .iv_robustness import run_iv_robustness_csv
from .live_indicators import FRED_HF_SERIES, build_fred_monthly_indicators_csv
from .local_sources import build_fiscal_indicator_csv, build_tic_row_indicator_csv, merge_indicator_csvs
from .lp import run_local_projections_csv, run_lp_iv_csv, run_lp_iv_placebo_csv
from .manifest import write_file_manifest, write_spec_manifest
from .method_status import write_method_status_csv
from .model_panel import assemble_model_panel_csv
from .outcomes import FRED_OUTCOME_SERIES, build_monthly_outcomes_csv
from .pipeline import run_monthly_proxy_pipeline
from .pretrend import add_pretrend_controls_csv
from .readiness import readiness_payload
from .reporting import (
    compare_iv_robustness_summaries,
    compare_lp_results,
    compare_monthly_proxies,
    export_thesis_status_report,
    export_publication_tables,
    summarize_estimates_bundle,
    summarize_panel_csv,
    summarize_proxy_run_cli,
)
from .shocks import build_named_residual_shock_csv, build_residual_shock_csv
from .upstream import resolve_repos
from .weekly import build_weekly_channel_panel_csv, build_weekly_state_csv
from .channel_context import build_auction_context


def _split_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_int_arg(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tdchf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the synthetic monthly benchmarking demo")
    demo.add_argument("--out", default="output/demo", help="Output directory")

    build = subparsers.add_parser("build-proxy", help="Build monthly proxy data products")
    build.add_argument("--out", default="data/processed", help="Output directory")
    build.add_argument("--monthly-indicators", default=None, help="Wide monthly indicator CSV")
    build.add_argument("--quarterly-anchors", default=None, help="Optional tdcest-style quarterly anchor CSV")
    build.add_argument(
        "--benchmark-method",
        choices=["residual_spread", "denton"],
        default="residual_spread",
        help="Monthly benchmarking method",
    )
    build.add_argument("--no-fill-missing", action="store_true", help="Require every component in monthly indicator CSV")
    build.add_argument("--method-label", default=None, help="Override method label in metadata")

    envelope = subparsers.add_parser("build-envelope", help="Build method-spread envelope from monthly indicators")
    envelope.add_argument("--out", default="data/processed/envelope", help="Output directory")
    envelope.add_argument("--monthly-indicators", default=None, help="Wide monthly indicator CSV")
    envelope.add_argument("--quarterly-anchors", default=None, help="Optional tdcest-style quarterly anchor CSV")

    subparsers.add_parser("inspect-upstream", help="Print resolved sibling repo contracts")
    subparsers.add_parser("doctor", help="Check upstream source readiness")

    fred = subparsers.add_parser("download-fred", help="Download FRED graph CSV series into one wide CSV")
    fred.add_argument("series", nargs="+", help="FRED series ids")
    fred.add_argument("--out", required=True, help="Output CSV path")

    hf_fred = subparsers.add_parser("download-hf-fred", help="Download default high-frequency FRED source series")
    hf_fred.add_argument("--out", default="data/raw/fred_hf_sources.csv", help="Output CSV path")

    outcome_fred = subparsers.add_parser("download-outcome-fred", help="Download default FRED outcome/control series")
    outcome_fred.add_argument("--out", default="data/raw/fred_outcome_sources.csv", help="Output CSV path")

    fred_ind = subparsers.add_parser("build-fred-indicators", help="Transform raw FRED source CSV into monthly indicators")
    fred_ind.add_argument("--raw", default="data/raw/fred_hf_sources.csv", help="Raw wide FRED source CSV")
    fred_ind.add_argument("--out", default="data/processed/fred_monthly_indicators.csv", help="Monthly indicator CSV")
    fred_ind.add_argument("--metadata", default="data/processed/fred_monthly_indicator_metadata.csv", help="Metadata CSV")

    outcomes = subparsers.add_parser("build-outcomes", help="Transform raw FRED outcome/control source CSV")
    outcomes.add_argument("--raw", default="data/raw/fred_outcome_sources.csv", help="Raw wide FRED source CSV")
    outcomes.add_argument("--out", default="data/processed/fred_monthly_outcomes.csv", help="Monthly outcomes CSV")
    outcomes.add_argument("--metadata", default="data/processed/fred_monthly_outcome_metadata.csv", help="Metadata CSV")

    weekly = subparsers.add_parser("build-weekly-state", help="Transform raw FRED source CSV into weekly channel state panel")
    weekly.add_argument("--raw", default="data/raw/fred_hf_sources.csv", help="Raw wide FRED source CSV")
    weekly.add_argument("--out", default="data/processed/tdc_weekly_state.csv", help="Weekly state CSV")

    weekly_panel = subparsers.add_parser("build-weekly-channel-panel", help="Build weekly changes/lags for channel LPs")
    weekly_panel.add_argument("inputs", nargs="+", help="Weekly wide CSVs to merge")
    weekly_panel.add_argument("--changes", default="", help="Comma-separated level columns to difference")
    weekly_panel.add_argument("--lags", default="", help="Comma-separated columns to lag")
    weekly_panel.add_argument("--out", required=True, help="Output weekly channel panel CSV")

    tic = subparsers.add_parser("build-tic-row-indicator", help="Normalize local TIC monthly ROW Treasury purchases")
    tic.add_argument("--raw", required=True, help="Local TIC extract CSV")
    tic.add_argument("--out", default="data/processed/tic_row_indicator.csv", help="Output indicator CSV")

    fiscal = subparsers.add_parser("build-fiscal-indicators", help="Normalize local DTS/MTS-style fiscal indicators")
    fiscal.add_argument("--raw", required=True, help="Local fiscal extract CSV")
    fiscal.add_argument("--out", default="data/processed/fiscal_monthly_indicators.csv", help="Output indicator CSV")

    fiscaldata = subparsers.add_parser("download-fiscaldata", help="Download a FiscalData API endpoint to CSV")
    fiscaldata.add_argument("--endpoint", required=True, help="Endpoint name below v1/accounting/dts")
    fiscaldata.add_argument("--out", required=True, help="Output CSV path")
    fiscaldata.add_argument("--fields", default="", help="Comma-separated fields to request")
    fiscaldata.add_argument("--filters", default="", help="Comma-separated FiscalData filters")
    fiscaldata.add_argument("--sort", default="record_date", help="FiscalData sort expression")
    fiscaldata.add_argument("--page-size", type=int, default=10_000, help="FiscalData page size")
    fiscaldata.add_argument("--manifest-json", default=None, help="Optional retrieval manifest JSON path")

    dts = subparsers.add_parser("download-dts-fiscaldata", help="Download default DTS FiscalData sources used by the proxy")
    dts.add_argument("--out-dir", default="data/raw/fiscaldata", help="Output raw source directory")
    dts.add_argument("--start-date", default="2005-01-01", help="First record_date to download")
    dts.add_argument("--page-size", type=int, default=10_000, help="FiscalData page size")

    dts_ind = subparsers.add_parser("build-dts-fiscal-indicators", help="Build monthly indicators from downloaded DTS FiscalData")
    dts_ind.add_argument("--operating-cash-balance", default="data/raw/fiscaldata/dts_operating_cash_balance.csv")
    dts_ind.add_argument("--fed-remit", default="data/raw/fiscaldata/dts_federal_reserve_earnings.csv")
    dts_ind.add_argument("--out", default="data/processed/dts_fiscal_monthly_indicators.csv")
    dts_ind.add_argument("--metadata", default="data/processed/dts_fiscal_monthly_indicator_metadata.csv")

    dts_tx = subparsers.add_parser("build-dts-transaction-indicators", help="Build monthly fiscal-flow indicators from full DTS deposits/withdrawals")
    dts_tx.add_argument("--transactions", default="data/raw/fiscaldata/dts_deposits_withdrawals_operating_cash.csv")
    dts_tx.add_argument("--out", default="data/processed/dts_transaction_monthly_indicators.csv")
    dts_tx.add_argument("--metadata", default="data/processed/dts_transaction_monthly_indicator_metadata.csv")

    merge = subparsers.add_parser("merge-indicators", help="Merge partial monthly indicator CSVs")
    merge.add_argument("inputs", nargs="+", help="Input indicator CSVs")
    merge.add_argument("--out", required=True, help="Output merged indicator CSV")

    auction = subparsers.add_parser("build-auction-context", help="Build quarterly auction-share context from tsyparty")
    auction.add_argument("--allocation", default=None, help="Optional primary_allocation.csv path")
    auction.add_argument("--out", default="data/processed/auction_context.csv", help="Output CSV")

    residual = subparsers.add_parser("build-residual-shock", help="Build expanding-window residual shock from a CSV")
    residual.add_argument("--data", required=True, help="Input wide CSV")
    residual.add_argument("--target", required=True, help="Target column")
    residual.add_argument("--predictors", required=True, help="Comma-separated predictor columns")
    residual.add_argument("--min-train-obs", type=int, default=24, help="Minimum training observations")
    residual.add_argument("--month-dummies", action="store_true", help="Include month-of-year dummies in the shock forecast")
    residual.add_argument("--trend", action="store_true", help="Include linear trend in the shock forecast")
    residual.add_argument("--out", required=True, help="Output CSV")

    named_residual = subparsers.add_parser("build-named-residual-shock", help="Build residual shock with explicit output column names")
    named_residual.add_argument("--data", required=True, help="Input wide CSV")
    named_residual.add_argument("--target", required=True, help="Target column")
    named_residual.add_argument("--predictors", required=True, help="Comma-separated predictor columns")
    named_residual.add_argument("--residual-column", required=True, help="Residual output column")
    named_residual.add_argument("--fitted-column", required=True, help="Fitted-value output column")
    named_residual.add_argument("--z-column", required=True, help="Standardized residual output column")
    named_residual.add_argument("--min-train-obs", type=int, default=24, help="Minimum training observations")
    named_residual.add_argument("--month-dummies", action="store_true", help="Include month-of-year dummies in the shock forecast")
    named_residual.add_argument("--trend", action="store_true", help="Include linear trend in the shock forecast")
    named_residual.add_argument("--out", required=True, help="Output CSV")

    lp = subparsers.add_parser("run-lp", help="Run local projections from a CSV")
    lp.add_argument("--data", required=True, help="Input wide CSV")
    lp.add_argument("--shock", required=True, help="Shock column")
    lp.add_argument("--outcomes", required=True, help="Comma-separated outcome columns")
    lp.add_argument("--controls", default="", help="Comma-separated control columns")
    lp.add_argument("--horizons", default="0,1,2,3,4,6,12", help="Comma-separated horizons")
    lp.add_argument("--out", required=True, help="Output CSV")
    lp.add_argument("--lead", action="store_true", help="Use lead-h response instead of cumulative h0-to-h")

    lpiv = subparsers.add_parser("run-lp-iv", help="Run manual 2SLS local projections from a CSV")
    lpiv.add_argument("--data", required=True, help="Input wide CSV")
    lpiv.add_argument("--treatment", required=True, help="Treatment column")
    lpiv.add_argument("--instruments", required=True, help="Comma-separated instrument columns")
    lpiv.add_argument("--outcomes", required=True, help="Comma-separated outcome columns")
    lpiv.add_argument("--controls", default="", help="Comma-separated control columns")
    lpiv.add_argument("--horizons", default="0,1,2,3,4,6,12", help="Comma-separated horizons")
    lpiv.add_argument("--out", required=True, help="Output CSV")
    lpiv.add_argument("--lead", action="store_true", help="Use lead-h response instead of cumulative h0-to-h")

    placebo = subparsers.add_parser("run-lp-iv-placebo", help="Run LP-IV placebo regressions against past outcomes")
    placebo.add_argument("--data", required=True, help="Input wide CSV")
    placebo.add_argument("--treatment", required=True, help="Treatment column")
    placebo.add_argument("--instruments", required=True, help="Comma-separated instrument columns")
    placebo.add_argument("--outcomes", required=True, help="Comma-separated outcome columns")
    placebo.add_argument("--controls", default="", help="Comma-separated control columns")
    placebo.add_argument("--horizons", default="1,2,3,4,6,12", help="Comma-separated backward placebo horizons")
    placebo.add_argument("--out", required=True, help="Output CSV")

    first = subparsers.add_parser("first-stage", help="Run first-stage diagnostics from a CSV")
    first.add_argument("--data", required=True, help="Input wide CSV")
    first.add_argument("--treatment", required=True, help="Treatment column")
    first.add_argument("--instruments", required=True, help="Comma-separated instrument columns")
    first.add_argument("--controls", default="", help="Comma-separated control columns")
    first.add_argument("--out", required=True, help="Output CSV")

    panel = subparsers.add_parser("assemble-panel", help="Join wide CSV files into one dated model panel")
    panel.add_argument("inputs", nargs="+", help="Input wide CSV files")
    panel.add_argument("--prefixes", default="", help="Comma-separated column prefixes, one per input")
    panel.add_argument("--lags", default="", help="Comma-separated columns to lag by one period")
    panel.add_argument("--diffs", default="", help="Comma-separated columns to difference")
    panel.add_argument("--require", default="", help="Comma-separated complete-case columns to require")
    panel.add_argument("--out", required=True, help="Output CSV")

    cal = subparsers.add_parser("add-calendar-controls", help="Add calendar/regime controls to a model panel")
    cal.add_argument("--data", required=True, help="Input wide CSV")
    cal.add_argument("--out", required=True, help="Output CSV")

    pretrend = subparsers.add_parser("add-pretrend-controls", help="Add deeper lags and rolling pretrend sums to a model panel")
    pretrend.add_argument("--data", required=True, help="Input wide CSV")
    pretrend.add_argument("--columns", required=True, help="Comma-separated columns to build pretrend controls from")
    pretrend.add_argument("--lags", default="2,3", help="Comma-separated positive lags to add")
    pretrend.add_argument("--windows", default="3,6", help="Comma-separated rolling pretrend windows")
    pretrend.add_argument("--out", required=True, help="Output CSV")

    summary = subparsers.add_parser("summarize-run", help="Summarize a proxy/envelope run directory")
    summary.add_argument("--run-dir", required=True, help="Run directory")
    summary.add_argument("--out-json", required=True, help="Output JSON summary")
    summary.add_argument("--out-md", default=None, help="Optional Markdown summary")

    panel_summary = subparsers.add_parser("summarize-panel", help="Summarize non-null coverage in a model panel")
    panel_summary.add_argument("--data", required=True, help="Input panel CSV")
    panel_summary.add_argument("--out", required=True, help="Output summary CSV")

    estimates = subparsers.add_parser("summarize-estimates", help="Write compact thesis tables from LP and first-stage CSVs")
    estimates.add_argument("--lp", required=True, help="LP or LP-IV result CSV")
    estimates.add_argument("--first-stage", default=None, help="Optional first-stage CSV")
    estimates.add_argument("--out-dir", required=True, help="Output directory")

    pub = subparsers.add_parser("export-publication-tables", help="Export publication-ready CSV/Markdown tables")
    pub.add_argument("--lp", required=True, help="LP or LP-IV result CSV")
    pub.add_argument("--bootstrap", default=None, help="Optional bootstrap result CSV")
    pub.add_argument("--first-stage", default=None, help="Optional first-stage CSV")
    pub.add_argument("--validation", default=None, help="Optional proxy validation CSV")
    pub.add_argument("--proxy-comparison", default=None, help="Optional quarterly proxy comparison CSV")
    pub.add_argument("--out-dir", required=True, help="Output directory")

    compare_proxy = subparsers.add_parser("compare-proxies", help="Compare two monthly proxy CSVs")
    compare_proxy.add_argument("--left", required=True, help="Left proxy CSV")
    compare_proxy.add_argument("--right", required=True, help="Right proxy CSV")
    compare_proxy.add_argument("--left-label", default="left", help="Left label")
    compare_proxy.add_argument("--right-label", default="right", help="Right label")
    compare_proxy.add_argument("--column", default="tdc_monthly", help="Proxy column")
    compare_proxy.add_argument("--out-dir", required=True, help="Output directory")

    figs = subparsers.add_parser("export-figures", help="Export thesis-ready PNG figures from generated artifacts")
    figs.add_argument("--proxy", required=True, help="Monthly proxy CSV")
    figs.add_argument("--components", default=None, help="Optional monthly components CSV")
    figs.add_argument("--envelope", default=None, help="Optional method envelope CSV")
    figs.add_argument("--lp", default=None, help="Optional LP/LP-IV result CSV")
    figs.add_argument("--out-dir", required=True, help="Output directory")

    boot = subparsers.add_parser("bootstrap-lp-iv", help="Moving-block bootstrap uncertainty for LP-IV estimates")
    boot.add_argument("--data", required=True, help="Input wide CSV")
    boot.add_argument("--treatment", required=True, help="Treatment column")
    boot.add_argument("--instruments", required=True, help="Comma-separated instrument columns")
    boot.add_argument("--outcomes", required=True, help="Comma-separated outcome columns")
    boot.add_argument("--controls", default="", help="Comma-separated control columns")
    boot.add_argument("--horizons", default="0,1,2,3,4,6,12", help="Comma-separated horizons")
    boot.add_argument("--replications", type=int, default=200, help="Bootstrap replications")
    boot.add_argument("--block-length", type=int, default=6, help="Moving block length")
    boot.add_argument("--seed", type=int, default=12345, help="RNG seed")
    boot.add_argument("--out", required=True, help="Output CSV")
    boot.add_argument("--lead", action="store_true", help="Use lead-h response instead of cumulative h0-to-h")

    ivrob = subparsers.add_parser("run-iv-robustness", help="Run LP-IV and first-stage variants across instrument sets")
    ivrob.add_argument("--data", required=True, help="Input wide CSV")
    ivrob.add_argument("--treatment", required=True, help="Treatment column")
    ivrob.add_argument(
        "--instrument-specs",
        required=True,
        help="Comma-separated specs like both=z1+z2,tga=z1,auction=z2",
    )
    ivrob.add_argument("--outcomes", required=True, help="Comma-separated outcome columns")
    ivrob.add_argument("--controls", default="", help="Comma-separated control columns")
    ivrob.add_argument("--horizons", default="0,1,2,3,4,6,12", help="Comma-separated horizons")
    ivrob.add_argument("--out-dir", required=True, help="Output directory")
    ivrob.add_argument("--lead", action="store_true", help="Use lead-h response instead of cumulative h0-to-h")

    ivcmp = subparsers.add_parser("compare-iv-robustness", help="Compare baseline and controlled IV robustness summaries")
    ivcmp.add_argument("--baseline", required=True, help="Baseline robustness summary CSV")
    ivcmp.add_argument("--controlled", required=True, help="Controlled robustness summary CSV")
    ivcmp.add_argument("--out-dir", required=True, help="Output directory")

    lpcmp = subparsers.add_parser("compare-lp-results", help="Compare two LP/LP-IV result CSVs")
    lpcmp.add_argument("--left", required=True, help="Left LP CSV")
    lpcmp.add_argument("--right", required=True, help="Right LP CSV")
    lpcmp.add_argument("--left-label", default="left", help="Left label")
    lpcmp.add_argument("--right-label", default="right", help="Right label")
    lpcmp.add_argument("--out-dir", required=True, help="Output directory")

    thesis = subparsers.add_parser("export-thesis-report", help="Export a consolidated Markdown thesis status report")
    thesis.add_argument("--out-md", required=True, help="Output Markdown path")
    thesis.add_argument("--publication-md", default=None, help="Publication table summary Markdown")
    thesis.add_argument("--iv-robustness-md", default=None, help="IV robustness Markdown")
    thesis.add_argument("--fiscal-flow-iv-md", default=None, help="DTS fiscal-flow IV robustness Markdown")
    thesis.add_argument("--category-flow-iv-md", default=None, help="DTS category-flow IV robustness Markdown")
    thesis.add_argument("--controlled-iv-md", default=None, help="Controlled IV comparison Markdown")
    thesis.add_argument("--lp-comparison-md", default=None, help="LP comparison Markdown")
    thesis.add_argument("--method-comparison-md", default=None, help="Proxy method comparison Markdown")
    thesis.add_argument("--identification-md", default=None, help="Identification risk Markdown")

    auction_shock = subparsers.add_parser("build-auction-size-shock", help="Build monthly auction-size surprise from local auction results")
    auction_shock.add_argument("--raw", required=True, help="Auction result CSV")
    auction_shock.add_argument("--out", default="data/processed/auction_size_shock.csv", help="Output shock CSV")
    auction_shock.add_argument("--date-column", default="issue_date", help="Auction settlement/issue date column")
    auction_shock.add_argument("--amount-column", default="offering_amt", help="Auction amount column")
    auction_shock.add_argument("--tenor-column", default="security_term", help="Tenor/grouping column")
    auction_shock.add_argument("--min-train-obs", type=int, default=12, help="Minimum training observations per tenor")

    tga_shock = subparsers.add_parser("build-tga-rebuild-shock", help="Build residual TGA/TOC shock from a monthly panel")
    tga_shock.add_argument("--data", required=True, help="Input wide CSV")
    tga_shock.add_argument("--target", default="minus_toc", help="Target cash-management column")
    tga_shock.add_argument("--predictors", default="", help="Comma-separated predictor columns")
    tga_shock.add_argument("--min-train-obs", type=int, default=24, help="Minimum training observations")
    tga_shock.add_argument("--out", required=True, help="Output shock CSV")

    shock_bundle = subparsers.add_parser("build-shock-bundle", help="Merge shock CSVs into one dated shock bundle")
    shock_bundle.add_argument("inputs", nargs="+", help="Input shock CSVs")
    shock_bundle.add_argument("--out", required=True, help="Output shock bundle CSV")
    shock_bundle.add_argument("--columns", default="", help="Comma-separated columns to keep after merging")

    spec = subparsers.add_parser("run-analysis-spec", help="Run a YAML analysis pipeline specification")
    spec.add_argument("spec", help="Analysis spec YAML")
    spec.add_argument("--only", default="", help="Comma-separated step ids to run")
    spec.add_argument("--skip-existing", action="store_true", help="Skip steps whose primary output already exists")

    manifest = subparsers.add_parser("write-manifest", help="Write hashes and metadata for generated/source files")
    manifest.add_argument("paths", nargs="+", help="Files to include")
    manifest.add_argument("--root", default=".", help="Root used for relative path display")
    manifest.add_argument("--out", required=True, help="Output manifest CSV")
    manifest.add_argument("--out-json", default=None, help="Optional output manifest JSON")

    spec_manifest = subparsers.add_parser("write-spec-manifest", help="Write a reproducibility manifest for a YAML analysis spec")
    spec_manifest.add_argument("spec", help="Analysis spec YAML")
    spec_manifest.add_argument("--out", required=True, help="Output manifest CSV")
    spec_manifest.add_argument("--out-md", default=None, help="Optional output manifest Markdown")

    methods = subparsers.add_parser("write-method-status", help="Write temporal-disaggregation method status CSV")
    methods.add_argument("--out", required=True, help="Output method status CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        report = run_demo(args.out)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-proxy":
        report = run_monthly_proxy_pipeline(
            out_dir=args.out,
            monthly_indicators_path=args.monthly_indicators,
            quarterly_anchors_path=args.quarterly_anchors,
            benchmark_method=args.benchmark_method,
            fill_missing=not args.no_fill_missing,
            method_label=args.method_label,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-envelope":
        report = build_method_envelope(
            out_dir=args.out,
            monthly_indicators_path=args.monthly_indicators,
            quarterly_anchors_path=args.quarterly_anchors,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "inspect-upstream":
        repos = resolve_repos()
        payload = {
            key: {
                "root": str(repo.root),
                "role": repo.role,
                "exists": repo.root.exists(),
                "preferred_outputs": {
                    name: {"path": str(path), "exists": path.exists()}
                    for name, path in repo.preferred_outputs.items()
                },
            }
            for key, repo in repos.items()
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "doctor":
        print(json.dumps(readiness_payload(), indent=2))
        return 0

    if args.command == "download-fred":
        path = write_fred_series_csv(args.series, args.out)
        print(json.dumps({"status": "ok", "out": str(path), "series": args.series}, indent=2))
        return 0

    if args.command == "download-hf-fred":
        path = write_fred_series_csv(FRED_HF_SERIES, args.out)
        print(json.dumps({"status": "ok", "out": str(path), "series": FRED_HF_SERIES}, indent=2))
        return 0

    if args.command == "download-outcome-fred":
        path = write_fred_series_csv(FRED_OUTCOME_SERIES, args.out)
        print(json.dumps({"status": "ok", "out": str(path), "series": FRED_OUTCOME_SERIES}, indent=2))
        return 0

    if args.command == "build-fred-indicators":
        report = build_fred_monthly_indicators_csv(args.raw, out_csv=args.out, metadata_csv=args.metadata)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-outcomes":
        report = build_monthly_outcomes_csv(args.raw, out_csv=args.out, metadata_csv=args.metadata)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-weekly-state":
        report = build_weekly_state_csv(args.raw, out_csv=args.out)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-weekly-channel-panel":
        report = build_weekly_channel_panel_csv(
            args.inputs,
            out_csv=args.out,
            change_columns=_split_csv_arg(args.changes) or None,
            lag_columns=_split_csv_arg(args.lags) or None,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-tic-row-indicator":
        report = build_tic_row_indicator_csv(args.raw, out_csv=args.out)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-fiscal-indicators":
        report = build_fiscal_indicator_csv(args.raw, out_csv=args.out)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "download-fiscaldata":
        report = write_fiscaldata_csv(
            args.endpoint,
            out_csv=args.out,
            fields=_split_csv_arg(args.fields),
            filters=_split_csv_arg(args.filters),
            sort=args.sort,
            page_size=args.page_size,
            manifest_json=args.manifest_json,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "download-dts-fiscaldata":
        report = download_default_dts_sources(out_dir=args.out_dir, start_date=args.start_date, page_size=args.page_size)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-dts-fiscal-indicators":
        report = build_dts_fiscal_indicators_csv(
            operating_cash_balance_csv=args.operating_cash_balance,
            fed_remit_csv=args.fed_remit,
            out_csv=args.out,
            metadata_csv=args.metadata,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-dts-transaction-indicators":
        report = build_dts_transaction_indicators_csv(
            args.transactions,
            out_csv=args.out,
            metadata_csv=args.metadata,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "merge-indicators":
        report = merge_indicator_csvs(args.inputs, out_csv=args.out)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-auction-context":
        report = build_auction_context(out_csv=args.out, allocation_csv=args.allocation)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-residual-shock":
        report = build_residual_shock_csv(
            args.data,
            target=args.target,
            predictors=_split_csv_arg(args.predictors),
            min_train_obs=args.min_train_obs,
            month_dummies=args.month_dummies,
            trend=args.trend,
            out_csv=args.out,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-named-residual-shock":
        report = build_named_residual_shock_csv(
            args.data,
            target=args.target,
            predictors=_split_csv_arg(args.predictors),
            residual_column=args.residual_column,
            fitted_column=args.fitted_column,
            z_column=args.z_column,
            min_train_obs=args.min_train_obs,
            month_dummies=args.month_dummies,
            trend=args.trend,
            out_csv=args.out,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "run-lp":
        report = run_local_projections_csv(
            args.data,
            shock_col=args.shock,
            outcome_cols=_split_csv_arg(args.outcomes),
            controls=_split_csv_arg(args.controls),
            horizons=_split_int_arg(args.horizons),
            out_csv=args.out,
            cumulative=not args.lead,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "run-lp-iv":
        report = run_lp_iv_csv(
            args.data,
            treatment_col=args.treatment,
            instrument_cols=_split_csv_arg(args.instruments),
            outcome_cols=_split_csv_arg(args.outcomes),
            controls=_split_csv_arg(args.controls),
            horizons=_split_int_arg(args.horizons),
            out_csv=args.out,
            cumulative=not args.lead,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "run-lp-iv-placebo":
        report = run_lp_iv_placebo_csv(
            args.data,
            treatment_col=args.treatment,
            instrument_cols=_split_csv_arg(args.instruments),
            outcome_cols=_split_csv_arg(args.outcomes),
            controls=_split_csv_arg(args.controls),
            placebo_horizons=_split_int_arg(args.horizons),
            out_csv=args.out,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "first-stage":
        report = run_first_stage_csv(
            args.data,
            treatment=args.treatment,
            instruments=_split_csv_arg(args.instruments),
            controls=_split_csv_arg(args.controls),
            out_csv=args.out,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "assemble-panel":
        prefixes = _split_csv_arg(args.prefixes)
        if not prefixes:
            prefixes = [""] * len(args.inputs)
        report = assemble_model_panel_csv(
            args.inputs,
            prefixes=prefixes,
            lags=_split_csv_arg(args.lags),
            diffs=_split_csv_arg(args.diffs),
            require=_split_csv_arg(args.require),
            out_csv=args.out,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "add-calendar-controls":
        report = add_calendar_controls_csv(args.data, out_csv=args.out)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "add-pretrend-controls":
        report = add_pretrend_controls_csv(
            args.data,
            columns=_split_csv_arg(args.columns),
            lags=_split_int_arg(args.lags),
            windows=_split_int_arg(args.windows),
            out_csv=args.out,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "summarize-run":
        report = summarize_proxy_run_cli(args.run_dir, out_json=args.out_json, out_md=args.out_md)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "summarize-panel":
        report = summarize_panel_csv(args.data, out_csv=args.out)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "summarize-estimates":
        report = summarize_estimates_bundle(args.lp, first_stage_csv=args.first_stage, out_dir=args.out_dir)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "export-publication-tables":
        report = export_publication_tables(
            lp_csv=args.lp,
            bootstrap_csv=args.bootstrap,
            first_stage_csv=args.first_stage,
            validation_csv=args.validation,
            proxy_comparison_csv=args.proxy_comparison,
            out_dir=args.out_dir,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "compare-proxies":
        report = compare_monthly_proxies(
            args.left,
            args.right,
            out_dir=args.out_dir,
            left_label=args.left_label,
            right_label=args.right_label,
            column=args.column,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "export-figures":
        report = export_thesis_figures(
            proxy_csv=args.proxy,
            components_csv=args.components,
            envelope_csv=args.envelope,
            lp_csv=args.lp,
            out_dir=args.out_dir,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "bootstrap-lp-iv":
        report = bootstrap_lp_iv_csv(
            args.data,
            treatment=args.treatment,
            instruments=_split_csv_arg(args.instruments),
            outcomes=_split_csv_arg(args.outcomes),
            controls=_split_csv_arg(args.controls),
            horizons=_split_int_arg(args.horizons),
            replications=args.replications,
            block_length=args.block_length,
            seed=args.seed,
            out_csv=args.out,
            cumulative=not args.lead,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "run-iv-robustness":
        report = run_iv_robustness_csv(
            args.data,
            treatment=args.treatment,
            instrument_specs=_split_csv_arg(args.instrument_specs),
            outcomes=_split_csv_arg(args.outcomes),
            controls=_split_csv_arg(args.controls),
            horizons=_split_int_arg(args.horizons),
            out_dir=args.out_dir,
            cumulative=not args.lead,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "compare-iv-robustness":
        report = compare_iv_robustness_summaries(args.baseline, args.controlled, out_dir=args.out_dir)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "compare-lp-results":
        report = compare_lp_results(
            args.left,
            args.right,
            out_dir=args.out_dir,
            left_label=args.left_label,
            right_label=args.right_label,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "export-thesis-report":
        report = export_thesis_status_report(
            out_md=args.out_md,
            publication_md=args.publication_md,
            iv_robustness_md=args.iv_robustness_md,
            fiscal_flow_iv_md=args.fiscal_flow_iv_md,
            category_flow_iv_md=args.category_flow_iv_md,
            controlled_iv_md=args.controlled_iv_md,
            lp_comparison_md=args.lp_comparison_md,
            method_comparison_md=args.method_comparison_md,
            identification_md=args.identification_md,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-auction-size-shock":
        report = build_auction_size_shock(
            args.raw,
            out_csv=args.out,
            date_column=args.date_column,
            amount_column=args.amount_column,
            tenor_column=args.tenor_column,
            min_train_obs=args.min_train_obs,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-tga-rebuild-shock":
        report = build_tga_rebuild_shock_csv(
            args.data,
            target=args.target,
            predictors=_split_csv_arg(args.predictors),
            min_train_obs=args.min_train_obs,
            out_csv=args.out,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "build-shock-bundle":
        report = build_shock_bundle_csv(args.inputs, out_csv=args.out, columns=_split_csv_arg(args.columns) or None)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "run-analysis-spec":
        report = run_analysis_spec(args.spec, only=_split_csv_arg(args.only) or None, skip_existing=args.skip_existing)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "write-manifest":
        report = write_file_manifest(args.paths, root=args.root, out_csv=args.out, out_json=args.out_json)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "write-spec-manifest":
        report = write_spec_manifest(args.spec, out_csv=args.out, out_md=args.out_md)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "write-method-status":
        report = write_method_status_csv(args.out)
        print(json.dumps(report, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
