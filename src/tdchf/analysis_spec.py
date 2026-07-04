from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .auction_shocks import build_auction_size_shock, build_shock_bundle_csv, build_tga_rebuild_shock_csv
from .accounting import export_accounting_decomposition
from .anchor_variants import run_anchor_variant_robustness_csv
from .bootstrap import bootstrap_lp_iv_csv
from .calendar_controls import add_calendar_controls_csv
from .closeout import (
    export_anchor_contract_audit,
    export_core_iv_vs_noniv_profiles,
    export_method_envelope_closeout,
    export_noniv_tdc_lp_closeout,
    export_placebo_summary,
    export_short_run_core_closeout,
    export_tga_reduced_form_closeout,
)
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
from .pretrend import add_lagged_factor_controls_csv, add_pretrend_controls_csv
from .regime import run_regime_exclusion_robustness_csv
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
from .thesis_outputs import export_anchor_variant_forest_plot, export_claim_status_table, export_short_run_profile_plot
from .weekly import build_weekly_channel_panel_csv, build_weekly_state_csv


def _load_spec(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Analysis spec must be a YAML mapping")
    return payload


def _resolve_path(value: str | Path | None, *, root: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path)


def _resolve_paths(values: list[str] | None, *, root: Path) -> list[str]:
    return [_resolve_path(value, root=root) or "" for value in values or []]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part) for part in value]
    raise TypeError(f"Expected list or comma-separated string, got {type(value).__name__}")


def _as_int_list(value: Any) -> list[int]:
    return [int(part) for part in _as_list(value)]


def _step_id(step: dict[str, Any], position: int) -> str:
    return str(step.get("id") or step.get("action") or f"step_{position + 1}")


def run_analysis_spec(
    spec_path: str | Path,
    *,
    only: list[str] | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    spec_file = Path(spec_path).expanduser().resolve()
    spec = _load_spec(spec_file)
    root = Path(spec.get("root", ".")).expanduser()
    if not root.is_absolute():
        root = (spec_file.parent / root).resolve()
    selected = set(only or [])
    reports: list[dict[str, Any]] = []

    steps = spec.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("Analysis spec `steps` must be a list")

    for position, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            raise ValueError(f"Step {position + 1} must be a mapping")
        step = dict(raw_step)
        ident = _step_id(step, position)
        action = step.get("action")
        if not action:
            raise ValueError(f"Step `{ident}` is missing an action")
        if step.get("enabled", True) is False:
            reports.append({"id": ident, "action": action, "status": "disabled"})
            continue
        if selected and ident not in selected:
            reports.append({"id": ident, "action": action, "status": "skipped_by_filter"})
            continue
        out = _resolve_path(step.get("out"), root=root)
        out_dir = _resolve_path(step.get("out_dir"), root=root)
        primary_output = out or out_dir
        if skip_existing and primary_output and Path(primary_output).exists():
            reports.append({"id": ident, "action": action, "status": "skipped_existing", "out": primary_output})
            continue

        if action == "download-fred":
            report = {"status": "ok", "out": str(write_fred_series_csv(_as_list(step["series"]), out or ""))}
        elif action == "download-hf-fred":
            report = {"status": "ok", "out": str(write_fred_series_csv(FRED_HF_SERIES, out or "")), "series": FRED_HF_SERIES}
        elif action == "download-outcome-fred":
            report = {"status": "ok", "out": str(write_fred_series_csv(FRED_OUTCOME_SERIES, out or "")), "series": FRED_OUTCOME_SERIES}
        elif action == "build-fred-indicators":
            report = build_fred_monthly_indicators_csv(
                _resolve_path(step.get("raw"), root=root) or "data/raw/fred_hf_sources.csv",
                out_csv=out or "data/processed/fred_monthly_indicators.csv",
                metadata_csv=_resolve_path(step.get("metadata"), root=root) or "data/processed/fred_monthly_indicator_metadata.csv",
            )
        elif action == "build-outcomes":
            report = build_monthly_outcomes_csv(
                _resolve_path(step.get("raw"), root=root) or "data/raw/fred_outcome_sources.csv",
                out_csv=out or "data/processed/fred_monthly_outcomes.csv",
                metadata_csv=_resolve_path(step.get("metadata"), root=root) or "data/processed/fred_monthly_outcome_metadata.csv",
            )
        elif action == "build-weekly-state":
            report = build_weekly_state_csv(_resolve_path(step.get("raw"), root=root) or "", out_csv=out or "")
        elif action == "build-weekly-channel-panel":
            report = build_weekly_channel_panel_csv(
                _resolve_paths(step.get("inputs"), root=root),
                out_csv=out or "",
                change_columns=_as_list(step.get("changes")) or None,
                lag_columns=_as_list(step.get("lags")) or None,
            )
        elif action == "build-proxy":
            report = run_monthly_proxy_pipeline(
                out_dir=out_dir or out or "data/processed",
                monthly_indicators_path=_resolve_path(step.get("monthly_indicators"), root=root),
                quarterly_anchors_path=_resolve_path(step.get("quarterly_anchors"), root=root),
                benchmark_method=str(step.get("benchmark_method", "residual_spread")),
                fill_missing=bool(step.get("fill_missing", True)),
                method_label=step.get("method_label"),
            )
        elif action == "build-envelope":
            report = build_method_envelope(
                out_dir=out_dir or out or "data/processed/envelope",
                monthly_indicators_path=_resolve_path(step.get("monthly_indicators"), root=root),
                quarterly_anchors_path=_resolve_path(step.get("quarterly_anchors"), root=root),
            )
        elif action == "build-tic-row-indicator":
            report = build_tic_row_indicator_csv(_resolve_path(step.get("raw"), root=root) or "", out_csv=out or "")
        elif action == "build-fiscal-indicators":
            report = build_fiscal_indicator_csv(_resolve_path(step.get("raw"), root=root) or "", out_csv=out or "")
        elif action == "download-fiscaldata":
            report = write_fiscaldata_csv(
                str(step["endpoint"]),
                out_csv=out or "",
                fields=_as_list(step.get("fields")),
                filters=_as_list(step.get("filters")),
                sort=str(step.get("sort", "record_date")),
                page_size=int(step.get("page_size", 10_000)),
                manifest_json=_resolve_path(step.get("manifest_json"), root=root),
            )
        elif action == "download-dts-fiscaldata":
            report = download_default_dts_sources(
                out_dir=out_dir or out or "data/raw/fiscaldata",
                start_date=str(step.get("start_date", "2005-01-01")),
                page_size=int(step.get("page_size", 10_000)),
            )
        elif action == "build-dts-fiscal-indicators":
            report = build_dts_fiscal_indicators_csv(
                operating_cash_balance_csv=_resolve_path(step.get("operating_cash_balance"), root=root)
                or "data/raw/fiscaldata/dts_operating_cash_balance.csv",
                fed_remit_csv=_resolve_path(step.get("fed_remit"), root=root),
                out_csv=out or "data/processed/dts_fiscal_monthly_indicators.csv",
                metadata_csv=_resolve_path(step.get("metadata"), root=root),
            )
        elif action == "build-dts-transaction-indicators":
            report = build_dts_transaction_indicators_csv(
                _resolve_path(step.get("transactions"), root=root) or "data/raw/fiscaldata/dts_deposits_withdrawals_operating_cash.csv",
                out_csv=out or "data/processed/dts_transaction_monthly_indicators.csv",
                metadata_csv=_resolve_path(step.get("metadata"), root=root),
            )
        elif action == "merge-indicators":
            report = merge_indicator_csvs(_resolve_paths(step.get("inputs"), root=root), out_csv=out or "")
        elif action == "assemble-panel":
            report = assemble_model_panel_csv(
                _resolve_paths(step.get("inputs"), root=root),
                prefixes=_as_list(step.get("prefixes")) or None,
                lags=_as_list(step.get("lags")),
                diffs=_as_list(step.get("diffs")),
                require=_as_list(step.get("require")),
                out_csv=out or "",
            )
        elif action == "add-calendar-controls":
            report = add_calendar_controls_csv(_resolve_path(step.get("data"), root=root) or "", out_csv=out or "")
        elif action == "add-pretrend-controls":
            report = add_pretrend_controls_csv(
                _resolve_path(step.get("data"), root=root) or "",
                columns=_as_list(step.get("columns")),
                lags=_as_int_list(step.get("lags")) or [2, 3],
                windows=_as_int_list(step.get("windows")) or [3, 6],
                out_csv=out or "",
            )
        elif action == "add-factor-controls":
            report = add_lagged_factor_controls_csv(
                _resolve_path(step.get("data"), root=root) or "",
                columns=_as_list(step.get("columns")),
                n_factors=int(step.get("n_factors", 3)),
                lag=int(step.get("lag", 1)),
                prefix=str(step.get("prefix", "factor")),
                out_csv=out or "",
            )
        elif action == "summarize-panel":
            report = summarize_panel_csv(_resolve_path(step.get("data"), root=root) or "", out_csv=out or "")
        elif action == "summarize-estimates":
            report = summarize_estimates_bundle(
                _resolve_path(step.get("lp"), root=root) or "",
                first_stage_csv=_resolve_path(step.get("first_stage"), root=root),
                out_dir=out_dir or out or "",
            )
        elif action == "export-publication-tables":
            report = export_publication_tables(
                lp_csv=_resolve_path(step.get("lp"), root=root) or "",
                bootstrap_csv=_resolve_path(step.get("bootstrap"), root=root),
                first_stage_csv=_resolve_path(step.get("first_stage"), root=root),
                validation_csv=_resolve_path(step.get("validation"), root=root),
                proxy_comparison_csv=_resolve_path(step.get("proxy_comparison"), root=root),
                out_dir=out_dir or out or "",
            )
        elif action == "compare-proxies":
            report = compare_monthly_proxies(
                _resolve_path(step.get("left"), root=root) or "",
                _resolve_path(step.get("right"), root=root) or "",
                out_dir=out_dir or out or "",
                left_label=str(step.get("left_label", "left")),
                right_label=str(step.get("right_label", "right")),
                column=str(step.get("column", "tdc_monthly")),
            )
        elif action == "export-figures":
            report = export_thesis_figures(
                proxy_csv=_resolve_path(step.get("proxy"), root=root) or "",
                components_csv=_resolve_path(step.get("components"), root=root),
                envelope_csv=_resolve_path(step.get("envelope"), root=root),
                lp_csv=_resolve_path(step.get("lp"), root=root),
                out_dir=out_dir or out or "",
            )
        elif action == "summarize-run":
            report = summarize_proxy_run_cli(
                _resolve_path(step.get("run_dir"), root=root) or "",
                out_json=_resolve_path(step.get("out_json"), root=root) or out or "",
                out_md=_resolve_path(step.get("out_md"), root=root),
            )
        elif action == "build-residual-shock":
            report = build_residual_shock_csv(
                _resolve_path(step.get("data"), root=root) or "",
                target=str(step["target"]),
                predictors=_as_list(step.get("predictors")),
                min_train_obs=int(step.get("min_train_obs", 24)),
                month_dummies=bool(step.get("month_dummies", False)),
                trend=bool(step.get("trend", False)),
                out_csv=out or "",
            )
        elif action == "build-named-residual-shock":
            report = build_named_residual_shock_csv(
                _resolve_path(step.get("data"), root=root) or "",
                target=str(step["target"]),
                predictors=_as_list(step.get("predictors")),
                residual_column=str(step["residual_column"]),
                fitted_column=str(step["fitted_column"]),
                z_column=str(step["z_column"]),
                min_train_obs=int(step.get("min_train_obs", 24)),
                month_dummies=bool(step.get("month_dummies", False)),
                trend=bool(step.get("trend", False)),
                out_csv=out or "",
            )
        elif action == "build-tga-rebuild-shock":
            report = build_tga_rebuild_shock_csv(
                _resolve_path(step.get("data"), root=root) or "",
                target=str(step.get("target", "minus_toc")),
                predictors=_as_list(step.get("predictors")),
                min_train_obs=int(step.get("min_train_obs", 24)),
                out_csv=out or "",
            )
        elif action == "build-auction-size-shock":
            report = build_auction_size_shock(
                _resolve_path(step.get("raw"), root=root) or "",
                out_csv=out or "",
                date_column=str(step.get("date_column", "issue_date")),
                amount_column=str(step.get("amount_column", "offering_amt")),
                tenor_column=str(step.get("tenor_column", "security_term")),
                min_train_obs=int(step.get("min_train_obs", 12)),
            )
        elif action == "build-shock-bundle":
            report = build_shock_bundle_csv(_resolve_paths(step.get("inputs"), root=root), out_csv=out or "", columns=_as_list(step.get("columns")) or None)
        elif action == "first-stage":
            report = run_first_stage_csv(
                _resolve_path(step.get("data"), root=root) or "",
                treatment=str(step["treatment"]),
                instruments=_as_list(step.get("instruments")),
                controls=_as_list(step.get("controls")),
                out_csv=out or "",
            )
        elif action == "run-lp":
            report = run_local_projections_csv(
                _resolve_path(step.get("data"), root=root) or "",
                shock_col=str(step["shock"]),
                outcome_cols=_as_list(step.get("outcomes")),
                controls=_as_list(step.get("controls")),
                horizons=_as_int_list(step.get("horizons")) or [0, 1, 2, 3, 4, 6, 12],
                out_csv=out or "",
                cumulative=bool(step.get("cumulative", True)),
            )
        elif action == "run-lp-iv":
            report = run_lp_iv_csv(
                _resolve_path(step.get("data"), root=root) or "",
                treatment_col=str(step["treatment"]),
                instrument_cols=_as_list(step.get("instruments")),
                outcome_cols=_as_list(step.get("outcomes")),
                controls=_as_list(step.get("controls")),
                horizons=_as_int_list(step.get("horizons")) or [0, 1, 2, 3, 4, 6, 12],
                out_csv=out or "",
                cumulative=bool(step.get("cumulative", True)),
            )
        elif action == "run-lp-iv-placebo":
            report = run_lp_iv_placebo_csv(
                _resolve_path(step.get("data"), root=root) or "",
                treatment_col=str(step["treatment"]),
                instrument_cols=_as_list(step.get("instruments")),
                outcome_cols=_as_list(step.get("outcomes")),
                controls=_as_list(step.get("controls")),
                placebo_horizons=_as_int_list(step.get("horizons")) or [1, 2, 3, 4, 6, 12],
                out_csv=out or "",
            )
        elif action == "bootstrap-lp-iv":
            report = bootstrap_lp_iv_csv(
                _resolve_path(step.get("data"), root=root) or "",
                treatment=str(step["treatment"]),
                instruments=_as_list(step.get("instruments")),
                outcomes=_as_list(step.get("outcomes")),
                controls=_as_list(step.get("controls")),
                horizons=_as_int_list(step.get("horizons")) or [0, 1, 2, 3, 4, 6, 12],
                replications=int(step.get("replications", 200)),
                block_length=int(step.get("block_length", 6)),
                seed=int(step.get("seed", 12345)),
                out_csv=out or "",
                cumulative=bool(step.get("cumulative", True)),
            )
        elif action == "run-iv-robustness":
            report = run_iv_robustness_csv(
                _resolve_path(step.get("data"), root=root) or "",
                treatment=str(step["treatment"]),
                instrument_specs=_as_list(step.get("instrument_specs")),
                outcomes=_as_list(step.get("outcomes")),
                controls=_as_list(step.get("controls")),
                horizons=_as_int_list(step.get("horizons")) or [0, 1, 2, 3, 4, 6, 12],
                out_dir=out_dir or out or "",
                cumulative=bool(step.get("cumulative", True)),
            )
        elif action == "run-regime-exclusion-robustness":
            report = run_regime_exclusion_robustness_csv(
                _resolve_path(step.get("data"), root=root) or "",
                treatment=str(step["treatment"]),
                instruments=_as_list(step.get("instruments")),
                outcomes=_as_list(step.get("outcomes")),
                controls=_as_list(step.get("controls")),
                horizons=_as_int_list(step.get("horizons")) or [0, 1, 2, 3, 4, 6, 12],
                out_dir=out_dir or out or "",
                cumulative=bool(step.get("cumulative", True)),
            )
        elif action == "run-anchor-variant-robustness":
            report = run_anchor_variant_robustness_csv(
                tdcest_estimates_csv=_resolve_path(step.get("tdcest_estimates"), root=root),
                monthly_indicators_csv=_resolve_path(step.get("monthly_indicators"), root=root) or "",
                base_panel_csv=_resolve_path(step.get("base_panel"), root=root) or "",
                out_dir=out_dir or out or "",
                variant_specs=_as_list(step.get("variant_specs")) or None,
                treatment=str(step.get("treatment", "tdc_monthly")),
                instruments=_as_list(step.get("instruments")) or ["tga_long_surprise_z"],
                outcomes=_as_list(step.get("outcomes")),
                controls=_as_list(step.get("controls")),
                horizons=_as_int_list(step.get("horizons")) or [0, 1, 2, 3, 4, 6, 12],
                bootstrap_outcomes=_as_list(step.get("bootstrap_outcomes")) or _as_list(step.get("outcomes")),
                bootstrap_replications=int(step.get("bootstrap_replications", 100)),
                block_length=int(step.get("block_length", 6)),
                seed=int(step.get("seed", 20260502)),
            )
        elif action == "export-accounting-decomposition":
            report = export_accounting_decomposition(
                _resolve_path(step.get("lp"), root=root) or "",
                horizons=_as_int_list(step.get("horizons")) or [0, 3, 6, 12],
                outcomes=_as_list(step.get("outcomes")) or None,
                out_dir=out_dir or out or "",
            )
        elif action == "compare-iv-robustness":
            report = compare_iv_robustness_summaries(
                _resolve_path(step.get("baseline"), root=root) or "",
                _resolve_path(step.get("controlled"), root=root) or "",
                out_dir=out_dir or out or "",
            )
        elif action == "compare-lp-results":
            report = compare_lp_results(
                _resolve_path(step.get("left"), root=root) or "",
                _resolve_path(step.get("right"), root=root) or "",
                out_dir=out_dir or out or "",
                left_label=str(step.get("left_label", "left")),
                right_label=str(step.get("right_label", "right")),
            )
        elif action == "export-claim-status":
            report = export_claim_status_table(
                anchor_dashboard_csv=_resolve_path(step.get("anchor_dashboard"), root=root) or "",
                regime_summary_csv=_resolve_path(step.get("regime_summary"), root=root),
                outcomes=_as_list(step.get("outcomes")) or None,
                out_csv=out or "",
                out_md=_resolve_path(step.get("out_md"), root=root),
            )
        elif action == "export-anchor-variant-forest":
            report = export_anchor_variant_forest_plot(
                anchor_dashboard_csv=_resolve_path(step.get("anchor_dashboard"), root=root) or "",
                outcomes=_as_list(step.get("outcomes")) or None,
                out_dir=out_dir or out or "",
            )
        elif action == "export-short-run-profile":
            report = export_short_run_profile_plot(
                anchor_dashboard_csv=_resolve_path(step.get("anchor_dashboard"), root=root) or "",
                outcomes=_as_list(step.get("outcomes")) or None,
                horizons=_as_int_list(step.get("horizons")) or None,
                out_dir=out_dir or out or "",
            )
        elif action == "export-anchor-contract-audit":
            report = export_anchor_contract_audit(
                monthly_proxy_csv=_resolve_path(step.get("monthly_proxy"), root=root) or "",
                tdcest_estimates_csv=_resolve_path(step.get("tdcest_estimates"), root=root),
                candidate_anchors=_as_list(step.get("candidate_anchors")) or None,
                out_csv=out or "",
                out_md=_resolve_path(step.get("out_md"), root=root),
            )
        elif action == "export-noniv-tdc-lp-closeout":
            report = export_noniv_tdc_lp_closeout(
                pretrend_panel_csv=_resolve_path(step.get("pretrend_panel"), root=root) or "",
                calendar_panel_csv=_resolve_path(step.get("calendar_panel"), root=root) or "",
                outcomes=_as_list(step.get("outcomes")) or None,
                horizons=_as_int_list(step.get("horizons")) or None,
                pretrend_controls=_as_list(step.get("pretrend_controls")),
                calendar_controls=_as_list(step.get("calendar_controls")),
                out_csv=out or "",
                out_md=_resolve_path(step.get("out_md"), root=root),
            )
        elif action == "export-tga-reduced-form-closeout":
            report = export_tga_reduced_form_closeout(
                pretrend_panel_csv=_resolve_path(step.get("pretrend_panel"), root=root) or "",
                calendar_panel_csv=_resolve_path(step.get("calendar_panel"), root=root) or "",
                outcomes=_as_list(step.get("outcomes")) or None,
                horizons=_as_int_list(step.get("horizons")) or None,
                pretrend_controls=_as_list(step.get("pretrend_controls")),
                calendar_controls=_as_list(step.get("calendar_controls")),
                out_csv=out or "",
                out_md=_resolve_path(step.get("out_md"), root=root),
            )
        elif action == "export-placebo-summary":
            report = export_placebo_summary(
                placebo_csv=_resolve_path(step.get("placebo"), root=root) or "",
                model=str(step.get("model", "long_tga_pretrend_iv")),
                out_csv=out or "",
                out_md=_resolve_path(step.get("out_md"), root=root),
            )
        elif action == "export-short-run-core-closeout":
            report = export_short_run_core_closeout(
                pretrend_iv_csv=_resolve_path(step.get("pretrend_iv"), root=root) or "",
                calendar_iv_csv=_resolve_path(step.get("calendar_iv"), root=root) or "",
                bootstrap_csv=_resolve_path(step.get("bootstrap"), root=root),
                placebo_csv=_resolve_path(step.get("placebo"), root=root),
                outcomes=_as_list(step.get("outcomes")) or None,
                horizons=_as_int_list(step.get("horizons")) or None,
                out_csv=out or "",
                out_md=_resolve_path(step.get("out_md"), root=root),
            )
        elif action == "export-method-envelope-closeout":
            report = export_method_envelope_closeout(
                proxy_monthly_comparison_csv=_resolve_path(step.get("proxy_monthly_comparison"), root=root) or "",
                proxy_quarterly_comparison_csv=_resolve_path(step.get("proxy_quarterly_comparison"), root=root) or "",
                out_csv=out or "",
                out_md=_resolve_path(step.get("out_md"), root=root),
            )
        elif action == "export-core-iv-vs-noniv-profiles":
            report = export_core_iv_vs_noniv_profiles(
                short_run_iv_csv=_resolve_path(step.get("short_run_iv"), root=root) or "",
                noniv_csv=_resolve_path(step.get("noniv"), root=root) or "",
                tga_rf_csv=_resolve_path(step.get("tga_rf"), root=root) or "",
                outcomes=_as_list(step.get("outcomes")) or None,
                horizons=_as_int_list(step.get("horizons")) or None,
                out_dir=out_dir or out or "",
            )
        elif action == "export-thesis-report":
            report = export_thesis_status_report(
                out_md=out or "",
                publication_md=_resolve_path(step.get("publication_md"), root=root),
                iv_robustness_md=_resolve_path(step.get("iv_robustness_md"), root=root),
                fiscal_flow_iv_md=_resolve_path(step.get("fiscal_flow_iv_md"), root=root),
                category_flow_iv_md=_resolve_path(step.get("category_flow_iv_md"), root=root),
                controlled_iv_md=_resolve_path(step.get("controlled_iv_md"), root=root),
                lp_comparison_md=_resolve_path(step.get("lp_comparison_md"), root=root),
                method_comparison_md=_resolve_path(step.get("method_comparison_md"), root=root),
                identification_md=_resolve_path(step.get("identification_md"), root=root),
            )
        elif action == "write-manifest":
            report = write_file_manifest(
                _resolve_paths(step.get("paths"), root=root),
                out_csv=out or "",
                root=root,
                out_json=_resolve_path(step.get("out_json"), root=root),
            )
        elif action == "write-spec-manifest":
            report = write_spec_manifest(
                _resolve_path(step.get("spec"), root=root) or str(spec_file),
                out_csv=out or "",
                out_md=_resolve_path(step.get("out_md"), root=root),
            )
        elif action == "write-method-status":
            report = write_method_status_csv(out or "")
        else:
            raise ValueError(f"Unknown analysis spec action: {action}")

        reports.append({"id": ident, "action": action, **report})

    run_report = {"status": "ok", "spec": str(spec_file), "root": str(root), "steps": reports}
    report_path = _resolve_path(spec.get("report"), root=root)
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run_report, indent=2) + "\n", encoding="utf-8")
        run_report["report"] = str(path)
    return run_report
