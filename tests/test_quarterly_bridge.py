from __future__ import annotations

import re

import pandas as pd
import pytest

import tdchf.quarterly_bridge as bridge


EXPECTED_COLUMNS = (
    "schema_version",
    "row_type",
    "lane",
    "status",
    "units",
    "input_bundle_sha256",
    "mapping_contract_sha256",
    "source_id",
    "source_path",
    "source_sha256",
    "source_retrieved_at",
    "source_endpoint",
    "source_units",
    "unit_scale_to_bn",
    "record_date",
    "h8_week_start",
    "h8_week_end",
    "aggregation_quarter",
    "aggregation_quarter_clock",
    "h8_endpoint_quarter",
    "quarter_end_date",
    "quarter_end_age_calendar_days",
    "quarter_end_age_bin",
    "h8_cross_quarter_flag",
    "h8_window_complete_flag",
    "stream_id",
    "component_id",
    "perimeter_role",
    "timing_status",
    "treatment_perimeter",
    "outcome_perimeter",
    "sign_convention",
    "amount_bn",
    "source_row_count",
    "source_category_count",
    "source_keyset_sha256",
    "coverage_class",
    "coverage_row_count",
    "coverage_amount_bn",
    "coverage_keyset_sha256",
    "coverage_start_date",
    "coverage_end_date",
    "reconciliation_level",
    "reference_id",
    "reference_value_bn",
    "aggregation_residual_bn",
    "residual_tolerance_bn",
    "anchor_id",
    "anchor_role",
    "anchor_contract_id",
    "anchor_source_column",
    "anchor_value_bn",
)

EXPECTED_STREAMS = (
    "gross_credits",
    "gross_drains",
    "gross_issuance",
    "gross_redemption",
    "sidecar_broad_outflows",
    "sidecar_interest_outflows",
)

EXPECTED_ROW_TYPES = (
    "event_daily",
    "h8_reconciliation",
    "quarter_reconciliation",
    "input_coverage",
    "quarter_anchor",
)

EXPECTED_COVERAGE_CLASSES = (
    "input_all",
    "eligible_observed",
    "mapped_observed",
    "excluded_reference_or_total",
    "excluded_seam_overlap",
    "excluded_outside_perimeter",
    "excluded_unmapped",
)

TRANSACTION_SOURCE = "dts_deposits_withdrawals_operating_cash"
REFUND_SOURCE = "dts_income_tax_refunds_issued"
TAX_SOURCE = "dts_federal_tax_deposits"


def _observed_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions = pd.DataFrame(
        {
            "record_date": ["2024-03-29"] * 14,
            "account_type": ["Federal Reserve Account"] * 13
            + ["Treasury General Account Total Withdrawals"],
            "transaction_type": [
                "Withdrawals",
                "Withdrawals",
                "Deposits",
                "Deposits",
                "Deposits",
                "Deposits",
                "Deposits",
                "Withdrawals",
                "Withdrawals",
                "Withdrawals",
                "Withdrawals",
                "Deposits",
                "Deposits",
                "Withdrawals",
            ],
            "transaction_catg": [
                "SSA - Benefits Payments",
                "Federal Salaries (EFT)",
                "Taxes - Withheld Individual/FICA",
                "Taxes - Non Withheld Ind/SECA",
                "Taxes - Corporate Income",
                "Taxes - Misc",
                "Public Debt Cash Issues (Table III-B)",
                "Public Debt Cash Redemp. (Table IIIB)",
                "HHS - Grants to States for Medicaid",
                "Interest on Treasury Securities",
                "IRS Tax Refunds",
                "Other Deposits",
                "Novel Mystery Flow",
                None,
            ],
            "transaction_today_amt": [
                "1000",
                "200",
                "500",
                "400",
                "300",
                "100",
                "300",
                "200",
                "900",
                "-100",
                "888",
                "700",
                "600",
                "9999",
            ],
            "src_line_nbr": range(1, 15),
        }
    )
    refunds = pd.DataFrame(
        {
            "record_date": ["2024-03-29", "2024-03-29"],
            "table_nbr": ["VI", "VI"],
            "sub_table_name": ["Classification", "Classification"],
            "tax_refund_type": ["Individual", "Novel Refund Kind"],
            "tax_refund_today_amt": ["50", "40"],
            "src_line_nbr": [1, 2],
        }
    )
    tax_deposits = pd.DataFrame(
        {
            "record_date": [
                "2023-02-13",
                "2023-02-13",
                "2023-02-13",
                "2023-02-14",
            ],
            "table_nbr": ["IV"] * 4,
            "sub_table_name": ["Classification"] * 4,
            "tax_deposit_type": [
                "Withheld Income and Employment Taxes",
                "Cash Federal Tax Deposits",
                "Novel Tax Kind",
                "Withheld Income and Employment Taxes",
            ],
            "tax_deposit_today_amt": ["1", "99", "3", "4"],
            "src_line_nbr": [1, 2, 3, 1],
        }
    )
    return transactions, refunds, tax_deposits


def _classify() -> pd.DataFrame:
    return bridge.classify_observed_sources(*_observed_inputs())


def _flatten_source_keys(values: pd.Series) -> list[str]:
    return [key for keys in values for key in keys]


def test_final_schema_and_contract_literals_are_pinned() -> None:
    assert bridge.SCHEMA_VERSION == "quarterly_hf_bridge_v1"
    assert tuple(bridge.BRIDGE_COLUMNS) == EXPECTED_COLUMNS
    assert tuple(bridge.STREAM_ORDER) == EXPECTED_STREAMS
    assert tuple(bridge.ROW_TYPES) == EXPECTED_ROW_TYPES
    assert tuple(bridge.COVERAGE_CLASSES) == EXPECTED_COVERAGE_CLASSES
    assert tuple(bridge.AGE_BIN_ORDER) == ("0-7", "8-30", "31-60", "61+")

    digest = bridge.mapping_contract_sha256()
    assert digest == bridge.mapping_contract_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_raw_keys_and_coverage_classes_form_an_exact_partition() -> None:
    transactions, refunds, tax_deposits = _observed_inputs()
    classified = bridge.classify_observed_sources(
        transactions,
        refunds,
        tax_deposits,
    )

    assert len(classified) == len(transactions) + len(refunds) + len(tax_deposits)
    assert classified["_raw_key"].is_unique
    assert not classified[
        ["source_id", "record_date", "src_line_nbr", "_raw_key", "coverage_class"]
    ].isna().any(axis=None)
    expected_keys = {
        f"{source_id}\x1f{pd.Timestamp(record_date).date().isoformat()}\x1f{src_line_nbr}"
        for source_id, record_date, src_line_nbr in classified[
            ["source_id", "record_date", "src_line_nbr"]
        ].itertuples(index=False, name=None)
    }
    assert set(classified["_raw_key"]) == expected_keys

    assert set(classified["coverage_class"]) == {
        "mapped_observed",
        "excluded_reference_or_total",
        "excluded_seam_overlap",
        "excluded_outside_perimeter",
        "excluded_unmapped",
    }
    mapped = classified.loc[classified["coverage_class"].eq("mapped_observed")]
    excluded = classified.loc[~classified["coverage_class"].eq("mapped_observed")]
    assert len(mapped) + len(excluded) == len(classified)
    assert set(mapped["_raw_key"]).isdisjoint(excluded["_raw_key"])
    assert set(mapped["stream_id"]) == set(EXPECTED_STREAMS)
    assert mapped[["stream_id", "component_id"]].notna().all(axis=None)
    assert classified.loc[
        classified["transaction_catg"].eq("Other Deposits"),
        "coverage_class",
    ].item() == "excluded_outside_perimeter"
    assert classified.loc[
        classified["tax_refund_type"].eq("Novel Refund Kind"),
        "coverage_class",
    ].item() == "excluded_unmapped"
    assert classified.loc[
        classified["tax_deposit_type"].eq("Novel Tax Kind"),
        "coverage_class",
    ].item() == "excluded_unmapped"
    assert classified.loc[
        classified["record_date"].eq(pd.Timestamp("2023-02-14"))
        & classified["source_id"].eq(TAX_SOURCE),
        "coverage_class",
    ].item() == "excluded_seam_overlap"

    interest = mapped.loc[
        mapped["stream_id"].eq("sidecar_interest_outflows"),
        "amount_bn",
    ]
    assert interest.item() == pytest.approx(-0.1)


def test_raw_identity_ignores_mutable_labels_and_duplicate_keys_fail() -> None:
    transactions, refunds, tax_deposits = _observed_inputs()
    duplicate = transactions.iloc[[0]].copy()
    duplicate["transaction_catg"] = "Changed Mutable Label"
    duplicate["transaction_type"] = "Deposits"
    transactions = pd.concat([transactions, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate.*raw key"):
        bridge.classify_observed_sources(
            transactions,
            refunds,
            tax_deposits,
        )


def test_event_daily_maps_all_six_streams_once_and_marks_incomplete_h8() -> None:
    classified = _classify()
    events = bridge.aggregate_event_daily(
        classified,
        last_complete_h8_week="2024-03-27",
    )

    public_columns = tuple(column for column in events if not column.startswith("_"))
    assert public_columns == EXPECTED_COLUMNS
    assert set(events["row_type"]) == {"event_daily"}
    assert set(events["schema_version"]) == {"quarterly_hf_bridge_v1"}
    assert set(events["stream_id"]) == set(EXPECTED_STREAMS)
    assert not events.duplicated(
        ["row_type", "source_id", "record_date", "component_id"]
    ).any()

    mapped_keys = set(
        classified.loc[
            classified["coverage_class"].eq("mapped_observed"),
            "_raw_key",
        ]
    )
    event_keys = _flatten_source_keys(events["_source_keys"])
    assert len(event_keys) == len(set(event_keys))
    assert set(event_keys) == mapped_keys
    assert int(events["source_row_count"].sum()) == len(mapped_keys)

    quarter_end_events = events.loc[
        events["record_date"].eq(pd.Timestamp("2024-03-29"))
    ]
    assert set(quarter_end_events["aggregation_quarter"]) == {"2024Q1"}
    assert set(quarter_end_events["h8_week_end"]) == {pd.Timestamp("2024-04-03")}
    assert set(quarter_end_events["h8_endpoint_quarter"]) == {"2024Q2"}
    assert quarter_end_events["h8_cross_quarter_flag"].all()
    assert not quarter_end_events["h8_window_complete_flag"].any()
    assert set(quarter_end_events["quarter_end_age_calendar_days"]) == {2}
    assert set(quarter_end_events["quarter_end_age_bin"]) == {"0-7"}

    complete = events.loc[
        events["record_date"].eq(pd.Timestamp("2023-02-13"))
    ]
    assert complete["h8_window_complete_flag"].all()
    assert set(complete["h8_week_start"]) == {pd.Timestamp("2023-02-09")}
    assert set(complete["h8_week_end"]) == {pd.Timestamp("2023-02-15")}


def test_settlement_quarter_and_ascii_age_bin_boundaries() -> None:
    quarter_end = pd.Timestamp("2024-06-30")
    ages = [0, 7, 8, 30, 31, 60, 61]
    transactions, refunds, tax_deposits = _observed_inputs()
    age_transactions = pd.DataFrame(
        {
            "record_date": [
                (quarter_end - pd.Timedelta(days=age)).date().isoformat()
                for age in ages
            ],
            "account_type": ["Federal Reserve Account"] * len(ages),
            "transaction_type": ["Withdrawals"] * len(ages),
            "transaction_catg": ["SSA - Benefits Payments"] * len(ages),
            "transaction_today_amt": ["1"] * len(ages),
            "src_line_nbr": range(100, 100 + len(ages)),
        }
    )
    transactions = pd.concat(
        [transactions, age_transactions],
        ignore_index=True,
    )
    classified = bridge.classify_observed_sources(
        transactions,
        refunds,
        tax_deposits,
    )
    events = bridge.aggregate_event_daily(
        classified,
        last_complete_h8_week="2024-07-03",
    )
    benefits = events.loc[
        events["source_id"].eq(TRANSACTION_SOURCE)
        & events["component_id"].eq("du_core_benefits")
        & events["aggregation_quarter"].eq("2024Q2")
    ].sort_values("quarter_end_age_calendar_days")

    assert benefits["quarter_end_age_calendar_days"].tolist() == ages
    assert benefits["quarter_end_age_bin"].tolist() == [
        "0-7",
        "0-7",
        "8-30",
        "8-30",
        "31-60",
        "31-60",
        "61+",
    ]
    assert set(benefits["aggregation_quarter"]) == {"2024Q2"}
    quarter_end_row = benefits.iloc[0]
    assert quarter_end_row["h8_week_end"] == pd.Timestamp("2024-07-03")
    assert quarter_end_row["h8_endpoint_quarter"] == "2024Q3"
    assert bool(quarter_end_row["h8_cross_quarter_flag"])


def _weekly_reference() -> pd.DataFrame:
    weekly = pd.DataFrame(
        {
            "h8_week_end": [
                pd.Timestamp("2023-02-15"),
                pd.Timestamp("2024-03-27"),
            ],
        }
    )
    for column in bridge.WEEKLY_REFERENCE_COLUMNS.values():
        weekly[column] = 0.0
    weekly.loc[
        weekly["h8_week_end"].eq(pd.Timestamp("2023-02-15")),
        "tax_receipts_bn",
    ] = 0.001
    return weekly


def test_six_stream_weekly_and_quarter_reconciliation_excludes_incomplete_h8() -> None:
    classified = _classify()
    events = bridge.aggregate_event_daily(
        classified,
        last_complete_h8_week="2024-03-27",
    )
    weekly = _weekly_reference()
    h8 = bridge._h8_reconciliation_rows(
        events,
        weekly,
        weekly_metadata={
            "source_path": "weekly.csv",
            "source_sha256": "a" * 64,
        },
    )

    assert set(h8["stream_id"]) == set(EXPECTED_STREAMS)
    assert set(h8["h8_week_end"]) == {
        pd.Timestamp("2023-02-15"),
        pd.Timestamp("2024-03-27"),
    }
    assert pd.Timestamp("2024-04-03") not in set(h8["h8_week_end"])
    assert h8["h8_window_complete_flag"].all()
    assert pd.to_numeric(h8["aggregation_residual_bn"]).abs().max() == 0.0

    quarter = bridge._quarter_reconciliation_rows(events, classified)
    assert set(quarter["stream_id"]) == set(EXPECTED_STREAMS)
    assert (
        quarter.groupby("aggregation_quarter")["stream_id"].nunique()
        .eq(len(EXPECTED_STREAMS))
        .all()
    )
    assert pd.to_numeric(quarter["aggregation_residual_bn"]).abs().max() == 0.0
    assert "2024Q1" in set(quarter["aggregation_quarter"])

    mismatched_weekly = weekly.copy()
    mismatched_weekly.loc[
        mismatched_weekly["h8_week_end"].eq(pd.Timestamp("2024-03-27")),
        "debt_issues_gross_bn",
    ] = 0.001
    with pytest.raises(AssertionError, match="gross_issuance"):
        bridge._h8_reconciliation_rows(
            events,
            mismatched_weekly,
            weekly_metadata={
                "source_path": "weekly.csv",
                "source_sha256": "a" * 64,
            },
        )

    mismatched_events = events.copy()
    row = mismatched_events["stream_id"].eq("gross_redemption").idxmax()
    mismatched_events.loc[row, "amount_bn"] += 0.001
    with pytest.raises(AssertionError, match="gross_redemption"):
        bridge._quarter_reconciliation_rows(mismatched_events, classified)


def _anchor_inputs() -> tuple[
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    period = pd.Period("2024Q1", freq="Q-DEC")
    tdc_estimates = pd.DataFrame(
        {
            "date": [period.end_time.normalize()],
            "tdc_tier2_interest_corrected_bank_only_ru_flow": [5000.0],
            "tdc_base_bank_only_ru_flow": [3000.0],
        }
    )
    handoff = {
        "artifacts": [
            {
                "artifact_key": (
                    "tdc_tier2_interest_corrected_bank_only_ru_flow"
                ),
                "current_role": "working_corrected_headline",
            },
            {"artifact_key": "tdc_base_bank_only_ru_flow"},
        ]
    }
    months = pd.date_range("2024-01-31", "2024-03-31", freq="ME")
    monthly_proxy = pd.DataFrame({"date": months, "tdc_monthly": 1000.0})
    monthly_components = pd.DataFrame(
        {
            "date": months,
            "fed_tsy": 1000.0,
            "banks_tsy": 0.0,
            "row_tsy": 0.0,
            "minus_toc": 0.0,
            "fed_remit_positive": 0.0,
        }
    )
    quarterly_sums = pd.DataFrame(
        {
            "date": [period.end_time.normalize()],
            "fed_tsy": [3000.0],
            "banks_tsy": [0.0],
            "row_tsy": [0.0],
            "minus_toc": [0.0],
            "fed_remit_positive": [0.0],
        }
    )
    validation = {
        "status": "ok",
        "method": "indicator_denton_headline",
        "diagnostics": [
            {
                "component": component,
                "status": "ok",
                "max_abs_quarterly_error": 0.0,
            }
            for component in (
                "fed_tsy",
                "banks_tsy",
                "row_tsy",
                "minus_toc",
                "fed_remit_positive",
            )
        ],
    }
    return (
        tdc_estimates,
        handoff,
        monthly_proxy,
        monthly_components,
        quarterly_sums,
        validation,
    )


def test_quarter_anchor_surface_has_exact_three_quarterly_comparisons() -> None:
    (
        tdc_estimates,
        handoff,
        monthly_proxy,
        monthly_components,
        quarterly_sums,
        validation,
    ) = _anchor_inputs()
    period = pd.Period("2024Q1", freq="Q-DEC")
    anchors = bridge._quarter_anchor_rows(
        tdc_estimates=tdc_estimates,
        handoff_payload=handoff,
        monthly_proxy=monthly_proxy,
        monthly_components=monthly_components,
        quarterly_sums=quarterly_sums,
        validation_payload=validation,
        estimates_metadata={
            "source_id": "tdcest_tdc_estimates",
            "source_path": "tdc_estimates.csv",
            "source_sha256": "b" * 64,
        },
        proxy_metadata={
            "source_id": "legacy_denton_monthly_proxy",
            "source_path": "tdc_monthly_proxy.csv",
            "source_sha256": "c" * 64,
        },
        first_period=period,
        last_period=period,
    )

    assert set(anchors["row_type"]) == {"quarter_anchor"}
    assert set(anchors["anchor_id"]) == {
        "tdc_tier2_interest_corrected_bank_only_ru_flow",
        "tdc_base_bank_only_ru_flow",
        "legacy_five_component_denton",
    }
    assert len(anchors) == 3
    assert set(anchors["aggregation_quarter"]) == {"2024Q1"}
    assert anchors[["record_date", "h8_week_start", "h8_week_end"]].isna().all(
        axis=None
    )

    current = anchors["anchor_id"].isin(
        {
            "tdc_tier2_interest_corrected_bank_only_ru_flow",
            "tdc_base_bank_only_ru_flow",
        }
    )
    assert anchors.loc[current, "aggregation_residual_bn"].isna().all()
    assert anchors.loc[current, "timing_status"].eq(
        "quarterly_no_intraquarter_timing"
    ).all()
    legacy = anchors.loc[
        anchors["anchor_id"].eq("legacy_five_component_denton")
    ].iloc[0]
    assert legacy["timing_status"] == "generated_not_observed"
    assert legacy["aggregation_residual_bn"] == pytest.approx(0.0)


def test_input_bundle_hash_is_order_invariant_and_mutation_sensitive() -> None:
    lineage = {
        "data/a.csv": "a" * 64,
        "external/b.json": "b" * 64,
    }
    digest = bridge._input_bundle_sha256(lineage)
    reordered = bridge._input_bundle_sha256(dict(reversed(tuple(lineage.items()))))
    mutated = bridge._input_bundle_sha256(
        {**lineage, "external/b.json": "c" * 64}
    )

    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert reordered == digest
    assert mutated != digest
