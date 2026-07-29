from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .disbursement import (
    SEAM_FIRST_TABLE_II,
    SEAM_LAST_DEDICATED,
    _tax_bucket_from_dedicated,
    _tax_bucket_from_table_ii,
    assign_h8_week,
    map_dts_category,
)
from .manifest import _sha256


SCHEMA_VERSION = "quarterly_hf_bridge_v1"
RESIDUAL_TOLERANCE_BN = 1e-9
NATIVE_LEGACY_TOLERANCE_MILLIONS = 1e-8
SIGN_CONVENTION = "source_signed_within_gross_class"
TREATMENT_PERIMETER = "dts_direct_cash_settlements_six_streams"
OUTCOME_PERIMETER = "h8_commercial_bank_deposits"
AGE_BIN_ORDER = ("0-7", "8-30", "31-60", "61+")

SCHEMA_COLUMNS = (
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
BRIDGE_COLUMNS = SCHEMA_COLUMNS

ROW_TYPES = (
    "event_daily",
    "h8_reconciliation",
    "quarter_reconciliation",
    "input_coverage",
    "quarter_anchor",
)
COVERAGE_CLASSES = (
    "input_all",
    "eligible_observed",
    "mapped_observed",
    "excluded_reference_or_total",
    "excluded_seam_overlap",
    "excluded_outside_perimeter",
    "excluded_unmapped",
)

STREAM_COMPONENTS = {
    "gross_credits": (
        "du_core_benefits",
        "du_core_refunds",
        "du_core_salaries_other",
    ),
    "gross_drains": (
        "tax_withheld",
        "tax_nonwithheld",
        "tax_corporate",
        "tax_other",
    ),
    "gross_issuance": ("debt_issues_gross",),
    "gross_redemption": ("debt_redemptions_gross",),
    "sidecar_broad_outflows": ("du_broad_outflows",),
    "sidecar_interest_outflows": ("interest_outflows",),
}
STREAM_ORDER = tuple(STREAM_COMPONENTS)
COMPONENT_TO_STREAM = {
    component: stream
    for stream, components in STREAM_COMPONENTS.items()
    for component in components
}
STREAM_ROLE = {
    stream: ("sidecar" if stream.startswith("sidecar_") else "required")
    for stream in STREAM_ORDER
}
WEEKLY_REFERENCE_COLUMNS = {
    "gross_credits": "du_core_outflows_bn",
    "gross_drains": "tax_receipts_bn",
    "gross_issuance": "debt_issues_gross_bn",
    "gross_redemption": "debt_redemptions_gross_bn",
    "sidecar_broad_outflows": "du_broad_outflows_bn",
    "sidecar_interest_outflows": "interest_outflows_bn",
}

SOURCE_TRANSACTION = "dts_deposits_withdrawals_operating_cash"
SOURCE_REFUNDS = "dts_income_tax_refunds_issued"
SOURCE_TAX = "dts_federal_tax_deposits"
EVENT_SOURCES = (SOURCE_TRANSACTION, SOURCE_REFUNDS, SOURCE_TAX)

_TRANSACTION_COMPONENTS = {
    "du_core_benefits": "du_core_benefits",
    "du_core_salaries_other": "du_core_salaries_other",
    "debt_issues_gross": "debt_issues_gross",
    "debt_redemptions_gross": "debt_redemptions_gross",
    "du_broad_outflows": "du_broad_outflows",
    "interest_outflows": "interest_outflows",
    "tax_withheld": "tax_withheld",
    "tax_nonwithheld": "tax_nonwithheld",
    "tax_corporate": "tax_corporate",
    "tax_other": "tax_other",
}
_REFERENCE_BUCKETS = {
    "refunds_table_ii_reference",
    "tax_table_reference",
    "table_ii_reference_total",
}
_OUTSIDE_BUCKETS = {"other_deposits", "other_withdrawals"}
_TAX_COMPONENTS = {
    "tax_withheld",
    "tax_nonwithheld",
    "tax_corporate",
    "tax_other",
}
_LEGACY_COMPONENTS = (
    "fed_tsy",
    "banks_tsy",
    "row_tsy",
    "minus_toc",
    "fed_remit_positive",
)


def _mapping_contract() -> dict[str, object]:
    return {
        "age_bins_calendar_days": {
            "0-7": [0, 7],
            "8-30": [8, 30],
            "31-60": [31, 60],
            "61+": [61, None],
        },
        "component_to_stream": COMPONENT_TO_STREAM,
        "h8_assignment_rule": (
            "h8_week_end=record_date+((Wednesday-weekday(record_date)) mod 7);"
            "h8_week_start=h8_week_end-6 calendar days"
        ),
        "outcome_perimeter": OUTCOME_PERIMETER,
        "quarter_clock": "calendar_quarter(record_date)",
        "schema_version": SCHEMA_VERSION,
        "sign_convention": SIGN_CONVENTION,
        "tax_seam": {
            "dedicated_last": SEAM_LAST_DEDICATED.date().isoformat(),
            "table_ii_first": SEAM_FIRST_TABLE_II.date().isoformat(),
        },
        "treatment_perimeter": TREATMENT_PERIMETER,
    }


def mapping_contract_sha256() -> str:
    payload = json.dumps(
        _mapping_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    source: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{source}: missing required columns {missing}")


def _strict_dates(values: pd.Series, *, source: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{source}: null required date")
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="coerce").dt.normalize()
    else:
        text = values.astype("string").str.strip()
        malformed_shape = ~text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
        parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce").dt.normalize()
        if malformed_shape.any():
            parsed.loc[malformed_shape] = pd.NaT
    if parsed.isna().any():
        bad = values.loc[parsed.isna()].astype("string").head(5).tolist()
        raise ValueError(f"{source}: malformed required dates {bad}")
    return parsed


def _clean_numeric(values: pd.Series, *, source: str, nullable: bool) -> pd.Series:
    text = (
        values.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    present = values.notna() & text.ne("")
    parsed = pd.to_numeric(text.where(present), errors="coerce")
    invalid = present & (parsed.isna() | ~np.isfinite(parsed))
    if invalid.any():
        bad = values.loc[invalid].astype("string").head(5).tolist()
        raise ValueError(f"{source}: malformed numeric values {bad}")
    if not nullable and parsed.isna().any():
        raise ValueError(f"{source}: null required numeric values")
    return parsed.astype("float64")


def _strict_amount_bn(values: pd.Series, *, source: str) -> pd.Series:
    return _clean_numeric(values, source=source, nullable=False) * 0.001


def _strict_line_numbers(values: pd.Series, *, source: str) -> pd.Series:
    parsed = _clean_numeric(values, source=f"{source}.src_line_nbr", nullable=False)
    if not np.equal(parsed, np.floor(parsed)).all():
        raise ValueError(f"{source}: src_line_nbr must be an integer")
    return parsed.astype("int64")


def _keyset_sha256(keys: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _raw_keys(
    source_id: str,
    dates: pd.Series,
    line_numbers: pd.Series,
) -> pd.Series:
    return pd.Series(
        [
            f"{source_id}\x1f{date.date().isoformat()}\x1f{int(line_number)}"
            for date, line_number in zip(dates, line_numbers, strict=True)
        ],
        index=dates.index,
        dtype="string",
    )


def _source_category(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    parts = [
        frame[column].astype("string").fillna("<NA>").str.strip()
        for column in columns
    ]
    result = parts[0]
    for part in parts[1:]:
        result = result + "\x1f" + part
    return result


def _metadata_for(
    source_id: str,
    metadata: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object]:
    values = dict((metadata or {}).get(source_id, {}))
    return {
        "source_id": source_id,
        "source_path": str(values.get("source_path", "")),
        "source_sha256": str(values.get("source_sha256", "")),
        "source_retrieved_at": str(values.get("source_retrieved_at", "")),
        "source_endpoint": str(values.get("source_endpoint", "")),
        "source_units": str(values.get("source_units", "$1,000,000")),
        "unit_scale_to_bn": float(values.get("unit_scale_to_bn", 0.001)),
    }


def _prepare_source(
    frame: pd.DataFrame,
    *,
    source_id: str,
    amount_column: str,
    category_columns: tuple[str, ...],
    metadata: Mapping[str, Mapping[str, object]] | None,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"{source_id}: required input is empty")
    _require_columns(
        frame,
        ["record_date", "src_line_nbr", amount_column, *category_columns],
        source=source_id,
    )
    out = frame.copy()
    out["record_date"] = _strict_dates(out["record_date"], source=source_id)
    out["src_line_nbr"] = _strict_line_numbers(
        out["src_line_nbr"],
        source=source_id,
    )
    out["amount_bn"] = _strict_amount_bn(
        out[amount_column],
        source=f"{source_id}.{amount_column}",
    )
    out["_raw_key"] = _raw_keys(
        source_id,
        out["record_date"],
        out["src_line_nbr"],
    )
    if out["_raw_key"].duplicated().any():
        bad = out.loc[
            out["_raw_key"].duplicated(keep=False),
            ["record_date", "src_line_nbr"],
        ].head(5)
        raise ValueError(
            f"{source_id}: duplicate raw keys {bad.to_dict(orient='records')}"
        )
    out["_source_category"] = _source_category(out, category_columns)
    for key, value in _metadata_for(source_id, metadata).items():
        out[key] = value
    out["coverage_class"] = "excluded_unmapped"
    out["stream_id"] = pd.NA
    out["component_id"] = pd.NA
    out["perimeter_role"] = pd.NA
    return out


def _admit_component(
    frame: pd.DataFrame,
    mask: pd.Series,
    components: pd.Series,
) -> None:
    frame.loc[mask, "coverage_class"] = "mapped_observed"
    frame.loc[mask, "component_id"] = components.loc[mask]
    frame.loc[mask, "stream_id"] = components.loc[mask].map(COMPONENT_TO_STREAM)
    frame.loc[mask, "perimeter_role"] = frame.loc[mask, "stream_id"].map(
        STREAM_ROLE
    )


def _classify_transactions(
    transactions: pd.DataFrame,
    metadata: Mapping[str, Mapping[str, object]] | None,
) -> pd.DataFrame:
    out = _prepare_source(
        transactions,
        source_id=SOURCE_TRANSACTION,
        amount_column="transaction_today_amt",
        category_columns=("transaction_type", "transaction_catg"),
        metadata=metadata,
    )
    _require_columns(out, ["account_type"], source=SOURCE_TRANSACTION)
    out["_bucket"] = [
        map_dts_category(transaction_type, category)
        for transaction_type, category in zip(
            out["transaction_type"],
            out["transaction_catg"],
            strict=True,
        )
    ]
    out["_table_ii_tax_component"] = out["transaction_catg"].map(
        _tax_bucket_from_table_ii
    ).astype("string").str.removesuffix("_bn")
    account_total = out["account_type"].astype("string").str.contains(
        r"Total Deposits|Total Withdrawals",
        case=False,
        na=False,
    )
    reference = account_total | out["_bucket"].isin(_REFERENCE_BUCKETS)
    out.loc[reference, "coverage_class"] = "excluded_reference_or_total"

    tax_candidate = (
        out["_bucket"].isin(_TAX_COMPONENTS)
        | out["_table_ii_tax_component"].notna()
    )
    seam_overlap = ~reference & tax_candidate & out["record_date"].lt(
        SEAM_FIRST_TABLE_II
    )
    out.loc[seam_overlap, "coverage_class"] = "excluded_seam_overlap"

    mapped_tax_outside_reference = (
        tax_candidate
        & out["_table_ii_tax_component"].isna()
        & out["record_date"].ge(SEAM_FIRST_TABLE_II)
    )
    outside = (
        ~reference
        & ~seam_overlap
        & (
            out["_bucket"].isin(_OUTSIDE_BUCKETS)
            | mapped_tax_outside_reference
        )
    )
    out.loc[outside, "coverage_class"] = "excluded_outside_perimeter"

    non_tax_mappable = (
        out["_bucket"].isin(_TRANSACTION_COMPONENTS)
        & ~out["_bucket"].isin(_TAX_COMPONENTS)
    )
    admitted_non_tax = (
        ~reference
        & ~seam_overlap
        & ~outside
        & non_tax_mappable
    )
    components = out["_bucket"].map(_TRANSACTION_COMPONENTS)
    _admit_component(out, admitted_non_tax, components)
    admitted_tax = (
        ~reference
        & ~seam_overlap
        & ~outside
        & out["_table_ii_tax_component"].notna()
        & out["record_date"].ge(SEAM_FIRST_TABLE_II)
    )
    _admit_component(out, admitted_tax, out["_table_ii_tax_component"])

    unmapped = out["coverage_class"].eq("excluded_unmapped")
    if unmapped.any():
        grouped = out.loc[unmapped].groupby("_source_category")["amount_bn"].sum()
        large = grouped.loc[grouped.abs().ge(10.0)]
        if not large.empty:
            raise ValueError(
                "DTS transaction crosswalk has unmapped categories above "
                f"$10bn signed-total threshold: {large.head(5).to_dict()}"
            )
    return out


def _known_refund_category(value: object) -> bool:
    text = str(value or "").strip().lower()
    if text in {
        "business",
        "individual",
        "irs tax refunds business",
        "irs tax refunds individual",
    }:
        return True
    return (
        text.startswith("taxes - business tax refunds")
        or text.startswith("taxes - individual tax refunds")
        or text.startswith("irs - economic impact payments")
        or text.startswith("irs - advanced child tax credit")
    )


def _classify_refunds(
    refunds: pd.DataFrame,
    metadata: Mapping[str, Mapping[str, object]] | None,
) -> pd.DataFrame:
    out = _prepare_source(
        refunds,
        source_id=SOURCE_REFUNDS,
        amount_column="tax_refund_today_amt",
        category_columns=("tax_refund_type",),
        metadata=metadata,
    )
    admitted = out["tax_refund_type"].map(_known_refund_category)
    components = pd.Series("du_core_refunds", index=out.index, dtype="string")
    _admit_component(out, admitted, components)
    return out


def _dedicated_tax_component(value: object) -> str | None:
    text = str(value or "").strip().lower()
    recognized = any(
        token in text
        for token in (
            "withheld income and employment",
            "individual income taxes",
            "corporation income taxes",
            "railroad retirement",
            "excise taxes",
            "federal unemployment",
            "estate and gift",
            "unclassified taxes",
        )
    )
    if not recognized:
        return None
    mapped = _tax_bucket_from_dedicated(value)
    return str(mapped).removesuffix("_bn") if mapped else None


def _dedicated_tax_reference(value: object) -> bool:
    text = str(value or "").strip().lower()
    return any(
        token in text
        for token in (
            "cash federal tax deposits",
            "inter-agency transfers",
            "these receipts were deposited in",
            "tax and loan accounts",
        )
    )


def _classify_tax(
    tax_deposits: pd.DataFrame,
    metadata: Mapping[str, Mapping[str, object]] | None,
) -> pd.DataFrame:
    out = _prepare_source(
        tax_deposits,
        source_id=SOURCE_TAX,
        amount_column="tax_deposit_today_amt",
        category_columns=("tax_deposit_type",),
        metadata=metadata,
    )
    components = out["tax_deposit_type"].map(_dedicated_tax_component)
    reference = out["tax_deposit_type"].map(_dedicated_tax_reference)
    out.loc[reference, "coverage_class"] = "excluded_reference_or_total"
    component_row = components.notna()
    seam_overlap = ~reference & component_row & out["record_date"].gt(
        SEAM_LAST_DEDICATED
    )
    out.loc[seam_overlap, "coverage_class"] = "excluded_seam_overlap"
    admitted = (
        ~reference
        & component_row
        & out["record_date"].le(SEAM_LAST_DEDICATED)
    )
    _admit_component(out, admitted, components.astype("string"))
    return out


def classify_observed_sources(
    transactions: pd.DataFrame,
    refunds: pd.DataFrame,
    tax_deposits: pd.DataFrame,
    source_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Classify every raw event-source key once, preserving signed source amounts."""
    classified = pd.concat(
        [
            _classify_transactions(transactions, source_metadata),
            _classify_refunds(refunds, source_metadata),
            _classify_tax(tax_deposits, source_metadata),
        ],
        ignore_index=True,
        sort=False,
    )
    if classified["_raw_key"].duplicated().any():
        raise AssertionError("raw keys are not globally unique after source classification")
    if not set(classified["coverage_class"]).issubset(
        set(COVERAGE_CLASSES) - {"input_all", "eligible_observed"}
    ):
        raise AssertionError("unknown raw coverage classification")
    mapped = classified["coverage_class"].eq("mapped_observed")
    if classified.loc[mapped, ["stream_id", "component_id"]].isna().any(axis=None):
        raise AssertionError("mapped rows require a stream and component")
    if classified.loc[~mapped, ["stream_id", "component_id"]].notna().any(axis=None):
        raise AssertionError("excluded rows may not carry a stream or component")
    if set(classified.loc[mapped, "component_id"]) != set(COMPONENT_TO_STREAM):
        missing = sorted(
            set(COMPONENT_TO_STREAM) - set(classified.loc[mapped, "component_id"])
        )
        raise ValueError(f"observed inputs are missing canonical components: {missing}")
    return classified.sort_values(
        ["source_id", "record_date", "src_line_nbr"],
    ).reset_index(drop=True)


def _quarter_end_age_bin(days: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(days, errors="coerce")
    if numeric.isna().any() or numeric.lt(0).any():
        raise ValueError("quarter-end age must be a nonnegative integer")
    labels = np.select(
        [
            numeric.le(7),
            numeric.between(8, 30),
            numeric.between(31, 60),
            numeric.ge(61),
        ],
        AGE_BIN_ORDER,
        default="",
    )
    if (labels == "").any():
        raise AssertionError("quarter-end age bin mapping is incomplete")
    return pd.Series(labels, index=days.index, dtype="string")


def _blank_rows(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{column: pd.NA for column in SCHEMA_COLUMNS} for _ in range(count)]
    )


def aggregate_event_daily(
    classified: pd.DataFrame,
    *,
    last_complete_h8_week: object,
) -> pd.DataFrame:
    """Aggregate admitted raw keys to source/date/canonical-component event rows."""
    _require_columns(
        classified,
        [
            "_raw_key",
            "_source_category",
            "record_date",
            "amount_bn",
            "source_id",
            "source_path",
            "source_sha256",
            "source_retrieved_at",
            "source_endpoint",
            "source_units",
            "unit_scale_to_bn",
            "coverage_class",
            "stream_id",
            "component_id",
            "perimeter_role",
        ],
        source="classified_observed_sources",
    )
    mapped = classified.loc[
        classified["coverage_class"].eq("mapped_observed")
    ].copy()
    if mapped.empty:
        raise ValueError("classified observed source surface has no mapped rows")
    last_complete = pd.Timestamp(last_complete_h8_week).normalize()
    if pd.isna(last_complete) or last_complete.weekday() != 2:
        raise ValueError("last_complete_h8_week must be a Wednesday")

    group_keys = [
        "source_id",
        "source_path",
        "source_sha256",
        "source_retrieved_at",
        "source_endpoint",
        "source_units",
        "unit_scale_to_bn",
        "record_date",
        "stream_id",
        "component_id",
        "perimeter_role",
    ]
    grouped = (
        mapped.groupby(group_keys, dropna=False, sort=True)
        .agg(
            amount_bn=("amount_bn", "sum"),
            source_row_count=("_raw_key", "size"),
            source_category_count=("_source_category", "nunique"),
        )
        .reset_index()
    )
    key_lists = (
        mapped.groupby(group_keys, dropna=False, sort=True)["_raw_key"]
        .apply(list)
        .rename("_source_keys")
        .reset_index()
    )
    grouped = grouped.merge(
        key_lists,
        on=group_keys,
        how="left",
        validate="one_to_one",
    )
    grouped["source_keyset_sha256"] = grouped["_source_keys"].map(_keyset_sha256)

    rows = _blank_rows(len(grouped))
    for column in (
        "source_id",
        "source_path",
        "source_sha256",
        "source_retrieved_at",
        "source_endpoint",
        "source_units",
        "unit_scale_to_bn",
        "record_date",
        "stream_id",
        "component_id",
        "perimeter_role",
        "amount_bn",
        "source_row_count",
        "source_category_count",
        "source_keyset_sha256",
    ):
        rows[column] = grouped[column].to_numpy()
    rows["_source_keys"] = grouped["_source_keys"].to_list()
    rows["schema_version"] = SCHEMA_VERSION
    rows["row_type"] = "event_daily"
    rows["lane"] = "observed_dts_direct"
    rows["status"] = "mapped_observed"
    rows["units"] = "USD_billions"
    rows["aggregation_quarter_clock"] = "record_date"
    rows["timing_status"] = "observed_settlement_date"
    rows["treatment_perimeter"] = TREATMENT_PERIMETER
    rows["outcome_perimeter"] = OUTCOME_PERIMETER
    rows["sign_convention"] = SIGN_CONVENTION

    rows["record_date"] = pd.to_datetime(rows["record_date"]).dt.normalize()
    rows["h8_week_end"] = rows["record_date"].map(assign_h8_week)
    rows["h8_week_start"] = rows["h8_week_end"] - pd.Timedelta(days=6)
    record_quarter = rows["record_date"].dt.to_period("Q-DEC")
    endpoint_quarter = rows["h8_week_end"].dt.to_period("Q-DEC")
    rows["aggregation_quarter"] = record_quarter.astype(str)
    rows["h8_endpoint_quarter"] = endpoint_quarter.astype(str)
    rows["quarter_end_date"] = record_quarter.dt.end_time.dt.normalize()
    rows["quarter_end_age_calendar_days"] = (
        rows["quarter_end_date"] - rows["record_date"]
    ).dt.days.astype("int64")
    rows["quarter_end_age_bin"] = _quarter_end_age_bin(
        rows["quarter_end_age_calendar_days"]
    )
    rows["h8_cross_quarter_flag"] = rows["aggregation_quarter"].ne(
        rows["h8_endpoint_quarter"]
    )
    rows["h8_window_complete_flag"] = rows["h8_week_end"].le(last_complete)

    if not rows["h8_week_end"].dt.weekday.eq(2).all():
        raise AssertionError("event H.8 endpoints must be Wednesday")
    inside = rows["record_date"].between(
        rows["h8_week_start"],
        rows["h8_week_end"],
    )
    if not inside.all():
        raise AssertionError("event record_date lies outside its H.8 window")
    if rows.duplicated(
        ["row_type", "source_id", "record_date", "component_id"]
    ).any():
        raise AssertionError("event_daily uniqueness key is not unique")

    event_keys = [key for keys in rows["_source_keys"] for key in keys]
    mapped_keys = mapped["_raw_key"].astype(str).tolist()
    if len(event_keys) != len(set(event_keys)):
        raise AssertionError("event keysets overlap")
    if set(event_keys) != set(mapped_keys):
        raise AssertionError("event keyset union differs from mapped raw keyset")
    if int(rows["source_row_count"].sum()) != len(mapped):
        raise AssertionError("event source_row_count does not conserve mapped rows")
    return rows.sort_values(
        ["record_date", "source_id", "component_id"]
    ).reset_index(drop=True)


def _coverage_rows(classified: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    empty_hash = _keyset_sha256([])
    exclusion_classes = (
        "excluded_reference_or_total",
        "excluded_seam_overlap",
        "excluded_outside_perimeter",
        "excluded_unmapped",
    )
    for source_id in EVENT_SOURCES:
        source = classified.loc[classified["source_id"].eq(source_id)].copy()
        if source.empty:
            raise ValueError(f"{source_id}: source disappeared before coverage accounting")
        mapped = source.loc[source["coverage_class"].eq("mapped_observed")]
        partitions = {
            "input_all": source,
            "eligible_observed": mapped,
            "mapped_observed": mapped,
            **{
                coverage_class: source.loc[
                    source["coverage_class"].eq(coverage_class)
                ]
                for coverage_class in exclusion_classes
            },
        }
        partition_keys: list[str] = []
        for coverage_class in ("mapped_observed", *exclusion_classes):
            partition_keys.extend(
                partitions[coverage_class]["_raw_key"].astype(str).tolist()
            )
        all_keys = source["_raw_key"].astype(str).tolist()
        if len(partition_keys) != len(set(partition_keys)):
            raise AssertionError(f"{source_id}: coverage partition overlaps")
        if set(partition_keys) != set(all_keys):
            raise AssertionError(f"{source_id}: coverage partition is incomplete")
        if len(partition_keys) != len(source):
            raise AssertionError(f"{source_id}: coverage partition count mismatch")
        if (
            len(partitions["eligible_observed"])
            != len(partitions["mapped_observed"])
            or _keyset_sha256(partitions["eligible_observed"]["_raw_key"])
            != _keyset_sha256(partitions["mapped_observed"]["_raw_key"])
        ):
            raise AssertionError(f"{source_id}: eligible and mapped keysets differ")

        metadata = source.iloc[0]
        for coverage_class in COVERAGE_CLASSES:
            subset = partitions[coverage_class]
            row = {column: pd.NA for column in SCHEMA_COLUMNS}
            row.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_type": "input_coverage",
                    "lane": "observed_dts_direct",
                    "status": "coverage_accounted",
                    "units": "USD_billions",
                    "source_id": source_id,
                    "source_path": metadata["source_path"],
                    "source_sha256": metadata["source_sha256"],
                    "source_retrieved_at": metadata["source_retrieved_at"],
                    "source_endpoint": metadata["source_endpoint"],
                    "source_units": metadata["source_units"],
                    "unit_scale_to_bn": metadata["unit_scale_to_bn"],
                    "timing_status": "source_coverage",
                    "treatment_perimeter": TREATMENT_PERIMETER,
                    "outcome_perimeter": OUTCOME_PERIMETER,
                    "sign_convention": SIGN_CONVENTION,
                    "coverage_class": coverage_class,
                    "coverage_row_count": int(len(subset)),
                    "coverage_amount_bn": float(subset["amount_bn"].sum()),
                    "coverage_keyset_sha256": (
                        _keyset_sha256(subset["_raw_key"])
                        if not subset.empty
                        else empty_hash
                    ),
                    "coverage_start_date": (
                        subset["record_date"].min() if not subset.empty else pd.NaT
                    ),
                    "coverage_end_date": (
                        subset["record_date"].max() if not subset.empty else pd.NaT
                    ),
                }
            )
            rows.append(row)
    output = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    if output.duplicated(["source_id", "coverage_class"]).any():
        raise AssertionError("input_coverage uniqueness key is not unique")
    return output


def _expected_last_complete_week(classified: pd.DataFrame) -> pd.Timestamp:
    latest_by_active_source = {}
    for source_id in (SOURCE_TRANSACTION, SOURCE_REFUNDS):
        dates = classified.loc[
            classified["source_id"].eq(source_id),
            "record_date",
        ]
        if dates.empty:
            raise ValueError(f"{source_id}: missing active-source coverage")
        latest_by_active_source[source_id] = pd.Timestamp(dates.max()).normalize()
    if len(set(latest_by_active_source.values())) != 1:
        raise ValueError(
            "active transaction/refund source coverage endpoints are not aligned: "
            f"{latest_by_active_source}"
        )
    latest = next(iter(latest_by_active_source.values()))
    endpoint = assign_h8_week(latest)
    if latest < endpoint:
        endpoint -= pd.Timedelta(days=7)
    return endpoint


def _weekly_frame(
    weekly_flows: pd.DataFrame,
    *,
    expected_last_complete: pd.Timestamp,
) -> pd.DataFrame:
    _require_columns(
        weekly_flows,
        ["date", *WEEKLY_REFERENCE_COLUMNS.values()],
        source="dts_weekly_flow_decomposition",
    )
    if weekly_flows.empty:
        raise ValueError("dts_weekly_flow_decomposition: required input is empty")
    weekly = weekly_flows.copy()
    weekly["h8_week_end"] = _strict_dates(
        weekly["date"],
        source="dts_weekly_flow_decomposition.date",
    )
    if weekly["h8_week_end"].duplicated().any():
        raise ValueError("dts_weekly_flow_decomposition: duplicate week endpoints")
    if not weekly["h8_week_end"].dt.weekday.eq(2).all():
        raise ValueError("dts_weekly_flow_decomposition: endpoints must be Wednesday")
    ordered = weekly["h8_week_end"].sort_values()
    if len(ordered) > 1 and not ordered.diff().dropna().eq(
        pd.Timedelta(days=7)
    ).all():
        raise ValueError("dts_weekly_flow_decomposition: weekly coverage has gaps")
    for column in WEEKLY_REFERENCE_COLUMNS.values():
        weekly[column] = _clean_numeric(
            weekly[column],
            source=f"dts_weekly_flow_decomposition.{column}",
            nullable=False,
        )
    actual_last = weekly["h8_week_end"].max()
    if actual_last != expected_last_complete:
        raise ValueError(
            "weekly completeness reference is stale: "
            f"expected {expected_last_complete.date()}, got {actual_last.date()}"
        )
    return weekly.sort_values("h8_week_end").reset_index(drop=True)


def _h8_reconciliation_rows(
    event_daily: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    weekly_metadata: Mapping[str, object],
    atol: float = RESIDUAL_TOLERANCE_BN,
) -> pd.DataFrame:
    last_complete = weekly["h8_week_end"].max()
    complete_events = event_daily.loc[
        event_daily["h8_week_end"].le(last_complete)
    ].copy()
    observed_weeks = set(complete_events["h8_week_end"])
    reference_weeks = set(weekly["h8_week_end"])
    if not observed_weeks.issubset(reference_weeks):
        missing = sorted(observed_weeks - reference_weeks)[:5]
        raise ValueError(f"complete event weeks are missing from weekly reference: {missing}")
    totals = complete_events.groupby(
        ["h8_week_end", "stream_id"],
        sort=True,
    )["amount_bn"].sum()
    rows: list[dict[str, object]] = []
    for weekly_row in weekly.itertuples(index=False):
        h8_week_end = pd.Timestamp(weekly_row.h8_week_end).normalize()
        for stream_id in STREAM_ORDER:
            weekly_column = WEEKLY_REFERENCE_COLUMNS[stream_id]
            amount = float(totals.get((h8_week_end, stream_id), 0.0))
            reference = float(getattr(weekly_row, weekly_column))
            residual = amount - reference
            if not np.isfinite(residual) or abs(residual) > atol:
                raise AssertionError(
                    "H.8 stream reconciliation failed for "
                    f"{h8_week_end.date()} {stream_id}: {residual}"
                )
            row = {column: pd.NA for column in SCHEMA_COLUMNS}
            row.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_type": "h8_reconciliation",
                    "lane": "observed_dts_direct",
                    "status": "reconciled",
                    "units": "USD_billions",
                    "source_id": "dts_weekly_flow_decomposition",
                    "source_path": weekly_metadata["source_path"],
                    "source_sha256": weekly_metadata["source_sha256"],
                    "source_units": "USD_billions",
                    "unit_scale_to_bn": 1.0,
                    "h8_week_start": h8_week_end - pd.Timedelta(days=6),
                    "h8_week_end": h8_week_end,
                    "h8_endpoint_quarter": str(h8_week_end.to_period("Q-DEC")),
                    "h8_window_complete_flag": True,
                    "stream_id": stream_id,
                    "perimeter_role": STREAM_ROLE[stream_id],
                    "timing_status": "observed_settlement_reconciliation",
                    "treatment_perimeter": TREATMENT_PERIMETER,
                    "outcome_perimeter": OUTCOME_PERIMETER,
                    "sign_convention": SIGN_CONVENTION,
                    "amount_bn": amount,
                    "reconciliation_level": "h8_week",
                    "reference_id": weekly_column,
                    "reference_value_bn": reference,
                    "aggregation_residual_bn": residual,
                    "residual_tolerance_bn": atol,
                }
            )
            rows.append(row)
    output = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    if output.duplicated(["h8_week_end", "stream_id"]).any():
        raise AssertionError("h8_reconciliation uniqueness key is not unique")
    return output


def _quarter_reconciliation_rows(
    event_daily: pd.DataFrame,
    classified: pd.DataFrame,
    *,
    atol: float = RESIDUAL_TOLERANCE_BN,
) -> pd.DataFrame:
    mapped = classified.loc[
        classified["coverage_class"].eq("mapped_observed")
    ].copy()
    mapped["_quarter"] = mapped["record_date"].dt.to_period("Q-DEC").astype(str)
    raw_amounts = mapped.groupby(["_quarter", "stream_id"])["amount_bn"].sum()
    raw_counts = mapped.groupby(["_quarter", "stream_id"])["_raw_key"].size()
    raw_categories = mapped.groupby(
        ["_quarter", "stream_id"]
    )["_source_category"].nunique()
    raw_keysets = mapped.groupby(
        ["_quarter", "stream_id"]
    )["_raw_key"].apply(_keyset_sha256)
    event_amounts = event_daily.groupby(
        ["aggregation_quarter", "stream_id"]
    )["amount_bn"].sum()
    first = mapped["record_date"].min().to_period("Q-DEC")
    last = mapped["record_date"].max().to_period("Q-DEC")
    rows: list[dict[str, object]] = []
    for period in pd.period_range(first, last, freq="Q-DEC"):
        quarter = str(period)
        for stream_id in STREAM_ORDER:
            key = (quarter, stream_id)
            amount = float(event_amounts.get(key, 0.0))
            reference = float(raw_amounts.get(key, 0.0))
            residual = amount - reference
            if not np.isfinite(residual) or abs(residual) > atol:
                raise AssertionError(
                    "quarter stream reconciliation failed for "
                    f"{quarter} {stream_id}: {residual}"
                )
            row = {column: pd.NA for column in SCHEMA_COLUMNS}
            row.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_type": "quarter_reconciliation",
                    "lane": "observed_dts_direct",
                    "status": "reconciled",
                    "units": "USD_billions",
                    "aggregation_quarter": quarter,
                    "aggregation_quarter_clock": "record_date",
                    "quarter_end_date": period.end_time.normalize(),
                    "stream_id": stream_id,
                    "perimeter_role": STREAM_ROLE[stream_id],
                    "timing_status": "observed_settlement_reconciliation",
                    "treatment_perimeter": TREATMENT_PERIMETER,
                    "outcome_perimeter": OUTCOME_PERIMETER,
                    "sign_convention": SIGN_CONVENTION,
                    "amount_bn": amount,
                    "source_row_count": int(raw_counts.get(key, 0)),
                    "source_category_count": int(raw_categories.get(key, 0)),
                    "source_keyset_sha256": (
                        str(raw_keysets.get(key))
                        if key in raw_keysets
                        else _keyset_sha256([])
                    ),
                    "reconciliation_level": "record_date_quarter",
                    "reference_id": "independent_raw_record_date_grouping",
                    "reference_value_bn": reference,
                    "aggregation_residual_bn": residual,
                    "residual_tolerance_bn": atol,
                }
            )
            rows.append(row)
    output = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    if output.duplicated(["aggregation_quarter", "stream_id"]).any():
        raise AssertionError("quarter_reconciliation uniqueness key is not unique")
    return output


def _walk_json(value: object) -> Iterable[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _validate_handoff_contract(payload: object) -> None:
    tier2_id = "tdc_tier2_interest_corrected_bank_only_ru_flow"
    base_id = "tdc_base_bank_only_ru_flow"
    mappings = list(_walk_json(payload))
    tier2_matches = [
        mapping
        for mapping in mappings
        if mapping.get("artifact_key") == tier2_id
        or mapping.get("series_key") == tier2_id
        or mapping.get("primary_artifact_key") == tier2_id
    ]
    if not tier2_matches:
        raise ValueError("TDCest handoff does not name the exact Tier-2 anchor")
    if not any(
        mapping.get("current_role") == "working_corrected_headline"
        or mapping.get("primary_role") == "headline_default"
        for mapping in tier2_matches
    ):
        raise ValueError(
            "TDCest handoff does not designate the exact Tier-2 anchor as current"
        )
    if not any(
        base_id in {
            mapping.get("artifact_key"),
            mapping.get("series_key"),
            mapping.get("comparison_artifact_key"),
        }
        for mapping in mappings
    ):
        raise ValueError("TDCest handoff does not name the current base comparison")


def _validate_legacy_attestation(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("legacy validation report must be a JSON object")
    if payload.get("status") != "ok":
        raise ValueError("legacy validation report status is not ok")
    if payload.get("method") != "indicator_denton_headline":
        raise ValueError("legacy validation report method is unexpected")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, list):
        raise ValueError("legacy validation diagnostics are missing")
    by_component = {
        item.get("component"): item
        for item in diagnostics
        if isinstance(item, dict)
    }
    if set(by_component) != set(_LEGACY_COMPONENTS):
        raise ValueError("legacy validation component set is not exact")
    for component in _LEGACY_COMPONENTS:
        item = by_component[component]
        if item.get("status") != "ok":
            raise ValueError(f"legacy validation failed for {component}")
        value = float(item.get("max_abs_quarterly_error", np.inf))
        if not np.isfinite(value) or value > NATIVE_LEGACY_TOLERANCE_MILLIONS:
            raise ValueError(
                f"legacy validation residual exceeds tolerance for {component}: {value}"
            )


def _anchor_base_row(
    *,
    period: pd.Period,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    row = {column: pd.NA for column in SCHEMA_COLUMNS}
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "row_type": "quarter_anchor",
            "lane": "quarterly_comparison",
            "status": "anchor_available",
            "units": "USD_billions",
            "source_id": metadata["source_id"],
            "source_path": metadata["source_path"],
            "source_sha256": metadata["source_sha256"],
            "source_retrieved_at": metadata.get("source_retrieved_at", ""),
            "source_endpoint": metadata.get("source_endpoint", ""),
            "source_units": "$1,000,000",
            "unit_scale_to_bn": 0.001,
            "aggregation_quarter": str(period),
            "aggregation_quarter_clock": "quarter_end_anchor",
            "quarter_end_date": period.end_time.normalize(),
            "perimeter_role": "comparison",
            "treatment_perimeter": "quarterly_tdc_comparison_only",
            "outcome_perimeter": "not_applicable_quarterly_anchor",
            "source_row_count": 1,
        }
    )
    return row


def _current_anchor_rows(
    tdc_estimates: pd.DataFrame,
    handoff_payload: object,
    *,
    estimates_metadata: Mapping[str, object],
    first_period: pd.Period,
    last_period: pd.Period,
) -> list[dict[str, object]]:
    tier2_column = "tdc_tier2_interest_corrected_bank_only_ru_flow"
    base_column = "tdc_base_bank_only_ru_flow"
    _validate_handoff_contract(handoff_payload)
    _require_columns(
        tdc_estimates,
        ["date", tier2_column, base_column],
        source="tdc_estimates",
    )
    estimates = tdc_estimates.copy()
    estimates["_date"] = _strict_dates(estimates["date"], source="tdc_estimates.date")
    if estimates["_date"].duplicated().any():
        raise ValueError("tdc_estimates: duplicate quarter-end dates")
    periods = estimates["_date"].dt.to_period("Q-DEC")
    if not estimates["_date"].eq(periods.dt.end_time.dt.normalize()).all():
        raise ValueError("tdc_estimates: every date must be a calendar quarter end")
    estimates["_period"] = periods
    for column in (tier2_column, base_column):
        estimates[column] = _clean_numeric(
            estimates[column],
            source=f"tdc_estimates.{column}",
            nullable=True,
        )
    selected = estimates.loc[
        estimates["_period"].between(first_period, last_period)
    ].copy()
    rows: list[dict[str, object]] = []
    contracts = (
        (
            tier2_column,
            "working_corrected_headline",
            "tdcest_handoff_working_corrected_headline_v1",
        ),
        (
            base_column,
            "base_comparison",
            "tdcest_handoff_base_comparison_v1",
        ),
    )
    for column, role, contract_id in contracts:
        available = selected.loc[selected[column].notna()]
        for _, source_row in available.iterrows():
            period = source_row["_period"]
            value_bn = float(source_row[column]) * 0.001
            row = _anchor_base_row(
                period=period,
                metadata=estimates_metadata,
            )
            row.update(
                {
                    "timing_status": "quarterly_no_intraquarter_timing",
                    "amount_bn": value_bn,
                    "anchor_id": column,
                    "anchor_role": role,
                    "anchor_contract_id": contract_id,
                    "anchor_source_column": column,
                    "anchor_value_bn": value_bn,
                }
            )
            rows.append(row)
    return rows


def _strict_month_end_dates(frame: pd.DataFrame, *, source: str) -> pd.Series:
    dates = _strict_dates(frame["date"], source=f"{source}.date")
    if dates.duplicated().any():
        raise ValueError(f"{source}: duplicate dates")
    if not dates.eq(dates + pd.offsets.MonthEnd(0)).all():
        raise ValueError(f"{source}: every date must be month end")
    return dates


def _legacy_anchor_rows(
    monthly_proxy: pd.DataFrame,
    monthly_components: pd.DataFrame,
    quarterly_sums: pd.DataFrame,
    validation_payload: object,
    *,
    proxy_metadata: Mapping[str, object],
    first_period: pd.Period,
    last_period: pd.Period,
) -> list[dict[str, object]]:
    _validate_legacy_attestation(validation_payload)
    _require_columns(
        monthly_proxy,
        ["date", "tdc_monthly"],
        source="legacy_monthly_proxy",
    )
    _require_columns(
        monthly_components,
        ["date", *_LEGACY_COMPONENTS],
        source="legacy_monthly_components",
    )
    _require_columns(
        quarterly_sums,
        ["date", *_LEGACY_COMPONENTS],
        source="legacy_quarterly_sums",
    )

    proxy = monthly_proxy.copy()
    components = monthly_components.copy()
    proxy["_date"] = _strict_month_end_dates(proxy, source="legacy_monthly_proxy")
    components["_date"] = _strict_month_end_dates(
        components,
        source="legacy_monthly_components",
    )
    proxy["tdc_monthly"] = _clean_numeric(
        proxy["tdc_monthly"],
        source="legacy_monthly_proxy.tdc_monthly",
        nullable=True,
    )
    for component in _LEGACY_COMPONENTS:
        components[component] = _clean_numeric(
            components[component],
            source=f"legacy_monthly_components.{component}",
            nullable=True,
        )
    monthly = proxy[["_date", "tdc_monthly"]].merge(
        components[["_date", *_LEGACY_COMPONENTS]],
        on="_date",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not monthly["_merge"].eq("both").all():
        raise ValueError("legacy proxy/component monthly date sets differ")
    monthly = monthly.drop(columns="_merge")
    monthly["_period"] = monthly["_date"].dt.to_period("Q-DEC")
    monthly = monthly.loc[
        monthly["_period"].between(first_period, last_period)
    ].copy()
    if monthly.empty:
        return []
    if monthly[["tdc_monthly", *_LEGACY_COMPONENTS]].isna().any(axis=None):
        raise ValueError("legacy selected monthly comparison values are incomplete")
    monthly_component_total = monthly.loc[:, list(_LEGACY_COMPONENTS)].sum(
        axis=1,
        min_count=len(_LEGACY_COMPONENTS),
    )
    monthly_identity_error = monthly["tdc_monthly"] - monthly_component_total
    if monthly_identity_error.abs().gt(NATIVE_LEGACY_TOLERANCE_MILLIONS).any():
        raise AssertionError("legacy monthly proxy does not equal its five components")
    counts = monthly.groupby("_period")["_date"].size()
    if not counts.eq(3).all():
        bad = counts.loc[~counts.eq(3)].head(5).to_dict()
        raise ValueError(f"legacy comparison needs exactly three months per quarter: {bad}")

    anchors = quarterly_sums.copy()
    anchors["_date"] = _strict_dates(
        anchors["date"],
        source="legacy_quarterly_sums.date",
    )
    if anchors["_date"].duplicated().any():
        raise ValueError("legacy_quarterly_sums: duplicate quarter ends")
    anchors["_period"] = anchors["_date"].dt.to_period("Q-DEC")
    if not anchors["_date"].eq(
        anchors["_period"].dt.end_time.dt.normalize()
    ).all():
        raise ValueError("legacy_quarterly_sums: dates must be quarter ends")
    for component in _LEGACY_COMPONENTS:
        anchors[component] = _clean_numeric(
            anchors[component],
            source=f"legacy_quarterly_sums.{component}",
            nullable=True,
        )
    anchors = anchors.loc[
        anchors["_period"].between(first_period, last_period)
    ].copy()
    periods = sorted(monthly["_period"].unique())
    anchors = anchors.loc[anchors["_period"].isin(periods)]
    if set(anchors["_period"]) != set(periods):
        raise ValueError("legacy quarterly anchors do not cover selected proxy quarters")
    if anchors[list(_LEGACY_COMPONENTS)].isna().any(axis=None):
        raise ValueError("legacy selected quarterly component anchors are incomplete")

    component_sums = monthly.groupby("_period")[list(_LEGACY_COMPONENTS)].sum()
    anchors_indexed = anchors.set_index("_period")[list(_LEGACY_COMPONENTS)]
    anchors_indexed = anchors_indexed.reindex(component_sums.index)
    component_errors = component_sums - anchors_indexed
    if component_errors.abs().gt(NATIVE_LEGACY_TOLERANCE_MILLIONS).any(axis=None):
        worst = float(component_errors.abs().max().max())
        raise AssertionError(
            f"legacy monthly components fail quarterly identities: {worst}"
        )
    proxy_sums = monthly.groupby("_period")["tdc_monthly"].sum()
    anchor_totals = anchors_indexed.sum(
        axis=1,
        min_count=len(_LEGACY_COMPONENTS),
    )
    total_errors = proxy_sums - anchor_totals
    if total_errors.abs().gt(NATIVE_LEGACY_TOLERANCE_MILLIONS).any():
        raise AssertionError("legacy monthly proxy fails its five-component anchor")

    rows: list[dict[str, object]] = []
    for period in component_sums.index:
        amount_bn = float(proxy_sums.loc[period]) * 0.001
        anchor_bn = float(anchor_totals.loc[period]) * 0.001
        residual_bn = amount_bn - anchor_bn
        row = _anchor_base_row(period=period, metadata=proxy_metadata)
        row.update(
            {
                "lane": "generated_quarterly_comparison",
                "status": "legacy_contract_reconciled",
                "timing_status": "generated_not_observed",
                "amount_bn": amount_bn,
                "source_row_count": 3,
                "source_category_count": len(_LEGACY_COMPONENTS),
                "aggregation_residual_bn": residual_bn,
                "residual_tolerance_bn": RESIDUAL_TOLERANCE_BN,
                "anchor_id": "legacy_five_component_denton",
                "anchor_role": "legacy_generated_comparison",
                "anchor_contract_id": "legacy_five_component_denton_v1",
                "anchor_source_column": "tdc_monthly",
                "anchor_value_bn": anchor_bn,
            }
        )
        rows.append(row)
    return rows


def _quarter_anchor_rows(
    *,
    tdc_estimates: pd.DataFrame,
    handoff_payload: object,
    monthly_proxy: pd.DataFrame,
    monthly_components: pd.DataFrame,
    quarterly_sums: pd.DataFrame,
    validation_payload: object,
    estimates_metadata: Mapping[str, object],
    proxy_metadata: Mapping[str, object],
    first_period: pd.Period,
    last_period: pd.Period,
) -> pd.DataFrame:
    rows = [
        *_current_anchor_rows(
            tdc_estimates,
            handoff_payload,
            estimates_metadata=estimates_metadata,
            first_period=first_period,
            last_period=last_period,
        ),
        *_legacy_anchor_rows(
            monthly_proxy,
            monthly_components,
            quarterly_sums,
            validation_payload,
            proxy_metadata=proxy_metadata,
            first_period=first_period,
            last_period=last_period,
        ),
    ]
    output = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    if output.empty:
        raise ValueError("quarter anchor/comparison surface is empty")
    if output.duplicated(["aggregation_quarter", "anchor_id"]).any():
        raise AssertionError("quarter_anchor uniqueness key is not unique")
    current = output["anchor_id"].isin(
        {
            "tdc_tier2_interest_corrected_bank_only_ru_flow",
            "tdc_base_bank_only_ru_flow",
        }
    )
    if output.loc[current, "aggregation_residual_bn"].notna().any():
        raise AssertionError("current quarterly comparisons may not carry residuals")
    if output.loc[current, "timing_status"].ne(
        "quarterly_no_intraquarter_timing"
    ).any():
        raise AssertionError("current anchors claim an unapproved timing status")
    if output.loc[
        output["anchor_id"].eq("legacy_five_component_denton"),
        "timing_status",
    ].ne("generated_not_observed").any():
        raise AssertionError("legacy comparison timing status is not explicit")
    return output


def _load_json(path: Path, *, source: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source}: invalid required JSON") from exc


def _resolved_file(path_value: str | Path, *, source: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{source}: required input does not exist: {path}")
    return path


def _manifest_metadata(
    manifest_payload: object,
    *,
    source_id: str,
    amount_column: str,
    source_path: str,
    source_sha256: str,
    observed_rows: int,
) -> dict[str, object]:
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"{source_id}: manifest must be a JSON object")
    if int(manifest_payload.get("rows", -1)) != observed_rows:
        raise ValueError(f"{source_id}: manifest row count is stale")
    endpoint = str(manifest_payload.get("endpoint", "")).strip()
    retrieved_at = str(manifest_payload.get("retrieved_at", "")).strip()
    formats = (
        manifest_payload.get("meta", {}).get("dataFormats", {})
        if isinstance(manifest_payload.get("meta"), dict)
        else {}
    )
    source_units = str(formats.get(amount_column, "")).strip()
    if not endpoint or not retrieved_at or source_units != "$1,000,000":
        raise ValueError(f"{source_id}: manifest retrieval/unit metadata is incomplete")
    return {
        "source_path": source_path,
        "source_sha256": source_sha256,
        "source_retrieved_at": retrieved_at,
        "source_endpoint": endpoint,
        "source_units": source_units,
        "unit_scale_to_bn": 0.001,
    }


def _normalized_lineage_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    if (
        not normalized
        or normalized.startswith("../")
        or "/../" in normalized
        or normalized.startswith("/")
    ):
        raise ValueError(f"invalid normalized lineage path: {value!r}")
    return normalized


def _input_bundle_sha256(lineage: Mapping[str, str]) -> str:
    normalized = {
        _normalized_lineage_path(path): str(digest).lower()
        for path, digest in lineage.items()
    }
    if len(normalized) != len(lineage):
        raise ValueError("input bundle has duplicate normalized paths")
    for path, digest in normalized.items():
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"input bundle has invalid SHA-256 for {path}")
    payload = "\n".join(
        f"{path}\t{normalized[path]}" for path in sorted(normalized)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _crosswalk_gate(crosswalk: pd.DataFrame) -> None:
    _require_columns(
        crosswalk,
        ["bucket", "above_threshold", "mapped"],
        source="dts_category_crosswalk",
    )

    def truthy(values: pd.Series) -> pd.Series:
        return values.astype("string").str.strip().str.lower().isin(
            {"true", "1", "yes"}
        )

    failures = crosswalk.loc[
        crosswalk["bucket"].astype("string").eq("unmapped")
        & truthy(crosswalk["above_threshold"])
        & ~truthy(crosswalk["mapped"])
    ]
    if not failures.empty:
        raise ValueError(
            "dts_category_crosswalk unmapped_above_threshold gate failed"
        )


def _format_output(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    date_columns = (
        "record_date",
        "h8_week_start",
        "h8_week_end",
        "quarter_end_date",
        "coverage_start_date",
        "coverage_end_date",
    )
    for column in date_columns:
        parsed = pd.to_datetime(output[column], errors="coerce")
        output[column] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), "")
    for column in ("h8_cross_quarter_flag", "h8_window_complete_flag"):
        output[column] = output[column].map(
            lambda value: (
                ""
                if pd.isna(value)
                else ("true" if bool(value) else "false")
            )
        )
    output["quarter_end_age_bin"] = (
        output["quarter_end_age_bin"].astype("string").fillna("")
    )
    return output


def _validate_final_output(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != SCHEMA_COLUMNS:
        raise AssertionError("final bridge column order differs from v1 schema")
    if set(frame["row_type"]) != set(ROW_TYPES):
        raise AssertionError("final bridge row-type set is not exact")
    for column in ("input_bundle_sha256", "mapping_contract_sha256"):
        values = frame[column].astype("string")
        if not values.str.fullmatch(r"[0-9a-f]{64}", na=False).all():
            raise AssertionError(f"{column} is not a lowercase SHA-256 on every row")
        if values.nunique() != 1:
            raise AssertionError(f"{column} differs across output rows")
    source_hashes = frame["source_sha256"].dropna().astype("string")
    source_hashes = source_hashes.loc[source_hashes.ne("")]
    if not source_hashes.str.fullmatch(r"[0-9a-f]{64}", na=False).all():
        raise AssertionError("source_sha256 contains a malformed value")

    uniqueness = {
        "event_daily": ["source_id", "record_date", "component_id"],
        "h8_reconciliation": ["h8_week_end", "stream_id"],
        "quarter_reconciliation": ["aggregation_quarter", "stream_id"],
        "input_coverage": ["source_id", "coverage_class"],
        "quarter_anchor": ["aggregation_quarter", "anchor_id"],
    }
    for row_type, key in uniqueness.items():
        selected = frame.loc[frame["row_type"].eq(row_type)]
        if selected.duplicated(key).any():
            raise AssertionError(f"{row_type} uniqueness key failed")

    events = frame.loc[frame["row_type"].eq("event_daily")]
    if not set(events["quarter_end_age_bin"].astype(str)).issubset(
        set(AGE_BIN_ORDER)
    ):
        raise AssertionError("event_daily contains an unknown age bin")
    if set(events["stream_id"]) != set(STREAM_ORDER):
        raise AssertionError("event_daily stream set is not exact")
    if events["sign_convention"].ne(SIGN_CONVENTION).any():
        raise AssertionError("event_daily sign convention changed")
    if events["anchor_id"].notna().any() or events["anchor_value_bn"].notna().any():
        raise AssertionError("observed event rows carry quarterly anchors")
    incomplete_endpoints = set(
        events.loc[
            events["h8_window_complete_flag"].eq(False),
            "h8_week_end",
        ]
    )
    h8_endpoints = set(
        frame.loc[
            frame["row_type"].eq("h8_reconciliation"),
            "h8_week_end",
        ]
    )
    if incomplete_endpoints & h8_endpoints:
        raise AssertionError("incomplete H.8 windows have reconciliation rows")

    current_anchor = frame[
        "anchor_id"
    ].isin(
        {
            "tdc_tier2_interest_corrected_bank_only_ru_flow",
            "tdc_base_bank_only_ru_flow",
        }
    )
    if frame.loc[current_anchor, "aggregation_residual_bn"].notna().any():
        raise AssertionError("current TDC comparisons carry accounting residuals")
    legacy_monthly_dates = frame.loc[
        frame["row_type"].eq("quarter_anchor"),
        ["record_date", "h8_week_start", "h8_week_end"],
    ]
    if legacy_monthly_dates.notna().any(axis=None):
        raise AssertionError("quarterly anchors claim event or H.8 timing")


def build_quarterly_hf_bridge_csv(
    *,
    transactions_csv: str | Path,
    refunds_csv: str | Path,
    tax_deposits_csv: str | Path,
    weekly_flows_csv: str | Path,
    crosswalk_csv: str | Path,
    tdc_estimates_csv: str | Path,
    tdc_handoff_json: str | Path,
    legacy_proxy_csv: str | Path,
    legacy_components_csv: str | Path,
    legacy_quarterly_sums_csv: str | Path,
    legacy_validation_json: str | Path,
    out_csv: str | Path,
    transactions_manifest_json: str | Path | None = None,
    refunds_manifest_json: str | Path | None = None,
    tax_deposits_manifest_json: str | Path | None = None,
) -> dict[str, object]:
    """Write the fail-closed deterministic OPEN-03 bridge without refitting."""
    specs = {
        "transactions": (
            transactions_csv,
            "data/raw/fiscaldata/dts_deposits_withdrawals_operating_cash.csv",
        ),
        "refunds": (
            refunds_csv,
            "data/raw/fiscaldata/dts_income_tax_refunds_issued.csv",
        ),
        "tax_deposits": (
            tax_deposits_csv,
            "data/raw/fiscaldata/dts_federal_tax_deposits.csv",
        ),
        "weekly_flows": (
            weekly_flows_csv,
            "data/processed/dts_weekly_flow_decomposition.csv",
        ),
        "crosswalk": (
            crosswalk_csv,
            "data/processed/dts_category_crosswalk.csv",
        ),
        "tdc_estimates": (
            tdc_estimates_csv,
            "external/tdcest/data/processed/tdc_estimates.csv",
        ),
        "tdc_handoff": (
            tdc_handoff_json,
            "external/tdcest/data/processed/tdc_downstream_handoff_bundle.json",
        ),
        "legacy_proxy": (
            legacy_proxy_csv,
            "data/processed/fred_denton_proxy/tdc_monthly_proxy.csv",
        ),
        "legacy_components": (
            legacy_components_csv,
            "data/processed/fred_denton_proxy/tdc_monthly_components.csv",
        ),
        "legacy_quarterly_sums": (
            legacy_quarterly_sums_csv,
            "data/processed/fred_denton_proxy/tdc_monthly_quarterly_sums.csv",
        ),
        "legacy_validation": (
            legacy_validation_json,
            "data/processed/fred_denton_proxy/tdc_validation_report.json",
        ),
    }
    resolved = {
        key: _resolved_file(path, source=key)
        for key, (path, _) in specs.items()
    }
    manifest_values = {
        "transactions_manifest": (
            transactions_manifest_json
            or f"{transactions_csv}.manifest.json",
            "data/raw/fiscaldata/dts_deposits_withdrawals_operating_cash.csv.manifest.json",
        ),
        "refunds_manifest": (
            refunds_manifest_json or f"{refunds_csv}.manifest.json",
            "data/raw/fiscaldata/dts_income_tax_refunds_issued.csv.manifest.json",
        ),
        "tax_deposits_manifest": (
            tax_deposits_manifest_json
            or f"{tax_deposits_csv}.manifest.json",
            "data/raw/fiscaldata/dts_federal_tax_deposits.csv.manifest.json",
        ),
    }
    resolved_manifests = {
        key: _resolved_file(path, source=key)
        for key, (path, _) in manifest_values.items()
    }

    frames = {
        key: pd.read_csv(resolved[key], low_memory=False)
        for key in (
            "transactions",
            "refunds",
            "tax_deposits",
            "weekly_flows",
            "crosswalk",
            "tdc_estimates",
            "legacy_proxy",
            "legacy_components",
            "legacy_quarterly_sums",
        )
    }
    json_payloads = {
        "tdc_handoff": _load_json(
            resolved["tdc_handoff"],
            source="tdc_handoff",
        ),
        "legacy_validation": _load_json(
            resolved["legacy_validation"],
            source="legacy_validation",
        ),
        **{
            key: _load_json(path, source=key)
            for key, path in resolved_manifests.items()
        },
    }
    _crosswalk_gate(frames["crosswalk"])

    lineage = {
        virtual_path: _sha256(resolved[key])
        for key, (_, virtual_path) in specs.items()
    }
    lineage.update(
        {
            virtual_path: _sha256(resolved_manifests[key])
            for key, (_, virtual_path) in manifest_values.items()
        }
    )
    mapping_sources = {
        "src/tdchf/quarterly_bridge.py": Path(__file__).resolve(),
        "src/tdchf/disbursement.py": Path(__file__).with_name(
            "disbursement.py"
        ).resolve(),
    }
    for virtual_path, path in mapping_sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"mapping source missing: {path}")
        lineage[virtual_path] = _sha256(path)
    input_bundle = _input_bundle_sha256(lineage)
    mapping_hash = mapping_contract_sha256()

    source_metadata = {
        SOURCE_TRANSACTION: _manifest_metadata(
            json_payloads["transactions_manifest"],
            source_id=SOURCE_TRANSACTION,
            amount_column="transaction_today_amt",
            source_path=specs["transactions"][1],
            source_sha256=lineage[specs["transactions"][1]],
            observed_rows=len(frames["transactions"]),
        ),
        SOURCE_REFUNDS: _manifest_metadata(
            json_payloads["refunds_manifest"],
            source_id=SOURCE_REFUNDS,
            amount_column="tax_refund_today_amt",
            source_path=specs["refunds"][1],
            source_sha256=lineage[specs["refunds"][1]],
            observed_rows=len(frames["refunds"]),
        ),
        SOURCE_TAX: _manifest_metadata(
            json_payloads["tax_deposits_manifest"],
            source_id=SOURCE_TAX,
            amount_column="tax_deposit_today_amt",
            source_path=specs["tax_deposits"][1],
            source_sha256=lineage[specs["tax_deposits"][1]],
            observed_rows=len(frames["tax_deposits"]),
        ),
    }
    classified = classify_observed_sources(
        frames["transactions"],
        frames["refunds"],
        frames["tax_deposits"],
        source_metadata,
    )
    last_complete = _expected_last_complete_week(classified)
    weekly = _weekly_frame(
        frames["weekly_flows"],
        expected_last_complete=last_complete,
    )
    event_daily = aggregate_event_daily(
        classified,
        last_complete_h8_week=last_complete,
    )
    coverage_rows = _coverage_rows(classified)
    weekly_metadata = {
        "source_path": specs["weekly_flows"][1],
        "source_sha256": lineage[specs["weekly_flows"][1]],
    }
    h8_rows = _h8_reconciliation_rows(
        event_daily,
        weekly,
        weekly_metadata=weekly_metadata,
    )
    quarter_rows = _quarter_reconciliation_rows(event_daily, classified)

    observed_periods = classified.loc[
        classified["coverage_class"].eq("mapped_observed"),
        "record_date",
    ].dt.to_period("Q-DEC")
    first_period = observed_periods.min()
    last_period = observed_periods.max()
    anchor_rows = _quarter_anchor_rows(
        tdc_estimates=frames["tdc_estimates"],
        handoff_payload=json_payloads["tdc_handoff"],
        monthly_proxy=frames["legacy_proxy"],
        monthly_components=frames["legacy_components"],
        quarterly_sums=frames["legacy_quarterly_sums"],
        validation_payload=json_payloads["legacy_validation"],
        estimates_metadata={
            "source_id": "tdcest_tdc_estimates",
            "source_path": specs["tdc_estimates"][1],
            "source_sha256": lineage[specs["tdc_estimates"][1]],
        },
        proxy_metadata={
            "source_id": "legacy_denton_monthly_proxy",
            "source_path": specs["legacy_proxy"][1],
            "source_sha256": lineage[specs["legacy_proxy"][1]],
        },
        first_period=first_period,
        last_period=last_period,
    )

    event_output = event_daily.drop(columns=["_source_keys"])
    output = pd.concat(
        [
            event_output,
            h8_rows,
            quarter_rows,
            coverage_rows,
            anchor_rows,
        ],
        ignore_index=True,
        sort=False,
    )
    output["input_bundle_sha256"] = input_bundle
    output["mapping_contract_sha256"] = mapping_hash
    output = output.loc[:, list(SCHEMA_COLUMNS)]
    _validate_final_output(output)

    row_order = {value: index for index, value in enumerate(ROW_TYPES)}
    stream_order = {value: index for index, value in enumerate(STREAM_ORDER)}
    coverage_order = {
        value: index for index, value in enumerate(COVERAGE_CLASSES)
    }
    anchor_order = {
        "tdc_tier2_interest_corrected_bank_only_ru_flow": 0,
        "tdc_base_bank_only_ru_flow": 1,
        "legacy_five_component_denton": 2,
    }
    output["_row_order"] = output["row_type"].map(row_order)
    output["_stream_order"] = output["stream_id"].map(stream_order)
    output["_coverage_order"] = output["coverage_class"].map(coverage_order)
    output["_anchor_order"] = output["anchor_id"].map(anchor_order)
    output = output.sort_values(
        [
            "_row_order",
            "record_date",
            "source_id",
            "component_id",
            "h8_week_end",
            "aggregation_quarter",
            "_stream_order",
            "_coverage_order",
            "_anchor_order",
        ],
        na_position="last",
        kind="mergesort",
    ).drop(
        columns=[
            "_row_order",
            "_stream_order",
            "_coverage_order",
            "_anchor_order",
        ]
    )
    output = _format_output(output)
    output = output.where(pd.notna(output), "")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    output.to_csv(temp_path, index=False, encoding="utf-8", lineterminator="\n")
    temp_path.replace(out_path)
    output_sha256 = _sha256(out_path)
    residuals = pd.to_numeric(
        output["aggregation_residual_bn"],
        errors="coerce",
    ).abs()
    row_counts = {
        key: int(value)
        for key, value in output["row_type"].value_counts().sort_index().items()
    }
    coverage_lookup = output.loc[
        output["row_type"].eq("input_coverage")
        & output["coverage_class"].eq("mapped_observed"),
        "coverage_row_count",
    ]
    return {
        "status": "ok",
        "out": out_path.as_posix(),
        "output_sha256": output_sha256,
        "rows": int(len(output)),
        "row_counts": row_counts,
        "mapped_observed_raw_rows": int(
            pd.to_numeric(coverage_lookup).sum()
        ),
        "event_daily_rows": int(row_counts.get("event_daily", 0)),
        "first_observed_date": event_daily["record_date"].min().date().isoformat(),
        "latest_observed_date": event_daily["record_date"].max().date().isoformat(),
        "last_complete_h8_week": last_complete.date().isoformat(),
        "input_bundle_sha256": input_bundle,
        "mapping_contract_sha256": mapping_hash,
        "max_abs_aggregation_residual_bn": float(residuals.max()),
        "lineage_inputs": len(lineage),
    }
