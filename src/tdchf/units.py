from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


TDC_TREATMENT_COLUMNS = {"tdc_monthly"}
MONETARY_FLOW_OUTCOMES = {
    "bank_credit",
    "broad_deposits",
    "broad_non_large_time_deposits",
    "auto_loans",
    "closed_end_residential_loans",
    "commercial_industrial_loans",
    "construction_land_development_loans",
    "consumer_loans",
    "cre_multifamily_loans",
    "cre_nonfarm_nonresidential_loans",
    "credit_card_revolving_loans",
    "deposits",
    "domestic_deposits",
    "domestic_non_large_time_deposits",
    "heloc_loans",
    "large_time_deposits",
    "institutional_mmf",
    "loans_to_commercial_banks",
    "loans_to_nondepository_financial_institutions",
    "onrrp",
    "other_consumer_loans",
    "reserves",
    "retail_mmf",
    "total_mmf",
}

MILLION_DOLLAR_FLOW_OUTCOMES = {
    "reserves",
}


def same_unit_multiplier(*, treatment: str, outcome: str) -> float:
    """Convert LP-IV coefficients to same-dollar pass-through when units differ.

    The thesis TDC proxy is anchored in millions of dollars, while the FRED
    H.8 banking and money-fund outcomes used here are in billions of dollars.
    A raw coefficient for these pairs is therefore billions per million;
    multiply by 1,000 to read it as dollars of outcome per dollar of TDC.
    Some Federal Reserve balance-sheet series, including reserve balances, are
    already reported in millions and need no rescaling.
    """

    if treatment in TDC_TREATMENT_COLUMNS and outcome in MILLION_DOLLAR_FLOW_OUTCOMES:
        return 1.0
    if treatment in TDC_TREATMENT_COLUMNS and outcome in MONETARY_FLOW_OUTCOMES:
        return 1000.0
    return 1.0


def same_unit_interpretation(*, treatment: str, outcome: str) -> str:
    if treatment in TDC_TREATMENT_COLUMNS and outcome in MONETARY_FLOW_OUTCOMES:
        return "outcome dollars per 1 TDC dollar"
    return "native coefficient units"


def add_same_unit_columns(
    frame: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str = "outcome",
    value_columns: Iterable[str] = ("beta", "se", "lower95", "upper95"),
) -> pd.DataFrame:
    out = frame.copy()
    if outcome_col not in out.columns:
        return out
    multipliers = out[outcome_col].map(lambda outcome: same_unit_multiplier(treatment=treatment_col, outcome=str(outcome)))
    out["same_unit_multiplier"] = multipliers
    out["same_unit_interpretation"] = [
        same_unit_interpretation(treatment=treatment_col, outcome=str(outcome))
        for outcome in out[outcome_col]
    ]
    for column in value_columns:
        if column in out.columns:
            out[f"same_unit_{column}"] = out[column] * multipliers
    return out
