from __future__ import annotations

import numpy as np
import pandas as pd

from tdchf.disbursement import (
    assign_h8_week,
    build_fiscal_calendar_weekly,
    build_stitched_tax_daily,
    build_weekly_flow_decomposition,
    estimate_disbursement_lps,
    map_dts_category,
    seam_diagnostic,
)


def test_assign_h8_week_uses_thursday_to_wednesday_window() -> None:
    assert assign_h8_week("2024-01-03") == pd.Timestamp("2024-01-03")
    assert assign_h8_week("2024-01-04") == pd.Timestamp("2024-01-10")
    assert assign_h8_week("2024-01-06") == pd.Timestamp("2024-01-10")


def test_tax_stitch_has_pre_and_post_seam_coverage() -> None:
    tx = pd.DataFrame(
        {
            "record_date": ["2023-02-14", "2023-02-14", "2023-02-14"],
            "transaction_type": ["Deposits", "Deposits", "Deposits"],
            "transaction_catg": [
                "Taxes - Withheld Individual/FICA",
                "Taxes - Non Withheld Ind/SECA Electronic",
                "Taxes - Corporate Income",
            ],
            "transaction_today_amt": ["1000", "200", "50"],
        }
    )
    dedicated = pd.DataFrame(
        {
            "record_date": ["2023-02-13", "2023-02-13", "2023-02-13"],
            "tax_deposit_type": [
                "Withheld Income and Employment Taxes",
                "Individual Income Taxes",
                "Corporation Income Taxes",
            ],
            "tax_deposit_today_amt": ["900", "180", "45"],
        }
    )

    stitched = build_stitched_tax_daily(tx, dedicated)
    diag = seam_diagnostic(stitched)

    assert set(stitched["source"]) == {"dedicated_federal_tax_deposits", "table_ii_tax_categories"}
    assert {"tax_withheld_bn", "tax_nonwithheld_bn", "tax_corporate_bn"}.issubset(set(stitched["tax_bucket"]))
    assert diag["verdict"] == "stitched"


def test_tax_stitch_excludes_dedicated_subtotal_rows() -> None:
    tx = pd.DataFrame(
        {
            "record_date": ["2023-02-14"],
            "transaction_type": ["Deposits"],
            "transaction_catg": ["Taxes - Withheld Individual/FICA"],
            "transaction_today_amt": ["1000"],
        }
    )
    dedicated = pd.DataFrame(
        {
            "record_date": ["2022-12-30"] * 4,
            "tax_deposit_type": [
                "Cash Federal Tax Deposits",
                "Withheld Income and Employment Taxes",
                "Individual Income Taxes",
                "Inter-agency Transfers",
            ],
            "tax_deposit_today_amt": ["4150", "3122", "591", "125"],
        }
    )

    stitched = build_stitched_tax_daily(tx, dedicated)
    annual = stitched.loc[pd.to_datetime(stitched["record_date"]).dt.year.eq(2022)].groupby("tax_bucket")["amount_bn"].sum()

    assert round(float(annual["tax_nonwithheld_bn"]), 3) == 0.591
    assert round(float(annual["tax_withheld_bn"]), 3) == 3.122


def test_real_tax_anchor_continuity_uses_component_rows() -> None:
    tx = pd.read_csv("data/raw/fiscaldata/dts_deposits_withdrawals_operating_cash.csv", parse_dates=["record_date"], low_memory=False)
    dedicated = pd.read_csv("data/raw/fiscaldata/dts_federal_tax_deposits.csv", parse_dates=["record_date"], low_memory=False)
    stitched = build_stitched_tax_daily(tx, dedicated)
    annual = (
        stitched.assign(year=pd.to_datetime(stitched["record_date"]).dt.year)
        .pivot_table(index="year", columns="tax_bucket", values="amount_bn", aggfunc="sum", fill_value=0.0)
    )

    assert 500 < float(annual.loc[2022, "tax_nonwithheld_bn"]) < 700
    for bucket in ["tax_withheld_bn", "tax_nonwithheld_bn", "tax_corporate_bn", "tax_other_bn"]:
        vals = annual.loc[[2022, 2023, 2024], bucket].replace(0, np.nan).dropna()
        ratios = vals.iloc[1:].to_numpy() / vals.iloc[:-1].to_numpy()
        assert (ratios > 0.2).all()
        assert (ratios < 5.0).all()


def test_crosswalk_mapping_keeps_core_and_broad_split() -> None:
    assert map_dts_category("Withdrawals", "SSA - Benefits Payments") == "du_core_benefits"
    assert map_dts_category("Withdrawals", "Federal Salaries (EFT)") == "du_core_salaries_other"
    assert map_dts_category("Withdrawals", "HHS - Grants to States for Medicaid") == "du_broad_outflows"
    assert map_dts_category("Deposits", "Taxes - Withheld Individual/FICA") == "tax_withheld"
    assert map_dts_category("Deposits", "Public Debt Cash Issues (Table III-B)") == "debt_issues_gross"
    assert map_dts_category("Deposits", "Unknown Giant Category") == "unmapped"


def test_weekly_decomposition_writes_reconciled_core_panel(tmp_path) -> None:
    transactions = tmp_path / "tx.csv"
    pd.DataFrame(
        {
            "record_date": ["2024-01-04", "2024-01-05", "2024-01-10", "2024-01-10"],
            "transaction_type": ["Withdrawals", "Deposits", "Deposits", "Withdrawals"],
            "transaction_catg": [
                "SSA - Benefits Payments",
                "Taxes - Withheld Individual/FICA",
                "Public Debt Cash Issues (Table III-B)",
                "Public Debt Cash Redemp. (Table IIIB)",
            ],
            "transaction_today_amt": ["1000", "500", "300", "200"],
        }
    ).to_csv(transactions, index=False)
    refunds = tmp_path / "refunds.csv"
    pd.DataFrame(
        {
            "record_date": ["2024-01-05"],
            "tax_refund_type": ["Taxes - Individual Tax Refunds (EFT)"],
            "tax_refund_today_amt": ["50"],
        }
    ).to_csv(refunds, index=False)
    tax = tmp_path / "tax.csv"
    pd.DataFrame(
        {
            "record_date": ["2023-02-13", "2023-02-13"],
            "tax_deposit_type": ["Withheld Income and Employment Taxes", "Cash Federal Tax Deposits"],
            "tax_deposit_today_amt": ["1", "99"],
        }
    ).to_csv(tax, index=False)
    ocb = tmp_path / "ocb.csv"
    pd.DataFrame(
        {
            "record_date": ["2024-01-03", "2024-01-10"],
            "account_type": ["Treasury General Account (TGA) Closing Balance"] * 2,
            "open_today_bal": ["10000", "9600"],
        }
    ).to_csv(ocb, index=False)

    report = build_weekly_flow_decomposition(
        transactions_csv=transactions,
        refunds_csv=refunds,
        tax_deposits_csv=tax,
        operating_cash_balance_csv=ocb,
        out_csv=tmp_path / "weekly.csv",
        crosswalk_csv=tmp_path / "crosswalk.csv",
    )
    out = pd.read_csv(tmp_path / "weekly.csv", parse_dates=["date"]).set_index("date")

    assert report["unmapped_above_threshold"] == 0
    assert out.loc[pd.Timestamp("2024-01-10"), "du_core_benefits_bn"] == 1.0
    assert out.loc[pd.Timestamp("2024-01-10"), "du_core_refunds_bn"] == 0.05
    assert out.loc[pd.Timestamp("2024-01-10"), "tax_receipts_bn"] == 0.5
    assert round(float(out.loc[pd.Timestamp("2024-01-10"), "debt_net_bn"]), 6) == 0.1


def test_calendar_places_known_shifted_tax_deadline_in_h8_week(tmp_path) -> None:
    build_fiscal_calendar_weekly(start="2020-07-01", end="2020-07-22", out_csv=tmp_path / "cal.csv")
    cal = pd.read_csv(tmp_path / "cal.csv", parse_dates=["date"]).set_index("date")

    assert cal.loc[pd.Timestamp("2020-07-15"), "tax_due_week"] == 1
    assert cal.loc[pd.Timestamp("2020-07-08"), "tax_due_week"] == 0


def test_disbursement_lp_recovers_planted_beta() -> None:
    n = 80
    dates = pd.date_range("2022-01-05", periods=n, freq="W-WED")
    rng = np.random.default_rng(123)
    core = rng.normal(4.0, 1.0, n)
    tax = rng.normal(3.0, 0.4, n)
    flows = pd.DataFrame(
        {
            "du_core_outflows_bn": core,
            "tax_receipts_bn": tax,
            "du_broad_outflows_bn": rng.normal(0, 0.1, n),
            "interest_outflows_bn": rng.normal(0, 0.1, n),
            "debt_issues_gross_bn": rng.normal(0, 0.1, n),
            "debt_redemptions_gross_bn": rng.normal(0, 0.1, n),
        },
        index=dates,
    )
    deposits = pd.Series(100.0, index=dates)
    deposits = deposits + 0.7 * flows["du_core_outflows_bn"].shift(0).fillna(0).cumsum()
    weekly = pd.DataFrame({"broad_deposits_nsa": deposits}, index=dates)
    calendar = pd.DataFrame({"date": dates, "tax_due_week": 0, "coupon_week": 0, "ssa_cycle_payment_count": 0})

    estimates = estimate_disbursement_lps(flows, calendar, weekly)
    row = estimates.loc[
        estimates["outcome"].eq("deposits_dpsacb")
        & estimates["spec_type"].eq("LP_lead_controlled")
        & estimates["sample"].eq("full")
        & estimates["treatment_id"].eq("du_core_outflows_bn")
        & estimates["horizon"].eq(0)
    ].iloc[0]

    assert round(float(row["beta"]), 1) == 0.7


def test_ex_pandemic_sample_does_not_pair_outcomes_across_hole() -> None:
    dates = pd.date_range("2019-01-02", periods=220, freq="W-WED")
    rng = np.random.default_rng(456)
    flows = pd.DataFrame(
        {
            "du_core_outflows_bn": rng.normal(1.0, 0.1, len(dates)),
            "tax_receipts_bn": rng.normal(1.0, 0.1, len(dates)),
            "du_broad_outflows_bn": rng.normal(0.0, 0.1, len(dates)),
            "interest_outflows_bn": rng.normal(0.0, 0.1, len(dates)),
            "debt_issues_gross_bn": rng.normal(0.0, 0.1, len(dates)),
            "debt_redemptions_gross_bn": rng.normal(0.0, 0.1, len(dates)),
        },
        index=dates,
    )
    deposits = pd.Series(np.arange(len(dates), dtype=float), index=dates)
    weekly = pd.DataFrame({"broad_deposits_nsa": deposits}, index=dates)
    calendar = pd.DataFrame({"date": dates, "tax_due_week": 0, "coupon_week": 0, "ssa_cycle_payment_count": 0})

    estimates = estimate_disbursement_lps(flows, calendar, weekly)
    ex_row = estimates.loc[
        estimates["outcome"].eq("deposits_dpsacb")
        & estimates["spec_type"].eq("LP_lead_controlled")
        & estimates["sample"].eq("ex_pandemic")
        & estimates["horizon"].eq(8)
        & estimates["treatment_id"].eq("du_core_outflows_bn")
    ].iloc[0]
    full_n = estimates.loc[
        estimates["outcome"].eq("deposits_dpsacb")
        & estimates["spec_type"].eq("LP_lead_controlled")
        & estimates["sample"].eq("full")
        & estimates["horizon"].eq(8)
        & estimates["treatment_id"].eq("du_core_outflows_bn"),
        "n",
    ].iloc[0]

    assert int(ex_row["n"]) < int(full_n)
    assert int(ex_row["n"]) <= int(full_n) - 50
