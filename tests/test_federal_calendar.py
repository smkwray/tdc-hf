from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tdchf.disbursement import build_fiscal_calendar_weekly
from tdchf.federal_calendar import (
    INDIVIDUAL_TAX_CATEGORIES,
    TaxDeadlineCategory as Category,
    federal_holidays,
    tax_deadline_dates,
    tax_deadlines,
)


def _due(year: int, category: Category) -> set[date]:
    return {item.due_date for item in tax_deadlines(year) if item.category is category}


def test_2021_postponement_moves_only_the_individual_annual_payment() -> None:
    # IR-2021-59: first-quarter 2021 estimates stayed due April 15; only the
    # individual 2020 balance due moved to May 17. Treating May 17 as the only
    # April-May deadline drops a real estimated-payment drain week.
    assert date(2021, 4, 15) in _due(2021, Category.INDIVIDUAL_ESTIMATED)
    assert date(2021, 5, 17) not in _due(2021, Category.INDIVIDUAL_ESTIMATED)
    assert _due(2021, Category.INDIVIDUAL_ANNUAL) == {date(2021, 5, 17)}
    assert date(2021, 4, 15) in _due(2021, Category.CORPORATE_ESTIMATED)
    assert _due(2021, Category.CORPORATE_ANNUAL) == {date(2021, 4, 15)}


def test_2020_postponement_moves_annual_and_first_two_estimates_to_july_15() -> None:
    # Notice 2020-23 covers Form 1040 and 1120 payments and Form 1040-ES and
    # 1120-W installments due April 1 through July 14, 2020.
    for category in Category:
        due = _due(2020, category)
        assert date(2020, 4, 15) not in due
        assert date(2020, 6, 15) not in due
        assert date(2020, 7, 15) in due
    assert _due(2020, Category.INDIVIDUAL_ESTIMATED) == {date(2020, 1, 15), date(2020, 7, 15), date(2020, 9, 15)}
    assert _due(2020, Category.CORPORATE_ESTIMATED) == {date(2020, 7, 15), date(2020, 9, 15), date(2020, 12, 15)}
    july_installments = [
        item
        for item in tax_deadlines(2020)
        if item.category is Category.INDIVIDUAL_ESTIMATED and item.due_date == date(2020, 7, 15)
    ]
    assert {item.statutory_date for item in july_installments} == {date(2020, 4, 15), date(2020, 6, 15)}


def test_weekend_and_legal_holiday_deadlines_move_to_next_business_day() -> None:
    # IRC section 7503: Saturday June 15, 2024 is due Monday June 17.
    assert date(2024, 6, 17) in _due(2024, Category.INDIVIDUAL_ESTIMATED)
    # Sunday January 15, 2023 followed by Martin Luther King Jr. Day.
    assert date(2023, 1, 17) in _due(2023, Category.INDIVIDUAL_ESTIMATED)
    # District of Columbia Emancipation Day observed Friday April 15, 2022
    # moves every April 15 deadline to Monday April 18.
    for category in Category:
        assert date(2022, 4, 18) in _due(2022, category)
        assert date(2022, 4, 15) not in _due(2022, category)
    # The national 2006 deadline ignored Emancipation Day (IR-2006-37); 2007 used it (IR-2007-15).
    assert _due(2006, Category.INDIVIDUAL_ANNUAL) == {date(2006, 4, 17)}
    assert _due(2007, Category.INDIVIDUAL_ANNUAL) == {date(2007, 4, 17)}
    # IR-2018-100 gave every April 17, 2018 filer and payer one more day.
    assert _due(2018, Category.INDIVIDUAL_ANNUAL) == {date(2018, 4, 18)}
    assert date(2018, 4, 18) in _due(2018, Category.CORPORATE_ESTIMATED)


def test_corporate_annual_deadline_moves_to_april_for_tax_years_after_2015() -> None:
    # Pub. L. 114-41 section 2006: tax year 2015 returns were due March 15, 2016;
    # tax year 2016 returns were due in April 2017 (April 18 after the weekend and Emancipation Day).
    assert _due(2016, Category.CORPORATE_ANNUAL) == {date(2016, 3, 15)}
    assert _due(2017, Category.CORPORATE_ANNUAL) == {date(2017, 4, 18)}
    assert _due(2019, Category.CORPORATE_ANNUAL) == {date(2019, 4, 15)}
    assert date(2017, 3, 15) not in tax_deadline_dates([2017], list(Category))


def test_deadline_category_selection_fails_closed() -> None:
    with pytest.raises(ValueError):
        tax_deadline_dates([2021], [])
    with pytest.raises(ValueError):
        tax_deadline_dates([2021], ["payroll_deposit"])


def test_weekly_tax_controls_use_individual_deadlines_only(tmp_path) -> None:
    out_csv = tmp_path / "cal.csv"
    build_fiscal_calendar_weekly(start="2021-03-31", end="2021-12-22", out_csv=out_csv)
    cal = pd.read_csv(out_csv, parse_dates=["date"]).set_index("date")

    expected = tax_deadline_dates([2021], INDIVIDUAL_TAX_CATEGORIES)
    assert expected == {date(2021, 1, 15), date(2021, 4, 15), date(2021, 5, 17), date(2021, 6, 15), date(2021, 9, 15)}
    # Thursday April 15, 2021 opens the H.8 week ending April 21.
    assert cal.loc[pd.Timestamp("2021-04-21"), "tax_due_week"] == 1
    assert cal.loc[pd.Timestamp("2021-04-21"), "tax_due_count"] == 1
    assert cal.loc[pd.Timestamp("2021-05-19"), "tax_due_week"] == 1
    # The corporate-only December installment is a documented exclusion from the existing control.
    assert cal.loc[pd.Timestamp("2021-12-15"), "tax_due_week"] == 0
    assert int(cal["tax_due_count"].sum()) == 4


def test_juneteenth_is_a_federal_holiday_only_from_2021() -> None:
    # Pub. L. 117-17 created the holiday on June 17, 2021; earlier June 19ths
    # were ordinary business days for Treasury payment and salary schedules.
    assert not any(day.month == 6 and day.day in {18, 19, 20} for day in federal_holidays(2020))
    assert date(2021, 6, 18) in federal_holidays(2021)
    assert date(2022, 6, 20) in federal_holidays(2022)


def test_weekly_holiday_count_excludes_pre_2021_juneteenth(tmp_path) -> None:
    out_csv = tmp_path / "cal.csv"
    build_fiscal_calendar_weekly(start="2020-06-10", end="2021-06-30", out_csv=out_csv)
    cal = pd.read_csv(out_csv, parse_dates=["date"]).set_index("date")

    # Friday June 19, 2020 falls in the H.8 week ending June 24, 2020.
    assert cal.loc[pd.Timestamp("2020-06-24"), "federal_holiday_count"] == 0
    # Friday June 18, 2021, the first observance, falls in the week ending June 23, 2021.
    assert cal.loc[pd.Timestamp("2021-06-23"), "federal_holiday_count"] == 1


def test_2026_deadlines_follow_the_statutory_rule_and_map_to_h8_weeks() -> None:
    # The 2026Q2-vintage refresh adds 2026 deadline events only through this rule:
    # no national postponement is encoded for 2026, January 15 falls on a Thursday
    # and so opens the H.8 week ending January 21, and June 15 and September 15
    # are weekdays, so no section 7503 shift applies.
    from tdchf.disbursement import assign_h8_week

    got = {(item.category, item.due_date) for item in tax_deadlines(2026)}
    assert got == {
        (Category.INDIVIDUAL_ESTIMATED, date(2026, 1, 15)),
        (Category.INDIVIDUAL_ESTIMATED, date(2026, 4, 15)),
        (Category.INDIVIDUAL_ESTIMATED, date(2026, 6, 15)),
        (Category.INDIVIDUAL_ESTIMATED, date(2026, 9, 15)),
        (Category.INDIVIDUAL_ANNUAL, date(2026, 4, 15)),
        (Category.CORPORATE_ESTIMATED, date(2026, 4, 15)),
        (Category.CORPORATE_ESTIMATED, date(2026, 6, 15)),
        (Category.CORPORATE_ESTIMATED, date(2026, 9, 15)),
        (Category.CORPORATE_ESTIMATED, date(2026, 12, 15)),
        (Category.CORPORATE_ANNUAL, date(2026, 4, 15)),
    }
    assert all(item.due_date == item.statutory_date for item in tax_deadlines(2026))
    weeks = {assign_h8_week(due).date() for _, due in got if due <= date(2026, 9, 30)}
    assert weeks == {date(2026, 1, 21), date(2026, 4, 15), date(2026, 6, 17), date(2026, 9, 16)}
