"""Federal holidays and federal income-tax payment deadlines.

The deadline calendar encodes national rules for calendar-year individuals and
calendar-year C corporations. Regional disaster postponements under IRC section
7508A, state holidays observed by a single IRS campus (for example, the 2006
Patriots' Day relief in IR-2006-37), and fiscal-year filers are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Iterable


class TaxDeadlineCategory(str, Enum):
    INDIVIDUAL_ESTIMATED = "individual_estimated"
    INDIVIDUAL_ANNUAL = "individual_annual"
    CORPORATE_ESTIMATED = "corporate_estimated"
    CORPORATE_ANNUAL = "corporate_annual"


INDIVIDUAL_TAX_CATEGORIES = frozenset(
    {TaxDeadlineCategory.INDIVIDUAL_ESTIMATED, TaxDeadlineCategory.INDIVIDUAL_ANNUAL}
)


@dataclass(frozen=True)
class TaxDeadline:
    category: TaxDeadlineCategory
    statutory_date: date
    due_date: date


def nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=((weekday - first.weekday()) % 7) + 7 * (nth - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year + int(month == 12), 1 if month == 12 else month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def observed_fixed_holiday(date_value: date) -> date:
    # 5 U.S.C. 6103(b) and Executive Order 11582: a fixed-date holiday falling
    # on Saturday is observed the preceding Friday; on Sunday, the next Monday.
    if date_value.weekday() == 5:
        return date_value - timedelta(days=1)
    if date_value.weekday() == 6:
        return date_value + timedelta(days=1)
    return date_value


def federal_holidays(year: int) -> set[date]:
    # Legal public holidays listed in 5 U.S.C. 6103(a).
    holidays = {
        observed_fixed_holiday(date(year, 1, 1)),
        nth_weekday(year, 1, 0, 3),
        nth_weekday(year, 2, 0, 3),
        last_weekday(year, 5, 0),
        observed_fixed_holiday(date(year, 7, 4)),
        nth_weekday(year, 9, 0, 1),
        nth_weekday(year, 10, 0, 2),
        observed_fixed_holiday(date(year, 11, 11)),
        nth_weekday(year, 11, 3, 4),
        observed_fixed_holiday(date(year, 12, 25)),
    }
    # Juneteenth National Independence Day Act (Pub. L. 117-17, June 17,
    # 2021) added June 19; it was first observed on Friday, June 18, 2021.
    if year >= 2021:
        holidays.add(observed_fixed_holiday(date(year, 6, 19)))
    return holidays


def dc_emancipation_day(year: int) -> date | None:
    # District of Columbia Emancipation Day (April 16) is a District legal
    # holiday, observed on Friday when April 16 is a Saturday (IRS Notice
    # 2011-17) and on Monday when it is a Sunday. The IRS first applied it to
    # the national deadline for 2007 (IR-2007-15); the 2006 national deadline
    # was Monday, April 17 (IR-2006-37).
    if year < 2007:
        return None
    return observed_fixed_holiday(date(year, 4, 16))


def tax_legal_holidays(year: int) -> set[date]:
    # IRC section 7503 counts legal holidays in the District of Columbia.
    holidays = federal_holidays(year)
    emancipation = dc_emancipation_day(year)
    if emancipation is not None:
        holidays.add(emancipation)
    return holidays


def next_business_day(date_value: date, holidays: set[date]) -> date:
    # IRC section 7503: an act due on a Saturday, Sunday, or legal holiday is
    # timely on the next succeeding day that is none of these.
    out = date_value
    while out.weekday() >= 5 or out in holidays:
        out += timedelta(days=1)
    return out


def _statutory_dates(category: TaxDeadlineCategory, year: int) -> list[date]:
    if category is TaxDeadlineCategory.INDIVIDUAL_ESTIMATED:
        # IRC section 6654(c)(2): installments on April 15, June 15, September
        # 15, and January 15 of the following taxable year.
        return [date(year, 1, 15), date(year, 4, 15), date(year, 6, 15), date(year, 9, 15)]
    if category is TaxDeadlineCategory.INDIVIDUAL_ANNUAL:
        # IRC sections 6072(a) and 6151(a): the calendar-year return and its
        # balance due on April 15.
        return [date(year, 4, 15)]
    if category is TaxDeadlineCategory.CORPORATE_ESTIMATED:
        # IRC section 6655(c)(2): installments on the 15th day of the 4th, 6th,
        # 9th, and 12th months of the taxable year.
        return [date(year, 4, 15), date(year, 6, 15), date(year, 9, 15), date(year, 12, 15)]
    # IRC section 6072(b), as amended by section 2006 of the Surface
    # Transportation and Veterans Health Care Choice Improvement Act of 2015
    # (Pub. L. 114-41): C corporation returns for taxable years beginning after
    # December 31, 2015 are due on the 15th day of the 4th month, earlier ones on
    # the 15th day of the 3rd month. The first calendar-year return under the new
    # rule (tax year 2016) was due in 2017; IRC section 6151(a) ties payment to it.
    return [date(year, 3 if year <= 2016 else 4, 15)]


def _national_postponement(category: TaxDeadlineCategory, due: date) -> date:
    # IR-2018-100: after an IRS systems outage, individuals and businesses with a
    # filing or payment due date of April 17, 2018 had until April 18, 2018.
    if due == date(2018, 4, 17):
        return date(2018, 4, 18)
    # IRS Notice 2020-23, section III.A: Form 1040 and Form 1120 payments and
    # Form 1040-ES and Form 1120-W estimated payments due on or after April 1,
    # 2020 and before July 15, 2020 are postponed to July 15, 2020.
    if date(2020, 4, 1) <= due < date(2020, 7, 15):
        return date(2020, 7, 15)
    # IR-2021-59 ("Tax Day for individuals extended to May 17"): individual
    # 2020 income-tax payments due April 15, 2021 move to May 17, 2021; the
    # relief does not apply to estimated tax payments or to corporations.
    if category is TaxDeadlineCategory.INDIVIDUAL_ANNUAL and due == date(2021, 4, 15):
        return date(2021, 5, 17)
    return due


def tax_deadlines(year: int) -> tuple[TaxDeadline, ...]:
    """Return every deadline whose statutory date falls in calendar ``year``."""
    holidays = tax_legal_holidays(year) | tax_legal_holidays(year + 1)
    deadlines = [
        TaxDeadline(
            category=category,
            statutory_date=statutory,
            due_date=_national_postponement(category, next_business_day(statutory, holidays)),
        )
        for category in TaxDeadlineCategory
        for statutory in _statutory_dates(category, year)
    ]
    return tuple(sorted(deadlines, key=lambda item: (item.due_date, item.category.value, item.statutory_date)))


def tax_deadline_dates(years: Iterable[int], categories: Iterable[TaxDeadlineCategory | str]) -> set[date]:
    """Return due dates for the named categories; unknown or empty sets fail."""
    wanted = frozenset(TaxDeadlineCategory(category) for category in categories)
    if not wanted:
        raise ValueError("at least one tax deadline category is required")
    return {item.due_date for year in years for item in tax_deadlines(year) if item.category in wanted}
