from __future__ import annotations

from pathlib import Path

import pandas as pd

from tdchf.qra_borrowing import QRA_REQUIRED_COLUMNS, parse_borrowing_release


FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_borrowing_release_higher_revision() -> None:
    path = FIXTURES / "qra_borrowing_release.html"
    parsed = parse_borrowing_release(path.read_text(encoding="utf-8"), source_path=path)

    assert parsed.quarter == "2026Q2"
    assert parsed.release_date == "2026-05-04"
    assert parsed.announced_net_borrowing_bn == 189.0
    assert parsed.prior_estimate_bn == 110.0
    assert parsed.surprise_bn == 79.0
    assert parsed.tga_assumption_announced_bn == 900.0


def test_parse_paydown_release_signs_net_borrowing_surprise() -> None:
    path = FIXTURES / "qra_paydown_release.txt"
    parsed = parse_borrowing_release(path.read_text(encoding="utf-8"), source_path=path)

    assert parsed.quarter == "2014Q2"
    assert parsed.announced_net_borrowing_bn == -78.0
    assert parsed.prior_estimate_bn == -40.0
    assert parsed.surprise_bn == -38.0


def test_parse_trillion_amount_rounds_at_parse_boundary() -> None:
    parsed = parse_borrowing_release(
        "July 31, 2023. During the July - September 2023 quarter, Treasury "
        "expects to borrow $1.007 trillion in privately-held net marketable debt, "
        "assuming an end-of-September cash balance of $650 billion. "
        "This borrowing estimate is $274 billion higher than announced in May.",
    )

    assert parsed.quarter == "2023Q3"
    assert parsed.announced_net_borrowing_bn == 1007.0
    assert parsed.prior_estimate_bn == 733.0


def test_qra_borrowing_output_schema_and_arithmetic() -> None:
    path = Path("data/processed/qra_borrowing_surprise.csv")
    assert path.exists(), "generated QRA borrowing surprise CSV is required for anti-drift coverage"

    df = pd.read_csv(path)
    assert list(df.columns) == QRA_REQUIRED_COLUMNS
    assert not df.duplicated(["event_id", "quarter"]).any()
    assert df["release_date"].is_monotonic_increasing

    parsed = df.loc[df["prior_source"].eq("parsed_prior_release")]
    assert not parsed.empty
    diff = parsed["announced_net_borrowing_bn"] - parsed["prior_estimate_bn"]
    assert (diff - parsed["surprise_bn"]).abs().max() <= 1.5
    assert "verified" not in set(df["parse_quality"].dropna())


def test_qra_borrowing_output_pins_independently_verified_values() -> None:
    path = Path("data/processed/qra_borrowing_surprise.csv")
    assert path.exists(), "generated QRA borrowing surprise CSV is required for anti-drift pins"

    df = pd.read_csv(path).set_index("quarter")
    expected = {
        "2011Q1": -194.0,
        "2014Q3": 22.0,
        "2016Q1": 85.0,
        "2019Q3": 274.0,
        "2020Q2": 3055.0,
        "2021Q2": 368.0,
        "2023Q3": 274.0,
        "2024Q3": -106.0,
        "2025Q2": 391.0,
    }
    for quarter, surprise in expected.items():
        assert quarter in df.index
        assert abs(float(df.loc[quarter, "surprise_bn"]) - surprise) <= 1.0
