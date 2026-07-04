from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

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


def test_qra_borrowing_output_schema_and_arithmetic() -> None:
    path = Path("data/processed/qra_borrowing_surprise.csv")
    if not path.exists():
        pytest.skip("generated QRA borrowing surprise CSV is not present")

    df = pd.read_csv(path)
    assert list(df.columns) == QRA_REQUIRED_COLUMNS
    assert not df.duplicated(["event_id", "quarter"]).any()
    assert df["release_date"].is_monotonic_increasing

    complete = df.dropna(subset=["announced_net_borrowing_bn", "prior_estimate_bn", "surprise_bn"])
    diff = complete["announced_net_borrowing_bn"] - complete["prior_estimate_bn"]
    assert (diff - complete["surprise_bn"]).abs().max() < 1e-9
