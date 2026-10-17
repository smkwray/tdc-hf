from pathlib import Path

import pytest


@pytest.fixture
def quarterly_anchors_path(tmp_path: Path) -> Path:
    path = tmp_path / "quarterly_anchors.csv"
    path.write_text(
        "date,fed_tsy_tx,bank_depository_tsy_tx,row_tsy_tx,"
        "minus_treasury_operating_cash_tx,fed_remit_positive\n"
        "2024-06-30,36,21,10,2,0\n"
        "2024-03-31,33,18,12,-9,3\n",
        encoding="utf-8",
    )
    return path
