from __future__ import annotations

import pandas as pd

from tdchf.channel_context import build_auction_context
from tdchf.local_sources import build_fiscal_indicator_csv, build_tic_row_indicator_csv, merge_indicator_csvs


def test_build_tic_row_indicator_csv_from_official_private(tmp_path) -> None:
    raw = tmp_path / "tic.csv"
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "official_net_purchases": [1.0],
            "private_net_purchases": [2.0],
        }
    ).to_csv(raw, index=False)

    report = build_tic_row_indicator_csv(raw, out_csv=tmp_path / "row.csv")
    out = pd.read_csv(tmp_path / "row.csv")

    assert report["source"] == "official_plus_private_net_purchases"
    assert out.loc[0, "row_tsy"] == 3.0


def test_build_fiscal_indicator_csv(tmp_path) -> None:
    raw = tmp_path / "fiscal.csv"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "operating_cash_balance": [10.0, 15.0],
            "federal_reserve_earnings": [-1.0, 2.0],
        }
    ).to_csv(raw, index=False)

    report = build_fiscal_indicator_csv(raw, out_csv=tmp_path / "fiscal_ind.csv")
    out = pd.read_csv(tmp_path / "fiscal_ind.csv")

    assert set(report["columns"]) == {"minus_toc", "fed_remit_positive"}
    assert out.loc[1, "minus_toc"] == -5.0
    assert out.loc[1, "fed_remit_positive"] == 2.0


def test_merge_indicator_csvs_last_duplicate_wins(tmp_path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame({"date": [pd.Timestamp("2024-01-31")], "fed_tsy": [1.0]}).to_csv(first, index=False)
    pd.DataFrame({"date": [pd.Timestamp("2024-01-31")], "fed_tsy": [2.0], "row_tsy": [3.0]}).to_csv(second, index=False)

    merge_indicator_csvs([first, second], out_csv=tmp_path / "merged.csv")
    out = pd.read_csv(tmp_path / "merged.csv")

    assert out.loc[0, "fed_tsy"] == 2.0
    assert out.loc[0, "row_tsy"] == 3.0


def test_build_auction_context(tmp_path) -> None:
    raw = tmp_path / "primary_allocation.csv"
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-03-31")],
            "buyer_class": ["dealers", "foreign_official"],
            "allotment_amount": [80.0, 20.0],
            "share_of_instrument": [0.8, 0.2],
            "instrument": ["all_instruments", "all_instruments"],
        }
    ).to_csv(raw, index=False)

    report = build_auction_context(out_csv=tmp_path / "auction.csv", allocation_csv=raw)
    out = pd.read_csv(tmp_path / "auction.csv")

    assert report["rows"] == 1
    assert out.loc[0, "auction_share_dealers"] == 0.8
