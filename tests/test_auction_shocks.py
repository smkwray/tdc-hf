from __future__ import annotations

import pandas as pd

from tdchf.auction_shocks import build_auction_size_shock, build_shock_bundle_csv, build_tga_rebuild_shock_csv


def test_build_auction_size_shock(tmp_path) -> None:
    dates = pd.date_range("2020-01-31", periods=20, freq="ME")
    raw = tmp_path / "auction.csv"
    pd.DataFrame(
        {
            "issue_date": dates,
            "offering_amt": [100.0 + i for i in range(20)],
            "security_term": ["4-Week"] * 20,
        }
    ).to_csv(raw, index=False)

    report = build_auction_size_shock(raw, out_csv=tmp_path / "auction_shock.csv", min_train_obs=6)
    out = pd.read_csv(tmp_path / "auction_shock.csv")

    assert report["status"] == "ok"
    assert "auction_size_surprise" in out.columns


def test_build_tga_rebuild_shock_csv(tmp_path) -> None:
    n = 20
    raw = tmp_path / "tga.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=n, freq="ME"),
            "minus_toc": range(n),
            "lag": range(n),
        }
    ).to_csv(raw, index=False)

    report = build_tga_rebuild_shock_csv(raw, predictors=["lag"], min_train_obs=8, out_csv=tmp_path / "tga_shock.csv")
    out = pd.read_csv(tmp_path / "tga_shock.csv")

    assert report["status"] == "ok"
    assert "tga_rebuild_surprise" in out.columns


def test_build_shock_bundle_csv(tmp_path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame({"date": [pd.Timestamp("2024-01-31")], "a": [1.0]}).to_csv(first, index=False)
    pd.DataFrame({"date": [pd.Timestamp("2024-01-31")], "b": [2.0]}).to_csv(second, index=False)

    report = build_shock_bundle_csv([first, second], out_csv=tmp_path / "bundle.csv")
    out = pd.read_csv(tmp_path / "bundle.csv")

    assert report["status"] == "ok"
    assert {"a", "b"}.issubset(out.columns)
