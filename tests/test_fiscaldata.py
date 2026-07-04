from __future__ import annotations

import pandas as pd

from tdchf import fiscaldata
from tdchf.fiscaldata import build_dts_fiscal_indicators_csv, build_dts_transaction_indicators_csv, iter_fiscaldata_rows, write_fiscaldata_csv


def test_iter_fiscaldata_rows_paginates(monkeypatch) -> None:
    seen: list[str] = []

    def fake_fetch(url: str) -> dict[str, object]:
        seen.append(url)
        if "page%5Bnumber%5D=1" in url:
            return {"data": [{"record_date": "2024-01-01"}], "meta": {"total-pages": 2}}
        return {"data": [{"record_date": "2024-01-02"}], "meta": {"total-pages": 2}}

    monkeypatch.setattr(fiscaldata, "fetch_fiscaldata_page", fake_fetch)

    rows, meta = iter_fiscaldata_rows("operating_cash_balance", filters=["record_date:gte:2024-01-01"])

    assert [row["record_date"] for row in rows] == ["2024-01-01", "2024-01-02"]
    assert meta["pages_downloaded"] == 2
    assert len(seen) == 2


def test_write_fiscaldata_csv_writes_manifest(monkeypatch, tmp_path) -> None:
    def fake_iter(*args, **kwargs):  # noqa: ANN002, ANN003
        return [{"record_date": "2024-01-01", "value": "1"}], {"first_url": "https://example.test/api", "total-pages": 1}

    monkeypatch.setattr(fiscaldata, "iter_fiscaldata_rows", fake_iter)

    report = write_fiscaldata_csv(
        "operating_cash_balance",
        out_csv=tmp_path / "raw.csv",
        fields=["record_date", "value"],
        manifest_json=tmp_path / "raw.manifest.json",
    )

    raw = pd.read_csv(tmp_path / "raw.csv")
    manifest = (tmp_path / "raw.manifest.json").read_text(encoding="utf-8")

    assert report["rows"] == 1
    assert raw.loc[0, "value"] == 1
    assert "retrieved_at" in manifest
    assert "https://example.test/api" in manifest


def test_build_dts_fiscal_indicators_csv(tmp_path) -> None:
    ocb = tmp_path / "ocb.csv"
    pd.DataFrame(
        {
            "record_date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-01-31", "2024-02-29", "2024-01-31", "2024-02-29"]),
            "account_type": [
                "Treasury General Account (TGA) Closing Balance",
                "Treasury General Account (TGA) Closing Balance",
                "Total TGA Deposits (Table II)",
                "Total TGA Deposits (Table II)",
                "Total TGA Withdrawals (Table II) (-)",
                "Total TGA Withdrawals (Table II) (-)",
            ],
            "open_today_bal": ["10", "15", "100", "120", "150", "80"],
            "close_today_bal": ["null", "null", "null", "null", "null", "null"],
        }
    ).to_csv(ocb, index=False)

    remit = tmp_path / "remit.csv"
    pd.DataFrame(
        {
            "record_date": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-02-01"]),
            "transaction_catg": ["Federal Reserve Earnings", "Federal Reserve Earnings", "Federal Reserve Earnings"],
            "transaction_today_amt": ["1", "2", "(3)"],
        }
    ).to_csv(remit, index=False)

    report = build_dts_fiscal_indicators_csv(
        operating_cash_balance_csv=ocb,
        fed_remit_csv=remit,
        out_csv=tmp_path / "indicators.csv",
        metadata_csv=tmp_path / "metadata.csv",
    )
    out = pd.read_csv(tmp_path / "indicators.csv")
    metadata = pd.read_csv(tmp_path / "metadata.csv")

    assert set(report["columns"]) == {
        "minus_toc",
        "tga_deposits",
        "tga_withdrawals",
        "net_tga_withdrawals",
        "fed_remit_positive",
    }
    assert out.loc[1, "minus_toc"] == -5.0
    assert out.loc[0, "net_tga_withdrawals"] == 50.0
    assert out.loc[1, "net_tga_withdrawals"] == -40.0
    assert out.loc[0, "fed_remit_positive"] == 3.0
    assert out.loc[1, "fed_remit_positive"] == 0.0
    assert set(metadata["component"]) == {
        "minus_toc",
        "tga_deposits",
        "tga_withdrawals",
        "net_tga_withdrawals",
        "fed_remit_positive",
    }


def test_build_dts_transaction_indicators_csv(tmp_path) -> None:
    raw = tmp_path / "transactions.csv"
    pd.DataFrame(
        {
            "record_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]),
            "transaction_type": ["Deposits", "Withdrawals", "Deposits", "Withdrawals", "Withdrawals"],
            "transaction_catg": [
                "Taxes - Individual Income and Employment Taxes, Not Withheld",
                "Interest on Treasury Securities",
                "Federal Reserve Earnings",
                "Federal Salaries (EFT)",
                "HHS - Federal Hospital Insr Trust Fund",
            ],
            "transaction_today_amt": ["100", "30", "4", "8", "12"],
        }
    ).to_csv(raw, index=False)

    report = build_dts_transaction_indicators_csv(
        raw,
        out_csv=tmp_path / "transaction_indicators.csv",
        metadata_csv=tmp_path / "transaction_metadata.csv",
    )
    out = pd.read_csv(tmp_path / "transaction_indicators.csv")
    metadata = pd.read_csv(tmp_path / "transaction_metadata.csv")

    assert report["status"] == "ok"
    assert out.loc[0, "dts_total_deposits"] == 104.0
    assert out.loc[0, "dts_total_withdrawals"] == 50.0
    assert out.loc[0, "dts_tax_deposits"] == 100.0
    assert out.loc[0, "dts_interest_withdrawals"] == 30.0
    assert out.loc[0, "dts_core_payment_withdrawals"] == 50.0
    assert out.loc[0, "dts_net_withdrawals"] == -54.0
    assert "dts_tax_deposits" in set(metadata["component"])
