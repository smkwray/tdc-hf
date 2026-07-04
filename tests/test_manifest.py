from __future__ import annotations

import json

from tdchf.manifest import build_file_manifest, write_file_manifest, write_spec_manifest


def test_write_file_manifest_records_hash_and_missing(tmp_path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")

    report = write_file_manifest(
        ["source.csv", "missing.csv"],
        root=tmp_path,
        out_csv=tmp_path / "manifest.csv",
        out_json=tmp_path / "manifest.json",
    )
    manifest = build_file_manifest(["source.csv"], root=tmp_path)

    assert report["status"] == "ok"
    assert report["missing"] == ["missing.csv"]
    assert (tmp_path / "manifest.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert len(manifest.loc[0, "sha256"]) == 64


def test_file_manifest_reads_companion_retrieval_metadata(tmp_path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "source.csv.manifest.json").write_text(
        json.dumps(
            {
                "retrieved_at": "2026-04-27T09:50:24+00:00",
                "url": "https://api.fiscaldata.treasury.gov/example",
                "endpoint": "operating_cash_balance",
            }
        ),
        encoding="utf-8",
    )

    manifest = build_file_manifest(["source.csv"], root=tmp_path)

    assert manifest.loc[0, "retrieved_at"] == "2026-04-27T09:50:24+00:00"
    assert manifest.loc[0, "source_endpoint"] == "operating_cash_balance"


def test_write_spec_manifest(tmp_path) -> None:
    spec = tmp_path / "spec.yml"
    spec.write_text(
        """
root: .
steps:
  - id: first
    action: build-proxy
    out_dir: data/processed/proxy
  - id: disabled
    action: build-auction-size-shock
    enabled: false
    out: data/processed/shock.csv
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = write_spec_manifest(spec, out_csv=tmp_path / "spec_manifest.csv", out_md=tmp_path / "spec_manifest.md")

    assert report["status"] == "ok"
    assert report["rows"] == 2
    assert (tmp_path / "spec_manifest.csv").exists()
    assert (tmp_path / "spec_manifest.md").exists()
