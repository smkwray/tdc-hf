from __future__ import annotations

from tdchf.envelope import build_method_envelope


def test_build_method_envelope(tmp_path, quarterly_anchors_path) -> None:
    report = build_method_envelope(out_dir=tmp_path, quarterly_anchors_path=quarterly_anchors_path)

    assert report["status"] == "ok"
    assert (tmp_path / "tdc_monthly_method_envelope.csv").exists()
