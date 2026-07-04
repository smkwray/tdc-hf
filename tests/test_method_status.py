from __future__ import annotations

from tdchf.method_status import temporal_disaggregation_method_status, write_method_status_csv


def test_method_status_marks_chow_lin_litterman_not_headline(tmp_path) -> None:
    status = temporal_disaggregation_method_status()

    assert status.loc[status["method"].eq("additive_denton"), "headline"].item() is True
    assert set(status.loc[status["status"].eq("stub_not_headline"), "method"]) == {"chow_lin", "litterman"}

    report = write_method_status_csv(tmp_path / "methods.csv")

    assert report["headline_method"] == "additive_denton"
    assert (tmp_path / "methods.csv").exists()
