from __future__ import annotations

from tdchf.upstream import load_tdcest_quarterly_anchors, resolve_repos


def test_resolve_repos_finds_tdcest_contract() -> None:
    repos = resolve_repos()

    assert "tdcest" in repos
    assert repos["tdcest"].root.name == "tdcest"
    assert "components" in repos["tdcest"].preferred_outputs


def test_load_tdcest_quarterly_anchors_has_expected_components() -> None:
    anchors = load_tdcest_quarterly_anchors()

    assert set(anchors) == {"fed_tsy", "banks_tsy", "row_tsy", "minus_toc", "fed_remit_positive"}
    assert anchors["fed_tsy"].index.is_monotonic_increasing
