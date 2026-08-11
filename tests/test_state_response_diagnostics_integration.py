from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tdchf.qra_event_lp import normalize_weekly_panel_units
from tdchf.state_response_diagnostics import (
    _calibrate_random_walk_placebo,
    _count_stability_failures,
    _estimate_state_interacted_response_diagnostic,
    build_state_response_weekly,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DATA = {
    "flows": PROJECT_ROOT / "data/processed/dts_weekly_flow_decomposition.csv",
    "calendar": PROJECT_ROOT / "data/processed/fiscal_calendar_weekly.csv",
    "weekly_panel": PROJECT_ROOT / "data/processed/tdc_weekly_channel_panel.csv",
    "state_sources": PROJECT_ROOT / "data/raw/fred_state_response_sources.csv",
}
pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in REQUIRED_DATA.values()),
    reason="requires project-local generated flow panels and raw state-source data",
)


def test_real_state_and_placebo_failures_reproduce(tmp_path) -> None:
    flows = pd.read_csv(
        REQUIRED_DATA["flows"],
        parse_dates=["date"],
    ).set_index("date")
    calendar = pd.read_csv(REQUIRED_DATA["calendar"], parse_dates=["date"])
    weekly_panel = normalize_weekly_panel_units(
        pd.read_csv(
            REQUIRED_DATA["weekly_panel"],
            parse_dates=["date"],
        ).set_index("date")
    )
    state = build_state_response_weekly(
        raw_state_csv=REQUIRED_DATA["state_sources"],
        out_csv=tmp_path / "state_response_weekly.csv",
        start=flows.index.min(),
        end=flows.index.max(),
    )
    real = _estimate_state_interacted_response_diagnostic(
        flows,
        calendar,
        weekly_panel,
        state,
    )
    calibration = _calibrate_random_walk_placebo(
        flows,
        calendar,
        weekly_panel,
        state,
        real,
    ).set_index("horizon")

    assert _count_stability_failures(real) == 2
    assert calibration.loc[4, "placebo_seed_count"] == 20
    assert calibration.loc[8, "placebo_seed_count"] == 20
    assert abs(calibration.loc[4, "placebo_false_positive_rate"] - 0.45) < 1e-9
    assert abs(calibration.loc[8, "placebo_false_positive_rate"] - 0.40) < 1e-9
    assert abs(calibration.loc[4, "placebo_effective_p"] - 0.25) < 1e-9
    assert abs(calibration.loc[8, "placebo_effective_p"] - 0.05) < 1e-9
