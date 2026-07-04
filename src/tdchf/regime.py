from __future__ import annotations

from pathlib import Path

import pandas as pd

from .calendar_controls import DEBT_CEILING_WINDOWS
from .first_stage import run_first_stage
from .indicators import read_wide_time_series_csv
from .lp import run_lp_iv


REGIME_EXCLUSIONS = {
    "exclude_gfc": [("2008-09-01", "2009-06-30")],
    "exclude_covid": [("2020-03-01", "2020-12-31")],
    "exclude_2023_bank_stress": [("2023-03-01", "2023-05-31")],
    "exclude_debt_ceiling": DEBT_CEILING_WINDOWS,
}


def _drop_windows(df: pd.DataFrame, windows: list[tuple[str, str]]) -> pd.DataFrame:
    out = df.copy()
    index = pd.DatetimeIndex(out.index)
    mask = pd.Series(False, index=out.index)
    for start, end in windows:
        start_m = pd.Timestamp(start).to_period("M").to_timestamp("M")
        end_m = pd.Timestamp(end).to_period("M").to_timestamp("M")
        mask |= (index >= start_m) & (index <= end_m)
    return out.loc[~mask].copy()


def run_regime_exclusion_robustness(
    df: pd.DataFrame,
    *,
    treatment: str,
    instruments: list[str],
    outcomes: list[str],
    controls: list[str],
    horizons: list[int],
    cumulative: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lp_frames: list[pd.DataFrame] = []
    fs_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    samples = {"full_sample": df, **{name: _drop_windows(df, windows) for name, windows in REGIME_EXCLUSIONS.items()}}
    for sample_name, sample in samples.items():
        first = run_first_stage(sample, treatment=treatment, instruments=instruments, controls=controls)
        first.insert(0, "sample", sample_name)
        fs_frames.append(first)

        lp = run_lp_iv(
            sample,
            treatment_col=treatment,
            instrument_cols=instruments,
            outcome_cols=outcomes,
            controls=controls,
            horizons=horizons,
            cumulative=cumulative,
            spec_name=f"lp_iv_{sample_name}",
        )
        lp.insert(0, "sample", sample_name)
        lp_frames.append(lp)

        joint = first.loc[first["excluded_instrument_f"].notna()]
        first_stage_f = float(joint.iloc[0]["excluded_instrument_f"]) if not joint.empty else float("nan")
        for outcome, group in lp.groupby("outcome"):
            h12 = group.loc[group["horizon"] == max(horizons)]
            row = h12.iloc[0] if not h12.empty else group.sort_values("horizon").iloc[-1]
            summary_rows.append(
                {
                    "sample": sample_name,
                    "outcome": outcome,
                    "reported_horizon": int(row["horizon"]),
                    "same_unit_beta": float(row.get("same_unit_beta", row["beta"])),
                    "same_unit_lower95": float(row.get("same_unit_lower95", row["lower95"])),
                    "same_unit_upper95": float(row.get("same_unit_upper95", row["upper95"])),
                    "n": int(row["n"]),
                    "first_stage_f": first_stage_f,
                    "hac_sig_95": bool(float(row["lower95"]) > 0 or float(row["upper95"]) < 0),
                }
            )

    return (
        pd.concat(lp_frames, ignore_index=True),
        pd.concat(fs_frames, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def run_regime_exclusion_robustness_csv(
    data_csv: str | Path,
    *,
    treatment: str,
    instruments: list[str],
    outcomes: list[str],
    controls: list[str],
    horizons: list[int],
    out_dir: str | Path,
    cumulative: bool = True,
) -> dict[str, object]:
    df = read_wide_time_series_csv(data_csv)
    lp, first_stage, summary = run_regime_exclusion_robustness(
        df,
        treatment=treatment,
        instruments=instruments,
        outcomes=outcomes,
        controls=controls,
        horizons=horizons,
        cumulative=cumulative,
    )
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    lp_path = root / "regime_exclusion_lp_iv.csv"
    fs_path = root / "regime_exclusion_first_stage.csv"
    summary_path = root / "regime_exclusion_summary.csv"
    lp.to_csv(lp_path, index=False)
    first_stage.to_csv(fs_path, index=False)
    summary.to_csv(summary_path, index=False)

    md_path = root / "regime_exclusion.md"
    lines = ["# Regime-Exclusion Robustness", "", "## Reported Horizon", ""]
    for row in summary.sort_values(["outcome", "sample"]).to_dict(orient="records"):
        lines.append(
            f"- `{row['outcome']}` / `{row['sample']}` h={int(row['reported_horizon'])}: "
            f"same-unit `{float(row['same_unit_beta']):.3g}` "
            f"[`{float(row['same_unit_lower95']):.3g}`, `{float(row['same_unit_upper95']):.3g}`], "
            f"F `{float(row['first_stage_f']):.3g}`, n `{int(row['n'])}`"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "out_dir": str(root),
        "lp": str(lp_path),
        "first_stage": str(fs_path),
        "summary": str(summary_path),
        "markdown": str(md_path),
        "rows": int(len(summary)),
    }
