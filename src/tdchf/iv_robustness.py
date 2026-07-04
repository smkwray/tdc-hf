from __future__ import annotations

from pathlib import Path

import pandas as pd

from .first_stage import run_first_stage
from .indicators import read_wide_time_series_csv
from .lp import run_lp_iv


def _safe_name(value: str) -> str:
    return value.replace(",", "_").replace(" ", "_").replace("/", "_")


def weak_iv_label(first_stage_f: float) -> str:
    if pd.isna(first_stage_f):
        return "unknown"
    if first_stage_f < 10:
        return "weak"
    if first_stage_f < 20:
        return "borderline"
    return "strong"


def run_iv_robustness(
    df: pd.DataFrame,
    *,
    treatment: str,
    instrument_sets: dict[str, list[str]],
    outcomes: list[str],
    controls: list[str],
    horizons: list[int],
    cumulative: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lp_frames: list[pd.DataFrame] = []
    fs_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for spec_name, instruments in instrument_sets.items():
        first = run_first_stage(df, treatment=treatment, instruments=instruments, controls=controls)
        first.insert(0, "iv_spec", spec_name)
        fs_frames.append(first)

        lp = run_lp_iv(
            df,
            treatment_col=treatment,
            instrument_cols=instruments,
            outcome_cols=outcomes,
            controls=controls,
            horizons=horizons,
            cumulative=cumulative,
            spec_name=f"lp_iv_{spec_name}",
        )
        lp.insert(0, "iv_spec", spec_name)
        lp_frames.append(lp)

        joint = first.loc[first["excluded_instrument_f"].notna()]
        first_stage_f = float(joint.iloc[0]["excluded_instrument_f"]) if not joint.empty else float("nan")
        weak_label = weak_iv_label(first_stage_f)
        for outcome, group in lp.groupby("outcome"):
            ranked = group.assign(abs_beta=group["beta"].abs()).sort_values(["abs_beta", "horizon"], ascending=[False, True])
            peak = ranked.iloc[0]
            summary_rows.append(
                {
                    "iv_spec": spec_name,
                    "instruments": ",".join(instruments),
                    "outcome": outcome,
                    "peak_horizon": int(peak["horizon"]),
                    "peak_beta": float(peak["beta"]),
                    "peak_lower95": float(peak["lower95"]),
                    "peak_upper95": float(peak["upper95"]),
                    "peak_hac_sig_95": bool(float(peak["lower95"]) > 0 or float(peak["upper95"]) < 0),
                    "first_stage_f": first_stage_f,
                    "weak_iv_label": weak_label,
                    "weak_iv_flag": weak_label in {"weak", "borderline"},
                    "n_at_peak": int(peak["n"]),
                }
            )

    lp_all = pd.concat(lp_frames, ignore_index=True) if lp_frames else pd.DataFrame()
    fs_all = pd.concat(fs_frames, ignore_index=True) if fs_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return lp_all, fs_all, summary


def run_iv_robustness_csv(
    data_csv: str | Path,
    *,
    treatment: str,
    instrument_specs: list[str],
    outcomes: list[str],
    controls: list[str],
    horizons: list[int],
    out_dir: str | Path,
    cumulative: bool = True,
) -> dict[str, object]:
    df = read_wide_time_series_csv(data_csv)
    parsed_specs: dict[str, list[str]] = {}
    for spec in instrument_specs:
        if "=" in spec:
            name, raw_instruments = spec.split("=", 1)
            instruments = [part.strip() for part in raw_instruments.split("+") if part.strip()]
        else:
            instruments = [part.strip() for part in spec.split("+") if part.strip()]
            name = _safe_name("_".join(instruments))
        if not instruments:
            raise ValueError(f"Instrument spec has no instruments: {spec}")
        parsed_specs[name.strip()] = instruments

    lp, first_stage, summary = run_iv_robustness(
        df,
        treatment=treatment,
        instrument_sets=parsed_specs,
        outcomes=outcomes,
        controls=controls,
        horizons=horizons,
        cumulative=cumulative,
    )
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    lp_path = root / "iv_robustness_lp_iv.csv"
    fs_path = root / "iv_robustness_first_stage.csv"
    summary_path = root / "iv_robustness_summary.csv"
    lp.to_csv(lp_path, index=False)
    first_stage.to_csv(fs_path, index=False)
    summary.to_csv(summary_path, index=False)

    md_path = root / "iv_robustness.md"
    lines = [
        "# IV Robustness",
        "",
        f"- Specifications: `{', '.join(parsed_specs)}`",
        f"- Outcomes: `{', '.join(outcomes)}`",
        f"- Horizons: `{', '.join(str(h) for h in horizons)}`",
        "",
        "## Peak Responses",
        "",
    ]
    for row in summary.sort_values(["outcome", "iv_spec"]).to_dict(orient="records"):
        sig = "*" if row["peak_hac_sig_95"] else ""
        lines.append(
            f"- `{row['outcome']}` / `{row['iv_spec']}` h={int(row['peak_horizon'])}: "
            f"beta `{float(row['peak_beta']):.6g}` "
            f"[`{float(row['peak_lower95']):.6g}`, `{float(row['peak_upper95']):.6g}`], "
            f"F `{float(row['first_stage_f']):.3f}` ({row['weak_iv_label']}){sig}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": "ok",
        "out_dir": str(root),
        "lp": str(lp_path),
        "first_stage": str(fs_path),
        "summary": str(summary_path),
        "markdown": str(md_path),
        "specs": list(parsed_specs),
        "lp_rows": int(len(lp)),
        "summary_rows": int(len(summary)),
    }
