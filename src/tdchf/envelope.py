from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pipeline import run_monthly_proxy_pipeline


def build_method_envelope(
    *,
    out_dir: str | Path = "data/processed/envelope",
    monthly_indicators_path: str | Path | None = None,
    quarterly_anchors_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(out_dir)
    methods = ["residual_spread", "denton"]
    reports: dict[str, object] = {}
    proxy_frames: list[pd.DataFrame] = []

    for method in methods:
        method_dir = root / method
        reports[method] = run_monthly_proxy_pipeline(
            out_dir=method_dir,
            monthly_indicators_path=monthly_indicators_path,
            quarterly_anchors_path=quarterly_anchors_path,
            benchmark_method=method,
            method_label=f"envelope_{method}",
        )
        proxy = pd.read_csv(method_dir / "tdc_monthly_proxy.csv", parse_dates=["date"]).set_index("date")
        proxy_frames.append(proxy.rename(columns={"tdc_monthly": method}))

    combined = pd.concat(proxy_frames, axis=1, sort=False).sort_index()
    combined["min"] = combined[methods].min(axis=1)
    combined["max"] = combined[methods].max(axis=1)
    combined["spread"] = combined["max"] - combined["min"]
    root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(root / "tdc_monthly_method_envelope.csv", index_label="date")
    return {
        "status": "ok",
        "methods": methods,
        "rows": int(len(combined)),
        "outputs": {"envelope": str(root / "tdc_monthly_method_envelope.csv")},
        "reports": reports,
    }
