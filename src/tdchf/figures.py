from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .indicators import read_wide_time_series_csv


def _save_line_plot(frame: pd.DataFrame, *, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    frame.plot(ax=ax, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frame.columns, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_lp_plot(lp: pd.DataFrame, *, outcome: str, path: Path) -> None:
    data = lp.loc[lp["outcome"] == outcome].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(data["horizon"], data["beta"], marker="o", linewidth=1.8, label="beta")
    ax.fill_between(data["horizon"], data["lower95"], data["upper95"], alpha=0.2, label="95% CI")
    ax.set_title(f"LP-IV response: {outcome}")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Cumulative response")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_thesis_figures(
    *,
    proxy_csv: str | Path,
    out_dir: str | Path,
    components_csv: str | Path | None = None,
    envelope_csv: str | Path | None = None,
    lp_csv: str | Path | None = None,
) -> dict[str, object]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    proxy = read_wide_time_series_csv(proxy_csv)
    proxy_path = out_root / "monthly_tdc_proxy.png"
    _save_line_plot(proxy[["tdc_monthly"]], path=proxy_path, title="Monthly TDC Proxy", ylabel="Monthly flow")
    outputs["monthly_tdc_proxy"] = str(proxy_path)

    if components_csv is not None and Path(components_csv).exists():
        components = read_wide_time_series_csv(components_csv)
        cols = [col for col in ["fed_tsy", "banks_tsy", "row_tsy", "minus_toc"] if col in components.columns]
        if cols:
            component_path = out_root / "monthly_component_contributions.png"
            _save_line_plot(components[cols], path=component_path, title="Monthly TDC Components", ylabel="Monthly flow")
            outputs["monthly_component_contributions"] = str(component_path)

    if envelope_csv is not None and Path(envelope_csv).exists():
        envelope = read_wide_time_series_csv(envelope_csv)
        cols = [col for col in ["denton", "residual_spread"] if col in envelope.columns]
        if cols:
            envelope_path = out_root / "proxy_method_envelope.png"
            _save_line_plot(envelope[cols], path=envelope_path, title="Monthly Proxy: Denton vs Residual-Spread", ylabel="Monthly flow")
            outputs["proxy_method_envelope"] = str(envelope_path)

    if lp_csv is not None and Path(lp_csv).exists():
        lp = pd.read_csv(lp_csv)
        for outcome in sorted(lp["outcome"].dropna().unique()):
            safe = str(outcome).replace("/", "_").replace(" ", "_")
            path = out_root / f"lp_iv_{safe}.png"
            _save_lp_plot(lp, outcome=str(outcome), path=path)
            outputs[f"lp_iv_{safe}"] = str(path)

    index_path = out_root / "figure_index.md"
    lines = ["# Figure Index", ""]
    for name, path in sorted(outputs.items()):
        lines.append(f"- `{name}`: `{path}`")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs["figure_index"] = str(index_path)
    return {"status": "ok", "out_dir": str(out_root), "figures": outputs, "count": len(outputs) - 1}
