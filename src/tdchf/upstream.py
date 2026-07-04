from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class UpstreamRepo:
    key: str
    root: Path
    role: str
    preferred_outputs: dict[str, Path]


def load_upstream_config(path: str | Path = "config/upstream_sources.yml") -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded


def resolve_repos(
    config_path: str | Path = "config/upstream_sources.yml",
    *,
    base_dir: str | Path | None = None,
) -> dict[str, UpstreamRepo]:
    config_file = Path(config_path)
    if base_dir is None:
        base = config_file.resolve().parent.parent
    else:
        base = Path(base_dir).resolve()
    config = load_upstream_config(config_file)

    repos: dict[str, UpstreamRepo] = {}
    for key, spec in (config.get("repos") or {}).items():
        root = (base / spec["root"]).resolve()
        outputs = {
            output_key: (root / output_path).resolve()
            for output_key, output_path in (spec.get("preferred_outputs") or {}).items()
        }
        repos[key] = UpstreamRepo(
            key=key,
            root=root,
            role=str(spec.get("role", "")),
            preferred_outputs=outputs,
        )
    return repos


def load_tdcest_quarterly_anchors(
    path: str | Path | None = None,
    *,
    config_path: str | Path = "config/upstream_sources.yml",
) -> dict[str, pd.Series]:
    """Load canonical quarterly component anchors from tdcest outputs."""
    if path is None:
        repos = resolve_repos(config_path)
        path = repos["tdcest"].preferred_outputs["components"]

    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    mapping = {
        "fed_tsy": "fed_tsy_tx",
        "banks_tsy": "bank_depository_tsy_tx",
        "row_tsy": "row_tsy_tx",
        "minus_toc": "minus_treasury_operating_cash_tx",
        "fed_remit_positive": "fed_remit_positive",
    }
    missing = [source for source in mapping.values() if source not in df.columns]
    if missing:
        raise KeyError(f"Missing required tdcest component columns: {missing}")
    return {
        component: pd.to_numeric(df[source], errors="coerce").rename(component)
        for component, source in mapping.items()
    }
