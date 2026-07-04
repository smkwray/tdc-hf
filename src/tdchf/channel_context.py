from __future__ import annotations

from pathlib import Path

import pandas as pd

from .upstream import resolve_repos


def build_auction_context(
    *,
    out_csv: str | Path,
    allocation_csv: str | Path | None = None,
) -> dict[str, object]:
    if allocation_csv is None:
        allocation_csv = resolve_repos()["tsyparty"].preferred_outputs["primary_allocation"]
    df = pd.read_csv(allocation_csv, parse_dates=["date"])
    required = {"date", "buyer_class", "allotment_amount", "share_of_instrument", "instrument"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing auction allocation columns: {sorted(missing)}")
    context = df[df["instrument"] == "all_instruments"].copy()
    wide = context.pivot_table(index="date", columns="buyer_class", values="share_of_instrument", aggfunc="mean")
    wide.columns = [f"auction_share_{col}" for col in wide.columns]
    wide = wide.sort_index()
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(path, index_label="date")
    return {"status": "ok", "out": str(path), "rows": int(len(wide)), "columns": list(wide.columns)}
