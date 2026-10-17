from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retrieval_metadata(path: Path) -> dict[str, object]:
    companion = path.with_suffix(path.suffix + ".manifest.json")
    if not companion.exists() and path.name.endswith(".manifest.json"):
        companion = path
    if not companion.exists():
        return {"retrieved_at": "", "source_url": "", "source_endpoint": ""}
    try:
        payload = json.loads(companion.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"retrieved_at": "", "source_url": "", "source_endpoint": ""}
    return {
        "retrieved_at": payload.get("retrieved_at", ""),
        "source_url": payload.get("url", ""),
        "source_endpoint": payload.get("endpoint", ""),
    }


def build_file_manifest(paths: list[str | Path], *, root: str | Path = ".") -> pd.DataFrame:
    base = Path(root).expanduser().resolve()
    rows: list[dict[str, object]] = []
    for value in paths:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        exists = path.exists()
        try:
            rel = str(path.relative_to(base))
        except ValueError:
            rel = str(path)
        row: dict[str, object] = {
            "path": rel,
            "absolute_path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists and path.is_file() else None,
            "modified_at": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat() if exists else "",
            "sha256": _sha256(path) if exists and path.is_file() else "",
            **(_retrieval_metadata(path) if exists and path.is_file() else {"retrieved_at": "", "source_url": "", "source_endpoint": ""}),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def write_file_manifest(
    paths: list[str | Path],
    *,
    out_csv: str | Path,
    root: str | Path = ".",
    out_json: str | Path | None = None,
) -> dict[str, object]:
    manifest = build_file_manifest(paths, root=root)
    csv_path = Path(out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(csv_path, index=False)
    if out_json is not None:
        json_path = Path(out_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(manifest.to_dict(orient="records"), indent=2) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "out": str(csv_path),
        "out_json": str(out_json) if out_json else "",
        "rows": int(len(manifest)),
        "missing": manifest.loc[~manifest["exists"], "path"].tolist(),
    }


def write_run_receipt(
    out_json: str | Path,
    *,
    inputs: list[str | Path],
    outputs: list[str | Path],
    steps: list[str] | None = None,
    vintage: str | None = None,
    root: str | Path = ".",
) -> dict[str, object]:
    """Bind a rebuild to its committed producer, environment, input and output hashes.

    Every listed file must exist, the producer must have no tracked edits, and an
    existing receipt is never replaced.
    """
    path = Path(out_json)
    if path.exists():
        raise FileExistsError(f"receipt already exists: {path}")
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    tracked = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository, text=True).strip()
    if tracked:
        raise ValueError("commit the producer before writing a run receipt")
    files = {}
    for role, values in (("inputs", inputs), ("outputs", outputs)):
        frame = build_file_manifest(values, root=root)
        missing = frame.loc[~frame["exists"], "path"].tolist()
        if missing:
            raise FileNotFoundError(f"run receipt {role} missing: {missing}")
        files[role] = frame.drop(columns=["exists", "modified_at"]).to_dict(orient="records")
    receipt = {
        "schema_version": "run_receipt_v1",
        "vintage": vintage,
        "producer_commit": commit,
        "producer_tracked_changes": False,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas", "statsmodels")}},
        "steps": list(steps or []),
        **files,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return {"status": "ok", "out": str(path), "sha256": _sha256(path), "inputs": len(files["inputs"]), "outputs": len(files["outputs"])}


def _primary_outputs(step: dict[str, Any]) -> list[str]:
    outputs: list[str] = []
    for key in ["out", "out_dir", "out_json", "out_md"]:
        if step.get(key):
            outputs.append(str(step[key]))
    return outputs


def write_spec_manifest(spec_path: str | Path, *, out_csv: str | Path, out_md: str | Path | None = None) -> dict[str, object]:
    spec_file = Path(spec_path).expanduser().resolve()
    spec = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("Analysis spec must be a YAML mapping")
    rows: list[dict[str, object]] = []
    for position, raw_step in enumerate(spec.get("steps", []), start=1):
        if not isinstance(raw_step, dict):
            continue
        step = dict(raw_step)
        rows.append(
            {
                "position": position,
                "id": step.get("id", ""),
                "action": step.get("action", ""),
                "enabled": bool(step.get("enabled", True)),
                "inputs": ",".join(str(value) for value in step.get("inputs", []) if value) if isinstance(step.get("inputs"), list) else "",
                "data": step.get("data", ""),
                "raw": step.get("raw", ""),
                "proxy": step.get("proxy", ""),
                "lp": step.get("lp", ""),
                "primary_outputs": ",".join(_primary_outputs(step)),
                "parameters_json": json.dumps(
                    {key: value for key, value in step.items() if key not in {"id", "action", "inputs"}},
                    sort_keys=True,
                ),
            }
        )
    manifest = pd.DataFrame(rows)
    csv_path = Path(out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(csv_path, index=False)

    md_path = None
    if out_md is not None:
        md_path = Path(out_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Analysis Spec Manifest", "", f"- Spec: `{spec_file}`", f"- Steps: `{len(manifest)}`", "", "## Steps", ""]
        for row in manifest.to_dict(orient="records"):
            status = "enabled" if row["enabled"] else "disabled"
            outputs = row["primary_outputs"] or ""
            lines.append(f"- `{row['position']}` `{row['id']}` `{row['action']}` ({status}) -> `{outputs}`")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": "ok",
        "out": str(csv_path),
        "out_md": str(md_path) if md_path else "",
        "rows": int(len(manifest)),
    }
