from __future__ import annotations

from dataclasses import asdict, dataclass

from .upstream import resolve_repos


@dataclass(frozen=True)
class ReadinessItem:
    repo: str
    item: str
    path: str
    exists: bool
    role: str


def upstream_readiness() -> list[ReadinessItem]:
    rows: list[ReadinessItem] = []
    for key, repo in resolve_repos().items():
        rows.append(ReadinessItem(key, "repo_root", str(repo.root), repo.root.exists(), repo.role))
        for item, path in repo.preferred_outputs.items():
            rows.append(ReadinessItem(key, item, str(path), path.exists(), repo.role))
    return rows


def readiness_payload() -> dict[str, object]:
    rows = upstream_readiness()
    return {
        "status": "ok" if all(row.exists for row in rows) else "missing",
        "items": [asdict(row) for row in rows],
    }
