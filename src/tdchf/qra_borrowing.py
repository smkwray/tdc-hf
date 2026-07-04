from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import pandas as pd


QRA_REQUIRED_COLUMNS = [
    "event_id",
    "quarter",
    "release_date",
    "announced_net_borrowing_bn",
    "prior_estimate_bn",
    "prior_release_date",
    "prior_source",
    "surprise_bn",
    "tga_assumption_announced_bn",
    "tga_assumption_prior_bn",
    "source_url",
    "source_kind",
    "parse_quality",
    "note",
]

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

MONTH_TO_QUARTER = {
    "march": "Q1",
    "june": "Q2",
    "september": "Q3",
    "december": "Q4",
}

QRAWATCH_FALLBACK_DOCS = {
    "qra_2015_08": "data/interim/qra_text/Financing_20Estimates_20August_202015_dd38a9657f.txt",
    "qra_2017_02": "data/raw/qra/files/jl0718_63556c5b3b.html",
    "qra_2017_08": "data/raw/qra/files/sm0138_cb0a9efeb1.html",
}

BUILTIN_FETCH_URLS = [
    # qrawatch registry gap: no current-quarter refunding event between 2021-08 and 2022-05.
    "https://home.treasury.gov/news/press-releases/jy0452",
    "https://home.treasury.gov/news/press-releases/jy0575",
    # Post-registry rows from Work Order 1.
    "https://home.treasury.gov/news/press-releases/sb0300",
    "https://home.treasury.gov/news/press-releases/sb0377",
    "https://home.treasury.gov/news/press-releases/sb0485",
]

MANUALLY_VERIFIED_QUARTERS = {
    "2011Q1",
    "2014Q3",
    "2016Q1",
    "2019Q3",
    "2020Q2",
    "2021Q2",
    "2021Q4",
    "2022Q1",
    "2023Q3",
    "2024Q3",
    "2025Q2",
    "2026Q1",
    "2026Q2",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self.parts)


@dataclass(frozen=True)
class BorrowingParse:
    quarter: str
    release_date: str
    announced_net_borrowing_bn: float
    prior_estimate_bn: float | None
    prior_release_date: str | None
    surprise_bn: float | None
    tga_assumption_announced_bn: float | None
    tga_assumption_prior_bn: float | None
    evidence_sentence: str
    parse_quality: str
    note: str = ""


@dataclass(frozen=True)
class EstimateParse:
    quarter: str
    amount_bn: float
    tga_assumption_bn: float | None
    evidence_sentence: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_text(text: str) -> str:
    text = unescape(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return _normalize_text(parser.text())


def document_to_text(payload: str, *, source_path: str | Path | None = None) -> str:
    suffix = Path(source_path).suffix.lower() if source_path is not None else ""
    if suffix in {".html", ".htm"} or "<html" in payload[:500].lower():
        return html_to_text(payload)
    return _normalize_text(payload)


def _money_to_bn(value: str, unit: str | None) -> float:
    number = float(value.replace(",", ""))
    if unit and unit.lower().startswith("trillion"):
        number *= 1000.0
    return number


def _quarter_from_end_month(end_month: str, year: str) -> str:
    q = MONTH_TO_QUARTER[end_month.lower()]
    return f"{year}{q}"


def _year_from_period(period: str, fallback_year: int) -> int:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", period)
    return int(years[-1]) if years else fallback_year


def _date_from_month_day_year(month: str, day: str, year: str) -> str:
    return f"{int(year):04d}-{MONTHS[month.lower()]:02d}-{int(day):02d}"


_DATE_PATTERNS = [
    re.compile(r"\b([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})\b"),
    re.compile(r"\bFOR IMMEDIATE RELEASE:\s*([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})\b", re.I),
]


def parse_release_date(text: str) -> str | None:
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return _date_from_month_day_year(*match.groups())
    return None


def parse_html_publication_date(payload: str) -> str | None:
    match = re.search(
        r"field--name-field-news-publication-date.*?<time[^>]+datetime=\"(\d{4}-\d{2}-\d{2})",
        payload,
        re.I | re.S,
    )
    if match:
        return match.group(1)
    match = re.search(r"<meta\s+property=\"og:updated_time\"\s+content=\"(\d{4}-\d{2}-\d{2})\"", payload, re.I)
    if match:
        return match.group(1)
    return None


_ESTIMATE_RE = re.compile(
    r"During the (?P<period>[^.]{0,120}?) quarter, Treasury "
    r"(?:expects to|estimated|will) (?P<verb>borrow|issue|pay down|paydown|borrowed|issued) "
    r"\$(?P<amount>[\d,.]+)\s*(?P<unit>trillion|billion)?"
    r"(?P<body>[^.]{0,260}?(?:net [^.]{0,120}?marketable debt|marketable debt)[^.]{0,260}?)"
    r"assuming an end-of-(?P<end_month>[A-Za-z]+) cash balance of "
    r"\$(?P<tga>[\d,.]+)\s*(?P<tga_unit>trillion|billion)?",
    re.I,
)

_REVISION_RE = re.compile(
    r"(?:This|The)\s+(?P<label>borrowing|pay down|paydown)\s+estimate\s+is\s+"
    r"\$(?P<amount>[\d,.]+)\s*(?P<unit>trillion|billion)?\s+"
    r"(?P<direction>higher|lower|larger|smaller)\s+than announced in\s+"
    r"(?P<month>[A-Za-z]+)(?:\s+(?P<year>\d{4}))?",
    re.I,
)


def _sentence_around(text: str, start: int, end: int) -> str:
    left = text.rfind(".", 0, start)
    right = text.find(".", end)
    if left == -1:
        left = max(0, start - 300)
    else:
        left += 1
    if right == -1:
        right = min(len(text), end + 300)
    else:
        right += 1
    return text[left:right].strip()


def _signed_announced(verb: str, amount_bn: float) -> float:
    return -amount_bn if "pay" in verb.lower() else amount_bn


def _signed_surprise(label: str, direction: str, amount_bn: float) -> float:
    sign = 1.0 if direction.lower() in {"higher", "larger"} else -1.0
    if "pay" in label.lower():
        sign *= -1.0
    return sign * amount_bn


def _prior_release_date(month: str, year: str | None, release_year: int, release_month: int | None) -> str:
    month_num = MONTHS[month.lower()]
    resolved_year = int(year) if year is not None else release_year
    if year is None and release_month is not None and month_num > release_month:
        resolved_year -= 1
    return f"{resolved_year:04d}-{month_num:02d}"


def _extract_prior_tga_from_cash_table(text: str) -> float | None:
    idx = text.lower().find("closing balance")
    if idx == -1:
        return None
    window = text[idx : idx + 500]
    amounts = re.findall(r"\$?\s*(-?[\d,]+)", window)
    if len(amounts) < 6:
        return None
    try:
        value = float(amounts[3].replace(",", ""))
    except ValueError:
        return None
    return value if abs(value) < 1500 else None


def parse_borrowing_release(text: str, *, source_path: str | Path | None = None) -> BorrowingParse:
    html_date = parse_html_publication_date(text)
    normalized = document_to_text(text, source_path=source_path)
    estimate = _ESTIMATE_RE.search(normalized)
    if estimate is None:
        return BorrowingParse(
            quarter="",
            release_date=html_date or parse_release_date(normalized) or "",
            announced_net_borrowing_bn=float("nan"),
            prior_estimate_bn=None,
            prior_release_date=None,
            surprise_bn=None,
            tga_assumption_announced_bn=None,
            tga_assumption_prior_bn=None,
            evidence_sentence="",
            parse_quality="missing",
            note="current-quarter borrowing estimate not parsed",
        )

    release_date = html_date or parse_release_date(normalized) or ""
    release_year = int(release_date[:4]) if release_date else int(re.search(r"\b(20\d{2}|19\d{2})\b", normalized).group(1))
    announced = _signed_announced(estimate.group("verb"), _money_to_bn(estimate.group("amount"), estimate.group("unit")))
    quarter_year = _year_from_period(estimate.group("period"), release_year)
    quarter = _quarter_from_end_month(estimate.group("end_month"), str(quarter_year))
    tga_current = _money_to_bn(estimate.group("tga"), estimate.group("tga_unit"))

    search_start = estimate.end()
    revision = _REVISION_RE.search(normalized, search_start, search_start + 700)
    if revision is None:
        return BorrowingParse(
            quarter=quarter,
            release_date=release_date,
            announced_net_borrowing_bn=announced,
            prior_estimate_bn=None,
            prior_release_date=None,
            surprise_bn=None,
            tga_assumption_announced_bn=tga_current,
            tga_assumption_prior_bn=_extract_prior_tga_from_cash_table(normalized),
            evidence_sentence=_sentence_around(normalized, estimate.start(), estimate.end()),
            parse_quality="parsed_unverified",
            note="revision sentence not parsed",
        )

    surprise = _signed_surprise(
        revision.group("label"),
        revision.group("direction"),
        _money_to_bn(revision.group("amount"), revision.group("unit")),
    )
    prior = announced - surprise
    evidence = _sentence_around(normalized, estimate.start(), revision.end())
    release_month = int(release_date[5:7]) if release_date else None
    prior_month = _prior_release_date(revision.group("month"), revision.group("year"), release_year, release_month)
    return BorrowingParse(
        quarter=quarter,
        release_date=release_date,
        announced_net_borrowing_bn=announced,
        prior_estimate_bn=prior,
        prior_release_date=prior_month,
        surprise_bn=surprise,
        tga_assumption_announced_bn=tga_current,
        tga_assumption_prior_bn=_extract_prior_tga_from_cash_table(normalized),
        evidence_sentence=evidence,
        parse_quality="parsed_unverified",
    )


def parse_borrowing_estimates(text: str, *, source_path: str | Path | None = None) -> list[EstimateParse]:
    normalized = document_to_text(text, source_path=source_path)
    release_date = parse_html_publication_date(text) or parse_release_date(normalized) or ""
    release_year = int(release_date[:4]) if release_date else int(re.search(r"\b(20\d{2}|19\d{2})\b", normalized).group(1))
    estimates: list[EstimateParse] = []
    for match in _ESTIMATE_RE.finditer(normalized):
        amount = _signed_announced(match.group("verb"), _money_to_bn(match.group("amount"), match.group("unit")))
        quarter_year = _year_from_period(match.group("period"), release_year)
        quarter = _quarter_from_end_month(match.group("end_month"), str(quarter_year))
        estimates.append(
            EstimateParse(
                quarter=quarter,
                amount_bn=amount,
                tga_assumption_bn=_money_to_bn(match.group("tga"), match.group("tga_unit")),
                evidence_sentence=_sentence_around(normalized, match.start(), match.end()),
            )
        )
    return estimates


def _read_sibling_commit(path: Path) -> str:
    git_path = path / ".git"
    try:
        if git_path.is_file():
            text = git_path.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                git_dir = Path(text.split(":", 1)[1].strip())
                head = git_dir / "HEAD"
                if head.exists():
                    return head.read_text(encoding="utf-8").strip()
        head = git_path / "HEAD"
        if head.exists():
            return head.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"
    return "unavailable"


def copy_qrawatch_inputs(
    *,
    qrawatch_root: str | Path,
    external_dir: str | Path,
) -> dict[str, object]:
    src_root = Path(qrawatch_root)
    out_root = Path(external_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    files = [
        src_root / "data/processed/qra_event_registry_v2.csv",
        src_root / "data/processed/qra_financing_release_map.csv",
    ]
    copied: list[dict[str, object]] = []
    for src in files:
        dst = out_root / src.name
        shutil.copy2(src, dst)
        copied.append(
            {
                "source_path": str(src),
                "copied_path": str(dst),
                "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "source": "qrawatch",
        "copy_date_utc": _utc_now_iso(),
        "sibling_git_commit": _read_sibling_commit(src_root),
        "files": copied,
    }
    manifest_path = out_root / "provenance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def fetch_url_to_cache(url: str, *, cache_dir: str | Path) -> Path:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    slug = url.rstrip("/").split("/")[-1] or hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    path = root / f"{slug}.html"
    if path.exists():
        return path
    request = Request(url, headers={"User-Agent": "tdc-hf qra borrowing parser"})
    with urlopen(request, timeout=90) as response:
        payload = response.read()
    path.write_bytes(payload)
    return path


def _load_registry(registry_csv: str | Path) -> pd.DataFrame:
    registry = pd.read_csv(registry_csv)
    return registry[
        ["event_id", "quarter", "financing_estimates_release_date_et", "financing_estimates_url", "policy_statement_url"]
    ].copy()


def _release_rows_from_qrawatch(
    *,
    copied_map_csv: str | Path,
    copied_registry_csv: str | Path,
    qrawatch_root: str | Path,
) -> list[dict[str, object]]:
    mapping = pd.read_csv(copied_map_csv)
    registry = _load_registry(copied_registry_csv)
    registry_by_quarter = registry.set_index("quarter", drop=False)
    rows: list[dict[str, object]] = []
    for _, row in mapping.iterrows():
        reg = registry_by_quarter.loc[row["quarter"]] if row["quarter"] in registry_by_quarter.index else None
        event_id = "" if reg is None else str(reg["event_id"])
        local_doc = row.get("source_doc_local")
        if pd.isna(local_doc) and event_id in QRAWATCH_FALLBACK_DOCS:
            local_doc = QRAWATCH_FALLBACK_DOCS[event_id]
        if pd.isna(local_doc):
            continue
        source_path = Path(qrawatch_root) / str(local_doc)
        if not source_path.exists():
            continue
        source_url = row.get("source_url")
        if pd.isna(source_url) and reg is not None:
            source_url = reg["financing_estimates_url"] if not pd.isna(reg["financing_estimates_url"]) else reg["policy_statement_url"]
        rows.append(
            {
                "event_id": event_id,
                "registry_release_date": "" if reg is None or pd.isna(reg["financing_estimates_release_date_et"]) else str(reg["financing_estimates_release_date_et"]),
                "source_url": str(source_url) if not pd.isna(source_url) else "",
                "source_path": source_path,
                "source_kind": "qrawatch_archive",
            }
        )
    return rows


def _release_rows_from_urls(urls: Iterable[str], *, cache_dir: str | Path) -> list[dict[str, object]]:
    rows = []
    for url in urls:
        path = fetch_url_to_cache(url, cache_dir=cache_dir)
        rows.append(
            {
                "event_id": "",
                "registry_release_date": "",
                "source_url": url,
                "source_path": path,
                "source_kind": "fetched_html",
            }
        )
    return rows


def _record_from_release(release: dict[str, object]) -> tuple[dict[str, object], str, list[EstimateParse]]:
    source_path = Path(release["source_path"])
    payload = source_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_borrowing_release(payload, source_path=source_path)
    estimates = parse_borrowing_estimates(payload, source_path=source_path)
    release_date = parsed.release_date or str(release.get("registry_release_date") or "")[:10]
    quality = parsed.parse_quality
    if quality == "parsed_unverified" and parsed.surprise_bn is not None:
        quality = "verified_manual" if parsed.quarter in MANUALLY_VERIFIED_QUARTERS else "parsed_ok"
    note = parsed.note
    row = {
        "event_id": release.get("event_id", ""),
        "quarter": parsed.quarter,
        "release_date": release_date,
        "announced_net_borrowing_bn": parsed.announced_net_borrowing_bn,
        "prior_estimate_bn": parsed.prior_estimate_bn,
        "prior_release_date": parsed.prior_release_date,
        "prior_source": "derived_from_revision" if parsed.prior_estimate_bn is not None else "",
        "surprise_bn": parsed.surprise_bn,
        "tga_assumption_announced_bn": parsed.tga_assumption_announced_bn,
        "tga_assumption_prior_bn": parsed.tga_assumption_prior_bn,
        "source_url": release.get("source_url", ""),
        "source_kind": release.get("source_kind", ""),
        "parse_quality": quality,
        "note": note,
    }
    return row, parsed.evidence_sentence, estimates


def _release_month(date_value: object) -> str:
    text = str(date_value or "")
    return text[:7] if len(text) >= 7 else ""


def _apply_independent_prior_parses(
    frame: pd.DataFrame,
    *,
    estimates_by_release_month: dict[str, list[EstimateParse]],
) -> pd.DataFrame:
    out = frame.copy()
    for idx, row in out.iterrows():
        prior_month = _release_month(row.get("prior_release_date"))
        quarter = str(row.get("quarter") or "")
        if not prior_month or not quarter:
            continue
        candidates = estimates_by_release_month.get(prior_month, [])
        match = next((estimate for estimate in candidates if estimate.quarter == quarter), None)
        if match is None:
            continue
        out.at[idx, "prior_estimate_bn"] = match.amount_bn
        out.at[idx, "prior_source"] = "parsed_prior_release"
        if pd.isna(row.get("tga_assumption_prior_bn")) and match.tga_assumption_bn is not None:
            out.at[idx, "tga_assumption_prior_bn"] = match.tga_assumption_bn
    return out


def build_qra_borrowing_surprise_csv(
    *,
    qrawatch_root: str | Path = "../qrawatch",
    external_dir: str | Path = "data/external/qrawatch",
    raw_cache_dir: str | Path = "data/raw/qra_borrowing",
    out_csv: str | Path = "data/processed/qra_borrowing_surprise.csv",
    notes_md: str | Path = "data/processed/qra_borrowing_surprise_notes.md",
    extra_urls: list[str] | None = None,
) -> dict[str, object]:
    manifest = copy_qrawatch_inputs(qrawatch_root=qrawatch_root, external_dir=external_dir)
    external = Path(external_dir)
    releases = _release_rows_from_qrawatch(
        copied_map_csv=external / "qra_financing_release_map.csv",
        copied_registry_csv=external / "qra_event_registry_v2.csv",
        qrawatch_root=qrawatch_root,
    )
    url_list = [*BUILTIN_FETCH_URLS, *(extra_urls or [])]
    releases.extend(_release_rows_from_urls(dict.fromkeys(url_list), cache_dir=raw_cache_dir))

    records: list[dict[str, object]] = []
    evidence_by_quarter: dict[str, str] = {}
    estimates_by_release_month: dict[str, list[EstimateParse]] = {}
    for release in releases:
        record, evidence, estimates = _record_from_release(release)
        records.append(record)
        if evidence:
            evidence_by_quarter[str(record["quarter"])] = evidence
        month = _release_month(record["release_date"])
        if month:
            estimates_by_release_month.setdefault(month, []).extend(estimates)

    frame = pd.DataFrame(records, columns=QRA_REQUIRED_COLUMNS)
    frame = frame.drop_duplicates(subset=["event_id", "quarter", "source_url"], keep="last")
    frame = _apply_independent_prior_parses(frame, estimates_by_release_month=estimates_by_release_month)
    frame = frame.sort_values(["release_date", "quarter"], kind="stable").reset_index(drop=True)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

    notes_path = Path(notes_md)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(_build_notes(frame, evidence_by_quarter, manifest), encoding="utf-8")

    return {
        "status": "ok",
        "out": str(out_path),
        "notes": str(notes_path),
        "rows": int(len(frame)),
        "quarters": [str(q) for q in frame["quarter"].dropna().tolist()],
        "parse_quality": frame["parse_quality"].value_counts(dropna=False).to_dict(),
    }


def _build_notes(frame: pd.DataFrame, evidence_by_quarter: dict[str, str], manifest: dict[str, object]) -> str:
    quality = frame["parse_quality"].value_counts(dropna=False).to_dict()
    gaps = frame.loc[frame["parse_quality"].eq("missing"), ["quarter", "parse_quality", "note"]]
    spot_quarters = [q for q in sorted(MANUALLY_VERIFIED_QUARTERS) if q in set(frame["quarter"].astype(str))]
    lines = [
        "# QRA Net Marketable Borrowing Surprise Construction",
        "",
        "Definition: for refunding event R and current quarter Q, surprise_bn = estimate_R(Q) - estimate_R_minus_1(Q).",
        "Treasury usually states the revision directly as the current-quarter borrowing estimate being higher or lower than announced at the prior refunding; the parser signs pay-down revisions as net borrowing.",
        "",
        "Sources: qrawatch registry and financing release map copied under data/external/qrawatch, qrawatch archived Treasury pages where available, and explicit Treasury pages fetched under data/raw/qra_borrowing for post-registry releases.",
        f"qrawatch sibling commit: {manifest.get('sibling_git_commit', 'unavailable')}",
        "",
        f"Rows: {len(frame)}",
        f"Parse quality: {quality}",
        "",
        "Known gaps and caveats: qrawatch's processed map starts at 2010Q1, so 2009 is not included in this first pass. The earlier 2021Q4/2022Q1 current-quarter gap has been fixed by direct Treasury fetches for `jy0452` and `jy0575`. Prior TGA assumptions are blank except where the prior Treasury release exposes a parseable forward estimate or cash-balance table.",
        "",
        "Registry alignment caveat: qrawatch event IDs point to the correct physical refunding release, but its registry quarter labels diverge from the current-quarter borrowing-estimate label for 2021+ events. This CSV's `quarter` is always the quarter covered by the current-quarter borrowing estimate.",
        "",
        "Definition caveat for 2025+ releases: the headline revision is used, not Treasury's separate cash-adjusted figure that excludes beginning-of-quarter cash-balance changes. Use the TGA-assumption columns for sensitivity work where populated.",
        "",
        "Prior-estimate construction: `prior_source=parsed_prior_release` means the prior value was independently parsed from the previous Treasury release's forward estimate for the same quarter; `derived_from_revision` means it remains announced minus Treasury's stated revision.",
        "",
        "## Spot Checks",
        "",
    ]
    for quarter in spot_quarters:
        match = frame.loc[frame["quarter"].astype(str).eq(quarter)]
        if match.empty:
            continue
        row = match.iloc[0]
        verdict = (
            f"- {quarter}: {row['parse_quality']}; announced {row['announced_net_borrowing_bn']} bn, "
            f"prior {row['prior_estimate_bn']} bn, surprise {row['surprise_bn']} bn. "
            f"Source: {row['source_url']}"
        )
        lines.append(verdict)
        evidence = evidence_by_quarter.get(quarter, "")
        if evidence:
            lines.append(f"  Evidence: \"{evidence}\"")
    lines.extend(["", "## Parse Gaps", ""])
    if gaps.empty:
        lines.append("None.")
    else:
        lines.extend(_simple_markdown_table(gaps))
    lines.extend(["", "## Second Source Disagreements", "", "None identified in this pass."])
    return "\n".join(lines) + "\n"


def _simple_markdown_table(frame: pd.DataFrame) -> list[str]:
    columns = [str(col) for col in frame.columns]
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in frame.to_numpy()]
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return out
