"""Preprocess raw SDP JSON into flat, ML-ready tables (CSV + Parquet).

Turns the nested API payloads written by :mod:`src.extract` into one tidy
DataFrame per module. The small, reusable helpers at the top (epoch parsing,
person flattening, HTML cleaning, language detection) are shared across every
module's row builder.

Usage examples::

    python -m src.preprocess                       # process everything in data/raw/
    python -m src.preprocess --modules requests problems
    python -m src.preprocess --sample "apiResponse (1).json:requests" \\
                             --sample "apiResponse.json:problems"
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("preprocess")

_ARABIC_RE = re.compile(r"[؀-ۿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
# resolution/workaround values that are clearly placeholder junk (e.g. "10", "16").
_JUNK_RE = re.compile(r"^\s*\d{1,3}\s*$")


# --- reusable field helpers ------------------------------------------------
def parse_epoch(field: Any) -> Optional[pd.Timestamp]:
    """Convert an SDP ``{display_value, value}`` time field to a Timestamp.

    ``value`` is epoch milliseconds as a string. Returns None for missing/null.
    """
    if not isinstance(field, dict):
        return None
    value = field.get("value")
    if value in (None, ""):
        return None
    try:
        return pd.to_datetime(int(value), unit="ms")
    except (ValueError, TypeError):
        return None


def pick(obj: Any, key: str = "name") -> Optional[Any]:
    """Safely read ``obj[key]`` when obj may be None / not a dict."""
    if isinstance(obj, dict):
        return obj.get(key)
    return None


def flatten_person(obj: Any, prefix: str) -> Dict[str, Any]:
    """Flatten a person object (requester/technician/…) to scalar columns."""
    dept = obj.get("department") if isinstance(obj, dict) else None
    return {
        f"{prefix}_name": pick(obj, "name"),
        f"{prefix}_email": pick(obj, "email_id"),
        f"{prefix}_id": pick(obj, "id"),
        f"{prefix}_dept": pick(dept, "name"),
        f"{prefix}_site": pick(pick(dept, "site"), "name") if isinstance(dept, dict) else None,
    }


def clean_html(text: Any) -> str:
    """Strip HTML tags/entities and collapse whitespace; keep Arabic text intact."""
    if not text or not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ")
    cleaned = cleaned.replace("\xa0", " ")  # &nbsp;
    return re.sub(r"\s+", " ", cleaned).strip()


def detect_lang(text: Any) -> str:
    """Classify text as 'ar', 'en', 'mixed' or 'unknown' by character ratio."""
    if not text or not isinstance(text, str):
        return "unknown"
    ar = len(_ARABIC_RE.findall(text))
    en = len(_LATIN_RE.findall(text))
    if ar == 0 and en == 0:
        return "unknown"
    if ar > 0 and en > 0:
        # "mixed" only when the minority is meaningful (>15% of letters).
        minority = min(ar, en) / (ar + en)
        return "mixed" if minority > 0.15 else ("ar" if ar > en else "en")
    return "ar" if ar > en else "en"


def is_junk(text: Any) -> bool:
    """True if a resolution/workaround value looks like placeholder junk."""
    return isinstance(text, str) and bool(_JUNK_RE.match(text))


def _time_features(created: Optional[pd.Timestamp], closed: Optional[pd.Timestamp]) -> Dict[str, Any]:
    feats: Dict[str, Any] = {
        "created_hour": created.hour if created is not None else None,
        "created_dayofweek": created.dayofweek if created is not None else None,
        "created_month": created.month if created is not None else None,
        "resolution_hours": None,
    }
    if created is not None and closed is not None:
        feats["resolution_hours"] = round((closed - created).total_seconds() / 3600.0, 2)
    return feats


# --- per-module row builders ----------------------------------------------
def flatten_request(rec: Dict[str, Any]) -> Dict[str, Any]:
    created = parse_epoch(rec.get("created_time"))
    subject = rec.get("subject") or ""
    short_desc = clean_html(rec.get("short_description"))
    row: Dict[str, Any] = {
        "id": rec.get("id"),
        "subject": subject,
        "short_description": short_desc,
        "status": pick(rec.get("status")),
        "priority": pick(rec.get("priority")),
        "group": pick(rec.get("group")),
        "template": pick(rec.get("template")),
        "site": pick(rec.get("site")),
        "is_service_request": rec.get("is_service_request"),
        "is_overdue": rec.get("is_overdue"),
        "cancel_requested": rec.get("cancel_requested"),
        "created_time": created,
        "due_by_time": parse_epoch(rec.get("due_by_time")),
    }
    row.update(flatten_person(rec.get("requester"), "requester"))
    row.update(flatten_person(rec.get("technician"), "technician"))
    row.update(flatten_person(rec.get("created_by"), "created_by"))
    row.update(_time_features(created, None))  # requests list has no closed_time
    row["subject_len"] = len(subject)
    row["desc_len"] = len(short_desc)
    row["lang"] = detect_lang(f"{subject} {short_desc}")
    return row


def flatten_problem(rec: Dict[str, Any]) -> Dict[str, Any]:
    reported = parse_epoch(rec.get("reported_time"))
    closed = parse_epoch(rec.get("closed_time"))
    title = rec.get("title") or ""
    description = clean_html(rec.get("description")) or clean_html(rec.get("short_description"))
    resolution = rec.get("resolution")
    row: Dict[str, Any] = {
        "id": rec.get("id"),
        "title": title,
        "description": description,
        "status": pick(rec.get("status")),
        "priority": pick(rec.get("priority")),
        "impact": pick(rec.get("impact")),
        "urgency": pick(rec.get("urgency")),
        "category": pick(rec.get("category")),
        "subcategory": pick(rec.get("subcategory")),
        "item": pick(rec.get("item")),
        "group": pick(rec.get("group")),
        "reported_time": reported,
        "closed_time": closed,
        "due_by_time": parse_epoch(rec.get("due_by_time")),
        "updated_time": parse_epoch(rec.get("updated_time")),
        "has_resolution": bool(resolution) and not is_junk(resolution),
        "resolution_is_junk": is_junk(resolution),
        "workaround_is_junk": is_junk(rec.get("workaround")),
        "notes_present": rec.get("notes_present"),
    }
    row.update(flatten_person(rec.get("reported_by"), "reported_by"))
    row.update(flatten_person(rec.get("technician"), "technician"))
    row.update(_time_features(reported, closed))
    row["title_len"] = len(title)
    row["desc_len"] = len(description)
    row["lang"] = detect_lang(f"{title} {description}")
    return row


def flatten_generic(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback flattener: keep scalars, reduce nested dicts to their 'name'."""
    row: Dict[str, Any] = {}
    for key, value in rec.items():
        if isinstance(value, dict):
            if "value" in value and "display_value" in value:
                row[key] = parse_epoch(value)
            else:
                row[key] = value.get("name", value.get("id"))
        elif isinstance(value, list):
            row[f"{key}_count"] = len(value)
        else:
            row[key] = value
    return row


BUILDERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "requests": flatten_request,
    "problems": flatten_problem,
}


# --- detail / notes enrichment --------------------------------------------
def _count_notes(bundle: Dict[str, Any]) -> Optional[int]:
    notes = bundle.get("notes")
    if isinstance(notes, dict):
        for value in notes.values():
            if isinstance(value, list):
                return len(value)
    return None


def _count_attachments(bundle: Dict[str, Any]) -> Optional[int]:
    atts = bundle.get("_attachments")
    if isinstance(atts, dict):
        for value in atts.values():
            if isinstance(value, list):
                return len(value)
    return None


def enrich_from_details(module: str, rows: List[Dict[str, Any]]) -> None:
    """Add notes_count / attachments_count and full description/resolution."""
    detail_dir = config.RAW_DIR / "details" / module
    if not detail_dir.exists():
        return
    for row in rows:
        detail_path = detail_dir / f"{row['id']}.json"
        if not detail_path.exists():
            continue
        bundle = json.loads(detail_path.read_text(encoding="utf-8"))
        row["notes_count"] = _count_notes(bundle)
        row["attachments_count"] = _count_attachments(bundle)
        detail = bundle.get("detail", {})
        payload = detail.get(module[:-1]) if isinstance(detail, dict) else None
        if isinstance(payload, dict):
            full_desc = clean_html(payload.get("description"))
            if full_desc:
                row["description"] = full_desc
            if payload.get("resolution"):
                res = pick(payload.get("resolution"), "content") or payload.get("resolution")
                row["resolution"] = clean_html(res) if isinstance(res, str) else res


# --- loading ---------------------------------------------------------------
def load_raw_records(module: str) -> List[Dict[str, Any]]:
    """Read every ``page_*.json`` under data/raw/<module>/ and flatten to records."""
    page_dir = config.RAW_DIR / module
    records: List[Dict[str, Any]] = []
    for page_file in sorted(page_dir.glob("page_*.json")):
        page = json.loads(page_file.read_text(encoding="utf-8"))
        records.extend(_records_from(page, module))
    return records


def load_sample_records(path: Path, module: str) -> List[Dict[str, Any]]:
    """Read a single API-response file (e.g. one of the provided samples)."""
    page = json.loads(path.read_text(encoding="utf-8"))
    return _records_from(page, module)


def _records_from(page: Dict[str, Any], module: str) -> List[Dict[str, Any]]:
    if module in page and isinstance(page[module], list):
        return page[module]
    for value in page.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    return []


def build_dataframe(module: str, records: List[Dict[str, Any]], *, enrich: bool) -> pd.DataFrame:
    builder = BUILDERS.get(module, flatten_generic)
    rows = [builder(rec) for rec in records]
    if enrich and module in BUILDERS:
        enrich_from_details(module, rows)
    df = pd.DataFrame(rows)
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")
    return df


def save_dataframe(df: pd.DataFrame, module: str) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.PROCESSED_DIR / f"{module}.csv"
    parquet_path = config.PROCESSED_DIR / f"{module}.parquet"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pyarrow missing / dtype issue shouldn't lose the CSV
        logger.warning("Parquet write failed for %s (%s); CSV still written", module, exc)
    logger.info("Wrote %s: %d rows, %d cols -> %s", module, len(df), df.shape[1], csv_path.name)


# --- CLI -------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten raw SDP JSON into ML-ready tables.")
    parser.add_argument(
        "--modules", nargs="*", default=None,
        help="Modules to process from data/raw/. Defaults to all present.",
    )
    parser.add_argument(
        "--sample", action="append", default=[],
        help="Process a single response file as 'path:module' (repeatable). "
             "Bypasses data/raw and detail enrichment.",
    )
    return parser.parse_args(argv)


def _discover_modules() -> List[str]:
    if not config.RAW_DIR.exists():
        return []
    return sorted(
        p.name for p in config.RAW_DIR.iterdir()
        if p.is_dir() and p.name != "details"
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.sample:
        for spec in args.sample:
            path_str, _, module = spec.rpartition(":")
            path = Path(path_str)
            if not path.is_absolute():
                path = config.PROJECT_ROOT / path
            records = load_sample_records(path, module)
            df = build_dataframe(module, records, enrich=False)
            save_dataframe(df, module)
        return

    modules = args.modules or _discover_modules()
    if not modules:
        logger.warning("No modules found under %s. Run the extractor first.", config.RAW_DIR)
        return
    for module in modules:
        records = load_raw_records(module)
        if not records:
            logger.warning("No records for %s; skipping", module)
            continue
        df = build_dataframe(module, records, enrich=True)
        save_dataframe(df, module)


if __name__ == "__main__":
    main()
