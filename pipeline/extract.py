"""Extractor: pull ServiceDesk Plus data to local raw JSON (GET-only).

The extractor is driven by a module table (:data:`MODULES`) so adding a new
module is a one-line change. It is idempotent: list pages and per-record detail
files are written to ``data/raw/`` and existing detail files are skipped on
re-run, so an interrupted pull can simply be restarted without losing work.

Usage examples::

    python -m pipeline.extract --modules requests,problems --limit 20 --skip-details
    python -m pipeline.extract                      # everything, with full details
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

import config
from pipeline.sdp_client import SDPClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("extract")


@dataclass(frozen=True)
class ModuleSpec:
    """Describes how to pull one module."""

    name: str
    # Sub-resources to pull per record (only when has_details is True).
    subresources: tuple = ()
    # Ticket-like modules get per-record detail pulls; reference tables don't.
    has_details: bool = False


# The module table. Ticket-like modules carry rich text + conversations; the
# reference modules are small lookup tables pulled list-only.
MODULES: Dict[str, ModuleSpec] = {
    "requests": ModuleSpec("requests", subresources=("notes", "_attachments"), has_details=True),
    "problems": ModuleSpec("problems", subresources=("notes", "_attachments"), has_details=True),
    "changes": ModuleSpec("changes", subresources=("notes", "_attachments"), has_details=True),
    "projects": ModuleSpec("projects", subresources=("_attachments",), has_details=True),
    "solutions": ModuleSpec("solutions", subresources=("_attachments",), has_details=True),
    # Reference / lookup tables (list-only).
    "requesters": ModuleSpec("requesters"),
    "technicians": ModuleSpec("technicians"),
    "groups": ModuleSpec("groups"),
    "categories": ModuleSpec("categories"),
    "sites": ModuleSpec("sites"),
}


@dataclass
class ExtractStats:
    pages: int = 0
    records: int = 0
    details: int = 0
    details_skipped: int = 0
    errors: List[str] = field(default_factory=list)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_list(client: SDPClient, spec: ModuleSpec, limit: Optional[int]) -> tuple:
    """Pull all list pages for a module. Returns (record_ids, stats)."""
    stats = ExtractStats()
    record_ids: List[str] = []
    out_dir = config.RAW_DIR / spec.name

    for page_num, page in enumerate(client.paginate(spec.name, limit=limit), start=1):
        _write_json(out_dir / f"page_{page_num:04d}.json", page)
        stats.pages += 1
        records = SDPClient._records_from(page, spec.name)
        stats.records += len(records)
        record_ids.extend(str(r["id"]) for r in records if r.get("id") is not None)
        logger.info("[%s] page %d: %d records", spec.name, page_num, len(records))

    return record_ids, stats


def extract_details(
    client: SDPClient,
    spec: ModuleSpec,
    record_ids: List[str],
    stats: ExtractStats,
    *,
    skip_notes: bool,
) -> None:
    """Pull each record's full detail + sub-resources into one JSON file.

    Existing files are skipped so re-runs are cheap and resumable.
    """
    out_dir = config.RAW_DIR / "details" / spec.name
    subs = () if skip_notes else spec.subresources

    for record_id in tqdm(record_ids, desc=f"{spec.name} details", unit="rec"):
        target = out_dir / f"{record_id}.json"
        if target.exists():
            stats.details_skipped += 1
            continue
        try:
            bundle: Dict[str, object] = {"detail": client.get_detail(spec.name, record_id)}
            for sub in subs:
                try:
                    bundle[sub] = client.get_subresource(spec.name, record_id, sub)
                except Exception as exc:  # a missing sub-resource shouldn't fail the record
                    bundle[sub] = {"_error": str(exc)}
            _write_json(target, bundle)
            stats.details += 1
        except Exception as exc:  # one bad record must not stop the rest
            msg = f"{spec.name}/{record_id}: {exc}"
            stats.errors.append(msg)
            logger.warning("detail failed for %s", msg)


def run(
    modules: List[str], *, limit: Optional[int], skip_details: bool, skip_notes: bool
) -> Dict[str, ExtractStats]:
    config.require_connection()
    client = SDPClient()
    results: Dict[str, ExtractStats] = {}

    for name in modules:
        spec = MODULES[name]
        logger.info("=== Extracting %s ===", name)
        record_ids, stats = extract_list(client, spec, limit)

        if spec.has_details and not skip_details:
            extract_details(client, spec, record_ids, stats, skip_notes=skip_notes)

        results[name] = stats
        logger.info(
            "[%s] done: %d records, %d details (%d skipped), %d errors",
            name, stats.records, stats.details, stats.details_skipped, len(stats.errors),
        )

    return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull ServiceDesk Plus data to raw JSON (GET-only).")
    parser.add_argument(
        "--modules",
        default="all",
        help="Comma-separated module names, or 'all'. Options: " + ", ".join(MODULES),
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap records per module (smoke test).")
    parser.add_argument("--skip-details", action="store_true", help="List pages only, no per-record detail.")
    parser.add_argument("--skip-notes", action="store_true", help="Detail bodies only, no notes/attachments.")
    return parser.parse_args(argv)


def resolve_modules(spec: str) -> List[str]:
    if spec.strip().lower() == "all":
        return list(MODULES)
    names = [m.strip() for m in spec.split(",") if m.strip()]
    unknown = [m for m in names if m not in MODULES]
    if unknown:
        raise SystemExit(f"Unknown module(s): {', '.join(unknown)}. Known: {', '.join(MODULES)}")
    return names


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    modules = resolve_modules(args.modules)
    results = run(
        modules,
        limit=args.limit,
        skip_details=args.skip_details,
        skip_notes=args.skip_notes,
    )
    total_errors = sum(len(s.errors) for s in results.values())
    logger.info("All done. Modules: %d, total errors: %d", len(results), total_errors)


if __name__ == "__main__":
    main()
