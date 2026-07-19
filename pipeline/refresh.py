"""One-shot data refresh: extract → preprocess → predict, with a status file.

Designed to run detached (the dashboard's "Refresh data" button launches it as
a background process) or manually::

    python -m pipeline.refresh

Progress is written to ``data/refresh_status.json`` so any dashboard session
can display the current state, and a second refresh refuses to start while one
is already running. All server traffic remains GET-only.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("refresh")

STATUS_PATH = config.DATA_DIR / "refresh_status.json"


def read_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_status(**fields) -> None:
    status = read_status()
    status.update(fields)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check for the recorded refresh process."""
    try:
        import psutil  # type: ignore  # optional; fall back if absent
        return psutil.pid_exists(pid)
    except ImportError:
        if os.name == "nt":
            import subprocess
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, check=False,
            ).stdout
            return str(pid) in out
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def run_refresh(modules: Optional[List[str]] = None, *, skip_extract: bool = False) -> None:
    """Run the full chain, updating the status file at each step."""
    from pipeline import extract as extract_mod
    from pipeline import preprocess as preprocess_mod

    _write_status(state="running", pid=os.getpid(), started=_now(),
                  finished=None, error=None, step="extract")
    try:
        if not skip_extract:
            names = modules or list(extract_mod.MODULES)
            extract_mod.run(names, limit=None, skip_details=False, skip_notes=False)

        _write_status(step="preprocess")
        preprocess_mod.main(["--anonymize"])

        _write_status(step="predict")
        try:
            from models import predict as predict_mod
            predict_mod.main([])
        except ImportError as exc:  # scikit-learn not installed yet
            logger.warning("prediction step skipped: %s", exc)

        _write_status(state="done", step=None, finished=_now())
        logger.info("Refresh complete.")
    except BaseException as exc:
        _write_status(state="failed", error=f"{type(exc).__name__}: {exc}", finished=_now())
        raise


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Refresh SDP data end-to-end (GET-only).")
    parser.add_argument("--modules", nargs="*", default=None,
                        help="Modules to extract (default: all).")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Only re-run preprocess + predict on existing raw data.")
    parser.add_argument("--force", action="store_true",
                        help="Start even if the status file claims a refresh is running.")
    args = parser.parse_args(argv)

    status = read_status()
    if status.get("state") == "running" and not args.force:
        pid = status.get("pid")
        if pid and _pid_alive(int(pid)):
            raise SystemExit(f"A refresh is already running (pid {pid}). Use --force to override.")
        logger.warning("Stale 'running' status (pid %s not alive) — proceeding.", pid)

    run_refresh(args.modules, skip_extract=args.skip_extract)


if __name__ == "__main__":
    main()
