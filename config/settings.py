"""Central configuration loaded from environment variables / a local .env file.

All ServiceDesk Plus connection settings live here so scripts never hard-code
secrets. Values are read once at import time via python-dotenv.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project layout ------------------------------------------------------------
# settings.py lives in <root>/config/, so the root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INSIGHTS_DIR = PROJECT_ROOT / "insights"
MODELS_DIR = PROJECT_ROOT / "models"

# Load .env sitting next to this file (if present). Real env vars win over it.
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


# Connection settings -------------------------------------------------------
BASE_URL = (os.getenv("SDP_BASE_URL") or "").rstrip("/")
AUTHTOKEN = os.getenv("SDP_AUTHTOKEN") or ""
VERIFY_SSL = _get_bool("SDP_VERIFY_SSL", False)
ROW_COUNT = min(_get_int("SDP_ROW_COUNT", 100), 100)  # server hard cap is 100
RATE_LIMIT_SLEEP = _get_float("SDP_RATE_LIMIT_SLEEP", 0.2)
TIMEOUT = _get_int("SDP_TIMEOUT", 60)
MAX_RETRIES = _get_int("SDP_MAX_RETRIES", 4)


def require_connection() -> None:
    """Raise a clear error if the settings needed to reach the API are missing.

    Only the extractor needs a live connection, so preprocess/eda don't call this.
    """
    missing = [
        name
        for name, value in (("SDP_BASE_URL", BASE_URL), ("SDP_AUTHTOKEN", AUTHTOKEN))
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required settings: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in the values."
        )
