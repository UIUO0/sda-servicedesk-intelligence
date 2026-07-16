"""Project configuration package.

Re-exports everything from :mod:`config.settings` so callers can simply do
``import config`` and read ``config.BASE_URL``, ``config.RAW_DIR``, etc.
"""
from config.settings import *  # noqa: F401,F403
from config.settings import (  # noqa: F401  explicit re-export for tooling
    AUTHTOKEN,
    BASE_URL,
    DATA_DIR,
    INSIGHTS_DIR,
    MAX_RETRIES,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RATE_LIMIT_SLEEP,
    RAW_DIR,
    ROW_COUNT,
    TIMEOUT,
    VERIFY_SSL,
    require_connection,
)
