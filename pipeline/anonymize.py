"""Anonymisation helpers for processed tables.

The raw ServiceDesk data carries real personal data (staff names, work e-mail
addresses, phone numbers). Any table that leaves this machine must have that
data replaced with stable surrogate IDs.

The mapping is deterministic *within a run* (the same person always maps to the
same surrogate, so grouping/counting still works) but is derived from a random
per-run salt, so the surrogates cannot be reversed by hashing a guessed name
unless the salt file is also present. The salt + mapping are written to
``data/processed/.anon_map.json`` which is git-ignored and never leaves the box.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

# Column suffixes that carry direct identifiers.
NAME_SUFFIXES = ("_name",)
EMAIL_SUFFIXES = ("_email",)
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\s\-()]{7,}\d)")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class Anonymizer:
    """Maps identifiers to stable surrogate IDs using a secret per-run salt."""

    def __init__(self, salt: Optional[str] = None) -> None:
        self.salt = salt or secrets.token_hex(16)
        self._map: Dict[str, str] = {}

    def surrogate(self, value: str, prefix: str) -> str:
        """Return a stable surrogate like ``PER_3f9a1c`` for ``value``."""
        key = f"{prefix}:{value.strip().lower()}"
        if key not in self._map:
            digest = hashlib.sha256(f"{self.salt}:{key}".encode("utf-8")).hexdigest()[:6]
            self._map[key] = f"{prefix}_{digest}"
        return self._map[key]

    def scrub_text(self, text: str) -> str:
        """Redact e-mails and phone numbers embedded in free text."""
        if not isinstance(text, str) or not text:
            return text
        text = EMAIL_PATTERN.sub(lambda m: self.surrogate(m.group(0), "EML"), text)
        text = PHONE_PATTERN.sub("[PHONE]", text)
        return text

    def save_map(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"salt": self.salt, "map": self._map}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def anonymize_frame(
    df: pd.DataFrame,
    anon: Anonymizer,
    *,
    text_columns: Iterable[str] = ("subject", "short_description", "description", "title", "resolution"),
) -> pd.DataFrame:
    """Return a copy of ``df`` with identifiers replaced by surrogates.

    * ``*_name``  → ``PER_xxxxxx``
    * ``*_email`` → ``EML_xxxxxx``
    * free-text columns have embedded e-mails/phones redacted
    * any ``*_phone``/``*_mobile`` column is dropped outright
    """
    out = df.copy()

    for col in out.columns:
        if any(col.endswith(s) for s in NAME_SUFFIXES):
            out[col] = out[col].map(lambda v: anon.surrogate(str(v), "PER") if pd.notna(v) else v)
        elif any(col.endswith(s) for s in EMAIL_SUFFIXES):
            out[col] = out[col].map(lambda v: anon.surrogate(str(v), "EML") if pd.notna(v) else v)

    # Phone-like columns carry no analytical value — drop them entirely.
    drop = [c for c in out.columns if c.endswith(("_phone", "_mobile"))]
    if drop:
        out = out.drop(columns=drop)

    for col in text_columns:
        if col in out.columns:
            out[col] = out[col].map(anon.scrub_text)

    return out
