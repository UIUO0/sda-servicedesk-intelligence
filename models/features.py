"""Shared feature contract + temporal split — the anti-leakage backbone.

Every model that predicts *at ticket-creation time* must train only on fields
that exist the moment a ticket is created. That rule is defined **once** here so
it is auditable in one place rather than re-litigated per model.

`CREATION_TIME_FEATURES` / `FORBIDDEN_AT_CREATION` are the contract;
`assert_no_leakage` is called by each trainer before `.fit()` so a leak fails
loudly instead of silently inflating a score (the lesson from the semantic-search
no-go: a documented refusal beats a flattering-but-wrong number).
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

# --- the contract ----------------------------------------------------------
# Available the instant a requester submits a ticket.
CREATION_TIME_FEATURES: List[str] = [
    "subject",              # free text, entered by requester
    "description",          # free text, entered by requester
    "short_description",    # free text, entered by requester
    "template",             # the service/catalog item chosen at submission
    "site",                 # ticket site, set from the request at submission
    "is_service_request",   # fixed by the chosen template at submission
    "requester_dept",       # requester attribute (their department)
    "requester_site",       # requester attribute (their site)
    "created_hour",         # derived from creation timestamp
    "created_dayofweek",    # derived from creation timestamp
    "created_month",        # derived from creation timestamp
    "subject_len",          # derived from subject text
    "desc_len",             # derived from description text
    "lang",                 # derived from the submitted text
]

# Populated only DURING processing — using any of these to predict at creation
# time is temporal leakage.
FORBIDDEN_AT_CREATION: List[str] = [
    "technician_name", "technician_email", "technician_id",
    "technician_dept", "technician_site",
    "status",               # moves across the lifecycle
    "priority",             # set during triage, not guaranteed at submission
    "due_by_time", "closed_time", "resolution_hours",
    "resolution",           # the outcome itself
    "notes_count",          # conversation accrues after creation
    "attachments_count",    # can grow after creation
    "is_overdue",           # only knowable after the SLA clock runs
    "cancel_requested",
    "created_by_name", "created_by_email", "created_by_id",  # may be a technician acting on behalf
    "created_by_dept", "created_by_site",
]


def assert_no_leakage(feature_cols: List[str]) -> None:
    """Raise if any feature is on the forbidden list. Call before every fit()."""
    leaked = [c for c in feature_cols if c in set(FORBIDDEN_AT_CREATION)]
    if leaked:
        raise ValueError(
            "Temporal leakage — these fields are not available at creation time: "
            + ", ".join(leaked)
        )


def build_text(df: pd.DataFrame, columns: Tuple[str, ...] = ("subject", "description")) -> pd.Series:
    """Concatenate creation-time text columns into one string per row."""
    assert_no_leakage(list(columns))
    parts = [df[c].fillna("").astype(str) for c in columns if c in df.columns]
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    return out.str.strip()


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2,
                   time_col: str = "created_time") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/test by time: oldest (1-test_frac) train, newest test.

    Simulates production — a model is always trained on the past and applied to
    the future — and is the only honest split for time-ordered tickets.
    """
    ordered = df.sort_values(time_col).reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_frac))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
