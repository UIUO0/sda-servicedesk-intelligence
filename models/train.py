"""Train imputation classifiers for fields SDP often leaves empty.

For each (module, target column) in :data:`TARGETS`, rows where the target is
present become training data: mixed Arabic/English ticket text is vectorised
with character n-gram TF-IDF (robust across both scripts without a tokenizer)
and fed to a logistic regression. Artifacts and evaluation metrics land in
``models/artifacts/`` so :mod:`models.predict` can later fill the gaps and the
dashboard can disclose model quality.

A model is only trained when there is enough signal to be honest about
(:data:`MIN_LABELED` labeled rows, ≥2 classes). Metrics are computed on a
held-out stratified split, never on training data.

Usage::

    python -m models.train                 # train everything trainable
    python -m models.train --modules requests
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")

ARTIFACTS_DIR = config.MODELS_DIR / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

MIN_LABELED = 100          # fewer labeled rows than this -> refuse to train
MIN_CLASS_SIZE = 10        # classes rarer than this are folded into "__other__"


@dataclass(frozen=True)
class TargetSpec:
    """One imputation target: which column to predict from which text."""

    module: str
    column: str
    text_columns: Tuple[str, ...]


# Fields observed to be frequently null in this SDP instance, predicted from
# the ticket's free text plus its categorical context.
TARGETS: List[TargetSpec] = [
    TargetSpec("requests", "priority", ("subject", "short_description", "template", "group")),
    TargetSpec("requests", "group", ("subject", "short_description", "template")),
    TargetSpec("problems", "category", ("title", "description")),
    TargetSpec("problems", "priority", ("title", "description")),
]


def build_text(df: pd.DataFrame, text_columns: Tuple[str, ...]) -> pd.Series:
    """Concatenate the available text/context columns into one string per row."""
    parts = [
        df[col].fillna("").astype(str)
        for col in text_columns
        if col in df.columns
    ]
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    out = parts[0]
    for part in parts[1:]:
        out = out + " " + part
    return out.str.strip()


def make_pipeline() -> Pipeline:
    """Char n-gram TF-IDF + logistic regression; language-agnostic (AR/EN)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                  min_df=2, max_features=200_000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def train_target(df: pd.DataFrame, spec: TargetSpec) -> Optional[Dict]:
    """Train one target; save artifact and return its metrics entry (or None)."""
    if spec.column not in df.columns:
        logger.info("[%s.%s] column absent — skipped", spec.module, spec.column)
        return None

    labeled = df[df[spec.column].notna()].copy()
    if len(labeled) < MIN_LABELED:
        logger.info("[%s.%s] only %d labeled rows (<%d) — skipped",
                    spec.module, spec.column, len(labeled), MIN_LABELED)
        return None

    y = labeled[spec.column].astype(str)
    # Fold classes too rare to learn or stratify on into a bucket the
    # predictor can treat as "no confident answer".
    counts = y.value_counts()
    rare = counts[counts < MIN_CLASS_SIZE].index
    y = y.where(~y.isin(rare), "__other__")
    if y.nunique() < 2:
        logger.info("[%s.%s] fewer than 2 usable classes — skipped", spec.module, spec.column)
        return None

    X = build_text(labeled, spec.text_columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    entry = {
        "module": spec.module,
        "column": spec.column,
        "text_columns": list(spec.text_columns),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_labeled": int(len(labeled)),
        "n_classes": int(y.nunique()),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACTS_DIR / f"{spec.module}__{spec.column}.joblib"
    joblib.dump({"pipeline": pipe, "spec": {
        "module": spec.module, "column": spec.column, "text_columns": list(spec.text_columns),
    }}, artifact)
    logger.info("[%s.%s] trained on %d rows, %d classes — acc %.3f, macro-F1 %.3f -> %s",
                spec.module, spec.column, len(labeled), y.nunique(),
                entry["accuracy"], entry["macro_f1"], artifact.name)
    return entry


def load_processed(module: str) -> Optional[pd.DataFrame]:
    path = config.PROCESSED_DIR / f"{module}.parquet"
    if not path.exists():
        csv = config.PROCESSED_DIR / f"{module}.csv"
        if not csv.exists():
            return None
        return pd.read_csv(csv)
    return pd.read_parquet(path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train imputation models for missing SDP fields.")
    parser.add_argument("--modules", nargs="*", default=None,
                        help="Restrict to these modules (default: all in TARGETS).")
    args = parser.parse_args(argv)

    metrics: List[Dict] = []
    tables: Dict[str, Optional[pd.DataFrame]] = {}
    for spec in TARGETS:
        if args.modules and spec.module not in args.modules:
            continue
        if spec.module not in tables:
            tables[spec.module] = load_processed(spec.module)
            if tables[spec.module] is None:
                logger.warning("[%s] no processed table — run preprocess first", spec.module)
        df = tables[spec.module]
        if df is None:
            continue
        entry = train_target(df, spec)
        if entry:
            metrics.append(entry)

    if metrics:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        existing = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else []
        merged = {(m["module"], m["column"]): m for m in existing}
        merged.update({(m["module"], m["column"]): m for m in metrics})
        METRICS_PATH.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=2),
                                encoding="utf-8")
        logger.info("Wrote metrics for %d model(s) -> %s", len(metrics), METRICS_PATH)
    else:
        logger.info("Nothing trained (not enough labeled data yet).")


if __name__ == "__main__":
    main()
