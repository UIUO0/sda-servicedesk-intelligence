"""Fill missing SDP fields with model predictions — clearly marked as such.

Applies the artifacts trained by :mod:`models.train` to the processed tables.
For every imputed column the output carries provenance, so a predicted value
can never masquerade as system data:

* ``{col}``            — original value, or the model's prediction where null
* ``{col}_source``     — ``"system"`` | ``"predicted"`` | ``"missing"``
* ``{col}_confidence`` — model probability for predicted rows (else NaN)

Predictions of the ``__other__`` bucket (rare classes folded at train time)
are treated as "no confident answer" and left missing.

Output: ``data/processed/{module}_enriched.parquet`` / ``.csv``.

Usage::

    python -m models.predict
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Optional

import joblib
import pandas as pd

import config
from models.train import ARTIFACTS_DIR, build_text, load_processed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("predict")


def apply_artifact(df: pd.DataFrame, artifact_path) -> Optional[str]:
    """Impute one column in-place from a saved artifact. Returns the column name."""
    bundle = joblib.load(artifact_path)
    pipe = bundle["pipeline"]
    spec = bundle["spec"]
    col = spec["column"]
    if col not in df.columns:
        return None

    source_col, conf_col = f"{col}_source", f"{col}_confidence"
    df[source_col] = "system"
    df.loc[df[col].isna(), source_col] = "missing"
    df[conf_col] = pd.NA

    missing = df[df[col].isna()]
    if missing.empty:
        logger.info("[%s.%s] nothing missing — no predictions needed", spec["module"], col)
        return col

    X = build_text(missing, tuple(spec["text_columns"]))
    proba = pipe.predict_proba(X)
    labels = pipe.classes_[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)

    filled = 0
    for idx, label, conf in zip(missing.index, labels, confidence):
        if label == "__other__":
            continue  # the model has no confident, nameable answer
        df.loc[idx, col] = label
        df.loc[idx, source_col] = "predicted"
        df.loc[idx, conf_col] = round(float(conf), 4)
        filled += 1

    logger.info("[%s.%s] filled %d of %d missing values (marked source=predicted)",
                spec["module"], col, filled, len(missing))
    return col


def enrich_module(module: str) -> bool:
    """Apply every artifact for ``module``; write the *_enriched table."""
    artifacts = sorted(ARTIFACTS_DIR.glob(f"{module}__*.joblib"))
    if not artifacts:
        logger.info("[%s] no trained models — run `python -m models.train` first", module)
        return False
    df = load_processed(module)
    if df is None:
        logger.warning("[%s] no processed table — run preprocess first", module)
        return False

    touched = [apply_artifact(df, a) for a in artifacts]
    if not any(touched):
        return False

    out = config.PROCESSED_DIR / f"{module}_enriched"
    df.to_parquet(out.with_suffix(".parquet"), index=False)
    df.to_csv(out.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    logger.info("[%s] wrote enriched table (%d rows) -> %s.parquet", module, len(df), out.name)
    return True


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Impute missing fields using trained models.")
    parser.add_argument("--modules", nargs="*", default=["requests", "problems"],
                        help="Modules to enrich (default: requests problems).")
    args = parser.parse_args(argv)
    if not ARTIFACTS_DIR.exists():
        logger.info("No artifacts directory — nothing to apply. Train models first.")
        return
    for module in args.modules:
        enrich_module(module)


if __name__ == "__main__":
    main()
