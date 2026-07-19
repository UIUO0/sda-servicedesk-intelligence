"""Semantic retrieval of past resolutions for new tickets — with honest eval.

Index: closed/resolved tickets that carry a usable resolution text. A query
(ticket subject + description) returns the k most similar past tickets and
their resolutions.

v1 uses character n-gram TF-IDF (language-agnostic across the corpus's
Arabic/English mix; the same recipe reached 87% accuracy on team
classification). If evaluation says it is not enough, the upgrade path is
multilingual sentence embeddings behind the same interface.

Evaluation protocol (reviewer-specified, decides go/no-go before any UI):
hold out N closed tickets, hide each one's resolution, query with its text
against an index that excludes it, and count a hit when any of the top-k
retrieved tickets carries an equivalent resolution (cosine similarity of the
resolution texts >= a threshold). Reports recall@1 / recall@k at several
thresholds plus a human-checkable examples file.

The corpus is the **anonymised** table, so evaluation artefacts are safe to
share.

Usage::

    python -m models.similar --build
    python -m models.similar --query "VPN لا يعمل من المنزل"
    python -m models.similar --evaluate            # 50 held-out tickets, k=5
"""
from __future__ import annotations

import argparse
import logging
import random
from datetime import datetime, timezone
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("similar")

INDEX_PATH = config.MODELS_DIR / "artifacts" / "similar_index.joblib"
EVAL_REPORT = config.INSIGHTS_DIR / "semantic_search_eval.md"

CLOSED_STATUSES = {"Closed", "Resolved"}
MIN_RESOLUTION_CHARS = 20  # boilerplate like "تم" / "Done" can't count as a hit


def load_corpus() -> pd.DataFrame:
    """Closed tickets with a usable resolution, from the anonymised table."""
    path = config.PROCESSED_DIR / "requests_anon.parquet"
    df = pd.read_parquet(path)
    df = df[df["status"].isin(CLOSED_STATUSES)]
    res = df["resolution"].fillna("").astype(str).str.strip()
    df = df[res.str.len() >= MIN_RESOLUTION_CHARS].copy()
    df["query_text"] = (
        df["subject"].fillna("").astype(str) + " " + df["description"].fillna("").astype(str)
    ).str.strip()
    df = df[df["query_text"].str.len() > 0]
    return df.reset_index(drop=True)


def _make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                           min_df=2, max_features=300_000, sublinear_tf=True)


def build_index() -> dict:
    corpus = load_corpus()
    if corpus.empty:
        raise SystemExit("No resolvable corpus — run preprocess first.")
    vectorizer = _make_vectorizer()
    matrix = vectorizer.fit_transform(corpus["query_text"])
    bundle = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "ids": corpus["id"].tolist(),
        "subjects": corpus["subject"].fillna("").tolist(),
        "resolutions": corpus["resolution"].fillna("").tolist(),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, INDEX_PATH)
    logger.info("Index built over %d resolved tickets -> %s", len(corpus), INDEX_PATH.name)
    return bundle


def query_index(text: str, k: int = 5, bundle: Optional[dict] = None) -> List[dict]:
    bundle = bundle or joblib.load(INDEX_PATH)
    vec = bundle["vectorizer"].transform([text])
    sims = linear_kernel(vec, bundle["matrix"]).ravel()
    order = np.argsort(-sims)[:k]
    return [
        {
            "id": bundle["ids"][i],
            "subject": bundle["subjects"][i],
            "resolution": bundle["resolutions"][i],
            "similarity": round(float(sims[i]), 4),
        }
        for i in order
    ]


def evaluate(n: int = 50, k: int = 5, seed: int = 42) -> dict:
    """Reviewer protocol: hidden resolutions, leave-one-out, recall@k."""
    corpus = load_corpus()
    logger.info("Corpus: %d resolved tickets with usable resolutions", len(corpus))

    vectorizer = _make_vectorizer()
    matrix = vectorizer.fit_transform(corpus["query_text"])

    # Separate space to judge "is this retrieved resolution equivalent to the
    # hidden one" — word-level so the judge differs from the retriever.
    res_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    res_matrix = res_vec.fit_transform(corpus["resolution"].fillna(""))

    rng = random.Random(seed)
    holdout = rng.sample(range(len(corpus)), min(n, len(corpus)))
    thresholds = (0.5, 0.6, 0.8)
    hits_at_k = {t: 0 for t in thresholds}
    hits_at_1 = {t: 0 for t in thresholds}
    examples = []

    for row in holdout:
        sims = linear_kernel(matrix[row], matrix).ravel()
        sims[row] = -1.0  # leave-one-out: the ticket must not retrieve itself
        top = np.argsort(-sims)[:k]
        res_sims = linear_kernel(res_matrix[row], res_matrix[top]).ravel()
        best_j = int(np.argmax(res_sims))
        for t in thresholds:
            if res_sims.max() >= t:
                hits_at_k[t] += 1
            if res_sims[0] >= t:
                hits_at_1[t] += 1
        examples.append({
            "id": corpus["id"].iloc[row],
            "subject": str(corpus["subject"].iloc[row])[:90],
            "hidden_resolution": str(corpus["resolution"].iloc[row])[:140],
            "best_match_subject": str(corpus["subject"].iloc[top[best_j]])[:90],
            "best_match_resolution": str(corpus["resolution"].iloc[top[best_j]])[:140],
            "resolution_similarity": round(float(res_sims.max()), 3),
        })

    n_eval = len(holdout)
    results = {
        "n": n_eval, "k": k, "corpus_size": len(corpus),
        "recall_at_k": {t: round(hits_at_k[t] / n_eval, 3) for t in thresholds},
        "recall_at_1": {t: round(hits_at_1[t] / n_eval, 3) for t in thresholds},
    }
    _write_report(results, examples)
    for t in thresholds:
        logger.info("threshold %.1f -> recall@%d: %.1f%% | recall@1: %.1f%%",
                    t, k, 100 * results["recall_at_k"][t], 100 * results["recall_at_1"][t])
    return results


def _write_report(results: dict, examples: List[dict]) -> None:
    lines = [
        "# Semantic search — hidden-resolution evaluation",
        "",
        f"*{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC — "
        f"corpus: {results['corpus_size']} resolved tickets (anonymised table), "
        f"holdout: {results['n']}, k={results['k']}, retriever: char TF-IDF*",
        "",
        "Protocol: each held-out ticket's resolution is hidden; its text queries an index",
        "that excludes it; a hit = any top-k neighbour's resolution is equivalent to the",
        "hidden one (word TF-IDF cosine ≥ threshold).",
        "",
        "| equivalence threshold | recall@1 | recall@" + str(results["k"]) + " |",
        "|---|---|---|",
    ]
    for t in sorted(results["recall_at_k"]):
        lines.append(f"| {t} | {results['recall_at_1'][t]:.0%} | {results['recall_at_k'][t]:.0%} |")
    lines += ["", "## Held-out examples (human check)", ""]
    for ex in examples:
        lines += [
            f"**#{ex['id']}** — {ex['subject']}",
            f"- hidden: {ex['hidden_resolution']}",
            f"- best match: {ex['best_match_subject']} → {ex['best_match_resolution']}",
            f"- resolution similarity: {ex['resolution_similarity']}",
            "",
        ]
    EVAL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote evaluation report -> %s", EVAL_REPORT)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Similar-ticket retrieval with hidden-resolution eval.")
    parser.add_argument("--build", action="store_true", help="Build/rebuild the retrieval index.")
    parser.add_argument("--query", type=str, default=None, help="Free-text query; prints top-k matches.")
    parser.add_argument("--evaluate", action="store_true", help="Run the recall@k protocol.")
    parser.add_argument("--n", type=int, default=50, help="Holdout size for --evaluate.")
    parser.add_argument("--k", type=int, default=5, help="Neighbours per query.")
    args = parser.parse_args(argv)

    if args.build:
        build_index()
    if args.evaluate:
        evaluate(n=args.n, k=args.k)
    if args.query:
        for hit in query_index(args.query, k=args.k):
            print(f"[{hit['similarity']:.3f}] #{hit['id']} {hit['subject'][:70]}")
            print(f"    -> {hit['resolution'][:160]}")
    if not (args.build or args.evaluate or args.query):
        parser.print_help()


if __name__ == "__main__":
    main()
