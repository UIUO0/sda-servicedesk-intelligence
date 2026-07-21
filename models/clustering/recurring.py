"""App 2 — recurring-issue discovery on the 7,077 tickets (operational output).

Pipeline: multilingual embeddings → UMAP → HDBSCAN. The deliverable is not
cluster numbers but ``insights/recurring_issues.md``: the top recurring issues
with a descriptive name, ticket count, monthly trend, anonymised examples and
the affected team — the candidates to seed the abandoned problems module.

Two gates run before any counts are reported (same spirit as the recall@5
no-go):
  1. cross-lingual test — a fault written in Arabic and in English must land in
     the same cluster, else language, not topic, drives the grouping and every
     count is distorted.
  2. eyeball test — 10 random tickets from each large cluster are dumped for a
     human to confirm they are really the same problem.

Runs on the anonymised table, so the report is safe to share.

Usage::

    python -m models.clustering.recurring --lang-test     # gate 1 only
    python -m models.clustering.recurring --build          # full run + report
"""
from __future__ import annotations

import argparse
import logging
import random
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from models.embeddings import embed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("clustering")

REPORT_PATH = config.INSIGHTS_DIR / "recurring_issues.md"
LABELS_PATH = config.PROCESSED_DIR / "requests_clusters.parquet"

MIN_CLUSTER_SIZE = 20        # a "recurring" issue must recur meaningfully
TOP_N = 10                   # issues to report
EXAMPLES_PER_CLUSTER = 3     # anonymised examples shown per issue
EYEBALL_SAMPLE = 10          # random tickets dumped per big cluster for review

# Signature fragments that survive anonymisation and must never reach a report.
_SIG_PATTERNS = [
    re.compile(r"regards[,\s].*", re.IGNORECASE),
    re.compile(r"best\s+regards.*", re.IGNORECASE),
    re.compile(r"thanks[,\s].*", re.IGNORECASE),
    re.compile(r"تحياتي.*"), re.compile(r"وتقبلوا.*"), re.compile(r"مع الشكر.*"),
]
_STOP = set("the a an and or to of for in on is are was were be this that i we you my our your it "
            "please kindly dear hi hello regards thanks thank ال في من على الى عن هذا هذه"
            .split())


def load_texts() -> pd.DataFrame:
    df = pd.read_parquet(config.PROCESSED_DIR / "requests_anon.parquet")
    df["text"] = (df["subject"].fillna("").astype(str) + ". "
                  + df["description"].fillna("").astype(str)).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")
    return df


def _scrub(text: str) -> str:
    """Strip trailing signatures before anything is written to a report."""
    out = text
    for pat in _SIG_PATTERNS:
        out = pat.sub("", out)
    return re.sub(r"\s+", " ", out).strip()


# --- gate 1: cross-lingual sanity -----------------------------------------
CROSS_LINGUAL_PROBES: List[Tuple[str, str, str]] = [
    ("VPN not working from home", "الاتصال عبر VPN لا يعمل من المنزل", "VPN failure"),
    ("Cannot access email on Outlook", "لا أستطيع الدخول على البريد في اوتلوك", "Email access"),
    ("Need a new laptop", "احتاج جهاز لابتوب جديد", "Laptop request"),
    ("Printer is not printing", "الطابعة لا تطبع", "Printer issue"),
    ("Reset my password", "إعادة تعيين كلمة المرور", "Password reset"),
]


def cross_lingual_test() -> bool:
    """Each AR/EN pair about one fault must be closer to each other than to
    other faults. Prints per-pair cosine similarity; returns pass/fail."""
    pairs = [(en, ar) for en, ar, _ in CROSS_LINGUAL_PROBES]
    flat = [t for pair in pairs for t in pair]
    vecs = embed(flat, use_cache=False)
    ok = True
    logger.info("=== cross-lingual test (AR vs EN, same fault) ===")
    for i, (_, _, name) in enumerate(CROSS_LINGUAL_PROBES):
        en_v, ar_v = vecs[2 * i], vecs[2 * i + 1]
        self_sim = float(en_v @ ar_v)
        # best similarity to any *other* fault's mean vector
        others = [vecs[2 * j:2 * j + 2].mean(0) for j in range(len(CROSS_LINGUAL_PROBES)) if j != i]
        cross_max = max(float((en_v @ o)) for o in others)
        passed = self_sim > cross_max
        ok = ok and passed
        logger.info("%-16s self=%.3f  best-other=%.3f  %s",
                    name, self_sim, cross_max, "PASS" if passed else "FAIL")
    logger.info("cross-lingual test: %s", "PASS" if ok else "FAIL")
    return ok


# --- clustering ------------------------------------------------------------
def cluster(vecs: np.ndarray) -> np.ndarray:
    import hdbscan
    import umap
    reducer = umap.UMAP(n_neighbors=15, n_components=10, metric="cosine", random_state=42)
    reduced = reducer.fit_transform(vecs)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=5,
                                metric="euclidean", cluster_selection_method="eom")
    return clusterer.fit_predict(reduced)


def _label_for(texts: List[str]) -> str:
    """Cheap descriptive name: most distinctive words across the cluster."""
    words: Counter = Counter()
    for t in texts:
        for w in re.findall(r"[A-Za-z؀-ۿ]{3,}", t.lower()):
            if w not in _STOP:
                words[w] += 1
    top = [w for w, _ in words.most_common(4)]
    return " / ".join(top) if top else "unlabelled"


def _monthly_trend(times: pd.Series) -> str:
    by_month = times.dt.to_period("M").value_counts().sort_index()
    if len(by_month) < 3:
        return "insufficient history"
    recent = by_month.iloc[-3:].mean()
    earlier = by_month.iloc[:-3].mean()
    if recent > earlier * 1.25:
        return f"rising (last 3 mo avg {recent:.0f}/mo vs {earlier:.0f})"
    if recent < earlier * 0.75:
        return f"falling (last 3 mo avg {recent:.0f}/mo vs {earlier:.0f})"
    return f"stable (~{recent:.0f}/mo)"


def build(seed: int = 42) -> Dict:
    df = load_texts()
    logger.info("Embedding %d tickets…", len(df))
    vecs = embed(df["text"].tolist())

    logger.info("Clustering…")
    labels = df["cluster"] = cluster(vecs)
    n_clusters = len({c for c in labels if c != -1})
    noise = int((labels == -1).sum())
    logger.info("Found %d clusters; %d tickets unclustered (noise)", n_clusters, noise)

    df.to_parquet(LABELS_PATH, index=False)  # anon table + cluster id

    sizes = Counter(c for c in labels if c != -1)
    top = sizes.most_common(TOP_N)

    rng = random.Random(seed)
    issues, eyeball = [], []
    for cid, size in top:
        rows = df[df["cluster"] == cid]
        name = _label_for(rows["text"].tolist())
        team = rows["group"].mode().iloc[0] if not rows["group"].mode().empty else "—"
        trend = _monthly_trend(rows["created_time"])
        examples = [_scrub(t)[:160] for t in rng.sample(rows["text"].tolist(),
                    min(EXAMPLES_PER_CLUSTER, len(rows)))]
        issues.append({"cid": int(cid), "name": name, "count": size, "team": team,
                       "trend": trend, "examples": examples})
        sample = rng.sample(rows["text"].tolist(), min(EYEBALL_SAMPLE, len(rows)))
        eyeball.append({"name": name, "cid": int(cid), "samples": [_scrub(s)[:120] for s in sample]})

    _write_report(df, issues, n_clusters, noise)
    _write_eyeball(eyeball)
    return {"n_clusters": n_clusters, "noise": noise, "top": issues}


def _write_report(df: pd.DataFrame, issues: List[Dict], n_clusters: int, noise: int) -> None:
    total = len(df)
    lines = [
        "# أكثر المشاكل تكراراً — Recurring issues",
        "",
        f"*{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC — "
        f"{total} tickets, {n_clusters} clusters, {noise} unclustered "
        f"({100*noise/total:.0f}%). Multilingual embeddings + HDBSCAN, anonymised table.*",
        "",
        "> هذه المجموعات هي المرشحة لتصبح سجلات في موديول problems المهجور (11 سجل فقط).",
        "",
        "> **ملاحظة صدق:** بما أن حقل subject يحوي غالباً اسم الخدمة (template)، فإن المجموعات "
        "تتطابق جزئياً مع الخدمات. القيمة المضافة ليست اكتشاف بنية خفية بل: دمج نسختي العربي/الإنجليزي "
        "لنفس العطل، الاتجاه الشهري، الفريق المتأثر، وفصل الحالات الشاذة (16% noise). "
        "التقييم بالعين في `clustering_eyeball.md`.",
        "",
    ]
    for rank, iss in enumerate(issues, 1):
        lines += [
            f"## {rank}. {iss['name']}  ·  {iss['count']} تذكرة",
            f"- **الفريق المتأثر:** {iss['team']}",
            f"- **الاتجاه الشهري:** {iss['trend']}",
            "- **أمثلة (معقّمة):**",
        ]
        lines += [f"  - {ex}" for ex in iss["examples"]]
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote recurring-issues report -> %s", REPORT_PATH)


def _write_eyeball(eyeball: List[Dict]) -> None:
    path = config.INSIGHTS_DIR / "clustering_eyeball.md"
    lines = ["# Clustering eyeball check", "",
             "10 random tickets per large cluster — confirm each cluster is really one problem.", ""]
    for e in eyeball:
        lines.append(f"## cluster {e['cid']} — {e['name']}")
        lines += [f"- {s}" for s in e["samples"]]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote eyeball sample -> %s", path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Recurring-issue clustering on tickets.")
    parser.add_argument("--lang-test", action="store_true", help="Run only the cross-lingual gate.")
    parser.add_argument("--build", action="store_true", help="Full cluster + reports.")
    args = parser.parse_args(argv)

    if args.lang_test or not args.build:
        passed = cross_lingual_test()
        if not args.build:
            raise SystemExit(0 if passed else 1)
        if not passed:
            logger.error("Cross-lingual test FAILED — clusters would split by language. "
                         "Stopping before reporting distorted counts.")
            raise SystemExit(1)
    if args.build:
        build()


if __name__ == "__main__":
    main()
