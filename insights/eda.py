"""Exploratory data analysis over the processed SDP tables.

Reads ``data/processed/*.parquet`` (falling back to CSV), writes distribution
plots to ``data/processed/eda/`` and a human-readable ``eda_summary.md`` that
reports label quality and recommends which ML target is realistic to start with.

Usage::

    python -m insights.eda
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eda")

sns.set_theme(style="whitegrid")
EDA_DIR = config.INSIGHTS_DIR


def load_table(module: str) -> Optional[pd.DataFrame]:
    parquet = config.PROCESSED_DIR / f"{module}.parquet"
    csv = config.PROCESSED_DIR / f"{module}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    return None


def _save_countplot(df: pd.DataFrame, column: str, title: str, filename: str, top: int = 15) -> None:
    if column not in df.columns or df[column].dropna().empty:
        return
    counts = df[column].fillna("<null>").value_counts().head(top)
    plt.figure(figsize=(9, max(3, 0.4 * len(counts))))
    sns.barplot(x=counts.values, y=counts.index.astype(str), color="#3b7dd8")
    plt.title(title)
    plt.xlabel("count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / filename, dpi=110)
    plt.close()


def _save_histogram(df: pd.DataFrame, column: str, title: str, filename: str) -> None:
    if column not in df.columns:
        return
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.histplot(series, bins=30, color="#3b7dd8")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(EDA_DIR / filename, dpi=110)
    plt.close()


def _null_pct(df: pd.DataFrame, column: str) -> Optional[float]:
    if column not in df.columns:
        return None
    return round(100.0 * df[column].isna().mean(), 1)


def _md_value_counts(df: pd.DataFrame, column: str, top: int = 10) -> str:
    if column not in df.columns:
        return f"- `{column}`: (missing)\n"
    vc = df[column].fillna("<null>").value_counts().head(top)
    lines = [f"- `{column}` (top {min(top, len(vc))}):"]
    for value, count in vc.items():
        lines.append(f"  - {value}: {count}")
    return "\n".join(lines) + "\n"


def analyze_requests(df: pd.DataFrame, out: List[str]) -> None:
    out.append("## requests\n")
    out.append(f"- rows: **{len(df)}**\n")
    for col in ("group", "template", "status", "site", "lang"):
        _save_countplot(df, col, f"requests by {col}", f"requests_{col}.png")
    _save_histogram(df, "subject_len", "requests subject length", "requests_subject_len.png")

    if "is_overdue" in df.columns:
        overdue = df["is_overdue"].fillna(False).mean() * 100
        out.append(f"- overdue: **{overdue:.1f}%**\n")
    out.append("\n### Label-quality check (requests)\n")
    for col in ("priority", "group", "technician_name", "template", "category"):
        pct = _null_pct(df, col)
        note = ""
        if pct is not None and pct >= 40:
            note = "  ⚠️ weak label (too many nulls)"
        out.append(f"- `{col}` null: {pct}%{note}\n")
    out.append("\n**Distributions**\n")
    out.append(_md_value_counts(df, "group"))
    out.append(_md_value_counts(df, "template"))
    out.append(_md_value_counts(df, "lang"))


def analyze_problems(df: pd.DataFrame, out: List[str]) -> None:
    out.append("## problems\n")
    out.append(f"- rows: **{len(df)}**\n")
    for col in ("category", "status", "group", "priority", "lang"):
        _save_countplot(df, col, f"problems by {col}", f"problems_{col}.png")
    _save_histogram(df, "resolution_hours", "problems resolution hours", "problems_resolution_hours.png")

    if "has_resolution" in df.columns:
        out.append(f"- with usable resolution: **{df['has_resolution'].fillna(False).mean() * 100:.1f}%**\n")
    if "resolution_is_junk" in df.columns:
        out.append(f"- junk resolution values: **{int(df['resolution_is_junk'].fillna(False).sum())}**\n")
    out.append("\n### Label-quality check (problems)\n")
    for col in ("category", "subcategory", "priority", "impact", "urgency", "technician_name", "closed_time"):
        pct = _null_pct(df, col)
        note = "  ⚠️ weak label (too many nulls)" if (pct is not None and pct >= 40) else ""
        out.append(f"- `{col}` null: {pct}%{note}\n")
    out.append("\n**Distributions**\n")
    out.append(_md_value_counts(df, "category"))
    out.append(_md_value_counts(df, "lang"))


def analyze_generic(module: str, df: pd.DataFrame, out: List[str]) -> None:
    out.append(f"## {module}\n")
    out.append(f"- rows: **{len(df)}**, cols: {df.shape[1]}\n")
    for col in ("status", "category", "lang"):
        _save_countplot(df, col, f"{module} by {col}", f"{module}_{col}.png")


def recommend(tables: dict, out: List[str]) -> None:
    out.append("\n## Recommendation (which ML target to start with)\n")
    req = tables.get("requests")
    prob = tables.get("problems")

    if req is not None and len(req) > 0:
        group_null = _null_pct(req, "group") or 0
        template_null = _null_pct(req, "template") or 0
        n_groups = req["group"].nunique() if "group" in req.columns else 0
        if group_null < 40 and n_groups >= 2:
            out.append(
                f"- **Ticket classification (group/template)** looks viable: "
                f"group null {group_null}%, template null {template_null}%, "
                f"{n_groups} distinct groups. Recommended first model — a multilingual "
                f"(AR+EN) text classifier on subject+description.\n"
            )
        overdue_rate = req["is_overdue"].fillna(False).mean() if "is_overdue" in req.columns else 0
        if 0.05 < overdue_rate < 0.95:
            out.append(
                f"- **SLA-breach prediction** is a solid alternative: overdue rate "
                f"{overdue_rate * 100:.1f}% gives usable class balance.\n"
            )
        else:
            out.append(
                f"- SLA-breach prediction: caution — overdue rate is "
                f"{overdue_rate * 100:.1f}% (very imbalanced on this data).\n"
            )

    if prob is not None:
        out.append(
            f"- problems table is small ({len(prob)} rows) and resolution/category are "
            f"sparse — better suited to clustering/topic-modelling than supervised learning.\n"
        )
    out.append(
        "\n> Note: these figures reflect whatever data is present. Re-run after a full "
        "extract before committing to a modelling target.\n"
    )


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    out: List[str] = ["# ServiceDesk Plus — EDA summary\n"]
    tables = {}

    for module in ("requests", "problems", "changes", "projects", "solutions"):
        df = load_table(module)
        if df is None:
            continue
        tables[module] = df
        if module == "requests":
            analyze_requests(df, out)
        elif module == "problems":
            analyze_problems(df, out)
        else:
            analyze_generic(module, df, out)
        out.append("\n")

    if not tables:
        logger.warning("No processed tables found in %s. Run preprocess first.", config.PROCESSED_DIR)
        return

    recommend(tables, out)
    summary_path = EDA_DIR / "eda_summary.md"
    summary_path.write_text("".join(out), encoding="utf-8")
    logger.info("Wrote EDA summary -> %s and plots -> %s", summary_path, EDA_DIR)


if __name__ == "__main__":
    main()
