"""Neon operational dashboard for ServiceDesk Plus data (Streamlit + Plotly).

Reads the processed tables written by :mod:`pipeline.preprocess` and presents
operational KPIs and interactive charts with an animated neon theme.

Run with::

    streamlit run src/dashboard.py

The data-shaping helpers (``load_table``, ``filter_requests``, ``kpis``,
``value_counts``, ``volume_over_time``) are pure functions kept apart from the
Streamlit calls so they can be unit-tested without a running server.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px

import config
from insights.dashboard_theme import (
    CATEGORICAL,
    CUSTOM_CSS,
    STATUS,
    kpi_card_html,
    register_plotly_template,
)

CLOSED_STATUSES = {"Closed", "Resolved"}


# --- pure data helpers -----------------------------------------------------
def load_table(module: str) -> Optional[pd.DataFrame]:
    """Load a processed table (parquet preferred, CSV fallback)."""
    parquet = config.PROCESSED_DIR / f"{module}.parquet"
    csv = config.PROCESSED_DIR / f"{module}.csv"
    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        return None
    for col in ("created_time", "due_by_time", "reported_time", "closed_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def filter_requests(
    df: pd.DataFrame,
    *,
    groups: Optional[List[str]] = None,
    statuses: Optional[List[str]] = None,
    sites: Optional[List[str]] = None,
    templates: Optional[List[str]] = None,
    date_range: Optional[tuple] = None,
) -> pd.DataFrame:
    """Apply sidebar filters to the requests table. Empty/None = no filter."""
    out = df
    if groups:
        out = out[out["group"].isin(groups)]
    if statuses:
        out = out[out["status"].isin(statuses)]
    if sites:
        out = out[out["site"].isin(sites)]
    if templates:
        out = out[out["template"].isin(templates)]
    if date_range and "created_time" in out.columns:
        start, end = date_range
        ts = out["created_time"]
        out = out[(ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end) + pd.Timedelta(days=1))]
    return out


def kpis(df: pd.DataFrame) -> Dict[str, str]:
    """Compute operational KPI strings for the requests table."""
    total = len(df)
    status = df["status"] if "status" in df.columns else pd.Series(dtype=object)
    closed = int(status.isin(CLOSED_STATUSES).sum())
    open_cnt = total - closed
    overdue = df["is_overdue"].fillna(False).mean() * 100 if "is_overdue" in df.columns else 0
    groups = df["group"].nunique() if "group" in df.columns else 0
    res = pd.to_numeric(df.get("resolution_hours"), errors="coerce").dropna() if "resolution_hours" in df.columns else pd.Series(dtype=float)
    avg_res = f"{res.mean():.1f}h" if not res.empty else "—"
    return {
        "total": f"{total:,}",
        "open": f"{open_cnt:,}",
        "closed_pct": f"{(closed / total * 100 if total else 0):.0f}%",
        "overdue": f"{overdue:.0f}%",
        "groups": f"{groups:,}",
        "avg_res": avg_res,
    }


def value_counts(df: pd.DataFrame, column: str, top: int = 12) -> pd.DataFrame:
    """Return a tidy count table for a categorical column."""
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])
    vc = df[column].fillna("—").value_counts().head(top).reset_index()
    vc.columns = [column, "count"]
    return vc


def volume_over_time(df: pd.DataFrame, time_col: str = "created_time", freq: str = "D") -> pd.DataFrame:
    """Aggregate ticket counts per time bucket."""
    if time_col not in df.columns or df[time_col].isna().all():
        return pd.DataFrame(columns=["period", "count"])
    ts = df.dropna(subset=[time_col]).set_index(time_col).resample(freq).size()
    out = ts.reset_index()
    out.columns = ["period", "count"]
    return out


# --- Streamlit app (import guarded so helpers stay testable) ---------------
def _run() -> None:
    import streamlit as st

    st.set_page_config(page_title="SDA ServiceDesk Intelligence", page_icon="⚡", layout="wide")
    template = register_plotly_template()

    # -- reusable chart helpers (closures over the registered template) -----
    def bar(df: pd.DataFrame, column: str, title: str, top: int = 12) -> None:
        st.markdown(f"### {title}")
        vc = value_counts(df, column, top=top)
        if vc.empty:
            st.caption("No data for this view.")
            return
        fig = px.bar(vc, x="count", y=column, orientation="h", template=template,
                     color=column, color_discrete_sequence=CATEGORICAL, text="count")
        fig.update_layout(showlegend=False, yaxis=dict(categoryorder="total ascending", title=None))
        fig.update_traces(hovertemplate="%{y}: %{x}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    def donut(df: pd.DataFrame, column: str, title: str) -> None:
        st.markdown(f"### {title}")
        vc = value_counts(df, column)
        if vc.empty:
            st.caption("No data for this view.")
            return
        fig = px.pie(vc, names=column, values="count", hole=0.55, template=template,
                     color_discrete_sequence=CATEGORICAL)
        fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{value}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    def kpi_row(cards: list) -> None:
        cols = st.columns(len(cards))
        for i, (col, (label, value, glow)) in enumerate(zip(cols, cards)):
            col.markdown(kpi_card_html(label, value, glow, delay=i * 0.08), unsafe_allow_html=True)

    def data_table(df: pd.DataFrame, name: str) -> None:
        with st.expander("View data table"):
            st.dataframe(df, use_container_width=True)
            st.download_button(f"Download {name} CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{name}.csv", mime="text/csv", key=f"dl_{name}")

    # -- header --
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="neon-title">⚡ ServiceDesk Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="neon-sub">Operational KPIs · ServiceDesk Plus</div>', unsafe_allow_html=True)

    requests = load_table("requests")
    if requests is None or requests.empty:
        st.warning("No processed data found. Run the extractor then `python -m pipeline.preprocess`.")
        return

    tab_req, tab_prob, tab_chg, tab_proj, tab_sol = st.tabs(
        ["🎫 Requests", "🛠️ Problems", "🔄 Changes", "📁 Projects", "📚 Solutions (KB)"]
    )

    # ================= REQUESTS =================
    with tab_req:
        st.sidebar.header("Filters · Requests")
        groups = st.sidebar.multiselect("Group", sorted(requests["group"].dropna().unique()))
        services = st.sidebar.multiselect("Service", sorted(requests["template"].dropna().unique()))
        statuses = st.sidebar.multiselect("Status", sorted(requests["status"].dropna().unique()))
        sites = (st.sidebar.multiselect("Site", sorted(requests["site"].dropna().unique()))
                 if "site" in requests else [])
        date_range = None
        if "created_time" in requests and requests["created_time"].notna().any():
            mn, mx = requests["created_time"].min().date(), requests["created_time"].max().date()
            picked = st.sidebar.date_input("Created between", (mn, mx), min_value=mn, max_value=mx)
            if isinstance(picked, tuple) and len(picked) == 2:
                date_range = picked

        df = filter_requests(requests, groups=groups, statuses=statuses, sites=sites,
                             templates=services, date_range=date_range)
        k = kpis(df)
        kpi_row([
            ("Total Tickets", k["total"], "cyan"), ("Open", k["open"], "magenta"),
            ("Closed", k["closed_pct"], "cyan"), ("Overdue (SLA)", k["overdue"], "magenta"),
            ("Avg Resolution", k["avg_res"], "purple"),
            ("Services", f'{requests["template"].nunique():,}', "purple"),
        ])

        st.markdown("### Ticket volume over time")
        vol = volume_over_time(df, freq="D")
        if not vol.empty:
            fig = px.area(vol, x="period", y="count", template=template)
            fig.update_traces(line_color=CATEGORICAL[0], fillcolor="rgba(57,135,229,0.25)",
                              mode="lines+markers", hovertemplate="%{x|%d %b}<br>%{y} tickets<extra></extra>")
            fig.update_layout(xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            bar(df, "template", "By service (catalog)")
        with c2:
            bar(df, "group", "By team")
        c3, c4 = st.columns(2)
        with c3:
            bar(df, "technician_name", "By technician (workload)")
        with c4:
            bar(df, "status", "By status")
        c5, c6 = st.columns(2)
        with c5:
            donut(df, "lang", "Language mix")
        with c6:
            bar(df, "site", "By site")
        data_table(df, "requests_filtered")

    # ================= PROBLEMS =================
    with tab_prob:
        problems = load_table("problems")
        if problems is None or problems.empty:
            st.info("No problems data yet. Run `python -m pipeline.extract --modules problems` then preprocess.")
        else:
            resolved = int(problems["status"].isin(CLOSED_STATUSES).sum()) if "status" in problems else 0
            has_res = problems["has_resolution"].fillna(False).mean() * 100 if "has_resolution" in problems else 0
            kpi_row([
                ("Total Problems", f"{len(problems):,}", "cyan"),
                ("Resolved/Closed", f"{resolved:,}", "magenta"),
                ("With Resolution", f"{has_res:.0f}%", "purple"),
            ])
            c1, c2 = st.columns(2)
            with c1:
                bar(problems, "category", "By category")
            with c2:
                bar(problems, "status", "By status")
            c3, c4 = st.columns(2)
            with c3:
                bar(problems, "group", "By team")
            with c4:
                bar(problems, "technician_name", "By technician")
            data_table(problems, "problems")

    # ============ CHANGES / PROJECTS / SOLUTIONS (generic) ============
    def generic_tab(module: str, label: str) -> None:
        table = load_table(module)
        if table is None or table.empty:
            st.info(f"No {label} data yet. Run `python -m pipeline.extract --modules {module}` then preprocess.")
            return
        kpi_row([("Total", f"{len(table):,}", "cyan"),
                 ("Columns", f"{table.shape[1]:,}", "purple")])
        for col in ("status", "category", "priority", "group", "technician_name"):
            if col in table.columns and table[col].notna().any():
                bar(table, col, f"By {col.replace('_', ' ')}")
        data_table(table, module)

    with tab_chg:
        generic_tab("changes", "changes")
    with tab_proj:
        generic_tab("projects", "projects")
    with tab_sol:
        generic_tab("solutions", "solutions (KB)")


if __name__ == "__main__":
    _run()
