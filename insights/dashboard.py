"""Operational dashboard for ServiceDesk Plus data (Streamlit, enterprise style).

Reads the processed tables written by :mod:`pipeline.preprocess` and presents
operational KPIs and charts using native Streamlit components only — theming
lives in ``.streamlit/config.toml``, no custom CSS or HTML.

Run with::

    streamlit run insights/dashboard.py

The data-shaping helpers (``load_table``, ``filter_requests``, ``kpis``,
``value_counts``, ``volume_over_time``) are pure functions kept apart from the
Streamlit calls so they can be unit-tested without a running server.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

import config

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


def split_predicted(df: pd.DataFrame, column: str) -> tuple:
    """Return (system-only rows, number of excluded predicted rows) for a column.

    Charts must never mix model-predicted values with system data — a footnote
    is not enough to stop a chart from misleading. Predictions stay available
    in the data table / exported CSV, clearly flagged by ``{column}_source``.
    """
    src = f"{column}_source"
    if src not in df.columns:
        return df, 0
    mask = df[src].eq("predicted")
    excluded = int(mask.sum())
    return (df[~mask], excluded) if excluded else (df, 0)


# --- Streamlit app (import guarded so helpers stay testable) ---------------
def _run() -> None:
    import subprocess
    import sys

    import streamlit as st

    from pipeline.refresh import read_status

    st.set_page_config(
        page_title="ServiceDesk intelligence",
        page_icon=":material/analytics:",
        layout="wide",
    )

    load = st.cache_data(ttl="5m", show_spinner=False)(load_table)

    def load_best(module: str) -> Optional[pd.DataFrame]:
        """Prefer the model-enriched table (carries *_source provenance)."""
        df = load(f"{module}_enriched")
        return df if df is not None else load(module)

    # -- reusable building blocks -------------------------------------------
    def count_chart(df: pd.DataFrame, column: str, title: str, top: int = 12) -> None:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            plot_df, excluded = split_predicted(df, column)
            vc = value_counts(plot_df, column, top=top)
            if vc.empty:
                st.caption("No data for this view.")
                return
            st.bar_chart(
                vc, x=column, y="count", horizontal=True, sort=False,
                x_label="", y_label="", height=max(180, 34 * len(vc)),
            )
            if excluded:
                st.caption(
                    f"System-entered values only ({len(plot_df):,} of {len(df):,} records). "
                    f"{excluded:,} model-predicted values are excluded from this chart "
                    "and available in the data table."
                )

    def data_table(df: pd.DataFrame, name: str) -> None:
        with st.expander("Data table", icon=":material/table_chart:"):
            st.dataframe(df, hide_index=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{name}.csv",
                mime="text/csv",
                icon=":material/download:",
                key=f"dl_{name}",
            )

    # -- header --------------------------------------------------------------
    st.title("ServiceDesk intelligence")
    st.caption("Operational overview · ManageEngine ServiceDesk Plus")

    # -- data refresh (sidebar) ----------------------------------------------
    with st.sidebar:
        st.subheader("Data")
        status = read_status()
        state = status.get("state")
        refreshing = state == "running"
        if refreshing:
            st.caption(f":material/progress_activity: Refreshing — step: {status.get('step') or '…'}")
        elif state == "done" and status.get("finished"):
            stamp = pd.to_datetime(status["finished"]).strftime("%b %d, %H:%M")
            st.caption(f":material/check_circle: Last refresh: {stamp} UTC")
        elif state == "failed":
            st.caption(f":material/error: Last refresh failed: {str(status.get('error'))[:90]}")

        if st.button("Refresh data", icon=":material/refresh:", disabled=refreshing):
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            log = open(config.DATA_DIR / "refresh.log", "ab")
            subprocess.Popen(
                [sys.executable, "-m", "pipeline.refresh"],
                cwd=str(config.PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            st.toast("Refresh started in the background. This can take a while "
                     "(the server throttles bulk listing).")
            import time as _time
            _time.sleep(1.5)  # let the child write its status file before rerun
            st.rerun()

        # When a refresh lands, drop cached tables once so new data shows up.
        if state == "done" and status.get("finished") and \
                st.session_state.get("_seen_refresh") != status["finished"]:
            st.session_state["_seen_refresh"] = status["finished"]
            load.clear()

    requests = load_best("requests")
    if requests is None or requests.empty:
        st.warning(
            "No processed data found. Use the **Refresh data** button, or run "
            "`python -m pipeline.refresh` from a terminal.",
            icon=":material/database:",
        )
        return

    tab_req, tab_prob, tab_chg, tab_proj, tab_sol = st.tabs([
        ":material/confirmation_number: Requests",
        ":material/build: Problems",
        ":material/change_circle: Changes",
        ":material/folder: Projects",
        ":material/menu_book: Solutions",
    ])

    # ================= REQUESTS =================
    with tab_req:
        with st.sidebar:
            st.subheader("Filters")
            groups = st.multiselect("Team", sorted(requests["group"].dropna().unique()))
            services = st.multiselect("Service", sorted(requests["template"].dropna().unique()))
            statuses = st.multiselect("Status", sorted(requests["status"].dropna().unique()))
            sites = (st.multiselect("Site", sorted(requests["site"].dropna().unique()))
                     if "site" in requests else [])
            date_range = None
            if "created_time" in requests and requests["created_time"].notna().any():
                mn = requests["created_time"].min().date()
                mx = requests["created_time"].max().date()
                picked = st.date_input("Created between", (mn, mx), min_value=mn, max_value=mx)
                if isinstance(picked, tuple) and len(picked) == 2:
                    date_range = picked

        df = filter_requests(requests, groups=groups, statuses=statuses, sites=sites,
                             templates=services, date_range=date_range)
        k = kpis(df)

        daily = volume_over_time(df, freq="D")["count"].tolist()
        with st.container(horizontal=True):
            st.metric("Total tickets", k["total"], border=True,
                      chart_data=daily, chart_type="area")
            st.metric("Open", k["open"], border=True)
            st.metric("Closed", k["closed_pct"], border=True)
            st.metric("Overdue (SLA)", k["overdue"], border=True)
            st.metric("Avg resolution", k["avg_res"], border=True)
            st.metric("Services", f'{requests["template"].nunique():,}', border=True)

        with st.container(border=True):
            st.markdown("**Ticket volume over time**")
            vol = volume_over_time(df, freq="D")
            if vol.empty:
                st.caption("No dated tickets in the current selection.")
            else:
                st.area_chart(vol, x="period", y="count", x_label="", y_label="", height=260)

        c1, c2 = st.columns(2)
        with c1:
            count_chart(df, "template", "By service")
            count_chart(df, "technician_name", "By technician")
            count_chart(df, "priority", "By priority")
            count_chart(df, "lang", "Language mix")
        with c2:
            count_chart(df, "group", "By team")
            count_chart(df, "status", "By status")
            count_chart(df, "site", "By site")

        data_table(df, "requests_filtered")

    # ================= PROBLEMS =================
    with tab_prob:
        problems = load_best("problems")
        if problems is None or problems.empty:
            st.info(
                "No problems data yet. Run `python -m pipeline.extract --modules problems`, then preprocess.",
                icon=":material/info:",
            )
        else:
            resolved = int(problems["status"].isin(CLOSED_STATUSES).sum()) if "status" in problems else 0
            has_res = problems["has_resolution"].fillna(False).mean() * 100 if "has_resolution" in problems else 0
            with st.container(horizontal=True):
                st.metric("Total problems", f"{len(problems):,}", border=True)
                st.metric("Resolved or closed", f"{resolved:,}", border=True)
                st.metric("With usable resolution", f"{has_res:.0f}%", border=True)
            c1, c2 = st.columns(2)
            with c1:
                count_chart(problems, "category", "By category")
                count_chart(problems, "priority", "By priority")
                count_chart(problems, "group", "By team")
            with c2:
                count_chart(problems, "status", "By status")
                count_chart(problems, "technician_name", "By technician")
            data_table(problems, "problems")

    # ============ CHANGES / PROJECTS / SOLUTIONS (generic) ============
    def generic_tab(module: str, label: str) -> None:
        table = load(module)
        if table is None or table.empty:
            st.info(
                f"No {label} data yet. Run `python -m pipeline.extract --modules {module}`, then preprocess.",
                icon=":material/info:",
            )
            return
        with st.container(horizontal=True):
            st.metric("Total records", f"{len(table):,}", border=True)
        cols = st.columns(2)
        idx = 0
        for col in ("status", "category", "priority", "group", "technician_name"):
            if col in table.columns and table[col].notna().any():
                with cols[idx % 2]:
                    count_chart(table, col, f"By {col.replace('_', ' ')}")
                idx += 1
        data_table(table, module)

    with tab_chg:
        generic_tab("changes", "changes")
    with tab_proj:
        generic_tab("projects", "projects")
    with tab_sol:
        generic_tab("solutions", "solutions")


if __name__ == "__main__":
    _run()
