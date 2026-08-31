"""Dashboard "Intelligence" tab — surfaces the three ML apps, honestly labelled.

Every model output shows its source and quality up front (per the display
policy): routing carries per-ticket confidence and a manual-routing fallback;
clustering is derived from ticket text; the forecast leads with the reliable
weekly pattern and marks the model estimate as high-error. Nothing here is
presented as a system-entered fact.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

import config

ARTIFACTS = config.MODELS_DIR / "artifacts"
WEEKDAY_PATTERN = {  # avg tickets/day from data — the stable, model-free signal
    "Sunday": 12.0, "Monday": 10.9, "Tuesday": 10.2, "Wednesday": 9.9,
    "Thursday": 8.7, "Friday": 0.2, "Saturday": 0.4,
}


def _read_json(path: Path) -> Optional[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def render(st) -> None:
    st.caption("Model-derived views. Every value here is a model output labelled with its "
               "source and quality — not a system-entered fact.")

    _routing_section(st)
    _clustering_section(st)
    _forecast_section(st)


# --- App 1: routing --------------------------------------------------------
def _routing_section(st) -> None:
    st.subheader(":material/alt_route: Auto-routing (assist)")
    metrics = _read_json(ARTIFACTS / "routing_metrics.json")
    if not metrics:
        st.info("Routing model not trained yet. Run `python -m models.routing.train --evaluate --build`.",
                icon=":material/info:")
        return

    with st.container(horizontal=True):
        st.metric("Majority baseline", f"{metrics['majority_baseline_acc']:.0%}", border=True,
                  help="Always predict the biggest team — the bar any model must clear.")
        st.metric("Macro-F1 (temporal test)", f"{metrics['macro_f1']:.2f}", border=True)
        for team, f1 in metrics["per_team_f1"].items():
            st.metric(f"F1 · {team}", f"{f1:.2f}", border=True,
                      delta="≥0.75" if f1 >= 0.75 else "below 0.75",
                      delta_color="normal" if f1 >= 0.75 else "inverse")
    verdict = "2 of 3 teams production-ready; System Admins below the 0.75 bar." if not metrics["passed"] \
        else "All top-3 teams clear the 0.75 bar."
    st.caption(f"Temporal split (train on oldest 80%, test on newest 20%). {verdict} "
               "Low-confidence tickets return **needs manual routing** instead of a guess.")

    with st.form("routing_demo"):
        txt = st.text_input("Try it — ticket subject/description",
                            placeholder="VPN not working from home / احتاج تركيب برنامج")
        submitted = st.form_submit_button("Suggest team", icon=":material/network_node:")
    if submitted and txt.strip():
        try:
            from models.routing.predict import predict
            r = predict({"subject": txt, "description": ""})
            if r["needs_manual_routing"]:
                st.warning(f"Needs manual routing — model not confident "
                           f"(best guess: {r['model_suggestion']}, {r['confidence']:.0%}).",
                           icon=":material/report:")
            else:
                st.success(f"Suggested team: **{r['team']}**  ·  confidence {r['confidence']:.0%}",
                           icon=":material/check_circle:")
        except Exception as exc:
            st.error(f"Model unavailable: {exc}")


# --- App 2: recurring issues ----------------------------------------------
def _clustering_section(st) -> None:
    st.subheader(":material/hub: Recurring issues")
    path = config.PROCESSED_DIR / "requests_clusters.parquet"
    if not path.exists():
        st.info("Clustering not run yet. Run `python -m models.clustering.recurring --build`.",
                icon=":material/info:")
        return
    df = pd.read_parquet(path)
    clustered = df[df["cluster"] != -1]
    noise = int((df["cluster"] == -1).sum())

    with st.container(horizontal=True):
        st.metric("Recurring-issue groups", f"{clustered['cluster'].nunique():,}", border=True)
        st.metric("Tickets grouped", f"{len(clustered):,}", border=True)
        st.metric("Unclustered (noise)", f"{noise/len(df):.0%}", border=True)

    # Top issues by size, named by dominant service/template.
    top = (clustered.groupby("cluster")
           .agg(count=("id", "size"),
                service=("template", lambda s: s.mode().iloc[0] if not s.mode().empty else "—"),
                team=("group", lambda s: s.mode().iloc[0] if not s.mode().empty else "—"))
           .sort_values("count", ascending=False).head(10).reset_index(drop=True))
    top.index = top.index + 1
    st.dataframe(top.rename(columns={"service": "Dominant service", "team": "Affected team",
                                     "count": "Tickets"}), width="stretch")
    st.caption("Derived by clustering ticket text (multilingual embeddings + HDBSCAN) — Arabic and "
               "English variants of one fault are merged. Full report with examples and the manual "
               "eyeball check: `insights/recurring_issues.md`. These groups are candidates to seed "
               "the problems module.")


# --- App 3: forecasting ----------------------------------------------------
def _forecast_section(st) -> None:
    st.subheader(":material/insights: Volume outlook")
    st.markdown("**Weekly pattern — the reliable signal (from history, no model)**")
    wk = pd.DataFrame({"day": list(WEEKDAY_PATTERN), "avg tickets/day": list(WEEKDAY_PATTERN.values())})
    st.bar_chart(wk, x="day", y="avg tickets/day", sort=False, x_label="", y_label="", height=220)
    st.caption("Sunday is the weekly peak; Friday/Saturday are near-zero (regional weekend). "
               "This seasonality is stable and is what capacity planning should use.")

    fc_path = config.PROCESSED_DIR / "forecast.parquet"
    metrics = _read_json(ARTIFACTS / "forecast_metrics.json")
    if fc_path.exists():
        st.markdown("**4-week daily forecast (model estimate — treat as indicative)**")
        fc = pd.read_parquet(fc_path)
        overall = fc[fc["group"] == "ALL"][["ds", "yhat", "yhat_lower", "yhat_upper"]]
        chart = overall.rename(columns={"ds": "date", "yhat": "forecast",
                                        "yhat_lower": "low", "yhat_upper": "high"}).set_index("date")
        st.line_chart(chart[["forecast", "low", "high"]], height=240)
        if metrics:
            worst = max(r["prophet_mape"] for r in metrics["results"])
            st.caption(f"Prophet with regional holidays/Ramadan. Daily error is high "
                       f"(MAPE up to {worst:.0%}) and Prophet only beats the naive baseline on "
                       f"{sum(r['winner']=='prophet' for r in metrics['results'])} of "
                       f"{len(metrics['results'])} series — so this is indicative, not precise. "
                       "The weekly pattern above is the dependable planning input.")
