"""App 3 — daily ticket-volume forecasting per team + overall.

Prophet with regional holidays/Ramadan (they break the weekly seasonality),
evaluated against an honest baseline: "same weekday last week". Per the
reviewer's rule, if Prophet does not beat the naive baseline by a clear MAPE
margin, the report says so plainly and the naive/known-peak view is what the
dashboard should show — no model for its own sake.

Evaluation is temporal: fit on all but the last ``HORIZON_EVAL`` days, forecast
them, compare to actuals.

Usage::

    python -m models.forecasting.forecast --evaluate
    python -m models.forecasting.forecast --build     # 4-week forward forecast
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("forecast")

FORECAST_PATH = config.PROCESSED_DIR / "forecast.parquet"
REPORT = config.INSIGHTS_DIR / "forecast_eval.md"
METRICS = config.MODELS_DIR / "artifacts" / "forecast_metrics.json"

TOP_TEAMS = ["IT Service Desk Team", "Network Team", "System Admins"]
HORIZON_EVAL = 28          # days held out for evaluation
HORIZON_FORECAST = 28      # days forecast forward (4 weeks)
MIN_MAPE_EDGE = 0.10       # Prophet must beat naive by >=10% relative MAPE to "win"


def daily_series(group: Optional[str] = None) -> pd.DataFrame:
    """Daily ticket counts as a Prophet frame (ds, y), zero-filled."""
    df = pd.read_parquet(config.PROCESSED_DIR / "requests_anon.parquet")
    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")
    df = df[df["created_time"].notna()]
    if group:
        df = df[df["group"] == group]
    daily = df.set_index("created_time").resample("D").size()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full, fill_value=0)
    return pd.DataFrame({"ds": daily.index, "y": daily.values})


def regional_holidays() -> pd.DataFrame:
    """Ramadan windows + Eids + National Day — they break weekly seasonality.

    Approximate civil dates (Ramadan/Eid drift ~11 days/yr); good enough to let
    Prophet down-weight those periods rather than fit them as normal weeks.
    """
    rows = []
    ramadans = {2024: ("2024-03-11", "2024-04-09"), 2025: ("2025-03-01", "2025-03-30"),
                2026: ("2026-02-18", "2026-03-19")}
    for _, (start, end) in ramadans.items():
        for d in pd.date_range(start, end):
            rows.append(("ramadan", d))
    fixed = {"eid_fitr": ["2024-04-10", "2025-03-31", "2026-03-20"],
             "eid_adha": ["2024-06-16", "2025-06-06", "2026-05-27"],
             "national_day": ["2023-09-23", "2024-09-23", "2025-09-23", "2026-09-23"],
             "founding_day": ["2024-02-22", "2025-02-22", "2026-02-22"]}
    for name, dates in fixed.items():
        for d in dates:
            rows.append((name, pd.Timestamp(d)))
    hol = pd.DataFrame(rows, columns=["holiday", "ds"])
    hol["lower_window"], hol["upper_window"] = 0, 1
    return hol


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """MAPE with a small floor so zero-ticket days don't explode the metric."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    denom = np.clip(actual, 1.0, None)
    return float(np.mean(np.abs(actual - pred) / denom))


def naive_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Same weekday last week: each future day = the value 7 days earlier."""
    hist = list(train["y"].values)
    out = []
    for _ in range(horizon):
        out.append(hist[-7] if len(hist) >= 7 else (hist[-1] if hist else 0))
        hist.append(out[-1])
    return np.array(out)


def prophet_forecast(train: pd.DataFrame, horizon: int):
    from prophet import Prophet
    m = Prophet(holidays=regional_holidays(), weekly_seasonality=True,
                yearly_seasonality=True, daily_seasonality=False,
                seasonality_mode="multiplicative")
    import logging as _lg
    _lg.getLogger("cmdstanpy").setLevel(_lg.WARNING)
    m.fit(train)
    future = m.make_future_dataframe(periods=horizon)
    fc = m.predict(future)
    return m, fc


def evaluate_one(group: Optional[str]) -> Dict:
    name = group or "ALL"
    s = daily_series(group)
    train, test = s.iloc[:-HORIZON_EVAL], s.iloc[-HORIZON_EVAL:]

    naive = naive_forecast(train, HORIZON_EVAL)
    naive_mape = _mape(test["y"].values, naive)

    _, fc = prophet_forecast(train, HORIZON_EVAL)
    prophet_pred = fc.tail(HORIZON_EVAL)["yhat"].clip(lower=0).values
    prophet_mape = _mape(test["y"].values, prophet_pred)

    edge = (naive_mape - prophet_mape) / naive_mape if naive_mape else 0.0
    winner = "prophet" if edge >= MIN_MAPE_EDGE else "naive"
    logger.info("[%s] naive MAPE=%.3f | prophet MAPE=%.3f | edge=%.1f%% -> %s",
                name, naive_mape, prophet_mape, 100 * edge, winner)
    return {"group": name, "naive_mape": round(naive_mape, 3),
            "prophet_mape": round(prophet_mape, 3), "edge": round(edge, 3), "winner": winner}


def evaluate() -> Dict:
    results = [evaluate_one(None)] + [evaluate_one(t) for t in TOP_TEAMS]
    prophet_wins = sum(r["winner"] == "prophet" for r in results)
    verdict = ("prophet" if prophet_wins > len(results) / 2 else "naive")
    payload = {"evaluated_at": datetime.now(timezone.utc).isoformat(),
               "horizon_days": HORIZON_EVAL, "min_edge": MIN_MAPE_EDGE,
               "results": results, "overall_recommendation": verdict}
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(payload)
    return payload


def _write_report(payload: Dict) -> None:
    lines = [
        "# Ticket-volume forecasting — evaluation",
        "",
        f"*{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC — "
        f"daily level, {payload['horizon_days']}-day holdout, Prophet vs. "
        f"naive (same weekday last week)*",
        "",
        "Prophet uses regional holidays + Ramadan (they break weekly seasonality). "
        f"Prophet is only preferred when it beats the naive baseline by ≥ {payload['min_edge']:.0%} "
        "relative MAPE — otherwise the naive baseline + the known Sunday peak are enough.",
        "",
        "| scope | naive MAPE | Prophet MAPE | Prophet edge | use |",
        "|---|---|---|---|---|",
    ]
    for r in payload["results"]:
        lines.append(f"| {r['group']} | {r['naive_mape']:.1%} | {r['prophet_mape']:.1%} | "
                     f"{r['edge']:+.1%} | {'**Prophet**' if r['winner']=='prophet' else 'naive'} |")
    verdict = payload["overall_recommendation"]
    lines += ["",
              f"**Overall: {'Prophet earns its place' if verdict=='prophet' else 'the naive baseline is enough'} "
              "on the majority of series.**", ""]
    if verdict == "naive":
        lines.append("> القرار: النموذج لا يتفوق بوضوح — الداشبورد يعرض خط الأساس الساذج + "
                     "الذروة المعروفة (الأحد) كحقيقة ثابتة بدون نموذج.")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote forecast report -> %s", REPORT)


def build_forward() -> None:
    """Produce the 4-week forward forecast for the dashboard (all series)."""
    frames = []
    for group in [None] + TOP_TEAMS:
        name = group or "ALL"
        s = daily_series(group)
        _, fc = prophet_forecast(s, HORIZON_FORECAST)
        tail = fc.tail(HORIZON_FORECAST)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        tail["group"] = name
        for c in ("yhat", "yhat_lower", "yhat_upper"):
            tail[c] = tail[c].clip(lower=0).round(1)
        frames.append(tail)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(FORECAST_PATH, index=False)
    logger.info("Wrote 4-week forward forecast (%d rows) -> %s", len(out), FORECAST_PATH)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Ticket-volume forecasting.")
    parser.add_argument("--evaluate", action="store_true", help="Prophet vs naive, temporal holdout.")
    parser.add_argument("--build", action="store_true", help="Write 4-week forward forecast.")
    args = parser.parse_args(argv)
    if args.evaluate or not args.build:
        evaluate()
    if args.build:
        build_forward()


if __name__ == "__main__":
    main()
