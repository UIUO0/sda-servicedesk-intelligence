"""Production predict() for team routing — returns team + confidence.

Low-confidence predictions deliberately return "needs manual routing" instead
of a guess, so the model never sends a ticket to the wrong team with false
certainty.
"""
from __future__ import annotations

from typing import Dict, Optional

import joblib

from models.routing.train import ARTIFACT, TEXT_COLS

NEEDS_MANUAL = "needs manual routing"
_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not ARTIFACT.exists():
            raise FileNotFoundError("Routing model not built. Run `python -m models.routing.train --build`.")
        _bundle = joblib.load(ARTIFACT)
    return _bundle


def predict(ticket: Dict[str, str]) -> Dict[str, object]:
    """Route one ticket. ``ticket`` needs at least 'subject' and/or 'description'.

    Returns {team, confidence, needs_manual_routing}. Below the model's
    confidence floor, ``team`` is the manual-routing sentinel.
    """
    bundle = _load()
    text = " ".join(str(ticket.get(c, "")) for c in bundle["text_cols"]).strip()
    proba = bundle["pipeline"].predict_proba([text])[0]
    classes = list(bundle["pipeline"].classes_)
    best = int(proba.argmax())
    team, conf = classes[best], float(proba[best])
    if conf < bundle["min_confidence"]:
        return {"team": NEEDS_MANUAL, "confidence": round(conf, 3),
                "needs_manual_routing": True, "model_suggestion": team}
    return {"team": team, "confidence": round(conf, 3), "needs_manual_routing": False}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "VPN connection is not working from home"
    print(predict({"subject": q, "description": ""}))
