from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from api.utils import BASELINE_DIR, SCORES_DIR, company_name

router = APIRouter()


def _call_pressure(ticker: str, call_date: str) -> float | None:
    safe = call_date.replace("-", "")
    p = SCORES_DIR / f"{ticker}_{safe}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("pressure_score")
        except Exception:
            pass
    return None


@router.get("/ceo/{ticker}")
def ceo_profile(ticker: str):
    """Full CEO baseline profile for a ticker."""
    ticker = ticker.upper()
    bp = BASELINE_DIR / f"{ticker}.json"
    if not bp.exists():
        raise HTTPException(status_code=404, detail=f"No baseline for {ticker}")

    baseline = json.loads(bp.read_text(encoding="utf-8"))
    pb = baseline.get("pressure_baseline") or {}
    centroid = baseline.get("centroid") or []
    per_call = baseline.get("per_call") or []

    calls = [
        {
            "date": c.get("call_date"),
            "quarter": c.get("quarter"),
            "year": c.get("year"),
            "n_qa_turns": c.get("n_answers"),
            "pressure_score": _call_pressure(ticker, c.get("call_date") or ""),
        }
        for c in per_call
    ]

    return {
        "ticker": ticker,
        "company_name": company_name(ticker),
        "n_calls": baseline.get("n_calls", len(per_call)),
        "pressure_mean": pb.get("mean"),
        "pressure_std": pb.get("std"),
        "centroid_dim": len(centroid),
        "calls": calls,
    }
