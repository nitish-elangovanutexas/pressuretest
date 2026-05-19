from __future__ import annotations

import json

from fastapi import APIRouter

from api.utils import SCORES_DIR, company_name

router = APIRouter()


@router.get("/flags")
def list_flags():
    """All calls across all tickers where flagged=true."""
    results = []
    for sf in sorted(SCORES_DIR.glob("*.json")):
        try:
            score = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not score.get("flagged"):
            continue
        ticker = (score.get("ticker") or sf.stem.split("_")[0]).upper()
        results.append(
            {
                "ticker": ticker,
                "company_name": company_name(ticker),
                "call_date": score.get("call_date"),
                "quarter": score.get("quarter"),
                "year": score.get("year"),
                "pressure_score": score.get("pressure_score"),
                "z_score": score.get("z_score"),
            }
        )
    return results
