from __future__ import annotations

import json

from fastapi import APIRouter

from api.utils import BASELINE_DIR, company_name

router = APIRouter()


@router.get("/tickers")
def list_tickers():
    """All tickers that have a baseline, with summary metadata."""
    results = []
    for bf in sorted(BASELINE_DIR.glob("*.json")):
        try:
            baseline = json.loads(bf.read_text(encoding="utf-8"))
        except Exception:
            continue
        ticker = bf.stem.upper()
        per_call = baseline.get("per_call") or []
        call_dates = [c["call_date"] for c in per_call if c.get("call_date")]
        latest = max(call_dates) if call_dates else None
        results.append(
            {
                "ticker": ticker,
                "company_name": company_name(ticker),
                "n_calls": baseline.get("n_calls", len(per_call)),
                "latest_call_date": latest,
            }
        )
    return results
