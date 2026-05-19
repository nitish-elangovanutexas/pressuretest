from __future__ import annotations

import json
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.utils import BASELINE_DIR, SCORES_DIR, TRANSCRIPTS_DIR

router = APIRouter()


@router.get("/calls/{ticker}")
def list_calls(ticker: str):
    """All scored calls for a ticker."""
    ticker = ticker.upper()
    results = []
    for sf in sorted(SCORES_DIR.glob(f"{ticker}_*.json")):
        try:
            score = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            continue
        results.append(
            {
                "ticker": score.get("ticker", ticker),
                "call_date": score.get("call_date"),
                "quarter": score.get("quarter"),
                "year": score.get("year"),
                "pressure_score": score.get("pressure_score"),
                "z_score": score.get("z_score"),
                "flagged": score.get("flagged"),
            }
        )
    return results


@router.get("/calls/{ticker}/latest")
def latest_call(ticker: str):
    """Full deviation report for the most recent scored call."""
    ticker = ticker.upper()
    files = sorted(SCORES_DIR.glob(f"{ticker}_*.json"))
    if not files:
        raise HTTPException(status_code=404, detail=f"No scored calls for {ticker}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


class ScoreRequest(BaseModel):
    call_date: str  # YYYY-MM-DD or YYYYMMDD


@router.post("/calls/{ticker}/score")
def score_call(ticker: str, body: ScoreRequest):
    """Run the deviation scorer on a specific call and return the report."""
    ticker = ticker.upper()
    safe_date = body.call_date.replace("-", "")

    tf = TRANSCRIPTS_DIR / f"{ticker}_{safe_date}.json"
    if not tf.exists():
        raise HTTPException(status_code=404, detail=f"Transcript not found: {ticker}_{safe_date}.json")

    bf = BASELINE_DIR / f"{ticker}.json"
    if not bf.exists():
        raise HTTPException(status_code=404, detail=f"No baseline for {ticker}")

    call = json.loads(tf.read_text(encoding="utf-8"))
    baseline = json.loads(bf.read_text(encoding="utf-8"))

    # Import here to avoid loading heavy ML models at startup.
    from nlp.deviation import compute_pressure, save_report
    from nlp.question_scorer import QuestionScorer

    try:
        report = compute_pressure(call, baseline, QuestionScorer())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    save_report(report, SCORES_DIR)
    return asdict(report)
