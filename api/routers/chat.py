from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.utils import BASELINE_DIR, SCORES_DIR, TRANSCRIPTS_DIR

router = APIRouter()

SYSTEM_PROMPT = (
    "You are PressureTest AI, an analyst assistant specializing in executive behavior "
    "during earnings calls. Answer questions grounded in the provided transcript and "
    "scoring data only."
)

MODEL = "claude-sonnet-4-20250514"


class ChatRequest(BaseModel):
    ticker: str
    question: str


def _build_context(ticker: str) -> str:
    parts: list[str] = []

    transcripts = sorted(TRANSCRIPTS_DIR.glob(f"{ticker}_*.json"))
    if transcripts:
        t = json.loads(transcripts[-1].read_text(encoding="utf-8"))
        parts.append(f"# Latest Transcript ({t.get('call_date')})\n")
        parts.append(f"Company: {t.get('company_name')}  Q{t.get('quarter')} {t.get('year')}\n\n")
        parts.append("## Prepared Remarks (excerpt)\n")
        parts.append((t.get("prepared_remarks") or "")[:3000])
        parts.append("\n\n## Q&A Turns\n")
        for turn in (t.get("qa_turns") or [])[:40]:
            name = turn.get("speaker_name", "")
            role = turn.get("speaker_role", "")
            text = turn.get("text", "")
            parts.append(f"**{name} ({role}):** {text}\n\n")

    bf = BASELINE_DIR / f"{ticker}.json"
    if bf.exists():
        b = json.loads(bf.read_text(encoding="utf-8"))
        pb = b.get("pressure_baseline") or {}
        parts.append("# CEO Baseline\n")
        parts.append(f"Calls analyzed: {b.get('n_calls')}\n")
        if pb.get("mean") is not None:
            parts.append(f"Historical pressure mean: {pb['mean']:.4f}\n")
            parts.append(f"Historical pressure std: {pb['std']:.4f}\n")

    scores = sorted(SCORES_DIR.glob(f"{ticker}_*.json"))
    if scores:
        s = json.loads(scores[-1].read_text(encoding="utf-8"))
        parts.append(f"\n# Latest Deviation Score ({s.get('call_date')})\n")
        parts.append(f"Pressure score: {s.get('pressure_score'):.4f}\n")
        parts.append(f"Z-score: {s.get('z_score'):.2f}\n")
        parts.append(f"Flagged: {s.get('flagged')}\n")
        parts.append(f"Cosine distance: {s.get('cosine_distance'):.4f}\n")
        parts.append(f"Sentiment shift: {s.get('sentiment_shift'):.4f}\n")

    return "".join(parts)


@router.post("/chat")
async def chat(body: ChatRequest):
    """Stream an AI analyst response grounded in transcript + scoring data."""
    ticker = body.ticker.upper()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    context = _build_context(ticker)
    if not context.strip():
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

    user_message = f"<context>\n{context}\n</context>\n\n{body.question}"

    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate():
        async with client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
