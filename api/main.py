"""PressureTest — Phase 3 FastAPI backend entry point."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when uvicorn launches from any cwd.
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import calls, ceo, chat, flags, stats, tickers

app = FastAPI(title="PressureTest API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tickers.router)
app.include_router(ceo.router)
app.include_router(calls.router)
app.include_router(flags.router)
app.include_router(stats.router)
app.include_router(chat.router)


@app.get("/health")
def health():
    return {"status": "ok"}
