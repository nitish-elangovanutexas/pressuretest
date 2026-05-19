"""Shared path constants and helpers for all routers."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_DIR = PROJECT_ROOT / "data" / "baselines"
SCORES_DIR = PROJECT_ROOT / "data" / "scores"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"


def company_name(ticker: str) -> str:
    for f in sorted(TRANSCRIPTS_DIR.glob(f"{ticker}_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if cn := data.get("company_name"):
                return cn
        except Exception:
            pass
    return ticker
