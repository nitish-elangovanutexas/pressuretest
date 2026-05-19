"""
PressureTest weekly scheduler.

Runs every Monday at 06:00 and refreshes transcript data for every ticker
that has a baseline in data/baselines/.  For each ticker:

  1. Ingest the latest transcript via FMP (live API).
  2. Rebuild the CEO baseline:  python -m nlp.baseline --ticker X
  3. Score the most-recent call: python -m nlp.deviation --ticker X --call <path>

All activity is logged to logs/scheduler.log (created if absent).

Usage:
    python scheduler.py          # runs indefinitely, fires each Monday at 06:00
    python scheduler.py --now    # run the refresh job immediately then exit
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import schedule

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent
_BASELINE_DIR = _ROOT / "data" / "baselines"
_TRANSCRIPTS_DIR = _ROOT / "data" / "transcripts"
_LOGS_DIR = _ROOT / "logs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOGS_DIR / "scheduler.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


log = logging.getLogger("scheduler")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tracked_tickers() -> list[str]:
    """Return all tickers that have a baseline file."""
    return sorted(p.stem.upper() for p in _BASELINE_DIR.glob("*.json"))


def _latest_transcript(ticker: str) -> Path | None:
    """Return the path to the most-recent transcript file for *ticker*."""
    files = sorted(_TRANSCRIPTS_DIR.glob(f"{ticker}_*.json"))
    return files[-1] if files else None


def _run(cmd: list[str], label: str) -> bool:
    """Run *cmd* as a subprocess; return True on success."""
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("%s failed (exit %d):\n%s", label, result.returncode, result.stderr)
        return False
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            log.info("  %s", line)
    return True


# ---------------------------------------------------------------------------
# Per-ticker refresh
# ---------------------------------------------------------------------------


def refresh_ticker(ticker: str) -> None:
    log.info("=== Refreshing %s ===", ticker)

    # 1. Ingest latest transcript via FMP
    ok = _run(
        [sys.executable, str(_ROOT / "main.py"),
         "--ticker", ticker, "--source", "fmp", "--max-transcripts", "1"],
        f"ingest {ticker}",
    )
    if not ok:
        log.warning("%s: ingest failed — skipping baseline rebuild and scoring", ticker)
        return

    # 2. Rebuild baseline
    ok = _run(
        [sys.executable, "-m", "nlp.baseline", "--ticker", ticker],
        f"baseline {ticker}",
    )
    if not ok:
        log.warning("%s: baseline rebuild failed — skipping scoring", ticker)
        return

    # 3. Score the most-recent call
    latest = _latest_transcript(ticker)
    if latest is None:
        log.warning("%s: no transcript found after ingest — cannot score", ticker)
        return

    _run(
        [sys.executable, "-m", "nlp.deviation",
         "--ticker", ticker, "--call", str(latest)],
        f"score {ticker}",
    )

    log.info("=== %s refresh complete ===", ticker)


# ---------------------------------------------------------------------------
# Weekly job
# ---------------------------------------------------------------------------


def weekly_refresh() -> None:
    tickers = _tracked_tickers()
    if not tickers:
        log.warning("No baselines found in %s — nothing to refresh", _BASELINE_DIR)
        return

    log.info("Weekly refresh started — %d ticker(s): %s", len(tickers), ", ".join(tickers))
    for ticker in tickers:
        try:
            refresh_ticker(ticker)
        except Exception as exc:
            log.error("Unexpected error refreshing %s: %s", ticker, exc, exc_info=True)
    log.info("Weekly refresh complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scheduler",
        description="PressureTest weekly refresh scheduler (Monday 06:00).",
    )
    p.add_argument(
        "--now",
        action="store_true",
        help="Run the refresh job immediately then exit (useful for testing).",
    )
    return p


def main() -> None:
    _setup_logging()
    args = _build_arg_parser().parse_args()

    if args.now:
        log.info("--now flag set: running weekly refresh immediately")
        weekly_refresh()
        return

    schedule.every().monday.at("06:00").do(weekly_refresh)
    log.info("Scheduler started — next run: Monday at 06:00")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
