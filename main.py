"""
PressureTest — Phase 1 + Phase 3 CLI entry point.

Fetches earnings call transcripts from SEC EDGAR and saves them as
structured JSON files ready for Phase 2 NLP modelling.

Usage:
    python main.py --ticker AAPL
    python main.py --ticker MSFT --max-filings 20 --out-dir data/transcripts
    python main.py --ticker NVDA --max-transcripts 1 --verbose
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date
from pathlib import Path

# Ensure the project root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).parent))

# Load .env so FMP_API_KEY and ANTHROPIC_API_KEY are available.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scraper.edgar_scraper import EdgarScraper
from scraper.fmp_scraper import FMPScraper
from scraper.hf_loader import list_tickers as hf_list_tickers
from scraper.hf_loader import load_ticker as hf_load_ticker
from scraper.motley_fool_scraper import MotleyFoolScraper
from scraper.transcript_parser import EarningsCallTranscript, TranscriptParser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third-party loggers unless in verbose mode
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


def _output_path(out_dir: Path, ticker: str, filing_date: str) -> Path:
    safe_date = filing_date.replace("-", "")
    return out_dir / f"{ticker.upper()}_{safe_date}.json"


def _save_transcript(transcript: EarningsCallTranscript, path: Path) -> None:
    payload = transcript.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript_parser = TranscriptParser()
    company_name = args.ticker.upper()
    filings = []

    if args.source == "fmp":
        # --- Source: Financial Modeling Prep (live) ---
        print(
            f"[PressureTest] Fetching transcripts for {args.ticker.upper()} "
            "from Financial Modeling Prep…"
        )
        async with FMPScraper() as fmp:
            company_name, filings = await fmp.fetch_transcripts(
                args.ticker,
                max_transcripts=args.max_transcripts,
            )
        if not filings:
            print(
                f"[ERROR] No transcripts returned by FMP for {args.ticker.upper()}. "
                "Check FMP_API_KEY and subscription tier, or use --source hf/edgar.",
                file=sys.stderr,
            )
            return 1

    elif args.source == "hf":
        # --- Source: HuggingFace dataset ---
        print(
            f"[PressureTest] Loading transcripts for {args.ticker.upper()} "
            "from HuggingFace dataset…"
        )
        company_name, filings = hf_load_ticker(
            args.ticker,
            max_transcripts=args.max_transcripts,
        )
        if not filings:
            print(
                f"[ERROR] No transcripts found for {args.ticker.upper()} "
                "in the HuggingFace dataset.",
                file=sys.stderr,
            )
            return 1

    elif args.source == "auto":
        # --- Source: FMP first, fall back to HuggingFace ---
        print(
            f"[PressureTest] Auto-mode: trying FMP for {args.ticker.upper()}…"
        )
        async with FMPScraper() as fmp:
            company_name, filings = await fmp.fetch_transcripts(
                args.ticker,
                max_transcripts=args.max_transcripts,
            )

        if not filings:
            print(
                f"[PressureTest] FMP returned 0 transcripts for {args.ticker.upper()}; "
                "falling back to HuggingFace…"
            )
            company_name, filings = hf_load_ticker(
                args.ticker,
                max_transcripts=args.max_transcripts,
            )

        if not filings:
            print(
                f"[ERROR] No transcripts found for {args.ticker.upper()} "
                "via FMP or HuggingFace.",
                file=sys.stderr,
            )
            return 1

    else:
        # --- Source: EDGAR → Motley Fool fallback (--source edgar) ---
        print(f"[PressureTest] Fetching transcripts for {args.ticker.upper()} from EDGAR…")
        async with EdgarScraper() as edgar:
            try:
                company_name, filings = await edgar.fetch_transcripts(
                    args.ticker,
                    max_filings_to_scan=args.max_filings,
                    max_transcripts=args.max_transcripts,
                )
            except ValueError as exc:
                logging.getLogger(__name__).warning("EDGAR lookup failed: %s", exc)

        # --- Motley Fool fallback ---
        if not filings:
            print(
                f"[PressureTest] EDGAR returned 0 transcripts for {args.ticker.upper()}; "
                "trying Motley Fool…"
            )
            async with MotleyFoolScraper() as mf_scraper:
                company_name, filings = await mf_scraper.fetch_transcripts(
                    args.ticker,
                    company_name,
                    max_transcripts=args.max_transcripts,
                )

        if not filings:
            print(
                f"[ERROR] No earnings call transcripts found for {args.ticker.upper()} "
                "on EDGAR or Motley Fool.",
                file=sys.stderr,
            )
            return 1

    print(f"[PressureTest] Found {len(filings)} transcript(s) for {company_name}.")

    for filing in filings:
        filing_date = date.fromisoformat(filing.filing_date)
        transcript = transcript_parser.parse(
            filing.raw_text,
            ticker=args.ticker,
            company_name=company_name,
            filing_date=filing_date,
        )

        out_path = _output_path(out_dir, args.ticker, filing.filing_date)
        _save_transcript(transcript, out_path)

        print(
            f"  ✓ {out_path.name}"
            f"  |  Q{transcript.quarter} {transcript.year}"
            f"  |  {len(transcript.qa_turns)} Q&A turns"
            f"  |  {len(transcript.prepared_remarks):,} chars prepared remarks"
        )

    return 0


async def run_all_tickers(args: argparse.Namespace) -> int:
    """Ingest → baseline → score for every HF ticker with >= 5 transcripts."""
    from nlp.baseline import build_baseline, load_transcripts_from_dir, save_baseline
    from nlp.deviation import compute_pressure, save_report
    from nlp.question_scorer import QuestionScorer

    log = logging.getLogger(__name__)
    transcript_dir = Path(args.out_dir)
    baseline_dir   = Path("data/baselines")
    scores_dir     = Path("data/scores")
    for d in (transcript_dir, baseline_dir, scores_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("[PressureTest] Discovering eligible tickers in the HuggingFace dataset…")
    try:
        eligible = hf_list_tickers(min_transcripts=5)
    except ImportError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    if not eligible:
        print("[ERROR] No tickers with >= 5 transcripts found.", file=sys.stderr)
        return 1

    total = len(eligible)
    print(f"[PressureTest] {total} tickers eligible — starting pipeline.\n")

    transcript_parser = TranscriptParser()
    scorer = QuestionScorer()
    ok = skipped = flagged_total = 0

    for i, ticker in enumerate(eligible, 1):
        print(f"[PressureTest] Processing {i} of {total}: {ticker}")
        try:
            # --- Ingest (HuggingFace) ---
            company_name, filings = hf_load_ticker(
                ticker, max_transcripts=args.max_transcripts
            )
            if not filings:
                log.warning("  %s: no transcripts found; skipping", ticker)
                skipped += 1
                continue

            for filing in filings:
                filing_date_obj = date.fromisoformat(filing.filing_date)
                transcript = transcript_parser.parse(
                    filing.raw_text,
                    ticker=ticker,
                    company_name=company_name,
                    filing_date=filing_date_obj,
                )
                out_path = _output_path(transcript_dir, ticker, filing.filing_date)
                _save_transcript(transcript, out_path)

            # --- Baseline ---
            transcripts = load_transcripts_from_dir(transcript_dir, ticker)
            baseline = build_baseline(transcripts, ticker)
            baseline_path = save_baseline(baseline, baseline_dir)

            # --- Score (latest ingested call) ---
            latest_files = sorted(transcript_dir.glob(f"{ticker.upper()}_*.json"))
            latest_call  = json.loads(latest_files[-1].read_text(encoding="utf-8"))
            baseline_dict = json.loads(baseline_path.read_text(encoding="utf-8"))
            report = compute_pressure(latest_call, baseline_dict, scorer)
            save_report(report, scores_dir)

            flag = "FLAGGED" if report.flagged else "ok"
            if report.flagged:
                flagged_total += 1
            log.info(
                "  %s  P=%.4f  z=%+.2f  %s",
                ticker, report.pressure_score, report.z_score, flag,
            )
            ok += 1

        except Exception as exc:
            log.warning("  SKIP %s: %s", ticker, exc)
            skipped += 1

    print(
        f"\n[PressureTest] Completed: {ok} succeeded, {skipped} skipped, "
        f"{flagged_total} flagged."
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pressuretest",
        description=(
            "Fetch and parse earnings call transcripts from SEC EDGAR.\n"
            "Outputs structured JSON files to data/transcripts/ by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ticker",
        metavar="SYMBOL",
        help="Stock ticker symbol (e.g. AAPL, MSFT, NVDA). Required unless --all-tickers.",
    )
    p.add_argument(
        "--all-tickers",
        action="store_true",
        help=(
            "Ingest, baseline, and score every ticker in the HuggingFace dataset "
            "that has >= 5 transcripts. --ticker and --source are ignored."
        ),
    )
    p.add_argument(
        "--max-filings",
        type=int,
        default=15,
        metavar="N",
        help="Max number of 8-K filings to scan per ticker (default: 15)",
    )
    p.add_argument(
        "--max-transcripts",
        type=int,
        default=5,
        metavar="N",
        help="Max number of transcripts to save per ticker (default: 5)",
    )
    p.add_argument(
        "--out-dir",
        default="data/transcripts",
        metavar="DIR",
        help="Output directory for JSON files (default: data/transcripts)",
    )
    p.add_argument(
        "--source",
        choices=["auto", "fmp", "edgar", "hf"],
        default="auto",
        metavar="SOURCE",
        help=(
            "Transcript source. "
            "'auto' (default) tries FMP live first, falls back to HuggingFace; "
            "'fmp' uses Financial Modeling Prep live API only; "
            "'hf' loads from HuggingFace (Rogersurf/earnings-call-transcripts); "
            "'edgar' uses SEC EDGAR with Motley Fool fallback."
        ),
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    _configure_logging(args.verbose)
    if args.all_tickers:
        sys.exit(asyncio.run(run_all_tickers(args)))
    if not args.ticker:
        _build_arg_parser().error(
            "--ticker is required unless --all-tickers is specified"
        )
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
