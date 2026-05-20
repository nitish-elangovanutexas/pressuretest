"""
HuggingFace dataset loader for earnings call transcripts.

Downloads Rogersurf/earnings-call-transcripts, filters by ticker symbol,
and returns FilingRecord instances compatible with TranscriptParser.

No network requests are made at import time; the dataset is only
fetched when load_ticker() is called.
"""
from __future__ import annotations

import logging
import re
from datetime import date

from scraper.edgar_scraper import FilingRecord

log = logging.getLogger(__name__)

_HF_DATASET = "Rogersurf/earnings-call-transcripts"

# Candidate column names for each field, tried in order of likelihood
_TICKER_COLS = ("ticker", "symbol", "stock_ticker", "company_ticker")
_DATE_COLS = ("date", "filing_date", "call_date", "transcript_date", "quarter_date")
_TEXT_COLS = ("transcript", "text", "raw_text", "content", "body")
_COMPANY_COLS = ("company", "company_name", "company_title", "name")


def _find_col(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate that matches a column name (case-insensitive)."""
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _normalise_date(raw: str) -> str:
    """
    Coerce a date-like string to YYYY-MM-DD.

    Handles ISO dates, YYYYMMDD, and quarter strings like "Q1 2024".
    Falls back to today's date if the input cannot be parsed.
    """
    raw = raw.strip()

    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw

    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # "Q3 2024" → last month of that quarter as a proxy date
    m = re.match(r"Q([1-4])\s*(\d{4})", raw, re.IGNORECASE)
    if m:
        q, year = int(m.group(1)), int(m.group(2))
        month = q * 3
        return f"{year}-{month:02d}-01"

    log.debug("Cannot parse date %r; using today", raw)
    return date.today().isoformat()


def load_ticker(
    ticker: str,
    *,
    max_transcripts: int = 5,
) -> tuple[str, list[FilingRecord]]:
    """
    Load up to *max_transcripts* FilingRecords for *ticker* from the HF dataset.

    Downloads Rogersurf/earnings-call-transcripts on first call (cached by
    the datasets library on subsequent runs).  Filters rows whose ticker
    column matches *ticker* (case-insensitive), sorts newest-first, and
    maps each row to a FilingRecord.

    Args:
        ticker:          Stock ticker symbol (e.g. "AAPL").
        max_transcripts: Maximum number of FilingRecords to return.

    Returns:
        ``(company_name, list[FilingRecord])`` with ``raw_text`` populated.
        Returns an empty list — not an exception — when no rows match.

    Raises:
        ImportError: If the ``datasets`` library is not installed.
        ValueError:  If no recognisable text column exists in the dataset.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required for --source hf. "
            "Install it with: pip install datasets"
        ) from exc

    log.info("Loading HF dataset %s …", _HF_DATASET)
    ds = load_dataset(_HF_DATASET, trust_remote_code=True)

    # Collect all splits into a flat list of dicts
    all_rows: list[dict] = []
    for split in ds.values():
        all_rows.extend(split)

    if not all_rows:
        log.warning("HF dataset is empty")
        return ticker.upper(), []

    cols: list[str] = list(ds[next(iter(ds))].column_names)
    ticker_col = _find_col(cols, _TICKER_COLS)
    date_col = _find_col(cols, _DATE_COLS)
    text_col = _find_col(cols, _TEXT_COLS)
    company_col = _find_col(cols, _COMPANY_COLS)

    if text_col is None:
        raise ValueError(
            f"Cannot find a text column in the HF dataset. "
            f"Available columns: {cols}"
        )

    log.debug(
        "Column mapping — ticker:%s date:%s text:%s company:%s",
        ticker_col, date_col, text_col, company_col,
    )

    ticker_upper = ticker.upper()
    matched: list[dict] = []
    for row in all_rows:
        if ticker_col is None:
            matched.append(row)
        elif str(row.get(ticker_col, "")).strip().upper() == ticker_upper:
            matched.append(row)

    log.info(
        "HF dataset: %d row(s) matched ticker %s (out of %d total)",
        len(matched), ticker_upper, len(all_rows),
    )

    if not matched:
        return ticker_upper, []

    # Sort newest-first when a date column is available
    if date_col:
        matched.sort(key=lambda r: str(r.get(date_col, "")), reverse=True)

    matched = matched[:max_transcripts]

    # Derive company name from the first matched row
    company_name = ticker_upper
    if company_col:
        candidate = str(matched[0].get(company_col, "")).strip()
        if candidate:
            company_name = candidate

    records: list[FilingRecord] = []
    for row in matched:
        raw_text = str(row.get(text_col, "")).strip()
        if not raw_text:
            log.debug("Skipping row with empty text field")
            continue

        date_str = (
            _normalise_date(str(row.get(date_col, "")))
            if date_col
            else date.today().isoformat()
        )
        acc_no = f"HF-{ticker_upper}-{date_str.replace('-', '')}"

        records.append(
            FilingRecord(
                accession_no=acc_no,
                filing_date=date_str,
                items="2.02",
                exhibit_url=None,
                raw_text=raw_text,
            )
        )
        log.info("HF transcript loaded: %s %s", ticker_upper, date_str)

    return company_name, records


def list_tickers(*, min_transcripts: int = 5) -> list[str]:
    """
    Return a sorted list of ticker symbols from the HF dataset that have at
    least *min_transcripts* rows.  Downloads and caches the dataset on first call.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required. Install with: pip install datasets"
        ) from exc

    from collections import Counter

    log.info("Scanning %s for ticker counts…", _HF_DATASET)
    ds = load_dataset(_HF_DATASET, trust_remote_code=True)

    all_rows: list[dict] = []
    for split in ds.values():
        all_rows.extend(split)

    if not all_rows:
        return []

    cols = list(ds[next(iter(ds))].column_names)
    ticker_col = _find_col(cols, _TICKER_COLS)
    if ticker_col is None:
        log.warning("No ticker column found in %s; returning empty list", _HF_DATASET)
        return []

    counts: Counter[str] = Counter()
    for row in all_rows:
        t = str(row.get(ticker_col, "")).strip().upper()
        if t:
            counts[t] += 1

    eligible = sorted(t for t, n in counts.items() if n >= min_transcripts)
    log.info("Found %d tickers with >= %d transcripts", len(eligible), min_transcripts)
    return eligible
