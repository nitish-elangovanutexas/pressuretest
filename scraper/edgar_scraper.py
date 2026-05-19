"""
EDGAR scraper for earnings call transcripts.

Queries SEC EDGAR APIs to locate and download 8-K filings that contain
earnings call transcripts (typically filed as Exhibit 99.1).

SEC rate-limit policy: max 10 requests/second; User-Agent must identify the
application and a contact address. This module enforces a 0.11 s inter-request
delay and retries on 429 with exponential back-off.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)

EDGAR_BASE = "https://www.sec.gov"
DATA_BASE = "https://data.sec.gov"

# SEC requires a descriptive User-Agent with a contact address.
DEFAULT_USER_AGENT = "PressureTest nlp-research contact@pressuretest.dev"
MIN_REQUEST_INTERVAL = 0.11  # seconds — keeps rate under 10 req/s


@dataclass
class FilingRecord:
    """Metadata for a single 8-K filing, populated incrementally."""

    accession_no: str          # e.g. "0000320193-24-000006"
    filing_date: str           # ISO date "YYYY-MM-DD"
    items: str                 # 8-K item codes, e.g. "2.02,9.01"
    exhibit_url: str | None = field(default=None)
    raw_text: str | None = field(default=None)


class EdgarScraper:
    """
    Async EDGAR scraper.  Use as an async context manager::

        async with EdgarScraper() as scraper:
            company_name, records = await scraper.fetch_transcripts("AAPL")
    """

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT) -> None:
        self._headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self._client: httpx.AsyncClient | None = None
        self._last_req_ts: float = 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "EdgarScraper":
        self._client = httpx.AsyncClient(
            headers=self._headers,
            follow_redirects=True,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    async def _get(self, url: str) -> httpx.Response:
        """Rate-limited GET with automatic 429 back-off (up to 4 attempts)."""
        assert self._client is not None, "Use EdgarScraper inside 'async with'"

        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - self._last_req_ts)
        if wait > 0:
            await asyncio.sleep(wait)

        for attempt in range(4):
            self._last_req_ts = time.monotonic()
            resp = await self._client.get(url)
            if resp.status_code == 429:
                delay = 2 ** attempt
                log.warning("EDGAR rate-limited; retrying in %ds", delay)
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            return resp

        raise RuntimeError(f"GET {url!r} failed after 4 attempts (persistent 429)")

    # ------------------------------------------------------------------
    # CIK resolution
    # ------------------------------------------------------------------

    async def ticker_to_cik(self, ticker: str) -> str:
        """
        Map a ticker symbol to a zero-padded 10-digit SEC CIK.

        Uses the authoritative SEC bulk ticker JSON published at
        https://www.sec.gov/files/company_tickers.json.
        """
        url = f"{EDGAR_BASE}/files/company_tickers.json"
        resp = await self._get(url)
        mapping: dict[str, dict] = resp.json()

        target = ticker.upper()
        for entry in mapping.values():
            if entry.get("ticker", "").upper() == target:
                return str(entry["cik_str"]).zfill(10)

        raise ValueError(
            f"Ticker {ticker!r} not found in SEC company tickers list. "
            "Verify the symbol is listed on a US exchange."
        )

    async def get_company_name(self, cik: str) -> str:
        """Return the official company name from the EDGAR submissions endpoint."""
        url = f"{DATA_BASE}/submissions/CIK{cik}.json"
        resp = await self._get(url)
        return resp.json()["name"]

    # ------------------------------------------------------------------
    # 8-K enumeration
    # ------------------------------------------------------------------

    async def list_earnings_8ks(
        self, cik: str, *, max_results: int = 20
    ) -> list[FilingRecord]:
        """
        Return recent 8-K filings that are likely earnings-related.

        Filters on Item 2.02 ("Results of Operations and Financial Condition"),
        which is the mandatory disclosure item for earnings releases and calls.
        Falls back to all 8-Ks when Item metadata is absent.
        """
        url = f"{DATA_BASE}/submissions/CIK{cik}.json"
        resp = await self._get(url)
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        items_list = recent.get("items", [])

        records: list[FilingRecord] = []
        for form, date_str, accession, items in zip(
            forms, dates, accessions, items_list
        ):
            if form != "8-K":
                continue
            items_str = str(items)
            if items_str and "2.02" not in items_str:
                continue
            records.append(
                FilingRecord(
                    accession_no=accession,
                    filing_date=date_str,
                    items=items_str,
                )
            )
            if len(records) >= max_results:
                break

        return records

    # ------------------------------------------------------------------
    # Exhibit discovery
    # ------------------------------------------------------------------

    async def find_exhibit_99_1(
        self, cik: str, record: FilingRecord
    ) -> str | None:
        """
        Locate the Exhibit 99.1 document URL from an 8-K filing index.

        Parses the filing's HTML index page, scanning for a table row whose
        Type column contains "EX-99.1".  Returns an absolute URL or None.
        """
        cik_int = int(cik)
        acc_nodash = record.accession_no.replace("-", "")
        index_url = (
            f"{EDGAR_BASE}/Archives/edgar/data/{cik_int}/"
            f"{acc_nodash}/{record.accession_no}-index.htm"
        )

        try:
            resp = await self._get(index_url)
        except httpx.HTTPStatusError as exc:
            log.debug("Index fetch failed for %s: %s", record.accession_no, exc)
            return None

        href = _extract_exhibit_href(resp.text, cik_int, acc_nodash)
        if href:
            return href

        log.debug("No Exhibit 99.1 found in index for %s", record.accession_no)
        return None

    # ------------------------------------------------------------------
    # Text extraction and cleaning
    # ------------------------------------------------------------------

    async def fetch_and_clean(self, url: str) -> str:
        """
        Download an EDGAR exhibit and return normalised plain text.

        Strips HTML markup (if present), normalises line endings, and
        collapses runs of blank lines to at most two newlines.
        """
        resp = await self._get(url)
        content_type = resp.headers.get("content-type", "").lower()
        raw = resp.text

        if "html" in content_type or _is_html(raw):
            raw = _strip_html(raw)

        raw = re.sub(r"\r\n|\r", "\n", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r" +\n", "\n", raw)
        raw = re.sub(r"\n{4,}", "\n\n\n", raw)
        return raw.strip()

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------

    async def fetch_transcripts(
        self,
        ticker: str,
        *,
        max_filings_to_scan: int = 15,
        max_transcripts: int = 5,
    ) -> tuple[str, list[FilingRecord]]:
        """
        Fetch recent earnings call transcripts for a ticker symbol.

        Scans up to *max_filings_to_scan* recent 8-Ks and returns up to
        *max_transcripts* FilingRecords with ``exhibit_url`` and ``raw_text``
        populated, ordered newest-first.

        Returns:
            (company_name, list[FilingRecord])
        """
        cik = await self.ticker_to_cik(ticker)
        company_name = await self.get_company_name(cik)
        candidates = await self.list_earnings_8ks(
            cik, max_results=max_filings_to_scan
        )

        log.info(
            "%s (%s): scanning %d candidate 8-K filings",
            ticker.upper(),
            company_name,
            len(candidates),
        )

        results: list[FilingRecord] = []
        for record in candidates:
            url = await self.find_exhibit_99_1(cik, record)
            if url is None:
                continue

            text = await self.fetch_and_clean(url)

            if not _looks_like_transcript(text):
                log.debug(
                    "Exhibit at %s does not look like a call transcript — skipping",
                    url,
                )
                continue

            record.exhibit_url = url
            record.raw_text = text
            results.append(record)
            log.info("Transcript found: %s %s", ticker.upper(), record.filing_date)

            if len(results) >= max_transcripts:
                break

        return company_name, results


# ------------------------------------------------------------------
# HTML utilities (module-level, also tested directly)
# ------------------------------------------------------------------

def _is_html(text: str) -> bool:
    """Return True if text looks like HTML."""
    return bool(re.search(r"<html|<!doctype", text[:500], re.IGNORECASE))


def _strip_html(html: str) -> str:
    """
    Lightweight HTML-to-text conversion with no external parser dependency.

    Handles the subset of HTML typically found in SEC EDGAR exhibit filings.
    """
    # Named entities
    entities = {
        "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&apos;": "'",
    }
    for entity, char in entities.items():
        html = html.replace(entity, char)

    # Numeric entities
    html = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), html)
    html = re.sub(
        r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), html
    )

    # Drop script and style blocks entirely
    html = re.sub(
        r"<(style|script)[^>]*>.*?</\1>",
        "",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Replace block-level elements with newlines
    html = re.sub(
        r"<(?:br|p|div|tr|li|h[1-6])\b[^>]*>",
        "\n",
        html,
        flags=re.IGNORECASE,
    )

    # Strip all remaining tags
    html = re.sub(r"<[^>]+>", "", html)
    return html


def _extract_exhibit_href(html: str, cik_int: int, acc_nodash: str) -> str | None:
    """
    Parse a filing index HTML page and return the Exhibit 99.1 document URL.

    Scans table rows for one whose content contains the string "EX-99.1",
    then extracts the first href from that row.
    """
    row_re = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    href_re = re.compile(r'href="([^"]+)"', re.IGNORECASE)

    for row_match in row_re.finditer(html):
        row = row_match.group(1)
        if not re.search(r"EX-99\.1", row, re.IGNORECASE):
            continue
        href_match = href_re.search(row)
        if not href_match:
            continue

        href = href_match.group(1)
        if href.startswith("http"):
            return href
        if href.startswith("/"):
            return f"{EDGAR_BASE}{href}"
        # Relative: construct from archive base
        base = f"{EDGAR_BASE}/Archives/edgar/data/{cik_int}/{acc_nodash}/"
        return base + href.lstrip("/")

    return None


def _looks_like_transcript(text: str) -> bool:
    """
    Heuristic filter: return True when text is plausibly a call transcript.

    A transcript typically has multiple speaker-attribution lines and
    keywords associated with live investor calls.
    """
    lower = text.lower()
    keyword_hits = sum(
        kw in lower
        for kw in [
            "operator",
            "earnings call",
            "conference call",
            "question",
            "analyst",
            "q&a",
            "please go ahead",
        ]
    )
    has_speaker_lines = bool(
        re.search(r"\n[A-Z][A-Za-z .,''\-]+(?::|--?)\s", text)
    )
    return keyword_hits >= 2 and has_speaker_lines


# ---------------------------------------------------------------------------
# Motley Fool fallback scraper
# ---------------------------------------------------------------------------

_MF_BASE = "https://www.fool.com"
_MF_TRANSCRIPT_BASE = f"{_MF_BASE}/earnings/call-transcripts"
_MF_SITEMAP_BASE = f"{_MF_BASE}/sitemap"
_MF_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 "
    "PressureTest/1.0 (contact@pressuretest.dev)"
)
_MF_MIN_INTERVAL = 1.0  # seconds — Motley Fool: max 1 req/s


class MotleyFoolScraper:
    """
    Async Motley Fool earnings call transcript scraper.

    Discovers transcript URLs via Motley Fool's monthly article sitemaps
    (``https://www.fool.com/sitemap/YYYY/MM``) and fetches each transcript
    page from the canonical path ``https://www.fool.com/earnings/call-transcripts/``.

    Returns the same ``FilingRecord`` type used by ``EdgarScraper`` so that
    callers can treat both sources uniformly.

    Usage::

        async with MotleyFoolScraper() as scraper:
            company_name, records = await scraper.fetch_transcripts(
                "AAPL", "Apple Inc.", max_transcripts=2
            )
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._last_req_ts: float = 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MotleyFoolScraper":
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": _MF_USER_AGENT,
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            },
            follow_redirects=True,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Rate-limited HTTP
    # ------------------------------------------------------------------

    async def _get(self, url: str) -> httpx.Response:
        """
        Rate-limited GET that enforces at most 1 request per second.

        Raises ``httpx.HTTPStatusError`` on 4xx/5xx responses so callers
        can handle failures without crashing the waterfall.
        """
        assert self._client is not None, "Use MotleyFoolScraper inside 'async with'"

        elapsed = time.monotonic() - self._last_req_ts
        if elapsed < _MF_MIN_INTERVAL:
            await asyncio.sleep(_MF_MIN_INTERVAL - elapsed)

        self._last_req_ts = time.monotonic()
        resp = await self._client.get(url)
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------
    # Transcript URL discovery
    # ------------------------------------------------------------------

    async def _find_transcript_links(
        self,
        ticker: str,
        max_results: int,
    ) -> list[tuple[str, str]]:
        """
        Return ``(url, date_str)`` pairs for Motley Fool transcript pages.

        Strategy: fetch monthly article sitemaps (newest month first) and
        search each for transcript URLs whose slug contains the ticker symbol
        as a hyphenated token (e.g. ``-aapl-`` for AAPL).  Stops as soon as
        ``max_results`` distinct URLs are collected or 13 months are exhausted.

        Sitemaps are hosted at ``https://www.fool.com/sitemap/YYYY/MM``.
        Each sitemap lists all articles published that month and is small
        enough to search with a single regex pass.
        """
        import datetime as _dt

        ticker_lower = ticker.lower()
        today = _dt.date.today()
        results: list[tuple[str, str]] = []
        seen: set[str] = set()

        for months_back in range(13):  # up to 12 months back
            # Derive YYYY/MM for (today − months_back months)
            total_months = today.year * 12 + (today.month - 1) - months_back
            year = total_months // 12
            month = total_months % 12 + 1

            sitemap_url = f"{_MF_SITEMAP_BASE}/{year}/{month:02d}"
            try:
                resp = await self._get(sitemap_url)
            except httpx.HTTPStatusError as exc:
                log.debug("MF sitemap fetch failed %s: %s", sitemap_url, exc)
                continue

            found = _mf_extract_ticker_urls(resp.text, ticker_lower)
            for url, date_str in found:
                if url not in seen:
                    seen.add(url)
                    results.append((url, date_str))

            if len(results) >= max_results:
                break

        # Sort newest-first and cap
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------

    async def fetch_transcripts(
        self,
        ticker: str,
        company_name: str,
        *,
        max_transcripts: int = 5,
    ) -> tuple[str, list[FilingRecord]]:
        """
        Fetch recent earnings call transcripts from Motley Fool.

        Searches monthly sitemaps for transcript URLs matching *ticker*,
        then downloads each page and extracts the transcript body.

        Args:
            ticker:          Stock ticker symbol (e.g. "AAPL").
            company_name:    Company name passed through to FilingRecord;
                             used as the returned ``company_name`` string.
            max_transcripts: Maximum number of transcripts to return.

        Returns:
            ``(company_name, list[FilingRecord])`` with ``raw_text`` and
            ``exhibit_url`` populated.  Returns an empty list — not an
            exception — when no transcripts are found.
        """
        log.info(
            "Motley Fool: searching for %s transcripts (max %d)",
            ticker.upper(),
            max_transcripts,
        )

        links = await self._find_transcript_links(ticker, max_transcripts)
        if not links:
            log.info("Motley Fool: no transcript URLs found for %s", ticker.upper())
            return company_name, []

        results: list[FilingRecord] = []
        for url, date_str in links:
            try:
                resp = await self._get(url)
            except httpx.HTTPStatusError as exc:
                log.warning("MF transcript fetch failed (%s): %s", url, exc)
                continue

            raw_text = _mf_extract_transcript_text(resp.text)
            if not raw_text:
                log.debug("No transcript body extracted from %s", url)
                continue

            record = FilingRecord(
                accession_no=f"MF-{ticker.upper()}-{date_str.replace('-', '')}",
                filing_date=date_str,
                items="2.02",
                exhibit_url=url,
                raw_text=raw_text,
            )
            results.append(record)
            log.info(
                "Motley Fool transcript found: %s %s", ticker.upper(), date_str
            )

        return company_name, results


# ---------------------------------------------------------------------------
# Motley Fool module-level helpers
# ---------------------------------------------------------------------------


def _mf_extract_ticker_urls(
    sitemap_text: str,
    ticker_lower: str,
) -> list[tuple[str, str]]:
    """
    Parse a Motley Fool monthly sitemap and return ``(url, date_str)`` pairs
    for earnings call transcript pages that contain the ticker symbol.

    Motley Fool transcript slugs embed the ticker as a hyphenated token:
    ``/earnings/call-transcripts/YYYY/MM/DD/{company}-{ticker}-qN-{year}-…/``

    Matching is done on the raw sitemap text so no XML parsing library is
    needed.
    """
    # Pattern: transcript path containing -ticker- as a word-boundary token
    pattern = re.compile(
        rf"({re.escape(_MF_TRANSCRIPT_BASE)}"
        rf"/(\d{{4}})/(\d{{2}})/(\d{{2}})"
        rf"/[^<\s]*-{re.escape(ticker_lower)}-[^<\s]*)",
        re.IGNORECASE,
    )

    results: list[tuple[str, str]] = []
    for m in pattern.finditer(sitemap_text):
        url = m.group(1).rstrip("/") + "/"
        date_str = f"{m.group(2)}-{m.group(3)}-{m.group(4)}"
        results.append((url, date_str))

    return results


def _mf_extract_transcript_text(html: str) -> str:
    """
    Extract the earnings call transcript body from a Motley Fool article page.

    Motley Fool transcript pages are structured as:
    - A preamble with participant list and AI-generated takeaways.
    - An ``<h2 id="full-conference-call-transcript">`` marker.
    - The full transcript as ``<p><strong>Speaker Name:</strong> text…</p>``
      paragraphs.
    - A boilerplate disclaimer ("This article is a transcript…").

    The function isolates the region between the h2 marker and the disclaimer,
    converts HTML formatting to the plain-text speaker-turn format expected by
    ``TranscriptParser``, and normalises whitespace.
    """
    # Drop scripts and style blocks — they add noise without content
    html = re.sub(
        r"<(style|script|nav|footer|header|aside)[^>]*>.*?</\1>",
        "",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Locate the transcript body start marker
    start_match = re.search(
        r'id=["\']full-conference-call-transcript["\']',
        html,
        re.IGNORECASE,
    )
    body = html[start_match.start():] if start_match else html

    # Trim at the boilerplate disclaimer that follows the last speaker turn
    for stop_pattern in [
        r"This article is a transcript of this conference call produced for",
        r"The Motley Fool has positions in and recommends",
        r"The Motley Fool has a disclosure policy",
    ]:
        stop_match = re.search(stop_pattern, body, re.IGNORECASE)
        if stop_match:
            body = body[: stop_match.start()]
            break

    # <strong>Speaker Name:</strong> → keep the text so "Speaker Name:" survives
    body = re.sub(r"<strong>(.*?)</strong>", r"\1", body, flags=re.DOTALL | re.IGNORECASE)

    # Block-level elements → newlines (mirrors EdgarScraper.fetch_and_clean)
    body = re.sub(
        r"<(?:br|p|div|tr|li|h[1-6])\b[^>]*>",
        "\n",
        body,
        flags=re.IGNORECASE,
    )

    # Strip all remaining tags
    body = re.sub(r"<[^>]+>", "", body)

    # Decode HTML entities
    entities = {
        "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&apos;": "'",
    }
    for ent, char in entities.items():
        body = body.replace(ent, char)
    body = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), body)
    body = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), body)

    # Normalise whitespace (same rules as EdgarScraper.fetch_and_clean)
    body = re.sub(r"\r\n|\r", "\n", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r" +\n", "\n", body)
    body = re.sub(r"\n{4,}", "\n\n\n", body)

    return body.strip()
