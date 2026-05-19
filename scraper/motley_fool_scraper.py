"""
Motley Fool earnings call transcript scraper.

Fallback source when EDGAR yields no transcripts (typical for large-cap companies
that distribute transcripts separately from their 8-K exhibit filings).

Rate-limit: 1 request/second maximum (enforced per-instance).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time

import httpx

from scraper.edgar_scraper import FilingRecord

log = logging.getLogger(__name__)

_MF_BASE = "https://www.fool.com"
_TRANSCRIPTS_URL = f"{_MF_BASE}/earnings/call-transcripts/"
_MIN_INTERVAL = 1.0  # seconds — enforces ≤ 1 req/s

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 "
    "PressureTest/1.0 (contact@pressuretest.dev)"
)

# Common corporate suffixes to strip from company-name token matching
_GENERIC_TOKENS = frozenset(
    {
        "corp", "corporation", "company", "companies", "inc", "incorporated",
        "limited", "ltd", "group", "holdings", "international", "industries",
    }
)


class MotleyFoolScraper:
    """
    Async Motley Fool transcript scraper.  Use as an async context manager::

        async with MotleyFoolScraper() as scraper:
            company_name, records = await scraper.fetch_transcripts(
                "AAPL", "Apple Inc.", max_transcripts=2
            )

    Returns the same ``FilingRecord`` type used by ``EdgarScraper``.
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._last_req_ts: float = 0.0

    async def __aenter__(self) -> "MotleyFoolScraper":
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": _USER_AGENT,
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

    async def _get(self, url: str) -> httpx.Response:
        """Rate-limited GET: enforces ≤ 1 request per second."""
        assert self._client is not None, "Use MotleyFoolScraper inside 'async with'"

        elapsed = time.monotonic() - self._last_req_ts
        if elapsed < _MIN_INTERVAL:
            await asyncio.sleep(_MIN_INTERVAL - elapsed)

        self._last_req_ts = time.monotonic()
        resp = await self._client.get(url)
        resp.raise_for_status()
        return resp

    async def fetch_transcripts(
        self,
        ticker: str,
        company_name: str,
        *,
        max_transcripts: int = 5,
    ) -> tuple[str, list[FilingRecord]]:
        """
        Search Motley Fool for recent earnings call transcripts.

        Args:
            ticker:          Stock ticker symbol (e.g. "AAPL").
            company_name:    Company name hint used to refine link matching.
            max_transcripts: Maximum number of transcripts to return.

        Returns:
            (company_name, list[FilingRecord]) with ``raw_text`` populated.
        """
        log.info("Motley Fool fallback: searching transcripts for %s", ticker.upper())

        transcript_links = await self._find_transcript_links(
            ticker, company_name, max_results=max_transcripts
        )

        results: list[FilingRecord] = []
        for url, date_str in transcript_links:
            try:
                resp = await self._get(url)
            except httpx.HTTPStatusError as exc:
                log.warning("Motley Fool page fetch failed (%s): %s", url, exc)
                continue

            raw_text = _extract_transcript_text(resp.text)
            if not raw_text:
                log.debug("No transcript text extracted from %s", url)
                continue

            acc_no = f"MF-{ticker.upper()}-{date_str.replace('-', '')}"
            record = FilingRecord(
                accession_no=acc_no,
                filing_date=date_str,
                items="2.02",
                exhibit_url=url,
                raw_text=raw_text,
            )
            results.append(record)
            log.info("Motley Fool transcript found: %s %s", ticker.upper(), date_str)

        return company_name, results

    async def _find_transcript_links(
        self,
        ticker: str,
        company_name: str,
        max_results: int,
    ) -> list[tuple[str, str]]:
        """
        Return (url, date_str) pairs for transcript pages matching the ticker.

        Tries a ticker-parameterised listing URL first; falls back to the
        main listing page and filters by ticker/company-name tokens.
        """
        ticker_upper = ticker.upper()
        results: list[tuple[str, str]] = []

        for listing_url in [
            f"{_TRANSCRIPTS_URL}?ticker={ticker_upper}",
            _TRANSCRIPTS_URL,
        ]:
            try:
                resp = await self._get(listing_url)
            except httpx.HTTPStatusError as exc:
                log.debug("Listing page failed %s: %s", listing_url, exc)
                continue

            results = _parse_transcript_links(resp.text, ticker_upper, company_name)
            if results:
                log.debug(
                    "Found %d transcript link(s) for %s at %s",
                    len(results),
                    ticker_upper,
                    listing_url,
                )
                break

        return results[:max_results]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_transcript_links(
    html: str,
    ticker: str,
    company_name: str,
) -> list[tuple[str, str]]:
    """
    Parse a Motley Fool transcripts listing page for links matching the ticker.

    Matching criteria (any one is sufficient):
      - Link text contains "(TICKER)" (e.g. "Apple (AAPL) Q1 2024 …")
      - URL slug contains "-ticker-" (e.g. "…apple-aapl-q1-2024-…")
      - A meaningful word from the company name appears in the URL slug.

    Returns a list of (absolute_url, date_str) sorted newest-first.
    """
    ticker_lower = ticker.lower()

    company_tokens = {
        w.lower()
        for w in re.findall(r"[a-z]+", company_name.lower())
        if len(w) > 3 and w.lower() not in _GENERIC_TOKENS
    }

    # Match anchor tags whose href is a transcript path (contains YYYY/MM/DD)
    link_re = re.compile(
        r'<a[^>]+href="(/earnings/call-transcripts/\d{4}/\d{2}/\d{2}/[^"#?]+)"[^>]*>'
        r"(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )

    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    for m in link_re.finditer(html):
        path = m.group(1).rstrip("/")
        raw_text = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        path_lower = path.lower()

        ticker_in_text = (
            f"({ticker})" in raw_text
            or f"({ticker_lower})" in raw_text.lower()
        )
        ticker_in_url = (
            f"-{ticker_lower}-" in path_lower
            or path_lower.endswith(f"-{ticker_lower}")
        )
        company_in_url = any(token in path_lower for token in company_tokens)

        if not (ticker_in_text or ticker_in_url or company_in_url):
            continue

        full_url = f"{_MF_BASE}{path}/"
        if full_url in seen:
            continue
        seen.add(full_url)

        date_str = _date_from_path(path) or ""
        results.append((full_url, date_str))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _date_from_path(path: str) -> str | None:
    """Extract YYYY-MM-DD from a Motley Fool URL path (/YYYY/MM/DD/slug)."""
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", path)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _extract_transcript_text(html: str) -> str:
    """
    Extract the earnings call body from a Motley Fool transcript page.

    Strips boilerplate (nav, ads, sidebar, scripts) and returns cleaned
    plain text in the same normalised form EdgarScraper produces.
    """
    # Drop non-content blocks entirely
    html = re.sub(
        r"<(style|script|nav|footer|header|aside)[^>]*>.*?</\1>",
        "",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Cascading attempt to isolate the main article body
    body = ""
    for pattern in [
        r"<article\b[^>]*>(.*?)</article>",
        r'<div[^>]+class="[^"]*article-body[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]+class="[^"]*body[^"]*"[^>]*>(.*?)</div>',
        r"<main\b[^>]*>(.*?)</main>",
    ]:
        match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        if match:
            body = match.group(1)
            break

    if not body:
        body = html

    # Block-level elements → newlines
    body = re.sub(
        r"<(?:br|p|div|tr|li|h[1-6])\b[^>]*>",
        "\n",
        body,
        flags=re.IGNORECASE,
    )

    # Strip remaining tags
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

    # Normalise whitespace (mirrors EdgarScraper.fetch_and_clean)
    body = re.sub(r"\r\n|\r", "\n", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r" +\n", "\n", body)
    body = re.sub(r"\n{4,}", "\n\n\n", body)

    return body.strip()
