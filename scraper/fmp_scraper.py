"""
Financial Modeling Prep (FMP) earnings call transcript scraper.

Fetches live earnings call transcripts from the FMP API and returns
FilingRecord instances compatible with TranscriptParser — the same
(company_name, list[FilingRecord]) tuple shape as hf_loader.py.

Primary endpoint (v3):
    GET /api/v3/earning_call_transcript/{ticker}?apikey={key}

Response shape (list, newest-first):
    [{"symbol": "AAPL", "quarter": 2, "year": 2023,
      "date": "2023-05-04 00:00:00", "content": "<full text>"},
     ...]

Company name is resolved via the stable profile endpoint:
    GET /stable/profile?symbol={ticker}&apikey={key}

Rate limit: 1 request/second (FMP basic-tier guideline).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import date

import httpx

from scraper.edgar_scraper import FilingRecord

log = logging.getLogger(__name__)

_FMP_BASE = "https://financialmodelingprep.com"
# Primary endpoint (v3 legacy); automatic fallback to stable if 403.
_TRANSCRIPT_PATH_V3 = "/api/v3/earning_call_transcript/{ticker}"
_TRANSCRIPT_PATH_STABLE = "/stable/earning-call-transcript"
_PROFILE_PATH = "/stable/profile"
_MIN_INTERVAL = 1.0  # seconds — FMP: max 1 req/s


def _parse_date(raw: str) -> str:
    """Convert 'YYYY-MM-DD HH:MM:SS' (or bare 'YYYY-MM-DD') to ISO date."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", raw.strip())
    return m.group(1) if m else date.today().isoformat()


class FMPScraper:
    """
    Async FMP transcript client.  Use as an async context manager::

        async with FMPScraper() as fmp:
            company_name, records = await fmp.fetch_transcripts("AAPL")
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("FMP_API_KEY", "")
        self._client: httpx.AsyncClient | None = None
        self._last_req_ts: float = 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "FMPScraper":
        self._client = httpx.AsyncClient(
            base_url=_FMP_BASE,
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

    async def _get(self, path: str, params: dict) -> httpx.Response:
        """Rate-limited GET; returns the response (caller checks status)."""
        assert self._client is not None, "Use FMPScraper inside 'async with'"
        elapsed = time.monotonic() - self._last_req_ts
        if elapsed < _MIN_INTERVAL:
            await asyncio.sleep(_MIN_INTERVAL - elapsed)
        self._last_req_ts = time.monotonic()
        params["apikey"] = self._api_key
        return await self._client.get(path, params=params)

    # ------------------------------------------------------------------
    # Company name lookup
    # ------------------------------------------------------------------

    async def _company_name(self, ticker: str) -> str:
        """Return company name from the FMP profile endpoint."""
        try:
            resp = await self._get(_PROFILE_PATH, {"symbol": ticker.upper()})
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list):
                    name = data[0].get("companyName") or data[0].get("name")
                    if name:
                        return str(name).strip()
        except Exception as exc:
            log.debug("FMP profile lookup failed for %s: %s", ticker, exc)
        return ticker.upper()

    # ------------------------------------------------------------------
    # Transcript payload fetch (v3 → stable fallback)
    # ------------------------------------------------------------------

    async def _fetch_payload(self, ticker_upper: str) -> list[dict] | None:
        """
        Try the v3 endpoint; if it returns 403 (legacy), try the stable endpoint.
        Returns a list of transcript dicts, an empty list when none exist, or
        None on an unrecoverable error/access denial.
        """
        # --- v3 ---
        v3_path = _TRANSCRIPT_PATH_V3.format(ticker=ticker_upper)
        try:
            resp = await self._get(v3_path, {})
        except Exception as exc:
            log.error("FMP v3 request failed for %s: %s", ticker_upper, exc)
            return None

        if resp.status_code == 200:
            try:
                return resp.json() or []
            except Exception as exc:
                log.error("FMP v3 response invalid JSON for %s: %s", ticker_upper, exc)
                return None

        if resp.status_code == 403:
            log.debug(
                "FMP v3 endpoint returned 403 for %s (legacy) — trying stable endpoint",
                ticker_upper,
            )
            # Fall through to stable endpoint below.

        elif resp.status_code == 402:
            log.warning(
                "FMP v3 endpoint requires a premium subscription (HTTP 402 for %s).",
                ticker_upper,
            )
            # Still try stable — it may have a different entitlement.

        else:
            log.warning(
                "FMP v3 returned HTTP %d for %s — trying stable endpoint",
                resp.status_code,
                ticker_upper,
            )

        # --- stable fallback ---
        try:
            resp2 = await self._get(
                _TRANSCRIPT_PATH_STABLE, {"symbol": ticker_upper}
            )
        except Exception as exc:
            log.error("FMP stable request failed for %s: %s", ticker_upper, exc)
            return None

        if resp2.status_code == 200:
            try:
                return resp2.json() or []
            except Exception as exc:
                log.error("FMP stable response invalid JSON for %s: %s", ticker_upper, exc)
                return None

        if resp2.status_code in (402, 403):
            log.warning(
                "FMP transcript access denied for %s (HTTP %d on both v3 and stable). "
                "Upgrade your FMP plan at https://financialmodelingprep.com/ "
                "or use --source hf or --source edgar.",
                ticker_upper,
                resp2.status_code,
            )
        else:
            log.warning(
                "FMP stable returned HTTP %d for %s — no transcripts available",
                resp2.status_code,
                ticker_upper,
            )
        return None

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------

    async def fetch_transcripts(
        self,
        ticker: str,
        *,
        max_transcripts: int = 5,
    ) -> tuple[str, list[FilingRecord]]:
        """
        Fetch recent earnings call transcripts for *ticker* from FMP.

        Args:
            ticker:          Stock ticker symbol (e.g. "AAPL").
            max_transcripts: Maximum number of FilingRecords to return.

        Returns:
            ``(company_name, list[FilingRecord])`` with ``raw_text`` set to the
            full transcript text.  Returns an empty list — not an exception —
            when the API returns no results or an access error.
        """
        if not self._api_key:
            log.error(
                "FMP_API_KEY is not set — cannot fetch transcripts from FMP. "
                "Add FMP_API_KEY to your .env file."
            )
            return ticker.upper(), []

        ticker_upper = ticker.upper()
        log.info("FMP: fetching transcripts for %s …", ticker_upper)

        # Try v3 endpoint first; if legacy-blocked (403), fall back to stable.
        payload = await self._fetch_payload(ticker_upper)
        if payload is None:
            return ticker_upper, []

        if not isinstance(payload, list) or not payload:
            log.info("FMP: no transcripts returned for %s", ticker_upper)
            return ticker_upper, []

        # Newest-first; FMP already returns them sorted, but be safe
        payload.sort(key=lambda r: str(r.get("date") or ""), reverse=True)
        payload = payload[:max_transcripts]

        company_name = await self._company_name(ticker_upper)

        records: list[FilingRecord] = []
        for item in payload:
            raw_text = str(item.get("content") or "").strip()
            if not raw_text:
                log.debug("FMP: empty content for %s — skipping", ticker_upper)
                continue

            date_str = _parse_date(str(item.get("date") or ""))
            quarter = item.get("quarter")
            year = item.get("year")

            # Embed quarter/year in the accession number for traceability
            acc_no = f"FMP-{ticker_upper}-{date_str.replace('-', '')}"
            if quarter and year:
                acc_no += f"-Q{quarter}{year}"

            records.append(
                FilingRecord(
                    accession_no=acc_no,
                    filing_date=date_str,
                    items="2.02",
                    exhibit_url=None,
                    raw_text=raw_text,
                )
            )
            log.info(
                "FMP transcript loaded: %s %s (Q%s %s)",
                ticker_upper,
                date_str,
                quarter,
                year,
            )

        return company_name, records
