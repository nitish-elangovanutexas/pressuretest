"""
Transcript parser for SEC EDGAR earnings call transcripts.

Responsibilities:
  1. Locate the boundary between prepared remarks and Q&A.
  2. Split Q&A text into discrete speaker turns.
  3. Attribute each turn to: CEO, CFO, Analyst, Operator, or Unknown.
  4. Emit a validated EarningsCallTranscript Pydantic model.

Design notes:
  - Pure text processing; no external NLP dependencies in Phase 1.
  - Multiple Q&A-boundary strategies tried in order of reliability.
  - Speaker role attribution uses title hints, an executive-name registry
    (built from the prepared-remarks section), and a hardcoded list of
    well-known analyst firm names.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

SpeakerRole = Literal["CEO", "CFO", "Analyst", "Operator", "Unknown"]

# Compiled title patterns for executive role detection
_CEO_RE = re.compile(
    r"\b(?:chief\s+executive|ceo|president\s+and\s+ceo|co-?ceo)\b",
    re.IGNORECASE,
)
_CFO_RE = re.compile(
    r"\b(?:chief\s+financial|cfo|chief\s+finance|"
    r"executive\s+vp.{0,20}finance|senior\s+vp.{0,20}finance)\b",
    re.IGNORECASE,
)

# Patterns that mark the beginning of the Q&A section
_QA_BOUNDARY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bquestions?\s+and\s+answers?\b", re.IGNORECASE),
    re.compile(r"\bq\s*&\s*a\b", re.IGNORECASE),
    re.compile(r"\bq\s+and\s+a\b", re.IGNORECASE),
    re.compile(
        r"\bwe\s+(?:will|would|are\s+going\s+to)\s+(?:now\s+)?"
        r"(?:begin|open|take)\s+(?:the\s+)?(?:floor\s+for\s+)?questions?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bopen(?:ing)?\s+(?:the\s+)?(?:floor|lines?|call)\s+"
        r"(?:for|to)\s+questions?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bplease\s+proceed\s+with\s+(?:the\s+)?questions?\b", re.IGNORECASE),
    re.compile(r"\byour\s+lines?\s+(?:are\s+)?(?:now\s+)?open\b", re.IGNORECASE),
]

# Well-known sell-side firms used to identify analyst speakers
_ANALYST_FIRMS_RE = re.compile(
    r"\b(?:goldman\s*sachs|morgan\s*stanley|jpmorgan|jp\s*morgan|"
    r"citigroup|citi(?:bank)?|barclays|ubs|credit\s*suisse|deutsche\s*bank|"
    r"bank\s+of\s+america|bofa|merrill\s*lynch|wells\s*fargo|raymond\s*james|"
    r"baird|needham|piper\s*sandler|oppenheimer|jefferies|cowen|td\s*cowen|"
    r"stifel|rbc\s*capital|keybanc|mizuho|bmo\s*capital|scotiabank|"
    r"loop\s*capital|truist|wedbush|bernstein|evercore|wolfe\s*research|"
    r"rosenblatt|canaccord|william\s*blair|guggenheim|hsbc)\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------------
# Pydantic output models
# ------------------------------------------------------------------


class SpeakerTurn(BaseModel):
    speaker_role: SpeakerRole
    speaker_name: str
    text: str
    turn_index: int


class EarningsCallTranscript(BaseModel):
    ticker: str
    company_name: str
    call_date: date
    quarter: int | None
    year: int | None
    prepared_remarks: str
    qa_turns: list[SpeakerTurn] = Field(default_factory=list)


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


class TranscriptParser:
    """
    Parses raw earnings call transcript text into an EarningsCallTranscript.

    Usage::

        parser = TranscriptParser()
        transcript = parser.parse(
            raw_text,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
    """

    def parse(
        self,
        raw_text: str,
        *,
        ticker: str,
        company_name: str,
        filing_date: date,
    ) -> EarningsCallTranscript:
        """
        Parse raw transcript text into a structured EarningsCallTranscript.

        Args:
            raw_text:     Cleaned transcript text (HTML already stripped).
            ticker:       Stock ticker symbol.
            company_name: Official company name from EDGAR.
            filing_date:  SEC filing date of the parent 8-K.
        """
        qa_boundary = self._find_qa_boundary(raw_text)

        if qa_boundary is None:
            log.warning(
                "No Q&A section detected in transcript — "
                "treating full text as prepared remarks"
            )
            prepared = raw_text
            qa_text = ""
        else:
            prepared = raw_text[:qa_boundary].strip()
            qa_text = raw_text[qa_boundary:].strip()

        exec_registry = _build_exec_registry(prepared)
        log.debug("Detected executives: %s", exec_registry)

        qa_turns = self._parse_qa_turns(qa_text, exec_registry)
        quarter, year = _extract_quarter_year(raw_text, filing_date)

        return EarningsCallTranscript(
            ticker=ticker.upper(),
            company_name=company_name,
            call_date=filing_date,
            quarter=quarter,
            year=year,
            prepared_remarks=prepared,
            qa_turns=qa_turns,
        )

    # ------------------------------------------------------------------
    # Q&A boundary detection
    # ------------------------------------------------------------------

    def _find_qa_boundary(self, text: str) -> int | None:
        """
        Return the character offset where the Q&A section begins, or None.

        Strategies (tried in order):
          1. Explicit header line matching known Q&A marker phrases.
          2. Operator line whose text signals the question period.
          3. "Q:" shorthand notation used by some transcript providers.
        """
        # Strategy 1: match Q&A header phrases near a line boundary
        for pattern in _QA_BOUNDARY_PATTERNS:
            for match in pattern.finditer(text):
                # Require the match to appear close to a line start
                line_start = text.rfind("\n", 0, match.start())
                leading = text[line_start + 1 : match.start()].strip()
                if len(leading) < 50:
                    return match.start()

        # Strategy 2: Operator line that explicitly invites questions
        op_qa = re.compile(
            r"\nOperator\s*:\s*[^\n]*"
            r"(?:question|Q&A|please go ahead|next question|open.*line)",
            re.IGNORECASE,
        )
        match = op_qa.search(text)
        if match:
            return match.start()

        # Strategy 3: "Q:" shorthand (used by Seeking Alpha-style transcripts)
        shorthand = re.compile(r"(?m)^\s*Q\s*:\s+\S")
        match = shorthand.search(text)
        if match:
            return match.start()

        return None

    # ------------------------------------------------------------------
    # Speaker turn splitting and attribution
    # ------------------------------------------------------------------

    def _parse_qa_turns(
        self, qa_text: str, exec_registry: dict[str, SpeakerRole]
    ) -> list[SpeakerTurn]:
        if not qa_text:
            return []

        raw_turns = _split_speaker_turns(qa_text)
        turns: list[SpeakerTurn] = []
        for idx, (name, title_hint, body) in enumerate(raw_turns):
            role = _attribute_role(name, title_hint, exec_registry)
            turns.append(
                SpeakerTurn(
                    speaker_role=role,
                    speaker_name=_normalise_name(name),
                    text=body.strip(),
                    turn_index=idx,
                )
            )
        return turns


# ------------------------------------------------------------------
# Speaker-turn splitting (module-level for testability)
# ------------------------------------------------------------------


def _split_speaker_turns(text: str) -> list[tuple[str, str, str]]:
    """
    Split Q&A text on speaker-change markers.

    Handles two common formats:
      • "Name [- Title]:↵body text"  — name-only line, body on subsequent lines
      • "Name [- Title]: body text"  — name and body on same line

    Returns a list of (raw_name, title_hint, body_text) tuples.
    """
    # Matches a line whose content before ":" looks like a speaker identifier:
    #   - Starts with a capital letter
    #   - May contain letters, spaces, commas, hyphens, apostrophes, periods
    #   - Total content before ":" is at most 120 chars
    speaker_re = re.compile(
        r"(?m)^(?P<tag>[A-Z][A-Za-z''.,\- ]{1,118}):(?P<inline>[^\n]*)$"
    )

    matches = list(speaker_re.finditer(text))

    # Fallback: "Q:" / "A:" shorthand
    if not matches:
        qa_re = re.compile(r"(?m)^(?P<tag>[QA])\s*:\s*(?P<inline>[^\n]*)$")
        matches = list(qa_re.finditer(text))

    if not matches:
        return [("Unknown", "", text)]

    turns: list[tuple[str, str, str]] = []
    for i, m in enumerate(matches):
        tag = m.group("tag").strip()
        inline = m.group("inline").strip()

        # Exclude lines that look like regular sentences (e.g., a company name
        # followed by a colon in the middle of a paragraph).
        if _is_false_positive_speaker(tag):
            continue

        name, title_hint = _parse_speaker_tag(tag)
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        post_body = text[m.end() : next_start].strip()

        body = (inline + "\n" + post_body).strip() if inline else post_body
        turns.append((name, title_hint, body))

    return turns if turns else [("Unknown", "", text)]


def _is_false_positive_speaker(tag: str) -> bool:
    """
    Reject tags that are clearly not speaker identifiers.

    Heuristics:
      - Contains digits (revenue figures, years embedded in the sentence)
      - Very short single lowercase-ish words
      - Longer than 100 chars (description text, not a name)
    """
    if len(tag) > 100:
        return True
    if re.search(r"\d", tag):
        return True
    return False


def _parse_speaker_tag(tag: str) -> tuple[str, str]:
    """
    Extract (name, title_hint) from a raw speaker tag string.

    Handles:
      "Tim Cook, Chief Executive Officer, Apple Inc." → ("Tim Cook", "Chief Executive Officer")
      "Erik Woodring - Morgan Stanley"                → ("Erik Woodring", "Morgan Stanley")
      "OPERATOR"                                      → ("Operator", "")
      "Tim Cook"                                      → ("Tim Cook", "")
    """
    # Em-dash or hyphen separator: "Name - Title"
    dash_match = re.match(r"^(.+?)\s*[-–]\s*(.+)$", tag)
    if dash_match:
        return dash_match.group(1).strip(), dash_match.group(2).strip()

    # Comma separator: "Name, Title, ..."
    parts = [p.strip() for p in tag.split(",", maxsplit=2)]
    if len(parts) >= 2:
        return parts[0], parts[1]

    return tag.strip(), ""


# ------------------------------------------------------------------
# Role attribution (module-level for direct testing)
# ------------------------------------------------------------------


def _attribute_role(
    name: str,
    title_hint: str,
    exec_registry: dict[str, SpeakerRole],
) -> SpeakerRole:
    """
    Map a speaker name and optional title hint to a SpeakerRole.

    Resolution order:
      1. Exact name matches ("Operator", "Q", "A").
      2. Executive name registry built from prepared remarks.
      3. Title-hint keyword matching (CEO/CFO patterns).
      4. Analyst firm name in name or title hint.
      5. Unknown fallback.
    """
    normalised = name.strip().upper()

    if normalised in ("OPERATOR", "OP"):
        return "Operator"
    if normalised in ("Q", "QUESTION"):
        return "Analyst"
    if normalised in ("A", "ANSWER"):
        # "A:" in Q/A shorthand belongs to the company side; CEO is safest default
        return "CEO"

    # Registry lookup (built from prepared-remarks intro lines)
    for reg_key, role in exec_registry.items():
        if reg_key in normalised or normalised in reg_key:
            return role

    # Title-hint keyword matching
    combined = f"{name} {title_hint}".lower()
    if _CEO_RE.search(combined):
        return "CEO"
    if _CFO_RE.search(combined):
        return "CFO"

    # Sell-side firm in name or title indicates an analyst
    if _ANALYST_FIRMS_RE.search(name) or _ANALYST_FIRMS_RE.search(title_hint):
        return "Analyst"

    return "Unknown"


# ------------------------------------------------------------------
# Executive name registry (built from prepared remarks)
# ------------------------------------------------------------------


def _build_exec_registry(prepared_remarks: str) -> dict[str, SpeakerRole]:
    """
    Scan the first ~3 000 chars of prepared remarks to build a
    {NORMALISED_NAME: role} registry.

    Handles two common formats:
      • "Name, Title ..."          — e.g. "Tim Cook, Chief Executive Officer"
      • "Title — Name"            — e.g. "Chief Executive Officer — Timothy D. Cook"
        (Motley Fool call-participants block uses em-dash or regular hyphen)
    """
    registry: dict[str, SpeakerRole] = {}
    snippet = prepared_remarks[:3000]

    # Format 1: "Name[,] Title" — original pattern.
    name_then_title = re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"  # proper name (2-4 words)
        r"[,\s]+"
        r"([^,\n.]{5,80})",                          # title phrase
    )
    for m in name_then_title.finditer(snippet):
        name_raw, title = m.group(1), m.group(2)
        if _CEO_RE.search(title):
            registry[name_raw.upper()] = "CEO"
        elif _CFO_RE.search(title):
            registry[name_raw.upper()] = "CFO"

    # Format 2: "Title — Name" (em-dash, en-dash, or hyphen; no newline between).
    # Matches the Motley Fool "Call participants" block.
    _NAME_FRAG = r"([A-Z][A-Za-z.'\-]+(?:[^\S\n]+[A-Z][A-Za-z.'\-]+){1,3})"
    title_then_name_patterns: list[tuple[re.Pattern, SpeakerRole]] = [
        (
            re.compile(
                r"(?:chief[^\S\n]+executive[^\S\n]+officer|ceo)"
                r"[^\S\n]*[—–\-][^\S\n]*" + _NAME_FRAG,
                re.IGNORECASE,
            ),
            "CEO",
        ),
        (
            re.compile(
                r"(?:chief[^\S\n]+financial[^\S\n]+officer|cfo)"
                r"[^\S\n]*[—–\-][^\S\n]*" + _NAME_FRAG,
                re.IGNORECASE,
            ),
            "CFO",
        ),
    ]
    for pattern, role in title_then_name_patterns:
        for m in pattern.finditer(snippet):
            registry[m.group(1).upper()] = role

    return registry


# ------------------------------------------------------------------
# Quarter / year extraction
# ------------------------------------------------------------------


def _extract_quarter_year(
    text: str, filing_date: date
) -> tuple[int | None, int | None]:
    """
    Infer fiscal quarter and calendar year from transcript text.

    Tries several patterns against the first 2 000 characters, then
    falls back to deriving the quarter from the filing month.
    """
    snippet = text[:2000]

    # "Q3 2024" or "Q3 FY2024"
    m = re.search(
        r"\bQ([1-4])\s*(?:fy\s*|fiscal\s*)?(\d{4})\b", snippet, re.IGNORECASE
    )
    if m:
        return int(m.group(1)), int(m.group(2))

    # "FY2024 Q3" or "fiscal year 2024 Q3"
    m = re.search(
        r"\b(?:fy\s*|fiscal\s*(?:year\s*)?)(\d{4})\s*Q([1-4])\b",
        snippet,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(2)), int(m.group(1))

    # "third quarter 2024" / "third quarter of fiscal 2024"
    ordinals = {"first": 1, "second": 2, "third": 3, "fourth": 4}
    m = re.search(
        r"\b(first|second|third|fourth)\s+quarter\s+"
        r"(?:of\s+)?(?:fiscal\s+)?(\d{4})\b",
        snippet,
        re.IGNORECASE,
    )
    if m:
        q = ordinals.get(m.group(1).lower())
        if q:
            return q, int(m.group(2))

    # Fallback: derive approximate quarter from filing month
    month = filing_date.month
    return (month - 1) // 3 + 1, filing_date.year


# ------------------------------------------------------------------
# Name normalisation
# ------------------------------------------------------------------


def _normalise_name(raw: str) -> str:
    """Title-case all-caps names; otherwise return stripped."""
    stripped = raw.strip()
    if stripped.isupper() and len(stripped) > 1:
        return stripped.title()
    return stripped
