"""
Unit tests for scraper/transcript_parser.py.

Run with:
    pytest tests/test_parser.py -v
    pytest tests/test_parser.py -v --tb=short
"""
from __future__ import annotations

from datetime import date

import pytest

from scraper.transcript_parser import (
    EarningsCallTranscript,
    SpeakerTurn,
    TranscriptParser,
    _attribute_role,
    _build_exec_registry,
    _extract_quarter_year,
    _parse_speaker_tag,
    _split_speaker_turns,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

SAMPLE_TRANSCRIPT = """\
Apple Inc.
Q1 Fiscal 2024 Earnings Conference Call
February 1, 2024

PREPARED REMARKS

Operator: Good afternoon. My name is Sarah and I will be your conference operator
today. Welcome everyone to the Apple Q1 fiscal 2024 earnings conference call.
Tim Cook, Chief Executive Officer, Apple Inc., will begin with some opening remarks.

Tim Cook, Chief Executive Officer, Apple Inc.:
Good afternoon and thank you for joining us. We are pleased to report revenue
of $119.6 billion, up 2% year over year, an all-time record for our December quarter.
Our installed base of active devices has now surpassed 2.2 billion.

Luca Maestri, Chief Financial Officer, Apple Inc.:
Thank you, Tim. Gross margin came in at 45.9%, a new all-time record for Apple.
Operating expenses were $14.5 billion, in line with our guidance.

QUESTION AND ANSWER SESSION

Operator: We will now begin the question and answer session. To ask a question,
please press star one on your telephone keypad. Our first question comes from
Erik Woodring of Morgan Stanley. Erik, please go ahead.

Erik Woodring - Morgan Stanley:
Hi, good afternoon and congrats on the results. Tim, can you talk about iPhone
demand trends in China and what you are seeing competitively there?

Tim Cook, Chief Executive Officer, Apple Inc.:
Sure, Erik. We saw some pressure in Mainland China but remain optimistic about
the long-term opportunity. We have a very loyal installed base there.

Operator: Our next question comes from Samik Chatterjee of JPMorgan.
Please go ahead.

Samik Chatterjee - JPMorgan:
Thanks. Luca, on the gross margin guidance for Q2, can you walk us through
the main puts and takes?

Luca Maestri, Chief Financial Officer, Apple Inc.:
Of course. We expect Q2 gross margin of approximately 46% to 47%, supported
by a favorable mix shift toward services.

Operator: Thank you all for participating. That concludes today's call.
"""

MINIMAL_PREPARED_ONLY = """\
Apple Inc.
Q2 2024 Earnings Call
April 30, 2024

Tim Cook - CEO:
Revenue was $90.8 billion, up 5% year over year. Services hit a new all-time record.
"""

QA_SHORTHAND_TRANSCRIPT = """\
Acme Corp Q3 2023 Earnings Call
July 28, 2023

Prepared remarks here. Revenue grew 12%.

Q: Can you discuss margin trends?
A: Sure, margins expanded 200 basis points year over year.

Q: What is your outlook for next quarter?
A: We expect continued improvement in the 15-17% range.
"""


@pytest.fixture
def parser() -> TranscriptParser:
    return TranscriptParser()


# ------------------------------------------------------------------
# Q&A boundary detection
# ------------------------------------------------------------------


class TestQaBoundaryDetection:
    def test_detects_explicit_qa_header(self, parser: TranscriptParser) -> None:
        boundary = parser._find_qa_boundary(SAMPLE_TRANSCRIPT)
        assert boundary is not None
        assert "QUESTION AND ANSWER SESSION" in SAMPLE_TRANSCRIPT[boundary : boundary + 60]

    def test_boundary_splits_text_correctly(self, parser: TranscriptParser) -> None:
        boundary = parser._find_qa_boundary(SAMPLE_TRANSCRIPT)
        prepared = SAMPLE_TRANSCRIPT[:boundary]
        qa = SAMPLE_TRANSCRIPT[boundary:]
        assert "Luca Maestri" in prepared
        assert "Erik Woodring" in qa

    def test_returns_none_when_no_qa_section(self, parser: TranscriptParser) -> None:
        assert parser._find_qa_boundary(MINIMAL_PREPARED_ONLY) is None

    def test_detects_operator_question_signal(self, parser: TranscriptParser) -> None:
        text = (
            "Prepared remarks.\n\n"
            "Operator: We will now begin the question and answer session.\n\n"
            "Analyst: My first question is about margins.\n"
        )
        boundary = parser._find_qa_boundary(text)
        assert boundary is not None

    def test_detects_qa_shorthand(self, parser: TranscriptParser) -> None:
        boundary = parser._find_qa_boundary(QA_SHORTHAND_TRANSCRIPT)
        assert boundary is not None
        assert "Q:" in QA_SHORTHAND_TRANSCRIPT[boundary : boundary + 10]


# ------------------------------------------------------------------
# Speaker tag parsing
# ------------------------------------------------------------------


class TestParseSpeakerTag:
    def test_dash_separator(self) -> None:
        name, title = _parse_speaker_tag("Erik Woodring - Morgan Stanley")
        assert name == "Erik Woodring"
        assert title == "Morgan Stanley"

    def test_comma_separator(self) -> None:
        name, title = _parse_speaker_tag(
            "Tim Cook, Chief Executive Officer, Apple Inc."
        )
        assert name == "Tim Cook"
        assert "Chief Executive Officer" in title

    def test_no_separator(self) -> None:
        name, title = _parse_speaker_tag("Operator")
        assert name == "Operator"
        assert title == ""

    def test_em_dash(self) -> None:
        name, title = _parse_speaker_tag("Jane Smith – Barclays")
        assert name == "Jane Smith"
        assert title == "Barclays"


# ------------------------------------------------------------------
# Speaker turn splitting
# ------------------------------------------------------------------


class TestSplitSpeakerTurns:
    def test_correct_turn_count(self) -> None:
        boundary_idx = SAMPLE_TRANSCRIPT.index("QUESTION AND ANSWER SESSION")
        qa_text = SAMPLE_TRANSCRIPT[boundary_idx:]
        turns = _split_speaker_turns(qa_text)
        # Operator intro + Erik + Tim + Operator + Samik + Luca + Operator closing
        assert len(turns) >= 5

    def test_turn_names_populated(self) -> None:
        boundary_idx = SAMPLE_TRANSCRIPT.index("QUESTION AND ANSWER SESSION")
        qa_text = SAMPLE_TRANSCRIPT[boundary_idx:]
        turns = _split_speaker_turns(qa_text)
        for name, _, body in turns:
            assert name.strip() != ""

    def test_qa_shorthand_fallback(self) -> None:
        text = "Q: What is your revenue outlook?\nA: We expect 10% growth.\n"
        turns = _split_speaker_turns(text)
        assert len(turns) == 2
        assert turns[0][0] == "Q"
        assert turns[1][0] == "A"

    def test_unknown_fallback_for_unstructured_text(self) -> None:
        turns = _split_speaker_turns("No speaker markers here at all.")
        assert len(turns) == 1
        assert turns[0][0] == "Unknown"


# ------------------------------------------------------------------
# Role attribution
# ------------------------------------------------------------------


class TestAttributeRole:
    def test_operator_exact(self) -> None:
        assert _attribute_role("Operator", "", {}) == "Operator"

    def test_operator_allcaps(self) -> None:
        assert _attribute_role("OPERATOR", "", {}) == "Operator"

    def test_ceo_via_title_long(self) -> None:
        assert _attribute_role("Tim Cook", "Chief Executive Officer", {}) == "CEO"

    def test_ceo_via_title_abbrev(self) -> None:
        assert _attribute_role("Tim Cook", "CEO", {}) == "CEO"

    def test_cfo_via_title(self) -> None:
        assert _attribute_role("Luca Maestri", "Chief Financial Officer", {}) == "CFO"

    def test_cfo_via_abbrev(self) -> None:
        assert _attribute_role("Luca Maestri", "CFO", {}) == "CFO"

    def test_analyst_via_firm_in_title(self) -> None:
        assert _attribute_role("Erik Woodring", "Morgan Stanley", {}) == "Analyst"

    def test_analyst_via_firm_in_name(self) -> None:
        assert _attribute_role("Goldman Sachs Analyst", "", {}) == "Analyst"

    def test_exec_registry_lookup(self) -> None:
        registry = {"TIM COOK": "CEO", "LUCA MAESTRI": "CFO"}
        assert _attribute_role("Tim Cook", "", registry) == "CEO"
        assert _attribute_role("Luca Maestri", "", registry) == "CFO"

    def test_q_shorthand_is_analyst(self) -> None:
        assert _attribute_role("Q", "", {}) == "Analyst"

    def test_a_shorthand_is_ceo(self) -> None:
        assert _attribute_role("A", "", {}) == "CEO"

    def test_unknown_fallback(self) -> None:
        assert _attribute_role("John Smith", "", {}) == "Unknown"


# ------------------------------------------------------------------
# Executive registry building
# ------------------------------------------------------------------


class TestBuildExecRegistry:
    def test_detects_ceo(self) -> None:
        text = "Tim Cook, Chief Executive Officer, Apple Inc."
        registry = _build_exec_registry(text)
        assert any("CEO" == v for v in registry.values())

    def test_detects_cfo(self) -> None:
        text = "Luca Maestri, Chief Financial Officer, Apple Inc."
        registry = _build_exec_registry(text)
        assert any("CFO" == v for v in registry.values())

    def test_empty_on_no_executives(self) -> None:
        text = "Revenue was $119 billion. Margins improved year over year."
        registry = _build_exec_registry(text)
        assert isinstance(registry, dict)


# ------------------------------------------------------------------
# Quarter/year extraction
# ------------------------------------------------------------------


class TestExtractQuarterYear:
    def test_q_notation(self) -> None:
        q, y = _extract_quarter_year("Q1 fiscal 2024 earnings call", date(2024, 2, 1))
        assert q == 1
        assert y == 2024

    def test_q_notation_no_fiscal(self) -> None:
        q, y = _extract_quarter_year("Q3 2023 results.", date(2023, 7, 30))
        assert q == 3
        assert y == 2023

    def test_ordinal_notation(self) -> None:
        q, y = _extract_quarter_year(
            "Welcome to the third quarter 2023 results call.", date(2023, 10, 28)
        )
        assert q == 3
        assert y == 2023

    def test_fy_notation(self) -> None:
        q, y = _extract_quarter_year("FY2024 Q2 earnings", date(2024, 5, 1))
        assert q == 2
        assert y == 2024

    def test_fallback_to_filing_date_q1(self) -> None:
        q, y = _extract_quarter_year("No quarter info.", date(2024, 2, 1))
        assert q == 1
        assert y == 2024

    def test_fallback_to_filing_date_q3(self) -> None:
        q, y = _extract_quarter_year("No quarter info.", date(2024, 8, 15))
        assert q == 3
        assert y == 2024


# ------------------------------------------------------------------
# Full pipeline integration
# ------------------------------------------------------------------


class TestFullPipeline:
    def test_output_type(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            SAMPLE_TRANSCRIPT,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
        assert isinstance(transcript, EarningsCallTranscript)

    def test_ticker_uppercased(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            SAMPLE_TRANSCRIPT,
            ticker="aapl",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
        assert transcript.ticker == "AAPL"

    def test_quarter_and_year(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            SAMPLE_TRANSCRIPT,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
        assert transcript.quarter == 1
        assert transcript.year == 2024

    def test_prepared_remarks_nonempty(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            SAMPLE_TRANSCRIPT,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
        assert len(transcript.prepared_remarks) > 50
        assert "Luca Maestri" in transcript.prepared_remarks

    def test_qa_turns_present(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            SAMPLE_TRANSCRIPT,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
        assert len(transcript.qa_turns) >= 4

    def test_turn_indices_sequential(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            SAMPLE_TRANSCRIPT,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
        indices = [t.turn_index for t in transcript.qa_turns]
        assert indices == list(range(len(indices)))

    def test_all_turns_have_role(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            SAMPLE_TRANSCRIPT,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
        valid_roles = {"CEO", "CFO", "Analyst", "Operator", "Unknown"}
        for turn in transcript.qa_turns:
            assert turn.speaker_role in valid_roles

    def test_analyst_turns_detected(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            SAMPLE_TRANSCRIPT,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
        roles = {t.speaker_role for t in transcript.qa_turns}
        assert "Analyst" in roles

    def test_operator_turns_detected(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            SAMPLE_TRANSCRIPT,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
        roles = {t.speaker_role for t in transcript.qa_turns}
        assert "Operator" in roles

    def test_speaker_names_nonempty(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            SAMPLE_TRANSCRIPT,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
        for turn in transcript.qa_turns:
            assert turn.speaker_name.strip() != ""

    def test_turn_text_nonempty(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            SAMPLE_TRANSCRIPT,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
        for turn in transcript.qa_turns:
            assert len(turn.text.strip()) > 0

    def test_prepared_only_transcript(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            MINIMAL_PREPARED_ONLY,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 4, 30),
        )
        assert transcript.qa_turns == []
        assert len(transcript.prepared_remarks) > 0

    def test_qa_shorthand_format(self, parser: TranscriptParser) -> None:
        transcript = parser.parse(
            QA_SHORTHAND_TRANSCRIPT,
            ticker="ACME",
            company_name="Acme Corp",
            filing_date=date(2023, 7, 28),
        )
        assert len(transcript.qa_turns) >= 2

    def test_model_serialisable(self, parser: TranscriptParser) -> None:
        """model_dump(mode='json') must not raise — covers date serialisation."""
        transcript = parser.parse(
            SAMPLE_TRANSCRIPT,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date=date(2024, 2, 1),
        )
        payload = transcript.model_dump(mode="json")
        assert payload["ticker"] == "AAPL"
        assert isinstance(payload["call_date"], str)
        assert isinstance(payload["qa_turns"], list)
