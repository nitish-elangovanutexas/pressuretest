"""
Analyst question difficulty scorer.

Maps a single analyst question (free text) to a difficulty score in [0, 1]:
    0.0 → soft, open-ended, friendly tone.
    1.0 → hard, confrontational, dense with technical/legal terminology.

Features combined:
    • negative sentiment    — FinBERT (ProsusAI/finbert) zero-shot P(negative).
    • hedge-word density    — softening words reduce difficulty.
    • terminology density   — accounting / legal jargon raises difficulty.
    • question length       — longer multi-part questions are harder.
    • lexical aggressiveness — direct / confrontational vocabulary.

Notes:
    * Zero-shot only — no fine-tuning.
    * FinBERT is lazily loaded the first time `.score()` is called.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

log = logging.getLogger(__name__)

FINBERT_MODEL_ID = "ProsusAI/finbert"

# ------------------------------------------------------------------
# Lexicons
# ------------------------------------------------------------------

_HEDGE_WORDS: frozenset[str] = frozenset(
    {
        "maybe", "perhaps", "possibly", "potentially", "might", "could",
        "would", "appear", "appears", "seem", "seems", "suggest", "suggests",
        "indicate", "indicates", "somewhat", "rather", "fairly", "roughly",
        "approximately", "around", "about", "generally", "typically",
        "usually", "often", "hopefully", "presumably", "arguably",
        "i think", "i believe", "i guess", "i suppose", "we hope",
        "sort of", "kind of", "a bit", "a little", "in general",
    }
)

_AGGRESSIVE_WORDS: frozenset[str] = frozenset(
    {
        "concern", "concerning", "concerned", "worry", "worried", "worrying",
        "problem", "problematic", "issue", "issues", "weakness", "weak",
        "failure", "failed", "fail", "miss", "missed", "missing", "shortfall",
        "decline", "declining", "declined", "challenge", "challenging",
        "pressure", "pressured", "criticism", "criticize", "disappointing",
        "disappointed", "unclear", "lack", "lacking", "headwind",
        "headwinds", "deteriorate", "deteriorating", "struggle",
        "struggling", "loss", "losses", "underperform", "underperforming",
        "risk", "risky", "doubt", "doubts", "skeptical", "skepticism",
        "pushback", "deceleration", "decelerating", "cut", "cuts", "slash",
        "slashed", "slowdown", "slowing", "warning", "warn", "warned",
        "alarm", "alarming", "concerned about", "worry about", "why did",
        "why are", "why is", "why have", "how do you justify",
        "how can you", "explain why", "what went wrong",
    }
)

_TERMINOLOGY_WORDS: frozenset[str] = frozenset(
    {
        # accounting
        "gaap", "non-gaap", "ebitda", "eps", "margin", "margins",
        "amortization", "amortize", "amortized", "depreciation",
        "impairment", "restatement", "writedown", "write-down", "goodwill",
        "accrual", "accruals", "deferred", "capex", "opex", "fcf",
        "free cash flow", "operating leverage", "deferred revenue",
        "backlog", "rpo", "arr", "mrr", "asc", "asc 606", "ifrs",
        "audit", "auditor", "covenant", "covenants", "default",
        "leverage ratio", "debt-to-equity", "net debt", "working capital",
        "dso", "dpo", "inventory turnover", "channel inventory",
        "deferred tax", "tax rate", "effective tax rate",
        "non-cash", "stock-based compensation", "sbc",
        # legal / regulatory
        "litigation", "lawsuit", "subpoena", "investigation", "doj",
        "sec", "ftc", "antitrust", "compliance", "consent decree",
        "settlement", "injunction", "patent", "infringement",
        "10-k", "10-q", "8-k", "material weakness", "going concern",
        "disclosure", "regulatory", "regulator",
    }
)

# Tokens the FinBERT pipeline emits — normalised to lower-case.
_FINBERT_NEG_LABEL = "negative"

# Calibration constants for length / density normalisation.
# Tuned to typical earnings-call analyst questions (40–150 words).
_LENGTH_SAT_WORDS = 120          # length score saturates at this many words
_HEDGE_SAT_RATE = 0.08           # hedge density saturates here
_TERM_SAT_RATE = 0.10            # terminology density saturates here
_AGGR_SAT_RATE = 0.08            # aggressive-word density saturates here

# Feature weights — must sum to 1.0.
_WEIGHTS = {
    "neg_sentiment": 0.30,
    "softness": 0.15,            # = 1 - hedge_density
    "terminology": 0.20,
    "length": 0.10,
    "aggressiveness": 0.25,
}


# ------------------------------------------------------------------
# Data class for feature breakdown
# ------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionFeatures:
    neg_sentiment: float
    hedge_density: float
    terminology_density: float
    length_score: float
    aggressiveness: float
    difficulty: float

    def to_dict(self) -> dict[str, float]:
        return {
            "neg_sentiment": self.neg_sentiment,
            "hedge_density": self.hedge_density,
            "terminology_density": self.terminology_density,
            "length_score": self.length_score,
            "aggressiveness": self.aggressiveness,
            "difficulty": self.difficulty,
        }


# ------------------------------------------------------------------
# FinBERT loader (lazy, cached)
# ------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_finbert():
    """Lazy-load the FinBERT sentiment pipeline; returns None on failure."""
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )
    except ImportError:
        log.warning(
            "transformers not installed; FinBERT sentiment will default to 0.0"
        )
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_ID)
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            top_k=None,        # return all class probabilities
            truncation=True,
            max_length=512,
        )
    except Exception as exc:                                  # pragma: no cover
        log.warning("Failed to initialise FinBERT (%s): %s", FINBERT_MODEL_ID, exc)
        return None


# ------------------------------------------------------------------
# Tokenisation helpers
# ------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _phrase_hits(text_lower: str, phrases: Iterable[str]) -> int:
    """Count occurrences of multi-word phrases inside lower-cased text."""
    return sum(text_lower.count(p) for p in phrases if " " in p or "-" in p)


def _word_hits(tokens: list[str], vocab: frozenset[str]) -> int:
    return sum(1 for t in tokens if t in vocab)


# ------------------------------------------------------------------
# FinBERT-based negative sentiment
# ------------------------------------------------------------------


def _negative_sentiment(text: str) -> float:
    """Return FinBERT P(negative) ∈ [0, 1]; 0.0 if model unavailable."""
    pipe = _load_finbert()
    if pipe is None:
        return 0.0
    try:
        raw = pipe(text)
    except Exception as exc:                                  # pragma: no cover
        log.warning("FinBERT inference failed: %s", exc)
        return 0.0

    # `pipeline(top_k=None)` returns either list[dict] or list[list[dict]]
    if raw and isinstance(raw[0], list):
        raw = raw[0]

    for entry in raw:
        if str(entry.get("label", "")).lower() == _FINBERT_NEG_LABEL:
            return float(entry.get("score", 0.0))
    return 0.0


# ------------------------------------------------------------------
# Public scorer
# ------------------------------------------------------------------


class QuestionScorer:
    """
    Stateless wrapper that computes a difficulty score for a question.

    Examples
    --------
    >>> scorer = QuestionScorer()
    >>> scorer.score("Great quarter, congrats!")          # ≈ very low
    >>> scorer.score(
    ...     "Why did gross margin compress 200 bps despite the favourable mix, "
    ...     "and how do you reconcile that with the prior guidance?"
    ... )                                                  # ≈ high
    """

    def score(self, question: str) -> float:
        """Return difficulty ∈ [0, 1]."""
        return self.score_with_features(question).difficulty

    def score_with_features(self, question: str) -> QuestionFeatures:
        if not question or not question.strip():
            return QuestionFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        tokens = _tokens(question)
        n_tokens = max(len(tokens), 1)
        text_lower = question.lower()

        # 1. negative sentiment (FinBERT, zero-shot)
        neg = _negative_sentiment(question)

        # 2. hedge density
        hedge_n = _word_hits(tokens, _HEDGE_WORDS) + _phrase_hits(text_lower, _HEDGE_WORDS)
        hedge_density = min(hedge_n / n_tokens, 1.0)
        hedge_density_norm = min(hedge_density / _HEDGE_SAT_RATE, 1.0)
        softness = 1.0 - hedge_density_norm

        # 3. terminology density
        term_n = _word_hits(tokens, _TERMINOLOGY_WORDS) + _phrase_hits(
            text_lower, _TERMINOLOGY_WORDS
        )
        term_density = min(term_n / n_tokens, 1.0)
        term_density_norm = min(term_density / _TERM_SAT_RATE, 1.0)

        # 4. length score (saturating)
        length_score = min(n_tokens / _LENGTH_SAT_WORDS, 1.0)

        # 5. aggressiveness
        aggr_n = _word_hits(tokens, _AGGRESSIVE_WORDS) + _phrase_hits(
            text_lower, _AGGRESSIVE_WORDS
        )
        aggr_density = min(aggr_n / n_tokens, 1.0)
        aggressiveness = min(aggr_density / _AGGR_SAT_RATE, 1.0)

        difficulty = (
            _WEIGHTS["neg_sentiment"] * neg
            + _WEIGHTS["softness"] * softness
            + _WEIGHTS["terminology"] * term_density_norm
            + _WEIGHTS["length"] * length_score
            + _WEIGHTS["aggressiveness"] * aggressiveness
        )
        difficulty = max(0.0, min(difficulty, 1.0))

        return QuestionFeatures(
            neg_sentiment=neg,
            hedge_density=hedge_density,
            terminology_density=term_density,
            length_score=length_score,
            aggressiveness=aggressiveness,
            difficulty=difficulty,
        )


__all__ = ["QuestionScorer", "QuestionFeatures", "FINBERT_MODEL_ID"]
