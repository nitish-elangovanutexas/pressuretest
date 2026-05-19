"""
CEO linguistic baseline builder.

Given a set of historical parsed transcripts for a single ticker, this module
extracts the CEO's Q&A responses, computes per-call linguistic statistics, and
stores a pooled baseline JSON in ``data/baselines/{TICKER}.json``.

The baseline includes:
    • per-call metrics — for downstream calibration of the pressure score.
    • aggregate stats  — mean / std for response length, hedge density,
                         FinBERT sentiment distribution, and
                         CEO-answer ↔ analyst-question semantic similarity.
    • centroid        — mean sentence-transformer embedding across all
                         historical CEO answers, used as the reference vector
                         for cosine distance in deviation.py.
    • pressure_baseline — mean / std of leave-one-out-style pressure scores,
                         giving deviation.py its 2-σ flagging threshold.

Run as a script::

    python -m nlp.baseline --ticker AAPL
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ._models import EMBEDDER_MODEL_ID, load_embedder, load_finbert
from .question_scorer import (
    FINBERT_MODEL_ID,
    QuestionScorer,
    _HEDGE_WORDS,
    _phrase_hits,
    _tokens,
    _word_hits,
)

log = logging.getLogger(__name__)

DEFAULT_BASELINE_DIR = Path("data/baselines")

# Minimum quality thresholds for baseline inclusion.
MIN_QA_TURNS = 10
MIN_PREPARED_CHARS = 5000

# Weighting used for the per-call pressure score during baseline calibration.
# Kept in sync with deviation.py.
_W_COSINE = 0.6
_W_SENTIMENT = 0.4


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class CallMetrics:
    """Per-call linguistic snapshot of CEO Q&A behaviour."""

    call_date: str
    quarter: int | None
    year: int | None
    n_answers: int
    mean_answer_length: float
    hedge_density: float
    pos_sentiment_mean: float
    neu_sentiment_mean: float
    neg_sentiment_mean: float
    qa_similarity_mean: float
    question_difficulty_mean: float
    answer_centroid: list[float] = field(default_factory=list)

    def to_jsonable(self, drop_centroid: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if drop_centroid:
            d.pop("answer_centroid", None)
        return d


@dataclass
class CEOBaseline:
    """Pooled baseline for a single ticker."""

    ticker: str
    n_calls: int
    n_skipped: int
    finbert_model: str
    embedder_model: str
    aggregate: dict[str, dict[str, float]]
    centroid: list[float]
    pressure_baseline: dict[str, float]
    per_call: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# Q&A pair extraction
# ------------------------------------------------------------------


# Phase-1 parser leaves many speakers tagged "Unknown" (Motley Fool transcripts
# use a "Chief Executive Officer — Name" header that the registry regex misses).
# We rebuild a CEO / management name set here so baseline construction is
# robust without modifying Phase 1.
# `[^\S\n]` = whitespace except newlines — keeps the name match on a single line.
_CEO_HEADER_RE = re.compile(
    r"(?:chief[^\S\n]+executive[^\S\n]+officer|ceo)[^\S\n]*[—–\-:,][^\S\n]*"
    r"([A-Z][A-Za-z.'\-]+(?:[^\S\n]+[A-Z][A-Za-z.'\-]+){1,3})",
    re.IGNORECASE,
)
_CFO_HEADER_RE = re.compile(
    r"(?:chief[^\S\n]+financial[^\S\n]+officer|cfo)[^\S\n]*[—–\-:,][^\S\n]*"
    r"([A-Z][A-Za-z.'\-]+(?:[^\S\n]+[A-Z][A-Za-z.'\-]+){1,3})",
    re.IGNORECASE,
)
_IR_HEADER_RE = re.compile(
    r"(?:investor[^\S\n]+relations|head[^\S\n]+of[^\S\n]+ir|"
    r"vp[^\S\n]+of[^\S\n]+ir|director,?[^\S\n]+investor[^\S\n]+relations)"
    r"[^\S\n]*[—–\-:,][^\S\n]*"
    r"([A-Z][A-Za-z.'\-]+(?:[^\S\n]+[A-Z][A-Za-z.'\-]+){1,3})",
    re.IGNORECASE,
)


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def _extract_known_speakers(prepared_remarks: str) -> dict[str, set[str]]:
    """Build {role: {normalised_name, ...}} from prepared-remarks headers."""
    snippet = prepared_remarks[:5000]
    return {
        "CEO": {_norm_name(m) for m in _CEO_HEADER_RE.findall(snippet)},
        "CFO": {_norm_name(m) for m in _CFO_HEADER_RE.findall(snippet)},
        "IR": {_norm_name(m) for m in _IR_HEADER_RE.findall(snippet)},
    }


def _classify_turn(
    turn: dict[str, Any], known: dict[str, set[str]]
) -> str:
    """Return one of: 'CEO', 'CFO', 'IR', 'Operator', 'Analyst'."""
    role = turn.get("speaker_role") or "Unknown"
    name = _norm_name(turn.get("speaker_name") or "")
    if role == "Operator":
        return "Operator"
    if role in ("CEO", "CFO"):
        return role
    if name and name in known["CEO"]:
        return "CEO"
    if name and name in known["CFO"]:
        return "CFO"
    if name and name in known["IR"]:
        return "IR"
    if role == "Analyst":
        return "Analyst"
    # Unknown speaker that isn't a known executive → treat as analyst question.
    return "Analyst"


def _iter_qa_pairs(transcript: dict[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield (analyst_question, ceo_answer) pairs from a transcript dict."""
    turns = transcript.get("qa_turns") or []
    known = _extract_known_speakers(transcript.get("prepared_remarks") or "")
    last_question: str | None = None
    for turn in turns:
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        kind = _classify_turn(turn, known)
        if kind == "Analyst":
            last_question = text
        elif kind == "CEO" and last_question:
            yield last_question, text
            last_question = None
        elif kind in ("CFO", "IR", "Operator"):
            # Don't pair management or operator turns with prior analyst text.
            continue


# ------------------------------------------------------------------
# Sentiment + embedding helpers
# ------------------------------------------------------------------


def _sentiment_distribution(text: str) -> tuple[float, float, float]:
    """Return (positive, neutral, negative) probabilities from FinBERT."""
    pipe = load_finbert()
    if pipe is None or not text.strip():
        return 0.0, 1.0, 0.0
    try:
        raw = pipe(text)
    except Exception as exc:                                  # pragma: no cover
        log.warning("FinBERT inference failed: %s", exc)
        return 0.0, 1.0, 0.0

    if raw and isinstance(raw[0], list):
        raw = raw[0]

    bucket = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for entry in raw:
        label = str(entry.get("label", "")).lower()
        if label in bucket:
            bucket[label] = float(entry.get("score", 0.0))
    return bucket["positive"], bucket["neutral"], bucket["negative"]


def _embed_many(texts: list[str]) -> list[list[float]] | None:
    """Encode texts to vectors; returns None if the embedder is unavailable."""
    embedder = load_embedder()
    if embedder is None or not texts:
        return None
    vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [vec.tolist() for vec in vectors]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    return [x / len(vectors) for x in acc]


def _hedge_density(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    n = _word_hits(tokens, _HEDGE_WORDS) + _phrase_hits(text.lower(), _HEDGE_WORDS)
    return n / len(tokens)


# ------------------------------------------------------------------
# Per-call metrics
# ------------------------------------------------------------------


def compute_call_metrics(
    transcript: dict[str, Any],
    question_scorer: QuestionScorer | None = None,
) -> CallMetrics | None:
    """
    Compute the linguistic snapshot for one parsed transcript.

    Returns None if the transcript contains no CEO answers to analyst
    questions, since such a call cannot inform the baseline.
    """
    pairs = list(_iter_qa_pairs(transcript))
    if not pairs:
        return None

    question_scorer = question_scorer or QuestionScorer()

    questions = [q for q, _ in pairs]
    answers = [a for _, a in pairs]

    answer_lengths = [len(_tokens(a)) for a in answers]
    hedge_rates = [_hedge_density(a) for a in answers]
    sentiments = [_sentiment_distribution(a) for a in answers]
    pos = [s[0] for s in sentiments]
    neu = [s[1] for s in sentiments]
    neg = [s[2] for s in sentiments]
    difficulties = [question_scorer.score(q) for q in questions]

    # Embeddings
    q_vecs = _embed_many(questions) or []
    a_vecs = _embed_many(answers) or []
    if q_vecs and a_vecs and len(q_vecs) == len(a_vecs):
        sims = [_cosine(q, a) for q, a in zip(q_vecs, a_vecs)]
    else:
        sims = []

    answer_centroid = _mean_vector(a_vecs) if a_vecs else []

    return CallMetrics(
        call_date=str(transcript.get("call_date") or ""),
        quarter=transcript.get("quarter"),
        year=transcript.get("year"),
        n_answers=len(answers),
        mean_answer_length=_mean(answer_lengths),
        hedge_density=_mean(hedge_rates),
        pos_sentiment_mean=_mean(pos),
        neu_sentiment_mean=_mean(neu),
        neg_sentiment_mean=_mean(neg),
        qa_similarity_mean=_mean(sims) if sims else 0.0,
        question_difficulty_mean=_mean(difficulties),
        answer_centroid=answer_centroid,
    )


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

_AGG_FIELDS = (
    "mean_answer_length",
    "hedge_density",
    "pos_sentiment_mean",
    "neu_sentiment_mean",
    "neg_sentiment_mean",
    "qa_similarity_mean",
    "question_difficulty_mean",
)


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _std(xs: list[float], mean: float | None = None) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean if mean is not None else _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _aggregate(per_call: list[CallMetrics]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for field_name in _AGG_FIELDS:
        values = [getattr(c, field_name) for c in per_call]
        m = _mean(values)
        out[field_name] = {"mean": m, "std": _std(values, m)}
    return out


def _pressure_baseline(
    per_call: list[CallMetrics],
    centroid: list[float],
    neg_mean: float,
) -> dict[str, float]:
    """
    Compute mean / std of historical pressure scores so that deviation.py
    can flag calls > 2 σ from the historical mean.
    """
    scores: list[float] = []
    for c in per_call:
        cos_dist = 1.0 - _cosine(c.answer_centroid, centroid) if centroid else 0.0
        sent_shift = abs(c.neg_sentiment_mean - neg_mean)
        difficulty_weight = 0.5 + 0.5 * c.question_difficulty_mean
        scores.append(difficulty_weight * (_W_COSINE * cos_dist + _W_SENTIMENT * sent_shift))
    m = _mean(scores)
    return {"mean": m, "std": _std(scores, m), "n": float(len(scores))}


# ------------------------------------------------------------------
# Top-level builder
# ------------------------------------------------------------------


def is_quality_transcript(transcript: dict[str, Any]) -> bool:
    """Return True only if a transcript meets the minimum quality bar."""
    n_turns = len(transcript.get("qa_turns") or [])
    prepared_len = len(transcript.get("prepared_remarks") or "")
    return n_turns >= MIN_QA_TURNS and prepared_len >= MIN_PREPARED_CHARS


def build_baseline(
    transcripts: list[dict[str, Any]],
    ticker: str,
) -> CEOBaseline:
    """Build a CEO linguistic baseline from a list of parsed transcript dicts."""
    if not transcripts:
        raise ValueError("Need at least one transcript to build a baseline")

    quality = [t for t in transcripts if is_quality_transcript(t)]
    n_skipped = len(transcripts) - len(quality)
    if n_skipped:
        log.info(
            "%s: skipped %d low-quality transcript(s) "
            "(< %d Q&A turns or < %d chars prepared remarks)",
            ticker.upper(), n_skipped, MIN_QA_TURNS, MIN_PREPARED_CHARS,
        )
    if not quality:
        raise ValueError(
            f"All {len(transcripts)} transcript(s) for {ticker} failed the "
            "quality filter (need >= 10 Q&A turns and >= 5000 chars)."
        )

    scorer = QuestionScorer()
    per_call: list[CallMetrics] = []
    for t in quality:
        metrics = compute_call_metrics(t, scorer)
        if metrics is not None:
            per_call.append(metrics)

    if not per_call:
        raise ValueError(
            f"None of the supplied transcripts for {ticker} contained "
            "CEO answers to analyst questions."
        )

    # Centroid across every CEO answer (weighted by number of answers per call).
    weighted: list[list[float]] = []
    for c in per_call:
        if not c.answer_centroid:
            continue
        weighted.extend([c.answer_centroid] * max(c.n_answers, 1))
    centroid = _mean_vector(weighted)

    aggregate = _aggregate(per_call)
    pressure = _pressure_baseline(per_call, centroid, aggregate["neg_sentiment_mean"]["mean"])

    return CEOBaseline(
        ticker=ticker.upper(),
        n_calls=len(per_call),
        n_skipped=n_skipped,
        finbert_model=FINBERT_MODEL_ID,
        embedder_model=EMBEDDER_MODEL_ID,
        aggregate=aggregate,
        centroid=centroid,
        pressure_baseline=pressure,
        per_call=[c.to_jsonable(drop_centroid=True) for c in per_call],
    )


def save_baseline(
    baseline: CEOBaseline,
    out_dir: Path = DEFAULT_BASELINE_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{baseline.ticker.upper()}.json"
    out_path.write_text(baseline.to_json(), encoding="utf-8")
    return out_path


def load_transcripts_from_dir(
    transcripts_dir: Path, ticker: str
) -> list[dict[str, Any]]:
    """Load every {TICKER}_*.json transcript file from a directory."""
    pattern = f"{ticker.upper()}_*.json"
    files = sorted(transcripts_dir.glob(pattern))
    out: list[dict[str, Any]] = []
    for fp in files:
        try:
            out.append(json.loads(fp.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            log.warning("Skipping malformed transcript %s: %s", fp, exc)
    return out


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def _discover_tickers(transcripts_dir: Path) -> list[str]:
    """Return sorted list of unique tickers found in the transcripts directory."""
    seen: set[str] = set()
    for fp in transcripts_dir.glob("*.json"):
        parts = fp.stem.split("_")
        if len(parts) >= 2:
            seen.add(parts[0].upper())
    return sorted(seen)


def _latest_transcript(transcripts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the transcript with the most recent call_date."""
    quality = [t for t in transcripts if is_quality_transcript(t)]
    if not quality:
        return None
    return max(quality, key=lambda t: str(t.get("call_date") or ""))


def _score_latest_against_baseline(
    transcript: dict[str, Any],
    baseline: CEOBaseline,
) -> tuple[float, float, bool]:
    """Return (P_score, z_score, flagged) for one transcript vs a built baseline."""
    from .deviation import compute_pressure  # lazy: avoids circular at module load
    report = compute_pressure(transcript, json.loads(baseline.to_json()))
    return report.pressure_score, report.z_score, report.flagged


def _print_summary_table(rows: list[dict[str, Any]]) -> None:
    cols = [
        ("ticker",        8),
        ("n_calls_used",  12),
        ("pressure_mean", 13),
        ("pressure_std",  12),
        ("latest_call",   12),
        ("P_score",        8),
        ("z_score",        8),
        ("flagged",        7),
    ]
    header = " | ".join(name.ljust(w) for name, w in cols)
    sep    = "-+-".join("-" * w for _, w in cols)
    print(header)
    print(sep)
    for r in rows:
        cells = [
            r["ticker"].ljust(cols[0][1]),
            str(r["n_calls_used"]).ljust(cols[1][1]),
            f"{r['pressure_mean']:.4f}".ljust(cols[2][1]),
            f"{r['pressure_std']:.4f}".ljust(cols[3][1]),
            str(r["latest_call"]).ljust(cols[4][1]),
            f"{r['P_score']:.4f}".ljust(cols[5][1]),
            f"{r['z_score']:+.2f}".ljust(cols[6][1]),
            ("YES" if r["flagged"] else "no").ljust(cols[7][1]),
        ]
        print(" | ".join(cells))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nlp.baseline",
        description="Build CEO linguistic baselines from historical transcripts.",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", help="Single stock ticker symbol")
    group.add_argument(
        "--all",
        action="store_true",
        help="Process every ticker found in --transcripts-dir and print summary table",
    )
    p.add_argument(
        "--transcripts-dir",
        default="data/transcripts",
        help="Directory containing parsed {TICKER}_*.json transcripts",
    )
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_BASELINE_DIR),
        help="Directory to write baseline JSON files to",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    transcripts_dir = Path(args.transcripts_dir)
    out_dir = Path(args.out_dir)

    tickers = (
        _discover_tickers(transcripts_dir) if args.all else [args.ticker.upper()]
    )
    if not tickers:
        raise SystemExit(f"No transcript files found in {transcripts_dir}")

    summary_rows: list[dict[str, Any]] = []

    for ticker in tickers:
        transcripts = load_transcripts_from_dir(transcripts_dir, ticker)
        if not transcripts:
            log.warning("No transcripts found for %s — skipping", ticker)
            continue

        try:
            baseline = build_baseline(transcripts, ticker)
        except ValueError as exc:
            log.warning("Could not build baseline for %s: %s", ticker, exc)
            continue

        out_path = save_baseline(baseline, out_dir)
        log.info(
            "Wrote %s  |  n_calls=%d  skipped=%d",
            out_path, baseline.n_calls, baseline.n_skipped,
        )

        latest = _latest_transcript(transcripts)
        if latest is None:
            p_score, z_score, flagged = 0.0, 0.0, False
            latest_date = "n/a"
        else:
            latest_date = str(latest.get("call_date") or "n/a")
            try:
                p_score, z_score, flagged = _score_latest_against_baseline(
                    latest, baseline
                )
            except Exception as exc:
                log.warning("Could not score latest call for %s: %s", ticker, exc)
                p_score, z_score, flagged = 0.0, 0.0, False

        summary_rows.append(
            {
                "ticker": ticker,
                "n_calls_used": baseline.n_calls,
                "pressure_mean": baseline.pressure_baseline["mean"],
                "pressure_std": baseline.pressure_baseline["std"],
                "latest_call": latest_date,
                "P_score": p_score,
                "z_score": z_score,
                "flagged": flagged,
            }
        )

    if args.all and summary_rows:
        print()
        _print_summary_table(summary_rows)
    elif summary_rows:
        r = summary_rows[0]
        print(
            f"[baseline] {r['ticker']}  n_calls={r['n_calls_used']}  "
            f"pressure mean={r['pressure_mean']:.4f}  std={r['pressure_std']:.4f}  "
            f"latest={r['latest_call']}  P={r['P_score']:.4f}  "
            f"z={r['z_score']:+.2f}  flagged={'YES' if r['flagged'] else 'no'}"
        )


if __name__ == "__main__":
    main()


__all__ = [
    "CEOBaseline",
    "CallMetrics",
    "build_baseline",
    "save_baseline",
    "load_transcripts_from_dir",
    "compute_call_metrics",
]
