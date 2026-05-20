"""
Pressure-score computation for a single earnings call.

Given a parsed call JSON and the CEO baseline JSON for the same ticker, this
module computes a pressure score::

    P = difficulty_weight × ( w_cos · cosine_distance(answer_centroid, baseline_centroid)
                            + w_sent · |neg_sentiment_call − neg_sentiment_baseline| )

where ``difficulty_weight = 0.5 + 0.5 × mean_question_difficulty`` so that
high-pressure responses to harder questions matter more than the same response
to a softball.

A call is flagged when ``P`` exceeds the historical mean by more than 2
standard deviations (μ and σ come from the baseline file).

Run as a script::

    python -m nlp.deviation --ticker AAPL --call data/transcripts/AAPL_20260430.json
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baseline import (
    DEFAULT_BASELINE_DIR,
    _W_COSINE,
    _W_SENTIMENT,
    _cosine,
    compute_call_metrics,
)
from .question_scorer import QuestionScorer

log = logging.getLogger(__name__)

DEFAULT_SCORES_DIR = Path("data/scores")
FLAG_THRESHOLD_SIGMA = 2.0


# ------------------------------------------------------------------
# Data class
# ------------------------------------------------------------------


@dataclass
class DeviationReport:
    ticker: str
    call_date: str
    quarter: int | None
    year: int | None
    pressure_score: float
    cosine_distance: float
    sentiment_shift: float
    question_difficulty_mean: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    flagged: bool
    flag_threshold_sigma: float
    components: dict[str, float]
    generated_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# Core computation
# ------------------------------------------------------------------


def compute_pressure(
    call: dict[str, Any],
    baseline: dict[str, Any],
    question_scorer: QuestionScorer | None = None,
) -> DeviationReport:
    """Compute a DeviationReport for one call against a CEO baseline."""
    metrics = compute_call_metrics(call, question_scorer or QuestionScorer())
    if metrics is None:
        raise ValueError(
            "Call contains no CEO answers to analyst questions — "
            "cannot compute a pressure score."
        )

    baseline_centroid: list[float] = baseline.get("centroid") or []
    aggregate: dict[str, dict[str, float]] = baseline.get("aggregate") or {}
    baseline_neg = aggregate.get("neg_sentiment_mean", {}).get("mean", 0.0)

    cos_dist = (
        1.0 - _cosine(metrics.answer_centroid, baseline_centroid)
        if (metrics.answer_centroid and baseline_centroid)
        else 0.0
    )
    sent_shift = abs(metrics.neg_sentiment_mean - baseline_neg)
    difficulty_weight = 0.5 + 0.5 * metrics.question_difficulty_mean
    pressure = difficulty_weight * (_W_COSINE * cos_dist + _W_SENTIMENT * sent_shift)

    pb = baseline.get("pressure_baseline") or {}
    mu = float(pb.get("mean", 0.0))
    sigma = float(pb.get("std", 0.0))
    if sigma > 0.0:
        z = (pressure - mu) / sigma
    else:
        z = 0.0
    flagged = z >= FLAG_THRESHOLD_SIGMA

    return DeviationReport(
        ticker=str(call.get("ticker") or baseline.get("ticker") or "").upper(),
        call_date=str(call.get("call_date") or ""),
        quarter=call.get("quarter"),
        year=call.get("year"),
        pressure_score=pressure,
        cosine_distance=cos_dist,
        sentiment_shift=sent_shift,
        question_difficulty_mean=metrics.question_difficulty_mean,
        baseline_mean=mu,
        baseline_std=sigma,
        z_score=z,
        flagged=bool(flagged),
        flag_threshold_sigma=FLAG_THRESHOLD_SIGMA,
        components={
            "w_cosine": _W_COSINE,
            "w_sentiment": _W_SENTIMENT,
            "difficulty_weight": difficulty_weight,
            "call_mean_answer_length": metrics.mean_answer_length,
            "call_hedge_density": metrics.hedge_density,
            "call_pos_sentiment_mean": metrics.pos_sentiment_mean,
            "call_neu_sentiment_mean": metrics.neu_sentiment_mean,
            "call_neg_sentiment_mean": metrics.neg_sentiment_mean,
            "call_qa_similarity_mean": metrics.qa_similarity_mean,
            "baseline_neg_sentiment_mean": baseline_neg,
        },
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# ------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------


def load_call(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_report(
    report: DeviationReport,
    out_dir: Path = DEFAULT_SCORES_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_date = (report.call_date or "unknown").replace("-", "")
    out_path = out_dir / f"{report.ticker}_{safe_date}.json"
    out_path.write_text(report.to_json(), encoding="utf-8")
    return out_path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _score_all_tickers(
    baseline_dir: Path,
    transcripts_dir: Path,
    out_dir: Path,
    scorer: QuestionScorer,
) -> None:
    """Score the latest transcript for every ticker that has a baseline."""
    baselines = sorted(baseline_dir.glob("*.json"))
    if not baselines:
        raise SystemExit(f"No baseline files found in {baseline_dir}")

    ok = skipped = 0
    for bf in baselines:
        ticker = bf.stem.upper()
        # Filenames are TICKER_YYYYMMDD.json — lexicographic sort gives chronological order.
        transcript_files = sorted(transcripts_dir.glob(f"{ticker}_*.json"))
        if not transcript_files:
            log.warning("SKIP %s — no transcripts in %s", ticker, transcripts_dir)
            skipped += 1
            continue

        call_path = transcript_files[-1]
        try:
            call = load_call(call_path)
            baseline = load_baseline(bf)
            report = compute_pressure(call, baseline, scorer)
        except Exception as exc:
            log.warning("SKIP %s — %s", ticker, exc)
            skipped += 1
            continue

        save_report(report, out_dir)
        flag = "FLAGGED" if report.flagged else "ok"
        print(
            f"[deviation] {ticker:<6}  {call_path.name:<32}  "
            f"P={report.pressure_score:.4f}  z={report.z_score:+.2f}  {flag}"
        )
        ok += 1

    print(f"\n[deviation] Done: {ok} scored, {skipped} skipped.")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nlp.deviation",
        description="Score earnings calls against CEO baselines.",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--all",
        action="store_true",
        help=(
            "Score the latest call for every ticker that has a baseline in "
            "--baseline-dir. Uses --transcripts-dir to find transcripts."
        ),
    )
    mode.add_argument(
        "--ticker",
        help="Single stock ticker symbol (requires --call).",
    )
    p.add_argument(
        "--call",
        help="Path to the parsed call JSON to score (required with --ticker).",
    )
    p.add_argument(
        "--transcripts-dir",
        default="data/transcripts",
        help="Directory of {TICKER}_*.json transcripts (used with --all; default: data/transcripts).",
    )
    p.add_argument(
        "--baseline-dir",
        default=str(DEFAULT_BASELINE_DIR),
        help="Directory containing {TICKER}.json baselines.",
    )
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_SCORES_DIR),
        help="Directory to write the deviation report JSON to.",
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

    if args.all:
        _score_all_tickers(
            baseline_dir=Path(args.baseline_dir),
            transcripts_dir=Path(args.transcripts_dir),
            out_dir=Path(args.out_dir),
            scorer=QuestionScorer(),
        )
        return

    if not args.call:
        _build_arg_parser().error("--call is required when using --ticker")

    call_path = Path(args.call)
    baseline_path = Path(args.baseline_dir) / f"{args.ticker.upper()}.json"
    if not call_path.exists():
        raise SystemExit(f"Call file not found: {call_path}")
    if not baseline_path.exists():
        raise SystemExit(
            f"Baseline file not found: {baseline_path}. "
            "Run `python -m nlp.baseline --ticker <T>` first."
        )

    call = load_call(call_path)
    baseline = load_baseline(baseline_path)
    report = compute_pressure(call, baseline)
    out_path = save_report(report, Path(args.out_dir))

    flag = "FLAGGED" if report.flagged else "ok"
    print(
        f"[deviation] Wrote {out_path}  |  P={report.pressure_score:.4f}  "
        f"|  z={report.z_score:+.2f}  |  {flag}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DeviationReport",
    "compute_pressure",
    "load_call",
    "load_baseline",
    "save_report",
    "FLAG_THRESHOLD_SIGMA",
]
