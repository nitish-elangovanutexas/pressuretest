# PressureTest

> **Does your CEO crack under pressure?** PressureTest quantifies executive stress during earnings calls by measuring how much a CEO's language deviates from their established communication baseline when facing tough analyst questions.

---

## What It Does

PressureTest is an end-to-end ML pipeline that ingests earnings call transcripts, builds a statistical linguistic baseline for each CEO, and scores new calls against that baseline to detect abnormal communication patterns. For each Q&A exchange, analyst questions are rated for aggressiveness using a composite NLP scorer (FinBERT sentiment + hedge/terminology density + lexical aggressiveness). CEO answers are embedded with a sentence transformer and their sentiment distribution is measured with FinBERT. A pressure score is computed as a difficulty-weighted combination of cosine distance from the historical answer centroid and sentiment shift from baseline. Calls that exceed two standard deviations from the CEO's historical mean are automatically flagged. A FastAPI backend exposes the scores and baselines, and includes a streaming AI analyst chat powered by Claude.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion | SEC EDGAR API, Motley Fool scraper, HuggingFace `datasets` (`Rogersurf/earnings-call-transcripts`) |
| Transcript parsing | Regex-based speaker segmentation, Pydantic data models |
| NLP / ML | `ProsusAI/finbert` (FinBERT), `sentence-transformers`, `transformers`, PyTorch, NumPy, SciPy |
| API | FastAPI, Uvicorn, Pydantic v2 |
| AI chat | Anthropic Claude (`claude-sonnet-4`) via streaming SSE |
| Testing | pytest, pytest-asyncio |
| Language | Python 3.13 |

---

## Project Architecture

### Phase 1 — Data Pipeline
Transcripts are fetched from SEC EDGAR 8-K filings, Motley Fool, or the HuggingFace dataset. A rule-based parser locates the prepared-remarks / Q&A boundary, splits the transcript into speaker turns, and attributes each turn to one of: `CEO`, `CFO`, `Analyst`, `Operator`, or `Unknown`. Structured JSON transcripts are written to `data/transcripts/`.

### Phase 2 — NLP Scoring
**Baseline builder** (`nlp/baseline.py`): For each ticker, historical transcripts are processed to extract all CEO Q&A answers. A sentence-transformer embedding centroid and FinBERT sentiment distribution are computed across all historical calls, producing a CEO-specific baseline stored in `data/baselines/`.

**Deviation scorer** (`nlp/deviation.py`): A new call is scored against the CEO baseline:

```
pressure = difficulty_weight × (0.6 × cosine_distance + 0.4 × sentiment_shift)
```

where `difficulty_weight = 0.5 + 0.5 × mean_question_difficulty`. Calls with a z-score ≥ 2σ are flagged.

**Question scorer** (`nlp/question_scorer.py`): Each analyst question is assigned a difficulty in `[0, 1]` from five features — FinBERT negative sentiment (30%), hedge-word softness (15%), financial/legal terminology density (20%), question length (10%), and aggressive vocabulary (25%).

### Phase 3 — FastAPI Backend
A REST API exposes tickers, CEO baselines, call scores, flagged events, and on-demand scoring. A `/chat` endpoint streams AI analyst commentary grounded in the transcript and scoring data via Claude.

---

## Running Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env (required for /chat endpoint)
```

**3. Build a CEO baseline** (needed before scoring)
```bash
python -m nlp.baseline --ticker AAPL
```

**4. Score a specific call**
```bash
python -m nlp.deviation --ticker AAPL --call data/transcripts/AAPL_20260430.json
```

**5. Start the API server**
```bash
uvicorn api.main:app --reload
# Server runs at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

---

## API Endpoints

### `GET /tickers`
All tickers with a built baseline.
```json
[
  {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "n_calls": 8,
    "latest_call_date": "2026-04-30"
  }
]
```

### `GET /ceo/{ticker}`
Full CEO linguistic baseline profile.
```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "n_calls": 8,
  "pressure_mean": 0.0216,
  "pressure_std": 0.0023,
  "centroid_dim": 384,
  "calls": [
    { "date": "2026-04-30", "quarter": 2, "year": 2026, "n_qa_turns": 26, "pressure_score": 0.0199 }
  ]
}
```

### `GET /calls/{ticker}/latest`
Full deviation report for the most recently scored call.
```json
{
  "ticker": "AAPL",
  "call_date": "2026-04-30",
  "quarter": 2,
  "year": 2026,
  "pressure_score": 0.0199,
  "cosine_distance": 0.0503,
  "sentiment_shift": 0.0103,
  "question_difficulty_mean": 0.162,
  "z_score": -0.71,
  "flagged": false,
  "flag_threshold_sigma": 2.0
}
```

### `GET /flags`
All calls across all tickers where `flagged = true` (z-score ≥ 2σ).

### `POST /calls/{ticker}/score`
Trigger on-demand scoring for a specific call date.
```json
{ "call_date": "2026-04-30" }
```

### `POST /chat`
Stream an AI analyst response grounded in transcript and scoring data.
```json
{ "ticker": "AAPL", "question": "Why was pressure lower this quarter?" }
```
Returns a server-sent event stream (`text/event-stream`).

---

## Screenshots

*Dashboard and score visualizations — coming soon.*

---

## Future Work

- **Live data ingestion** — replace batch scraping with a real-time pipeline that fetches and scores new earnings calls as they are published (e.g., triggered by an EDGAR RSS feed or market calendar).
- **Supervised fine-tuning (SFT)** — fine-tune a small language model on labeled earnings call exchanges to improve question difficulty scoring and CEO answer evasiveness detection beyond zero-shot FinBERT.
- **Broader ticker coverage** — scale beyond the current six tickers to the full S&P 500, enabling cross-sector CEO stress comparisons and sector-normalized flagging thresholds.

---

## Project Structure

```
pressuretest/
├── scraper/          # Phase 1: data ingestion & transcript parsing
│   ├── edgar_scraper.py
│   ├── motley_fool_scraper.py
│   ├── hf_loader.py
│   └── transcript_parser.py
├── nlp/              # Phase 2: baseline building & deviation scoring
│   ├── baseline.py
│   ├── deviation.py
│   ├── question_scorer.py
│   └── _models.py
├── api/              # Phase 3: FastAPI backend
│   ├── main.py
│   └── routers/
│       ├── tickers.py
│       ├── ceo.py
│       ├── calls.py
│       ├── flags.py
│       └── chat.py
├── data/
│   ├── transcripts/  # parsed call JSONs
│   ├── baselines/    # per-ticker CEO baselines
│   └── scores/       # deviation reports
└── tests/
```
