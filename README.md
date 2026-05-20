# PressureTest

> **Does your CEO crack under pressure?** PressureTest quantifies executive stress during earnings calls by measuring how much a CEO's language deviates from their established communication baseline when facing tough analyst questions.

**487 CEOs tracked · 2,604 calls analyzed · 4 flagged**

---

## What It Does

PressureTest is an end-to-end ML pipeline that ingests earnings call transcripts, builds a statistical linguistic baseline for each CEO, and scores new calls against that baseline to detect abnormal communication patterns.

For each Q&A exchange, analyst questions are rated for aggressiveness using a composite NLP scorer (FinBERT sentiment + hedge/terminology density + lexical aggressiveness). CEO answers are embedded with a sentence transformer and their sentiment distribution is measured with FinBERT. A pressure score is computed as a difficulty-weighted combination of cosine distance from the historical answer centroid and sentiment shift from baseline:

```
pressure = difficulty_weight × (0.6 × cosine_distance + 0.4 × sentiment_shift)
```

where `difficulty_weight = 0.5 + 0.5 × mean_question_difficulty`. Calls that exceed two standard deviations from the CEO's historical mean are automatically flagged.

---

## Validation: The Visa z = 47 Signal

The clearest proof the model works: when scored against Visa's historical baseline, the call where CEO Ryan McInerney faced sustained DOJ antitrust questions produced a z-score of **47** — 23× above the flag threshold of 2σ. Every other Visa call in the dataset sat between z = −0.8 and z = 1.4. The model had never been told about the antitrust investigation; it detected the linguistic deviation from his established baseline on its own.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion | SEC EDGAR API, Motley Fool scraper, HuggingFace `datasets`, FMP API |
| Transcript parsing | Regex-based speaker segmentation, Pydantic data models |
| NLP / ML | `ProsusAI/finbert`, `sentence-transformers`, `transformers`, PyTorch, NumPy, SciPy |
| Backend | FastAPI, Uvicorn, Pydantic v2 |
| Frontend | React 18, Vite, Tailwind CSS, Recharts, React Router |
| AI chat | Anthropic Claude (`claude-sonnet-4-6`) via streaming SSE |
| Scheduler | `schedule` library — weekly FMP refresh, Monday 06:00 |
| Language | Python 3.13, Node 20 |

---

## Architecture: Five Phases

### Phase 1 — Data Pipeline
Transcripts are fetched from SEC EDGAR 8-K filings, Motley Fool, or the HuggingFace `Rogersurf/earnings-call-transcripts` dataset. A rule-based parser locates the prepared-remarks / Q&A boundary, splits the transcript into speaker turns, and attributes each turn to `CEO`, `CFO`, `Analyst`, `Operator`, or `Unknown`. Structured JSON transcripts are written to `data/transcripts/`.

### Phase 2 — NLP Scoring
**Baseline builder** (`nlp/baseline.py`): For each ticker, all historical CEO Q&A answers are processed to produce an embedding centroid (sentence-transformer) and a FinBERT sentiment distribution, stored in `data/baselines/`.

**Deviation scorer** (`nlp/deviation.py`): A new call is scored against the CEO's baseline. Calls with z-score ≥ 2σ are written to `data/scores/` and marked `flagged: true`.

**Question scorer** (`nlp/question_scorer.py`): Each analyst question gets a difficulty in [0, 1] from five features — FinBERT negative sentiment (30%), hedge-word softness (15%), financial/legal terminology density (20%), question length (10%), and aggressive vocabulary (25%).

### Phase 3 — FastAPI Backend
A REST API exposes tickers, CEO baselines, call scores, flagged events, and on-demand scoring. A `/chat` endpoint streams AI analyst commentary grounded in transcript and scoring data via Claude.

### Phase 4 — React Frontend
A single-page dashboard built with React + Vite + Tailwind + Recharts. Pages: Dashboard (leaderboard + z-score chart), CEO Profiles (baseline history per ticker), Calls (full Q&A breakdown with per-turn scores), Flags (all flagged events), and About.

### Phase 5 — Weekly Scheduler
`scheduler.py` runs every Monday at 06:00, ingesting fresh transcripts via the FMP API, rebuilding baselines, and re-scoring for every tracked ticker. Run it immediately with `python scheduler.py --now`.

---

## Running Locally

**1. Clone and install Python dependencies**
```bash
git clone https://github.com/<you>/pressuretest.git
cd pressuretest
pip install -r requirements.txt
```

**2. Configure environment variables**
```bash
cp .env.example .env
```
Edit `.env` and add:
```
ANTHROPIC_API_KEY=sk-ant-...   # required for /chat endpoint
FMP_API_KEY=...                 # required for live data ingestion
```

**3. Start the API server**
```bash
uvicorn api.main:app --reload
# Runs at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

**4. Start the frontend**
```bash
cd frontend
npm install
npm run dev
# Runs at http://localhost:5173
```

**5. (Optional) Build a CEO baseline manually**
```bash
python -m nlp.baseline --ticker AAPL
python -m nlp.deviation --ticker AAPL --call data/transcripts/AAPL_20260430.json
```

**6. (Optional) Run the weekly scheduler immediately**
```bash
python scheduler.py --now
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/tickers` | All tickers with a built baseline |
| `GET` | `/ceo/{ticker}` | Full CEO linguistic baseline profile |
| `GET` | `/calls/{ticker}/latest` | Deviation report for the most recent call |
| `GET` | `/flags` | All flagged calls (z ≥ 2σ) across all tickers |
| `POST` | `/calls/{ticker}/score` | Trigger on-demand scoring for a call date |
| `POST` | `/chat` | Stream AI analyst commentary (SSE) |

---

## Project Structure

```
pressuretest/
├── scraper/              # Phase 1: data ingestion & transcript parsing
│   ├── edgar_scraper.py
│   ├── motley_fool_scraper.py
│   ├── hf_loader.py
│   ├── fmp_scraper.py
│   └── transcript_parser.py
├── nlp/                  # Phase 2: baseline building & deviation scoring
│   ├── baseline.py
│   ├── deviation.py
│   ├── question_scorer.py
│   └── _models.py
├── api/                  # Phase 3: FastAPI backend
│   ├── main.py
│   └── routers/
│       ├── tickers.py
│       ├── ceo.py
│       ├── calls.py
│       ├── flags.py
│       └── chat.py
├── frontend/             # Phase 4: React dashboard
│   └── src/
│       ├── pages/        # Dashboard, CEOProfiles, Calls, Flags, About
│       └── components/
├── scheduler.py          # Phase 5: weekly FMP refresh
├── data/
│   ├── transcripts/      # parsed call JSONs
│   ├── baselines/        # per-ticker CEO baselines
│   └── scores/           # deviation reports
└── tests/
```

---

## Future Work

- **Railway deployment** — containerize the FastAPI backend and deploy to Railway so the dashboard is publicly accessible; wire the frontend Vercel deployment to the live API URL.
- **Live FMP data** — upgrade to a paid FMP tier to enable the weekly scheduler in production, replacing the static HuggingFace dataset with real-time earnings call ingestion as calls are published.
- **Supervised fine-tuning** — fine-tune a small language model on labeled earnings call exchanges to improve question difficulty scoring and CEO answer evasiveness detection beyond zero-shot FinBERT.
- **S&P 500 coverage** — scale beyond the current 487 CEOs to the full index, enabling cross-sector CEO stress comparisons and sector-normalized flagging thresholds.
