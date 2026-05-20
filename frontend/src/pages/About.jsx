const SECTIONS = [
  {
    title: 'What is PressureTest?',
    body: `PressureTest is an end-to-end ML pipeline that quantifies executive stress during earnings calls.
It measures how much a CEO's language deviates from their established communication baseline when facing
tough analyst questions — surfacing statistical signals that may not be obvious from reading a transcript alone.
Built as a portfolio project demonstrating data ingestion, NLP scoring, REST API design, and React frontend development.`,
  },
  {
    title: 'How the scoring works',
    bullets: [
      'Analyst questions are rated for difficulty using five weighted features: FinBERT negative sentiment (30%), hedge-word softness (15%), financial/legal terminology density (20%), question length (10%), and aggressive vocabulary (25%).',
      'CEO answers are embedded with a sentence transformer (all-MiniLM-L6-v2) and their FinBERT sentiment distribution is computed.',
      'A pressure score P = difficulty_weight × (0.6 × cosine_distance + 0.4 × sentiment_shift) is computed against the CEO\'s historical baseline centroid.',
      'Calls that exceed two standard deviations (z ≥ 2σ) from the CEO\'s historical mean are automatically flagged.',
    ],
  },
  {
    title: 'Data sources',
    bullets: [
      'SEC EDGAR 8-K filings — primary source for official earnings call transcripts',
      'Motley Fool — supplementary transcript source',
      'HuggingFace Datasets — Rogersurf/earnings-call-transcripts for historical coverage',
      'Current coverage: AAPL, LLY, PFE, SHOP, GM, SBUX, DUOL',
    ],
  },
  {
    title: 'Tech stack',
    bullets: [
      'Pipeline: Python 3.13, httpx, Pydantic v2, datasets',
      'NLP / ML: ProsusAI/finbert, sentence-transformers (all-MiniLM-L6-v2), PyTorch, NumPy, SciPy',
      'API: FastAPI 0.111, Uvicorn, Anthropic Claude (streaming SSE chat)',
      'Frontend: React 18, Vite 5, Tailwind CSS 3, Recharts, React Router 6',
    ],
  },
]

function InfoCard({ title, body, bullets }) {
  return (
    <div className="bg-white rounded-xl border border-gray-100 p-6">
      <h2 className="text-sm font-semibold text-gray-900 mb-3">{title}</h2>
      {body && (
        <p className="text-sm text-gray-600 leading-relaxed whitespace-pre-line">{body}</p>
      )}
      {bullets && (
        <ul className="space-y-2">
          {bullets.map((b, i) => (
            <li key={i} className="flex gap-2 text-sm text-gray-600 leading-relaxed">
              <span className="text-blue-400 mt-0.5 flex-shrink-0 font-medium">—</span>
              <span>{b}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default function About() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">About PressureTest</h1>
        <p className="text-sm text-gray-500 mt-0.5">How it works and what powers it</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {SECTIONS.map(s => (
          <InfoCard key={s.title} {...s} />
        ))}
      </div>

      <div className="rounded-xl border p-5" style={{ backgroundColor: '#EFF6FF', borderColor: '#BFDBFE' }}>
        <p className="text-sm font-semibold mb-1" style={{ color: '#1E40AF' }}>Scoring formula</p>
        <code className="block text-sm font-mono mt-2 leading-relaxed" style={{ color: '#2563EB' }}>
          difficulty_weight = 0.5 + 0.5 × mean_question_difficulty<br />
          pressure_score    = difficulty_weight × (0.6 × cosine_distance + 0.4 × sentiment_shift)<br />
          flagged           = z_score ≥ 2.0σ
        </code>
      </div>
    </div>
  )
}
