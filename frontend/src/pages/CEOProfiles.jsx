import { useState, useEffect } from 'react'
import * as api from '../api/client'

function Spinner() {
  return (
    <div className="flex items-center justify-center h-48">
      <div className="w-8 h-8 rounded-full border-2 border-gray-200 border-t-blue-600 animate-spin" />
    </div>
  )
}

function fmt(n, d = 4) {
  return n != null ? Number(n).toFixed(d) : '—'
}

function CEOCard({ ticker, ceo, isSelected, onToggle }) {
  return (
    <div
      className={`bg-white rounded-xl border transition-all cursor-pointer ${
        isSelected ? 'border-blue-300 ring-2 ring-blue-100' : 'border-gray-100 hover:border-gray-200'
      }`}
      onClick={onToggle}
    >
      <div className="p-5">
        <div className="flex items-start justify-between mb-3">
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <span className="text-lg font-bold text-gray-900">{ticker.ticker}</span>
              <span className="text-xs text-blue-600 bg-blue-50 px-2 py-0.5 rounded font-medium">
                {ceo?.n_calls ?? ticker.n_calls} calls
              </span>
            </div>
            <p className="text-sm text-gray-500">{ticker.company_name}</p>
            <p className="text-xs text-gray-400 mt-0.5">{api.CEO_NAMES[ticker.ticker] || ''}</p>
          </div>
          <span className="text-gray-300 text-sm select-none mt-1">
            {isSelected ? '▲' : '▼'}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-xs text-gray-400 mb-1">Pressure Mean</p>
            <p className="text-sm font-semibold text-gray-900 font-mono">{fmt(ceo?.pressure_mean)}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-xs text-gray-400 mb-1">Pressure Std</p>
            <p className="text-sm font-semibold text-gray-900 font-mono">{fmt(ceo?.pressure_std)}</p>
          </div>
        </div>

        <p className="text-xs text-gray-400 mt-3">
          Latest call: <span className="text-gray-600">{ticker.latest_call_date || '—'}</span>
        </p>
      </div>
    </div>
  )
}

function CallHistoryPanel({ ceo }) {
  if (!ceo?.calls?.length) {
    return (
      <div className="bg-white rounded-xl border border-gray-100 px-5 py-6 text-sm text-gray-400">
        No call history available.
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl border border-blue-200 overflow-hidden">
      <div className="px-5 py-4 border-b border-gray-100 flex items-center gap-3">
        <div
          className="w-6 h-6 rounded text-white text-xs font-bold flex items-center justify-center flex-shrink-0"
          style={{ backgroundColor: '#2563EB' }}
        >
          {ceo.ticker?.[0]}
        </div>
        <div>
          <p className="text-sm font-semibold text-gray-900">{ceo.company_name}</p>
          <p className="text-xs text-gray-400">{api.CEO_NAMES[ceo.ticker] || ceo.ticker} &middot; {ceo.n_calls} calls analyzed</p>
        </div>
        <div className="ml-auto flex gap-6 text-right">
          <div>
            <p className="text-xs text-gray-400">Baseline Mean</p>
            <p className="text-sm font-mono font-semibold text-gray-900">{fmt(ceo.pressure_mean)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Baseline Std</p>
            <p className="text-sm font-mono font-semibold text-gray-900">{fmt(ceo.pressure_std)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Embedding Dim</p>
            <p className="text-sm font-mono font-semibold text-gray-900">{ceo.centroid_dim ?? '—'}</p>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-gray-400 border-b border-gray-100">
              {['Date', 'Quarter', 'Year', 'Q&A Turns', 'Pressure Score'].map(h => (
                <th key={h} className="px-5 py-3 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {[...ceo.calls].sort((a, b) => (b.date || '').localeCompare(a.date || '')).map((call, i) => (
              <tr key={i} className="hover:bg-gray-50/60 transition-colors">
                <td className="px-5 py-3 text-gray-700">{call.date}</td>
                <td className="px-5 py-3 text-gray-700">Q{call.quarter}</td>
                <td className="px-5 py-3 text-gray-700">{call.year}</td>
                <td className="px-5 py-3 text-gray-700">{call.n_qa_turns ?? '—'}</td>
                <td className="px-5 py-3 font-mono text-gray-700">
                  {call.pressure_score != null ? fmt(call.pressure_score) : <span className="text-gray-300">not scored</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function CEOProfiles() {
  const [tickers, setTickers]   = useState([])
  const [ceoData, setCeoData]   = useState({})
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)
  const [expanded, setExpanded] = useState(null)

  useEffect(() => {
    async function load() {
      try {
        const tickersData = await api.getTickers()
        setTickers(tickersData)

        const results = await Promise.allSettled(
          tickersData.map(t =>
            api.getCEO(t.ticker).then(data => ({ ticker: t.ticker, data }))
          )
        )
        const map = {}
        for (const r of results) {
          if (r.status === 'fulfilled') map[r.value.ticker] = r.value.data
        }
        setCeoData(map)
      } catch {
        setError('Could not load CEO profiles. Is the backend running?')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <Spinner />
  if (error) return (
    <div className="bg-amber-50 border border-amber-200 text-amber-800 px-5 py-4 rounded-xl text-sm">{error}</div>
  )

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">CEO Profiles</h1>
        <p className="text-sm text-gray-500 mt-0.5">Historical communication baselines per executive</p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {tickers.map(t => (
          <CEOCard
            key={t.ticker}
            ticker={t}
            ceo={ceoData[t.ticker]}
            isSelected={expanded === t.ticker}
            onToggle={() => setExpanded(expanded === t.ticker ? null : t.ticker)}
          />
        ))}
      </div>

      {expanded && ceoData[expanded] && (
        <CallHistoryPanel ceo={ceoData[expanded]} />
      )}

      {tickers.length === 0 && (
        <div className="bg-white rounded-xl border border-gray-100 px-5 py-12 text-center text-sm text-gray-400">
          No CEO baselines found. Run <code className="bg-gray-100 px-1 rounded font-mono">python -m nlp.baseline --ticker AAPL</code> to build one.
        </div>
      )}
    </div>
  )
}
