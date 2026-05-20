import { useState, useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import * as api from '../api/client'

function Spinner() {
  return (
    <div className="flex items-center justify-center h-48">
      <div className="w-8 h-8 rounded-full border-2 border-gray-200 border-t-blue-600 animate-spin" />
    </div>
  )
}

function StatusBadge({ flagged }) {
  return flagged
    ? <span className="inline-flex px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-700">FLAGGED</span>
    : <span className="inline-flex px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-700">OK</span>
}

function fmt(n, d = 4) {
  return n != null ? Number(n).toFixed(d) : '—'
}

function fmtZ(z) {
  if (z == null) return '—'
  return (z >= 0 ? '+' : '') + Number(z).toFixed(2)
}

const SORTS = {
  date:     (a, b) => (b.call_date || '').localeCompare(a.call_date || ''),
  pressure: (a, b) => (b.pressure_score || 0) - (a.pressure_score || 0),
  zscore:   (a, b) => (b.z_score || 0) - (a.z_score || 0),
}

function SortButton({ field, label, current, onChange }) {
  const active = current === field
  return (
    <button
      onClick={() => onChange(field)}
      className={`px-3 py-1.5 text-xs rounded-md font-medium transition-colors ${
        active
          ? 'bg-blue-600 text-white'
          : 'bg-white border border-gray-200 text-gray-600 hover:border-gray-300'
      }`}
    >
      {label}
    </button>
  )
}

export default function Calls() {
  const [searchParams] = useSearchParams()
  const [tickers, setTickers]     = useState([])
  const [callsMap, setCallsMap]   = useState({})
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState(null)
  const [sortBy, setSortBy]       = useState('date')
  const [filterTicker, setFilter] = useState(searchParams.get('ticker') || '')

  useEffect(() => {
    async function load() {
      try {
        const tickersData = await api.getTickers()
        setTickers(tickersData)

        const results = await Promise.allSettled(
          tickersData.map(t =>
            api.getCalls(t.ticker).then(data => ({ ticker: t.ticker, data }))
          )
        )
        const map = {}
        for (const r of results) {
          if (r.status === 'fulfilled') map[r.value.ticker] = r.value.data
        }
        setCallsMap(map)
      } catch {
        setError('Could not load calls. Is the backend running?')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const companyMap = useMemo(() => {
    const m = {}
    tickers.forEach(t => { m[t.ticker] = t.company_name })
    return m
  }, [tickers])

  const allCalls = useMemo(() => {
    const calls = Object.entries(callsMap).flatMap(([ticker, list]) =>
      list.map(c => ({ ...c, ticker, company_name: companyMap[ticker] || ticker }))
    )
    const sorted = [...calls].sort(SORTS[sortBy] || SORTS.date)
    return filterTicker ? sorted.filter(c => c.ticker === filterTicker) : sorted
  }, [callsMap, companyMap, sortBy, filterTicker])

  if (loading) return <Spinner />
  if (error) return (
    <div className="bg-amber-50 border border-amber-200 text-amber-800 px-5 py-4 rounded-xl text-sm">{error}</div>
  )

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Calls</h1>
        <p className="text-sm text-gray-500 mt-0.5">All scored earnings calls</p>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <select
          value={filterTicker}
          onChange={e => setFilter(e.target.value)}
          className="text-sm border border-gray-200 rounded-md px-3 py-2 bg-white focus:outline-none focus:border-blue-500 transition"
        >
          <option value="">All tickers</option>
          {tickers.map(t => (
            <option key={t.ticker} value={t.ticker}>
              {t.ticker} — {t.company_name}
            </option>
          ))}
        </select>

        <div className="flex items-center gap-1.5">
          <span className="text-xs text-gray-400 mr-1">Sort by:</span>
          <SortButton field="date"     label="Date"           current={sortBy} onChange={setSortBy} />
          <SortButton field="pressure" label="Pressure Score" current={sortBy} onChange={setSortBy} />
          <SortButton field="zscore"   label="Z-Score"        current={sortBy} onChange={setSortBy} />
        </div>

        <span className="text-sm text-gray-400 ml-auto">
          {allCalls.length} call{allCalls.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-gray-100">
        {allCalls.length === 0 ? (
          <div className="px-5 py-14 text-center space-y-1">
            <p className="text-sm text-gray-500 font-medium">No scored calls found</p>
            <p className="text-xs text-gray-400">
              Use <code className="bg-gray-100 px-1 rounded font-mono">POST /calls/&#123;ticker&#125;/score</code> to score a call, then refresh.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-400 border-b border-gray-100">
                  {['Ticker', 'CEO', 'Company', 'Call Date', 'Quarter', 'Pressure Score', 'Z-Score', 'Status'].map(h => (
                    <th key={h} className="px-5 py-3 font-medium whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {allCalls.map((call, i) => (
                  <tr key={i} className="hover:bg-gray-50/60 transition-colors">
                    <td className="px-5 py-3 font-semibold text-blue-600">{call.ticker}</td>
                    <td className="px-5 py-3 text-gray-700 whitespace-nowrap">{api.CEO_NAMES[call.ticker] || '—'}</td>
                    <td className="px-5 py-3 text-gray-400 text-xs whitespace-nowrap">{call.company_name}</td>
                    <td className="px-5 py-3 text-gray-700 whitespace-nowrap">{call.call_date}</td>
                    <td className="px-5 py-3 text-gray-700">Q{call.quarter} {call.year}</td>
                    <td className="px-5 py-3 font-mono text-gray-700">{fmt(call.pressure_score)}</td>
                    <td className="px-5 py-3 font-mono">
                      <span className={Math.abs(call.z_score || 0) >= 2 ? 'text-red-600 font-semibold' : 'text-gray-700'}>
                        {fmtZ(call.z_score)}
                      </span>
                    </td>
                    <td className="px-5 py-3">
                      <StatusBadge flagged={call.flagged} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
