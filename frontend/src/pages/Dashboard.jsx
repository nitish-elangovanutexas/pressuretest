import { useState, useEffect, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import * as api from '../api/client'

function Spinner() {
  return (
    <div className="flex items-center justify-center h-48">
      <div className="w-8 h-8 rounded-full border-2 border-gray-200 border-t-blue-600 animate-spin" />
    </div>
  )
}

function StatCard({ label, value, sub }) {
  return (
    <div className="bg-white rounded-xl border border-gray-100 p-5">
      <p className="text-xs text-gray-500 mb-1 font-medium uppercase tracking-wide">{label}</p>
      <p className="text-3xl font-bold text-gray-900 leading-none mb-1">{value ?? '—'}</p>
      {sub && <p className="text-xs text-gray-400 truncate">{sub}</p>}
    </div>
  )
}

function StatusBadge({ flagged }) {
  return flagged
    ? <span className="inline-flex px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-700">FLAGGED</span>
    : <span className="inline-flex px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-700">OK</span>
}

function fmt(n, d = 4) { return n != null ? Number(n).toFixed(d) : '—' }
function fmtZ(z) {
  if (z == null) return '—'
  return (z >= 0 ? '+' : '') + Number(z).toFixed(2)
}

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  return (
    <div className="bg-white border border-gray-200 rounded-lg px-3 py-2 text-xs shadow-md">
      <p className="font-semibold text-gray-900">
        {d.ticker}{d.flagged ? ' — FLAGGED' : ''}
      </p>
      <p className="text-gray-500">{d.date}</p>
      <p className="text-gray-700">P = {fmt(d.score)}</p>
    </div>
  )
}

export default function Dashboard() {
  const [tickers, setTickers]   = useState([])
  const [flags, setFlags]       = useState([])
  const [callsMap, setCallsMap] = useState({})
  const [appStats, setAppStats] = useState(null)
  const [loading, setLoading]   = useState(true)
  const [apiDown, setApiDown]   = useState(false)

  useEffect(() => {
    async function load() {
      try {
        const [tickersData, flagsData, statsData] = await Promise.all([
          api.getTickers(),
          api.getFlags(),
          api.getStats(),
        ])
        setTickers(tickersData)
        setFlags(flagsData)
        setAppStats(statsData)

        // Fix 1 + 3: fetch all scored calls per ticker (used for both chart and table)
        const results = await Promise.allSettled(
          tickersData.map(t =>
            api.getCalls(t.ticker).then(data => ({ ticker: t.ticker, data }))
          )
        )
        const map = {}
        for (const r of results) {
          if (r.status === 'fulfilled' && r.value.data.length > 0) {
            map[r.value.ticker] = r.value.data
          }
        }
        setCallsMap(map)
      } catch {
        setApiDown(true)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  // All scored calls sorted newest-first (for the table)
  const allCalls = useMemo(() => {
    const calls = Object.entries(callsMap).flatMap(([ticker, list]) =>
      list.map(c => ({ ...c, ticker }))
    )
    return calls.sort((a, b) => (b.call_date || '').localeCompare(a.call_date || ''))
  }, [callsMap])

  // Fix 1: latest scored call per ticker for the table
  const latestPerTicker = useMemo(() =>
    Object.entries(callsMap)
      .map(([ticker, calls]) => ({ ...calls[calls.length - 1], ticker }))
      .sort((a, b) => (b.call_date || '').localeCompare(a.call_date || ''))
  , [callsMap])

  // Fix 3: chart data — all calls as scatter dots sorted oldest→newest
  const chartData = useMemo(() =>
    [...allCalls]
      .sort((a, b) => (a.call_date || '').localeCompare(b.call_date || ''))
      .map(c => ({
        date:    c.call_date,
        score:   c.pressure_score,
        ticker:  c.ticker,
        flagged: c.flagged,
      }))
  , [allCalls])

  const avgPressure = allCalls.length
    ? allCalls.reduce((s, c) => s + (c.pressure_score || 0), 0) / allCalls.length
    : null

  const latestCall = latestPerTicker[0] || null

  if (loading) return <Spinner />

  if (apiDown) {
    return (
      <div className="bg-amber-50 border border-amber-200 text-amber-800 px-5 py-4 rounded-xl text-sm">
        <p className="font-semibold mb-1">Backend unavailable</p>
        <p>Make sure the FastAPI server is running on port 8000 (<code className="bg-amber-100 px-1 rounded font-mono">uvicorn api.main:app --reload</code>), then refresh.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-sm text-gray-500 mt-0.5">CEO earnings call pressure overview</p>
      </div>

      {/* Fix 2: Calls Analyzed uses /stats total_transcripts */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="CEOs Tracked"
          value={tickers.length}
          sub={`${appStats?.total_baselines ?? '—'} baselines built`}
        />
        <StatCard
          label="Calls Analyzed"
          value={appStats?.total_transcripts?.toLocaleString() ?? '—'}
          sub="total transcripts ingested"
        />
        <StatCard
          label="Flags Raised"
          value={flags.length}
          sub={flags.length === 0 ? 'all calls within 2σ' : `${flags.length} anomalous call${flags.length > 1 ? 's' : ''}`}
        />
        <StatCard
          label="Avg Pressure Score"
          value={avgPressure != null ? fmt(avgPressure) : '—'}
          sub={allCalls.length ? `from ${allCalls.length.toLocaleString()} scored calls` : 'no scored calls yet'}
        />
      </div>

      {/* Chart + Latest Call */}
      <div className="grid grid-cols-3 gap-4">

        {/* Fix 3: Scatter chart — one dot per scored call, colored by flagged status */}
        <div className="col-span-2 bg-white rounded-xl border border-gray-100 p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-gray-900">Pressure Score Timeline</h2>
            <div className="flex items-center gap-3 text-xs text-gray-400">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full inline-block bg-blue-500 opacity-60" />
                OK
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full inline-block bg-red-500" />
                FLAGGED
              </span>
            </div>
          </div>
          {chartData.length === 0 ? (
            <div className="h-48 flex flex-col items-center justify-center gap-2 text-sm text-gray-400">
              <p>No scored calls yet.</p>
              <p className="text-xs text-gray-300">
                Run <code className="bg-gray-100 text-gray-500 px-1 rounded font-mono">POST /calls/&#123;ticker&#125;/score</code> to populate the chart.
              </p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={chartData} margin={{ top: 4, right: 12, left: -8, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: '#9CA3AF' }}
                  tickLine={false}
                  axisLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontSize: 10, fill: '#9CA3AF' }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={v => v.toFixed(1)}
                />
                <Tooltip content={<ChartTooltip />} />
                <ReferenceLine
                  y={0.2}
                  stroke="#EF4444"
                  strokeDasharray="4 4"
                  label={{ value: '2σ ref', position: 'insideTopRight', fontSize: 9, fill: '#EF4444' }}
                />
                {/* Single line, strokeWidth=0 — only the custom dots are rendered */}
                <Line
                  dataKey="score"
                  strokeWidth={0}
                  isAnimationActive={false}
                  dot={({ cx, cy, payload, index }) => (
                    <circle
                      key={`dot-${index}`}
                      cx={cx}
                      cy={cy}
                      r={payload.flagged ? 5 : 3}
                      fill={payload.flagged ? '#EF4444' : '#2563EB'}
                      fillOpacity={payload.flagged ? 1 : 0.55}
                    />
                  )}
                  activeDot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Latest scored call */}
        <div className="bg-white rounded-xl border border-gray-100 p-5 flex flex-col">
          <h2 className="text-sm font-semibold text-gray-900 mb-4">Latest Scored Call</h2>
          {latestCall ? (
            <div className="flex-1 flex flex-col">
              <div className="flex items-start justify-between mb-1">
                <span className="text-2xl font-bold text-gray-900">{latestCall.ticker}</span>
                <StatusBadge flagged={latestCall.flagged} />
              </div>
              <p className="text-sm text-gray-400 mb-4">
                {latestCall.call_date} &middot; Q{latestCall.quarter} {latestCall.year}
              </p>
              <div className="border-t border-gray-50 pt-4 space-y-3 flex-1">
                {[
                  ['Pressure Score', fmt(latestCall.pressure_score), false],
                  ['Z-Score', fmtZ(latestCall.z_score), Math.abs(latestCall.z_score || 0) >= 2],
                  ['Cosine Distance', fmt(latestCall.cosine_distance), false],
                  ['Sentiment Shift', fmt(latestCall.sentiment_shift), false],
                ].map(([label, value, hl]) => (
                  <div key={label} className="flex justify-between text-sm">
                    <span className="text-gray-400">{label}</span>
                    <span className={`font-semibold font-mono ${hl ? 'text-red-600' : 'text-gray-900'}`}>{value}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-sm text-gray-400 flex-1">No scored calls available.</p>
          )}
        </div>
      </div>

      {/* Fix 1: Recent Calls — latest scored call per ticker */}
      <div className="bg-white rounded-xl border border-gray-100">
        <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-900">Recent Calls</h2>
          <span className="text-xs text-gray-400">{latestPerTicker.length} ticker{latestPerTicker.length !== 1 ? 's' : ''} scored</span>
        </div>
        {latestPerTicker.length === 0 ? (
          <div className="px-5 py-10 text-center text-sm text-gray-400">No scored calls to display.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-400 border-b border-gray-100">
                  {['Ticker', 'CEO', 'Call Date', 'Quarter', 'Pressure Score', 'Z-Score', 'Status'].map(h => (
                    <th key={h} className="px-5 py-3 font-medium whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {latestPerTicker.slice(0, 20).map((call, i) => (
                  <tr key={i} className="hover:bg-gray-50/60 transition-colors">
                    <td className="px-5 py-3 font-semibold text-blue-600">{call.ticker}</td>
                    <td className="px-5 py-3 text-gray-600 whitespace-nowrap">{api.CEO_NAMES[call.ticker] || '—'}</td>
                    <td className="px-5 py-3 text-gray-700 whitespace-nowrap">{call.call_date}</td>
                    <td className="px-5 py-3 text-gray-700">Q{call.quarter} {call.year}</td>
                    <td className="px-5 py-3 text-gray-700 font-mono">{fmt(call.pressure_score)}</td>
                    <td className="px-5 py-3 font-mono">
                      <span className={Math.abs(call.z_score || 0) >= 2 ? 'text-red-600 font-semibold' : 'text-gray-700'}>
                        {fmtZ(call.z_score)}
                      </span>
                    </td>
                    <td className="px-5 py-3"><StatusBadge flagged={call.flagged} /></td>
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
