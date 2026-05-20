import { useState, useEffect } from 'react'
import { getFlags } from '../api/client'

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

function fmtZ(z) {
  if (z == null) return '—'
  return (z >= 0 ? '+' : '') + Number(z).toFixed(2)
}

export default function Flags() {
  const [flags, setFlags]   = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState(null)

  useEffect(() => {
    getFlags()
      .then(setFlags)
      .catch(() => setError('Could not load flags. Is the backend running?'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Spinner />
  if (error) return (
    <div className="bg-amber-50 border border-amber-200 text-amber-800 px-5 py-4 rounded-xl text-sm">{error}</div>
  )

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Flags</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Calls where pressure score exceeded 2 standard deviations from the CEO baseline
        </p>
      </div>

      {flags.length === 0 ? (
        <div className="bg-white rounded-xl border border-gray-100 px-5 py-16 text-center">
          <div
            className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
            style={{ backgroundColor: '#F0FDF4' }}
          >
            <svg className="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <p className="text-base font-semibold text-gray-700 mb-1">No flags raised yet</p>
          <p className="text-sm text-gray-400">All scored calls are within 2σ of each CEO's baseline</p>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-gray-100 overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-100 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
            <span className="text-sm font-semibold text-gray-900">
              {flags.length} flagged call{flags.length !== 1 ? 's' : ''}
            </span>
            <span className="text-xs text-gray-400 ml-1">z-score &ge; 2.0</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-400 border-b border-gray-100">
                  {['Ticker', 'Company', 'Call Date', 'Quarter', 'Pressure Score', 'Z-Score'].map(h => (
                    <th key={h} className="px-5 py-3 font-medium whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {[...flags]
                  .sort((a, b) => (b.z_score || 0) - (a.z_score || 0))
                  .map((flag, i) => (
                    <tr key={i} className="hover:bg-red-50/30 transition-colors">
                      <td className="px-5 py-3 font-semibold text-blue-600">{flag.ticker}</td>
                      <td className="px-5 py-3 text-gray-400 text-xs whitespace-nowrap">{flag.company_name}</td>
                      <td className="px-5 py-3 text-gray-700 whitespace-nowrap">{flag.call_date}</td>
                      <td className="px-5 py-3 text-gray-700">Q{flag.quarter} {flag.year}</td>
                      <td className="px-5 py-3 font-mono font-semibold text-red-600">
                        {fmt(flag.pressure_score)}
                      </td>
                      <td className="px-5 py-3 font-mono font-semibold text-red-600">
                        {fmtZ(flag.z_score)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
