import { useState, useEffect, useRef } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { getTickers } from '../api/client'

const NAV = [
  { to: '/dashboard',    label: 'Dashboard' },
  { to: '/ceo-profiles', label: 'CEO Profiles' },
  { to: '/calls',        label: 'Calls' },
  { to: '/flags',        label: 'Flags' },
  { to: '/about',        label: 'About' },
]

export default function Sidebar() {
  const [query, setQuery]           = useState('')
  const [tickers, setTickers]       = useState([])
  const [showDropdown, setShowDropdown] = useState(false)
  const navigate = useNavigate()
  const dropdownRef = useRef(null)

  useEffect(() => {
    getTickers().then(setTickers).catch(() => {})
  }, [])

  useEffect(() => {
    function handleClick(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setShowDropdown(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const filtered = query.trim().length > 0
    ? tickers.filter(t =>
        t.ticker.toLowerCase().includes(query.toLowerCase()) ||
        (t.company_name || '').toLowerCase().includes(query.toLowerCase())
      )
    : []

  function selectTicker(ticker) {
    navigate(`/calls?ticker=${ticker}`)
    setQuery('')
    setShowDropdown(false)
  }

  return (
    <aside className="w-[220px] min-h-screen bg-white border-r border-gray-100 flex flex-col flex-shrink-0">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-gray-100">
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center text-white text-sm font-bold flex-shrink-0"
            style={{ backgroundColor: '#2563EB' }}
          >
            PT
          </div>
          <span className="font-semibold text-gray-900 text-sm">PressureTest</span>
        </div>
      </div>

      {/* Search */}
      <div className="px-3 py-3 border-b border-gray-100 relative" ref={dropdownRef}>
        <input
          type="text"
          placeholder="Search ticker..."
          value={query}
          onChange={e => { setQuery(e.target.value); setShowDropdown(true) }}
          onFocus={() => query.trim() && setShowDropdown(true)}
          className="w-full px-3 py-2 text-xs bg-gray-50 border border-gray-200 rounded-md focus:outline-none focus:ring-2 focus:border-blue-500 transition"
          style={{ '--tw-ring-color': 'rgba(37,99,235,0.15)' }}
        />
        {showDropdown && filtered.length > 0 && (
          <div className="absolute left-3 right-3 top-full mt-1 bg-white border border-gray-200 rounded-md shadow-lg z-50 overflow-hidden">
            {filtered.map(t => (
              <button
                key={t.ticker}
                onMouseDown={() => selectTicker(t.ticker)}
                className="w-full text-left px-3 py-2.5 text-xs hover:bg-gray-50 flex items-center justify-between gap-2"
              >
                <span className="font-semibold text-gray-900">{t.ticker}</span>
                <span className="text-gray-400 truncate text-right">{t.company_name}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-3 space-y-0.5">
        {NAV.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `block px-3 py-2.5 rounded-md text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-blue-50 text-blue-600'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-gray-100">
        <p className="text-xs text-gray-400">PressureTest v1.0</p>
      </div>
    </aside>
  )
}
