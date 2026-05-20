import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import CEOProfiles from './pages/CEOProfiles'
import Calls from './pages/Calls'
import Flags from './pages/Flags'
import About from './pages/About'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="ceo-profiles" element={<CEOProfiles />} />
          <Route path="calls" element={<Calls />} />
          <Route path="flags" element={<Flags />} />
          <Route path="about" element={<About />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
