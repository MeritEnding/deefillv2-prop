import { useEffect, useState } from 'react'
import './App.css'
import NavBar from './components/NavBar.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import Landing from './pages/Landing.jsx'
import Studio from './pages/Studio.jsx'
import { MODE_MAP } from './lib/modes.js'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8080'

function parseHash() {
  const raw = window.location.hash.replace(/^#/, '') || '/'
  const [path, query] = raw.split('?')
  const params = new URLSearchParams(query || '')
  return { path, mode: params.get('mode') }
}

export default function App() {
  const [{ path, mode }, setRoute] = useState(parseHash())
  const [engine, setEngine] = useState(null)

  useEffect(() => {
    const on = () => setRoute(parseHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  useEffect(() => {
    fetch(`${API_BASE}/api/health`).then((r) => r.json()).then((d) => setEngine(d.engine ?? null)).catch(() => setEngine(null))
  }, [])

  const navigate = (to) => { window.location.hash = to; window.scrollTo({ top: 0 }) }
  const modeId = MODE_MAP[mode] ? mode : 'object'
  const setModeId = (id) => navigate(`/studio?mode=${id}`)

  return (
    <>
      <NavBar route={path} navigate={navigate} />
      {path === '/studio' ? (
        <ErrorBoundary onReset={() => navigate('/studio')}>
          <Studio modeId={modeId} setModeId={setModeId} engine={engine} />
        </ErrorBoundary>
      ) : (
        <Landing navigate={navigate} />
      )}
    </>
  )
}
