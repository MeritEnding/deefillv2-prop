import { useEffect, useState } from 'react'
import './App.css'
import NavBar from './components/NavBar.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import Admin from './pages/Admin.jsx'
import Cases from './pages/Cases.jsx'
import Community from './pages/Community.jsx'
import Enhance from './pages/Enhance.jsx'
import Landing from './pages/Landing.jsx'
import Notices from './pages/Notices.jsx'
import Privacy from './pages/Privacy.jsx'
import PersonErase from './pages/PersonErase.jsx'
import PrivacyHub from './pages/PrivacyHub.jsx'
import Stats from './pages/Stats.jsx'
import Studio from './pages/Studio.jsx'
import Tips from './pages/Tips.jsx'
import { MODE_MAP } from './lib/modes.js'
import { getJSON } from './lib/api.js'

function parseHash() {
  const raw = window.location.hash.replace(/^#/, '') || '/'
  const [path, query] = raw.split('?')
  const params = new URLSearchParams(query || '')
  return { path, mode: params.get('mode') }
}

export default function App() {
  const [{ path, mode }, setRoute] = useState(parseHash())
  const [engine, setEngine] = useState(null) // 현재 선택된 모델 id
  const [engines, setEngines] = useState([]) // 사용 가능 모델 카탈로그

  useEffect(() => {
    const on = () => setRoute(parseHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  useEffect(() => {
    getJSON('/api/engines').then((d) => {
      const list = d.engines ?? []
      setEngines(list)
      const saved = localStorage.getItem('jium-engine')
      const usable = list.some((e) => e.id === saved && e.available)
      setEngine(usable ? saved : (d.default ?? null))
    }).catch(() => { setEngines([]); setEngine(null) })
  }, [])

  // 모델 선택을 기억한다(새로고침·페이지 이동에도 유지).
  const chooseEngine = (id) => { setEngine(id); localStorage.setItem('jium-engine', id) }

  const navigate = (to) => { window.location.hash = to; window.scrollTo({ top: 0 }) }
  const modeId = MODE_MAP[mode] ? mode : 'object'
  const setModeId = (id) => navigate(`/studio?mode=${id}`)

  return (
    <>
      <NavBar route={path} navigate={navigate} />
      {path === '/admin' ? (
        <Admin />
      ) : path === '/notices' ? (
        <Notices />
      ) : path === '/tips' ? (
        <Tips />
      ) : path === '/cases' ? (
        <Cases navigate={navigate} />
      ) : path === '/community' ? (
        <Community />
      ) : path === '/stats' ? (
        <Stats />
      ) : path === '/privacy' ? (
        <PrivacyHub navigate={navigate} />
      ) : path === '/enhance' ? (
        <ErrorBoundary onReset={() => navigate('/enhance')}>
          <Enhance key={mode ?? 'face'} engine={engine} modeId={mode ?? 'face'} navigate={navigate} />
        </ErrorBoundary>
      ) : path === '/privacy/person' ? (
        <ErrorBoundary onReset={() => navigate('/privacy/person')}>
          <PersonErase engine={engine} engines={engines} setEngine={chooseEngine} />
        </ErrorBoundary>
      ) : path === '/privacy/face' || path === '/privacy/plate' || path === '/privacy/text' ? (
        <ErrorBoundary onReset={() => navigate(path)}>
          <Privacy key={path} engine={engine} engines={engines} setEngine={chooseEngine} kind={path.split('/')[2]} />
        </ErrorBoundary>
      ) : path === '/studio' ? (
        <ErrorBoundary onReset={() => navigate('/studio')}>
          <Studio modeId={modeId} setModeId={setModeId} engine={engine} engines={engines} setEngine={chooseEngine} />
        </ErrorBoundary>
      ) : (
        <Landing navigate={navigate} />
      )}
    </>
  )
}
