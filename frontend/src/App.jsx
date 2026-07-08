import { useEffect, useState } from 'react'
import './App.css'
import NavBar from './components/NavBar.jsx'
import Paywall from './components/Paywall.jsx'
import Landing from './pages/Landing.jsx'
import Editor from './pages/Editor.jsx'
import Pricing from './pages/Pricing.jsx'
import { account, loadAccount } from './lib/account.js'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8080'

function useHashRoute() {
  const parse = () => window.location.hash.replace(/^#/, '') || '/'
  const [route, setRoute] = useState(parse())
  useEffect(() => {
    const on = () => setRoute(parse())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  const navigate = (to) => {
    window.location.hash = to
    window.scrollTo({ top: 0 })
  }
  return [route, navigate]
}

export default function App() {
  const [route, navigate] = useHashRoute()
  const [acc, setAcc] = useState(loadAccount())
  const [paywall, setPaywall] = useState(false)
  const [engine, setEngine] = useState(null)
  const [toast, setToast] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((r) => r.json())
      .then((d) => setEngine(d.engine ?? null))
      .catch(() => setEngine(null))
  }, [])

  const buy = (plan) => {
    setAcc(account.addCredits(acc, plan.credits))
    setPaywall(false)
    setToast(`${plan.name} 구매 완료 · 크레딧 ${plan.credits}장 충전됐어요`)
    setTimeout(() => setToast(null), 3200)
    if (route === '/pricing') navigate('/editor')
  }

  return (
    <>
      <NavBar route={route} navigate={navigate} acc={acc} />
      {route === '/editor' ? (
        <Editor acc={acc} setAcc={setAcc} engine={engine} requirePaywall={() => setPaywall(true)} />
      ) : route === '/pricing' ? (
        <Pricing acc={acc} onBuy={buy} navigate={navigate} />
      ) : (
        <Landing navigate={navigate} />
      )}
      {paywall && <Paywall onClose={() => setPaywall(false)} onBuy={buy} />}
      {toast && <div className="toast">{toast}</div>}
    </>
  )
}
