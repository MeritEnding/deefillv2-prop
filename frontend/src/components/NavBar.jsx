import { account } from '../lib/account.js'

export default function NavBar({ route, navigate, acc }) {
  const remaining = account.freeRemaining(acc)
  return (
    <header className="nav">
      <div className="nav-inner">
        <a className="brand" href="#/" onClick={() => navigate('/')}>
          <span className="brand-mark" />
          <span className="brand-name">지움</span>
        </a>
        <nav className="nav-links">
          <a className={route === '/' ? 'on' : ''} href="#/" onClick={() => navigate('/')}>
            홈
          </a>
          <a className={route === '/editor' ? 'on' : ''} href="#/editor" onClick={() => navigate('/editor')}>
            지우개
          </a>
          <a className={route === '/pricing' ? 'on' : ''} href="#/pricing" onClick={() => navigate('/pricing')}>
            요금제
          </a>
        </nav>
        <div className="nav-right">
          {acc.plan === 'pro' ? (
            <span className="pill pill-pro">크레딧 {acc.credits}</span>
          ) : (
            <span className="pill">오늘 무료 {remaining}/{account.FREE_DAILY_LIMIT}</span>
          )}
          <button className="btn btn-primary btn-sm" onClick={() => navigate('/editor')}>
            지우러 가기
          </button>
        </div>
      </div>
    </header>
  )
}
