export default function NavBar({ route, navigate }) {
  const link = (to, label) => (
    <a className={route === to ? 'on' : ''} href={`#${to}`} onClick={() => navigate(to)}>
      {label}
    </a>
  )
  return (
    <header className="nav">
      <div className="nav-inner">
        <a className="brand" href="#/" onClick={() => navigate('/')}>
          <img className="brand-logo" src="/logo-icon.png" alt="지움 로고" />
          <span className="brand-name">지움</span>
        </a>
        <nav className="nav-links">
          {link('/', '홈')}
          {link('/privacy', '개인정보 지우기')}
          {link('/studio', '스튜디오')}
        </nav>
        <div className="nav-right">
          <button className="btn btn-dark btn-sm" onClick={() => navigate('/privacy')}>
            무료로 시작하기
          </button>
        </div>
      </div>
    </header>
  )
}
