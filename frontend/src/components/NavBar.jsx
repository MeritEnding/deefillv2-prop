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
          <span className="brand-mark" />
          <span className="brand-name">지움</span>
        </a>
        <nav className="nav-links">
          {link('/', '홈')}
          {link('/studio', '스튜디오')}
        </nav>
        <div className="nav-right">
          <button className="btn btn-primary btn-sm" onClick={() => navigate('/studio')}>
            무료로 시작하기
          </button>
        </div>
      </div>
    </header>
  )
}
