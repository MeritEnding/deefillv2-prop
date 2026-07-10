export default function NavBar({ route, navigate }) {
  const link = (to, label) => {
    const active = route === to || (to !== '/' && route.startsWith(`${to}/`))
    return (
      <a className={active ? 'on' : ''} href={`#${to}`} onClick={() => navigate(to)}>
        {label}
      </a>
    )
  }
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
          {link('/enhance', '화질 복원')}
          {link('/cases', '고객 사례')}
          {link('/tips', '활용 팁')}
          {link('/community', '커뮤니티')}
          {link('/notices', '공지')}
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
