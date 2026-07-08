import BeforeAfter from '../components/BeforeAfter.jsx'

const USE_CASES = [
  {
    icon: '🛍️',
    title: '중고거래 사진',
    desc: '배경의 개인정보, 택배 송장, 잡동사니를 지우고 상품만 깔끔하게.',
  },
  {
    icon: '🏖️',
    title: '여행 사진',
    desc: '인생샷에 끼어든 낯선 행인을 지워 나만의 장면으로.',
  },
  {
    icon: '📦',
    title: '쇼핑몰 상품컷',
    desc: '워터마크·로고·먼지·반사광을 제거해 상세페이지 품질 업.',
  },
  {
    icon: '🏠',
    title: '부동산 매물',
    desc: '어질러진 물건이나 전선을 지워 넓고 정돈된 공간으로.',
  },
]

const STEPS = [
  { n: '01', title: '사진 업로드', desc: '지우고 싶은 사진을 끌어다 놓으세요.' },
  { n: '02', title: '쓱쓱 칠하기', desc: '지울 대상을 브러시로 칠하면 끝.' },
  { n: '03', title: '3초 만에 완성', desc: 'AI가 배경을 복원해 자연스럽게 지웁니다.' },
]

export default function Landing({ navigate }) {
  return (
    <main className="landing">
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">AI 인페인팅 · 객체 제거</span>
          <h1 className="hero-title">
            사진 속 <span className="grad">지우고 싶은 것</span>,<br />
            3초 만에 감쪽같이.
          </h1>
          <p className="hero-sub">
            사람, 물건, 글자, 워터마크까지. 지울 부분만 칠하면 AI가 뒤 배경을 복원해
            원래 없었던 것처럼 만들어 드립니다.
          </p>
          <div className="hero-cta">
            <button className="btn btn-primary btn-lg" onClick={() => navigate('/editor')}>
              무료로 지워보기
            </button>
            <button className="btn btn-ghost btn-lg" onClick={() => navigate('/pricing')}>
              요금제 보기
            </button>
          </div>
          <p className="hero-note">가입 없이 하루 3장 무료 · 카드 등록 불필요</p>
        </div>
        <div className="hero-demo">
          <BeforeAfter before="/demo-before.png" after="/demo-after.png" />
          <p className="hero-demo-cap">← 손잡이를 드래그해 비교해 보세요</p>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>이럴 때 씁니다</h2>
          <p className="section-sub">하나만 지워도 사진의 완성도가 달라집니다.</p>
        </div>
        <div className="cards">
          {USE_CASES.map((u) => (
            <article className="card usecase" key={u.title}>
              <div className="usecase-icon">{u.icon}</div>
              <h3>{u.title}</h3>
              <p>{u.desc}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>사용법은 세 단계</h2>
          <p className="section-sub">포토샵도, 설치도 필요 없습니다.</p>
        </div>
        <div className="steps">
          {STEPS.map((s) => (
            <div className="step" key={s.n}>
              <span className="step-n">{s.n}</span>
              <h3>{s.title}</h3>
              <p>{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="cta-band">
        <div className="cta-band-inner">
          <h2>지금 바로, 무료로.</h2>
          <p>업로드 한 장이면 충분합니다.</p>
          <button className="btn btn-primary btn-lg" onClick={() => navigate('/editor')}>
            지우개 열기
          </button>
        </div>
      </section>

      <footer className="footer">
        <div className="footer-inner">
          <div className="brand">
            <span className="brand-mark" />
            <span className="brand-name">지움</span>
          </div>
          <p className="footer-note">
            DeepFillv2 기반 인페인팅 엔진으로 구동됩니다. · 데모 프로젝트
          </p>
        </div>
      </footer>
    </main>
  )
}
