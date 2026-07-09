import BeforeAfter from '../components/BeforeAfter.jsx'
import { Icon } from '../components/icons.jsx'
import { MODES } from '../lib/modes.js'

const USE_CASES = [
  { icon: 'bag', title: '중고거래 사진', desc: '배경의 개인 정보, 택배 상자, 잡동사니를 깔끔하게 제거' },
  { icon: 'plane', title: '여행 사진', desc: '인생샷에 끼어든 낯선 행인이나 전봇대, 간판 등을 제거' },
  { icon: 'box', title: '쇼핑몰 상품컷', desc: '워터마크, 로고, 먼지, 반사광을 제거해 상품에만 집중' },
  { icon: 'building', title: '부동산 매물', desc: '어질러진 물건이나 전선을 지워 넓고 깔끔한 공간으로 연출' },
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
            사진 속 불필요한<br />
            요소를 <span className="accent">3초 만에</span><br />
            자연스럽게 제거하세요.
          </h1>
          <p className="hero-sub">
            사람, 물건, 글자, 워터마크까지.<br />
            원래 없었던 것처럼 깔끔하게 지워드립니다.
          </p>
          <div className="hero-cta">
            <button className="btn btn-primary btn-lg" onClick={() => navigate('/studio')}>무료로 시작하기</button>
            <button className="btn btn-outline btn-lg" onClick={() => navigate('/studio')}>사용 예시 보기</button>
          </div>
          <p className="hero-note">회원 가입 없이 · 설치 없이 브라우저에서 바로 · 완전 무료</p>
        </div>
        <div className="hero-demo">
          <BeforeAfter before="/demo-before.png" after="/demo-after.png" />
          <p className="hero-demo-cap">← 좌우로 드래그해서 비교해보세요</p>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>이럴 때 씁니다</h2>
          <p className="section-sub">하나만 지워도 사진의 완성도가 달라집니다.</p>
        </div>
        <div className="cards">
          {USE_CASES.map((u) => {
            const I = Icon[u.icon]
            return (
              <article className="card usecase" key={u.title}>
                <div className="tile"><I /></div>
                <h3>{u.title}</h3>
                <p>{u.desc}</p>
              </article>
            )
          })}
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>하나의 도구, 여섯 가지 쓰임</h2>
          <p className="section-sub">같은 AI 엔진으로 이만큼 됩니다.</p>
        </div>
        <div className="mode-grid">
          {MODES.map((m) => {
            const I = Icon[m.icon]
            return (
              <button className="mode-card" key={m.id} onClick={() => navigate('/studio?mode=' + m.id)}>
                <div className="tile tile-sm"><I /></div>
                <div>
                  <div className="mode-card-name">{m.name}</div>
                  <div className="mode-card-short">{m.short}</div>
                </div>
              </button>
            )
          })}
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
          <button className="btn btn-primary btn-lg" onClick={() => navigate('/studio')}>스튜디오 열기</button>
        </div>
      </section>

      <footer className="footer">
        <div className="footer-inner">
          <div className="brand"><span className="brand-mark" /><span className="brand-name">지움</span></div>
          <p className="footer-note">사전학습 LaMa · DeepFillv2 인페인팅 기반 · 데모 프로젝트</p>
        </div>
      </footer>
    </main>
  )
}
