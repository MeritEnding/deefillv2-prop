import { useRef, useState } from 'react'
import BeforeAfter from '../components/BeforeAfter.jsx'
import { Icon } from '../components/icons.jsx'

// 졸업연구(DeepFillv2 개선안) 복원 결과 — 마우스를 올리면 마스크 입력이 보인다
const GALLERY = ['12', '14', '6', '16', '2', '9', '17', '59'].map((n) => ({
  masked: `/gallery/g${n}-masked.png`,
  result: `/gallery/g${n}-result.png`,
}))

const METRICS = [
  { label: 'PSNR', before: '28.28', after: '29.01', note: '신호 대 잡음비 · 높을수록 원본에 가까움' },
  { label: 'SSIM', before: '0.914', after: '0.951', note: '구조적 유사도 · 1에 가까울수록 좋음' },
  { label: 'FID', before: '3.1', after: '2.8', note: '분포 거리 · 낮을수록 자연스러움' },
]

const USE_CASES = [
  { icon: 'shield', title: 'CCTV·현장 사진', desc: '보고서 첨부 전 얼굴을 자동 비식별' },
  { icon: 'building', title: '건설·안전 보고', desc: '작업자 얼굴을 지우고 배경 복원' },
  { icon: 'box', title: '보험·사고 접수', desc: '차량·인물 개인정보 정리' },
  { icon: 'bag', title: '커머스·매물 사진', desc: '워터마크·잡동사니는 스튜디오에서' },
]

const STEPS = [
  { img: '/mascot-select.png', n: '01', title: '업로드하면 자동 분석', desc: '얼굴 등 개인정보를 AI가 찾아냅니다.' },
  { img: '/mascot-erase.png', n: '02', title: '제거할 항목만 선택', desc: '박스를 눌러 지울 대상을 고르세요.' },
  { img: '/mascot-excited.png', n: '03', title: '지우고, 자연 복원', desc: '모자이크 대신 배경을 다시 그립니다.' },
]

function GalleryCarousel() {
  const railRef = useRef(null)
  const [progress, setProgress] = useState(0)

  const onScroll = () => {
    const el = railRef.current
    if (!el) return
    const max = el.scrollWidth - el.clientWidth
    setProgress(max > 0 ? el.scrollLeft / max : 0)
  }
  const shift = (dir) => railRef.current?.scrollBy({ left: dir * 540, behavior: 'smooth' })

  return (
    <>
      <div className="rail" ref={railRef} onScroll={onScroll}>
        {GALLERY.map((g) => (
          <figure className="rail-card" key={g.result}>
            <div className="rail-imgs">
              <img src={g.result} alt="복원 결과" loading="lazy" />
              <img className="rail-hover" src={g.masked} alt="마스크 입력" loading="lazy" />
            </div>
            <figcaption>
              <span>결손 복원</span>
              <em>마우스를 올리면 입력</em>
            </figcaption>
          </figure>
        ))}
      </div>
      <div className="rail-nav">
        <div className="rail-track"><i style={{ width: `${8 + progress * 92}%` }} /></div>
        <button aria-label="이전" onClick={() => shift(-1)}><Icon.undo width="15" height="15" /></button>
        <button aria-label="다음" onClick={() => shift(1)}><Icon.redo width="15" height="15" /></button>
      </div>
    </>
  )
}

export default function Landing({ navigate }) {
  return (
    <main className="landing">
      {/* 히어로 — 딥 네이비 패널 */}
      <section className="hero">
        <div className="hero-inner">
          <div className="hero-copy">
            <span className="hero-eyebrow">AI Privacy Redaction</span>
            <h1 className="hero-title">
              가리지 말고,<br />지우세요.
            </h1>
            <p className="hero-sub">
              모자이크는 티가 납니다. 지움은 사진 속 얼굴 등 개인정보를 자동으로 찾아
              제거하고, 그 자리를 배경으로 자연스럽게 복원합니다.
              CCTV 캡처·현장 사진·사고 접수, 그대로 보고서에 쓰세요.
            </p>
            <div className="hero-cta">
              <button className="btn btn-white btn-lg" onClick={() => navigate('/privacy')}>비식별화 시작</button>
              <button className="btn btn-hero-ghost btn-lg" onClick={() => navigate('/studio')}>수동 스튜디오</button>
              <span className="hero-free">가입 없음 · 무료</span>
            </div>
          </div>
          <div className="hero-visual">
            <figure className="polaroid polaroid-back">
              <img src="/logo-icon.png" alt="지움 로고" />
              <figcaption>before / after</figcaption>
            </figure>
            <figure className="polaroid polaroid-front">
              <img src="/mascot-hero.png" alt="마스코트 지우" />
              <figcaption>지우개 '지우'</figcaption>
            </figure>
          </div>
        </div>
      </section>

      {/* 복원 갤러리 — 연구 결과 */}
      <section className="section">
        <div className="section-head">
          <h2 className="sec-title">Gallery <i>|</i> <small>복원 갤러리</small></h2>
          <p className="section-sub">졸업연구에서 개선한 DeepFillv2 모델의 실제 복원 결과입니다.</p>
        </div>
        <GalleryCarousel />
      </section>

      {/* 데모 비교 */}
      <section className="section section-tint">
        <div className="section-inner demo-split">
          <div className="demo-copy">
            <h2 className="sec-title">Try it <i>|</i> <small>직접 비교</small></h2>
            <p className="section-sub">
              핸들을 좌우로 움직여 보세요.<br />
              길 한가운데 서 있던 사람이 사라졌습니다.
            </p>
            <button className="btn btn-dark" onClick={() => navigate('/studio')}>내 사진으로 해보기</button>
          </div>
          <div className="demo-media">
            <BeforeAfter before="/demo-before.png" after="/demo-after.png" />
          </div>
        </div>
      </section>

      {/* 연구 지표 */}
      <section className="section">
        <div className="section-head">
          <h2 className="sec-title">Research <i>|</i> <small>성능 개선 연구</small></h2>
          <p className="section-sub">
            DeepFillv2의 생성기 구조를 다중 브랜치로 변형하고 다양한 마스크 형태로 검증했습니다.
          </p>
        </div>
        <div className="metrics">
          {METRICS.map((m) => (
            <div className="metric" key={m.label}>
              <span className="metric-name">{m.label}</span>
              <div className="metric-vals">
                <span className="metric-before">{m.before}</span>
                <span className="metric-arrow">→</span>
                <span className="metric-after">{m.after}</span>
              </div>
              <p>{m.note}</p>
            </div>
          ))}
        </div>
        <p className="research-note">
          「DeepFillv2 인페인팅 구조 변형에 따른 성능 개선과 실증 구현」 · 황지윤, 이정훈 ·
          한밭대학교 정보통신공학과 졸업연구 (2024) · CelebA 32장 배치 평가 기준
        </p>
      </section>

      {/* 사용법 */}
      <section className="section section-tint">
        <div className="section-inner">
          <div className="section-head">
            <h2 className="sec-title">Pipeline <i>|</i> <small>처리 과정</small></h2>
            <p className="section-sub">탐지(YuNet) → 마스크 → 인페인팅(LaMa) 파이프라인이 자동으로 이어집니다.</p>
          </div>
          <div className="steps">
            {STEPS.map((s) => (
              <div className="step" key={s.n}>
                <img className="step-img" src={s.img} alt="" />
                <span className="step-n">{s.n}</span>
                <h3>{s.title}</h3>
                <p>{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 활용 */}
      <section className="section">
        <div className="section-head">
          <h2 className="sec-title">Use cases <i>|</i> <small>이럴 때 씁니다</small></h2>
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

      {/* 기업 연동 API */}
      <section className="section">
        <div className="api-split">
          <div>
            <h2 className="sec-title">For business <i>|</i> <small>연동 API</small></h2>
            <p className="section-sub">
              웹 화면 없이도 씁니다. 업로드 한 번이면 탐지부터 복원까지 —
              사내 시스템·업무 자동화에 REST API로 바로 붙이세요.
              키 인증과 사용량 집계를 지원합니다.
            </p>
            <ul className="api-points">
              <li>원콜 처리 — 탐지 → 제거 → 자연 복원</li>
              <li><code>X-API-Key</code> 인증 · 키별 사용량 조회(<code>/api/v1/usage</code>)</li>
              <li>처리 결과 헤더 — <code>X-Redacted-Count</code></li>
            </ul>
          </div>
          <pre className="api-code">{`curl -X POST http://localhost:8080/api/v1/redact \\
  -H "X-API-Key: <발급받은 키>" \\
  -F "image=@site-photo.jpg" \\
  -o redacted.png

# X-Redacted-Count: 3  (제거된 개인정보 수)`}</pre>
        </div>
      </section>

      {/* CTA */}
      <section className="cta-band">
        <div className="cta-band-inner">
          <h2>한 장 지워보면 압니다.</h2>
          <p>회원 가입 없이 브라우저에서 바로.</p>
          <button className="btn btn-white btn-lg" onClick={() => navigate('/privacy')}>무료로 시작하기</button>
        </div>
      </section>

      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-brand">
            <img src="/logo-icon.png" alt="지움 로고" />
            <div>
              <strong>지움</strong>
              <p>사진 속 불필요한 것을 지우는 가장 쉬운 방법</p>
            </div>
          </div>
          <div className="footer-cols">
            <div>
              <h4>Technology</h4>
              <p>LaMa · DeepFillv2 개선안</p>
              <p>FastAPI · Spring Boot · React</p>
            </div>
            <div>
              <h4>Research</h4>
              <p>DeepFillv2 인페인팅 구조 변형에<br />따른 성능 개선과 실증 구현</p>
              <p>황지윤 · 이정훈 (2024)</p>
            </div>
            <div>
              <h4>Operations</h4>
              <p><a href="#/admin">관리자 콘솔</a></p>
              <p>감사 로그 · 사용량 통계</p>
            </div>
          </div>
        </div>
        <p className="footer-fine">한밭대학교 정보통신공학과 졸업연구 기반 · 업로드한 사진은 처리 후 저장되지 않습니다.</p>
      </footer>
    </main>
  )
}
