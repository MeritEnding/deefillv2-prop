import { Icon } from '../components/icons.jsx'

// 지울 항목을 먼저 고르는 허브 — 항목별 전용 페이지로 안내한다
const ITEMS = [
  {
    to: '/privacy/face',
    icon: 'shield',
    tag: '자동 인식',
    auto: true,
    title: '얼굴 지우기',
    desc: '사진 속 모든 얼굴을 자동으로 찾아 표시합니다. 지울 것만 체크하면 원래 배경으로 복원돼요.',
    cta: '얼굴 지우러 가기',
  },
  {
    to: '/privacy/plate',
    icon: 'car',
    tag: '자동 인식 · 베타',
    auto: true,
    title: '차량 번호판 지우기',
    desc: '주차장·사고·거리 사진 속 번호판을 자동으로 찾아 흔적 없이 지웁니다.',
    cta: '번호판 지우러 가기',
  },
  {
    to: '/studio',
    icon: 'brush',
    tag: '클릭 · 붓질',
    auto: false,
    title: '그 밖의 개인정보',
    desc: '서류 정보, 명찰, 운송장처럼 자동으로 못 찾는 정보는 스튜디오에서 클릭 한 번 또는 붓질로 지웁니다.',
    cta: '스튜디오로 가기',
  },
]

export default function PrivacyHub({ navigate }) {
  return (
    <main className="studio page-narrow">
      <div className="section-head">
        <h2 className="sec-title">Privacy <i>|</i> <small>개인정보 지우기</small></h2>
        <p className="section-sub">무엇을 지울까요? 항목마다 전용 화면에서 찾고, 지우고, 복원까지 한 번에 끝냅니다.</p>
      </div>
      <div className="pvhub-grid">
        {ITEMS.map((it) => {
          const ItemIcon = Icon[it.icon] ?? Icon.shield
          return (
            <button key={it.to} className="pvhub-card" onClick={() => navigate(it.to)}>
              <div className="pvhub-head">
                <div className="tile tile-sm"><ItemIcon /></div>
                <span className={`rtag ${it.auto ? 'rtag-auto' : ''}`}>{it.tag}</span>
              </div>
              <h3>{it.title}</h3>
              <p>{it.desc}</p>
              <span className="pvhub-cta">{it.cta} →</span>
            </button>
          )
        })}
      </div>
      <p className="cover-note">
        무엇을 지우든 결과는 같습니다 — 지운 자리는 모자이크가 아니라 <strong>원래 배경</strong>으로 채워집니다.
      </p>
    </main>
  )
}
