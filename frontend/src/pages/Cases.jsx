// 고객 사례 — 도메인별 활용, 후기, 커뮤니티 인기글
const DOMAINS = [
  { k: '건설 · 안전', t: '현장 사진 보고', d: '작업자 얼굴을 자동으로 지우고 배경을 복원해, 안전 점검 보고서에 원본처럼 깨끗한 사진을 첨부합니다.' },
  { k: '보험 · 차량', t: '사고 접수 사진', d: '사고 차량 주변의 번호판과 행인 얼굴을 정리해 심사 서류의 개인정보 노출을 막습니다.' },
  { k: '공공 · 관제', t: 'CCTV 캡처 공유', d: '민원 회신이나 자료 제공 전에 제3자의 얼굴을 지웁니다. 모자이크가 아니라 기록물이 자연스럽습니다.' },
  { k: '커머스', t: '상품 사진 정리', d: '같은 위치에 박힌 공급사 워터마크를 일괄 지우기로 수십 장씩 한 번에 제거합니다.' },
  { k: '부동산', t: '매물 사진 연출', d: '생활감 있는 물건과 개인 사진 액자를 지워 넓고 정돈된 공간으로 보여줍니다.' },
  { k: '미디어 · 교육', t: '자료 화면 정리', d: '강의 자료나 기사 이미지 속 이름, 연락처, 로고를 지워 배포 가능한 자료로 만듭니다.' },
]

const REVIEWS = [
  { name: '건설사 안전팀 J', body: '보고서 한 건에 사진이 스무 장인데, 얼굴 모자이크에 쓰던 시간이 사실상 사라졌어요. 자동 탐지 체크만 하면 끝이라서요.', tag: '개인정보 지우기' },
  { name: '쇼핑몰 운영 M', body: '공급사 워터마크 지우는 게 일이었는데 일괄 지우기로 40장을 10분 안에 정리했습니다. 결과물이 티가 안 나는 게 제일 좋아요.', tag: '일괄 지우기' },
  { name: '손해사정 L', body: '사고 사진 속 번호판·행인 정리용으로 씁니다. 지운 자리가 배경으로 채워지니 서류가 깔끔합니다.', tag: '자동 탐지' },
  { name: '대학원생 P', body: '논문 그림에 들어간 기관 로고와 이름을 지우는 데 썼어요. 스마트 클릭이 영역을 알아서 잡아줘서 편했습니다.', tag: '스마트 클릭' },
]

const HOT_POSTS = [
  { title: '상품컷 40장 워터마크 10분 만에 정리했어요', name: '마케터K', likes: 12 },
  { title: '현장 사진 보고용으로 쓰고 있습니다', name: '안전관리자', likes: 9 },
  { title: '스마트 클릭 진짜 편하네요', name: '지우팬', likes: 7 },
]

export default function Cases({ navigate }) {
  return (
    <main className="studio page-narrow">
      <div className="section-head">
        <h2 className="sec-title">Cases <i>|</i> <small>고객 사례</small></h2>
        <p className="section-sub">개인정보를 다루는 곳이라면 어디든 — 도메인별 활용 모습입니다.</p>
      </div>

      <div className="domain-grid">
        {DOMAINS.map((c) => (
          <article className="tip-card" key={c.t}>
            <span className="rtag">{c.k}</span>
            <h3>{c.t}</h3>
            <p>{c.d}</p>
          </article>
        ))}
      </div>

      <h3 className="stats-h">사용 후기</h3>
      <div className="review-grid">
        {REVIEWS.map((r) => (
          <figure className="review-card" key={r.name}>
            <blockquote>“{r.body}”</blockquote>
            <figcaption>
              <strong>{r.name}</strong>
              <span className="rtag rtag-auto">{r.tag}</span>
            </figcaption>
          </figure>
        ))}
      </div>

      <h3 className="stats-h">커뮤니티 인기글</h3>
      <ul className="notice-list">
        {HOT_POSTS.map((p) => (
          <li key={p.title}>
            <button className="notice-row" onClick={() => navigate('/community')}>
              <span className="rtag">{p.name}</span>
              <strong>{p.title}</strong>
              <time>공감 {p.likes}</time>
            </button>
          </li>
        ))}
      </ul>
      <p className="research-note">후기와 사례는 데모 구성용 예시입니다. 커뮤니티에서 실제 사용담을 나눠주세요.</p>
    </main>
  )
}
