import { useState } from 'react'

// 공지사항 — 릴리스·운영 소식
const NOTICES = [
  {
    date: '2026-07-09', tag: '업데이트', title: '개인정보 지우기 화면에 관제 콘솔이 생겼어요',
    body: '업로드부터 복원까지 각 단계의 처리 시간과 탐지 결과가 화면 아래 콘솔에 실시간으로 기록됩니다. 탐지 목록에는 항목별 신뢰도 게이지가 함께 표시됩니다.',
  },
  {
    date: '2026-07-09', tag: '신기능', title: '차량 번호판 자동 인식을 시작했습니다',
    body: '얼굴에 이어 차량 번호판도 업로드 즉시 자동으로 찾아냅니다. 지우고 싶은 항목만 체크하면 그 자리는 원래 배경으로 복원됩니다. 번호판 인식은 베타 단계로, 정확도를 계속 높여가고 있어요.',
  },
  {
    date: '2026-07-09', tag: '신기능', title: '일괄 지우기 — 여러 장을 한 번에',
    body: '같은 위치에 워터마크나 로고가 박힌 사진 여러 장을 올리고 첫 장에만 칠하면, 나머지 전부에 같은 마스크가 적용되어 순차 처리됩니다. 쇼핑몰 상품컷 정리에 특히 유용합니다.',
  },
  {
    date: '2026-07-08', tag: '업데이트', title: '스마트 클릭 — 칠하지 말고 클릭 한 번',
    body: '스튜디오의 객체 지우개에 스마트 클릭 도구가 추가됐습니다. 지우고 싶은 대상을 클릭하면 AI가 영역을 자동으로 잡아줍니다. 결과가 마음에 안 들면 되돌리기로 즉시 취소할 수 있어요.',
  },
  {
    date: '2026-07-08', tag: '엔진', title: '복원 품질이 크게 좋아졌습니다 (LaMa 엔진 도입)',
    body: '사전학습 LaMa 인페인팅 엔진을 도입해 지운 자리가 훨씬 자연스럽게 복원됩니다. 졸업연구에서 개선한 DeepFillv2 실험 결과는 랜딩의 복원 갤러리에서 볼 수 있습니다.',
  },
]

export default function Notices() {
  const [open, setOpen] = useState(0)

  return (
    <main className="studio page-narrow">
      <div className="section-head">
        <h2 className="sec-title">Notice <i>|</i> <small>공지사항</small></h2>
        <p className="section-sub">지움의 새 기능과 운영 소식을 전해드립니다.</p>
      </div>
      <ul className="notice-list">
        {NOTICES.map((n, i) => (
          <li key={n.title} className={open === i ? 'open' : ''}>
            <button className="notice-row" onClick={() => setOpen(open === i ? -1 : i)}>
              <span className="rtag">{n.tag}</span>
              <strong>{n.title}</strong>
              <time>{n.date}</time>
            </button>
            {open === i && <p className="notice-body">{n.body}</p>}
          </li>
        ))}
      </ul>
    </main>
  )
}
