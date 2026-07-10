import { useEffect, useState } from 'react'
import { getJSON } from '../lib/api.js'

const ACTION_LABEL = { inpaint: '스튜디오 지우기', redact: '자동 비식별 처리', restore: '얼굴 복원' }

const RESEARCH = [
  { label: 'PSNR', v: '29.01', d: '복원 품질 (개선 전 28.28)' },
  { label: 'SSIM', v: '0.951', d: '구조 유사도 (개선 전 0.914)' },
  { label: 'FID', v: '2.8', d: '자연스러움 (개선 전 3.1)' },
]

// 서비스 처리 현황 + 연구 성능 지표
export default function Stats() {
  const [live, setLive] = useState(null)
  const [down, setDown] = useState(false)

  const load = () => {
    getJSON('/api/stats').then(setLive).catch(() => setDown(true))
  }
  useEffect(load, [])

  return (
    <main className="studio page-narrow">
      <div className="section-head">
        <h2 className="sec-title">Stats <i>|</i> <small>서비스 통계</small></h2>
        <p className="section-sub">이번 서버 세션의 실시간 처리 현황과 모델 성능 지표입니다.</p>
      </div>

      <h3 className="stats-h">실시간 처리 현황</h3>
      {down && <p className="admin-empty">서버에 연결할 수 없어 실시간 현황을 불러오지 못했어요.</p>}
      {live && (
        <div className="admin-stats">
          <div className="astat"><span>총 처리</span><strong>{live.total}</strong><em>건</em></div>
          {Object.entries(live.byAction ?? {}).map(([k, v]) => (
            <div className="astat" key={k}><span>{ACTION_LABEL[k] ?? k}</span><strong>{v}</strong><em>건</em></div>
          ))}
          <div className="astat"><span>평균 처리 시간</span><strong>{live.avgElapsedMs ? (live.avgElapsedMs / 1000).toFixed(1) : '—'}</strong><em>초</em></div>
        </div>
      )}
      {live && <button className="btn btn-outline btn-sm stats-refresh" onClick={load}>새로고침</button>}

      <h3 className="stats-h">모델 성능 (졸업연구 개선안)</h3>
      <div className="admin-stats">
        {RESEARCH.map((r) => (
          <div className="astat" key={r.label}>
            <span>{r.label}</span><strong>{r.v}</strong>
            <p className="astat-sub">{r.d}</p>
          </div>
        ))}
      </div>
      <p className="research-note">
        「DeepFillv2 인페인팅 구조 변형에 따른 성능 개선과 실증 구현」 · CelebA 32장 배치 평가 ·
        실시간 현황은 서버 재시작 시 초기화됩니다.
      </p>
    </main>
  )
}
