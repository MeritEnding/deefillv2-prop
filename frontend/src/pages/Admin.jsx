import { useState } from 'react'
import { Icon } from '../components/icons.jsx'
import { getJSON } from '../lib/api.js'

const ACTION_LABEL = { inpaint: '수동 인페인팅', redact: 'API 비식별화', restore: '얼굴 복원', upscale: '고화질 확대' }

// 운영자 콘솔: X-Admin-Key로 통계·감사 로그 조회
export default function Admin() {
  const [key, setKey] = useState(sessionStorage.getItem('adminKey') ?? '')
  const [stats, setStats] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const load = async (e) => {
    e?.preventDefault()
    if (!key || busy) return
    setBusy(true); setError(null)
    try {
      const data = await getJSON('/api/admin/stats', { headers: { 'X-Admin-Key': key } })
      sessionStorage.setItem('adminKey', key)
      setStats(data)
    } catch (err) { setError(err.message); setStats(null) } finally { setBusy(false) }
  }

  const logout = () => { sessionStorage.removeItem('adminKey'); setKey(''); setStats(null) }

  return (
    <main className="studio admin">
      <div className="studio-head">
        <div className="studio-title">
          <div className="tile tile-sm"><Icon.bolt /></div>
          <div>
            <h2>관리자 콘솔</h2>
            <p>처리 통계, API 키 사용량, 감사 로그를 확인합니다.</p>
          </div>
        </div>
        {stats && (
          <div className="btn-row">
            <button className="btn btn-outline btn-sm" onClick={() => load()} disabled={busy}>{busy ? '새로고침…' : '새로고침'}</button>
            <button className="btn btn-ghost btn-sm" onClick={logout}>잠금</button>
          </div>
        )}
      </div>

      {!stats ? (
        <form className="admin-gate" onSubmit={load}>
          <img src="/mascot-think.png" alt="" />
          <p>운영자 키를 입력하세요.</p>
          <div className="admin-gate-row">
            <input
              type="password"
              value={key}
              placeholder="X-Admin-Key"
              onChange={(e) => setKey(e.target.value)}
              autoFocus
            />
            <button className="btn btn-primary" disabled={!key || busy}>{busy ? '확인 중…' : '열기'}</button>
          </div>
          {error && <p className="error">{error}</p>}
        </form>
      ) : (
        <>
          <div className="admin-stats">
            <div className="astat">
              <span>총 처리</span>
              <strong>{stats.total}</strong>
              <em>건</em>
            </div>
            {Object.entries(stats.byAction ?? {}).map(([k, v]) => (
              <div className="astat" key={k}>
                <span>{ACTION_LABEL[k] ?? k}</span>
                <strong>{v}</strong>
                <em>건</em>
              </div>
            ))}
          </div>

          <div className="admin-grid">
            <section className="admin-card">
              <h3>API 키별 사용량</h3>
              {Object.keys(stats.byApiKey ?? {}).length === 0 ? (
                <p className="admin-empty">아직 API 호출이 없습니다.</p>
              ) : (
                <table>
                  <thead><tr><th>키</th><th>비식별화 건수</th></tr></thead>
                  <tbody>
                    {Object.entries(stats.byApiKey).map(([k, v]) => (
                      <tr key={k}><td><code>{k}</code></td><td>{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>

            <section className="admin-card">
              <h3>최근 처리 이력 <small>(최근 {stats.recent?.length ?? 0}건)</small></h3>
              {!stats.recent?.length ? (
                <p className="admin-empty">아직 처리 이력이 없습니다.</p>
              ) : (
                <table>
                  <thead><tr><th>시각</th><th>주체</th><th>작업</th><th>탐지</th><th>처리</th></tr></thead>
                  <tbody>
                    {stats.recent.map((r, i) => (
                      <tr key={i}>
                        <td>{new Date(r.at).toLocaleTimeString('ko-KR')}</td>
                        <td><code>{r.source}</code></td>
                        <td>{ACTION_LABEL[r.action] ?? r.action}</td>
                        <td>{r.action === 'redact' ? `${r.detected}건` : '—'}</td>
                        <td>{(r.elapsedMs / 1000).toFixed(1)}초</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>
          </div>
        </>
      )}
    </main>
  )
}
