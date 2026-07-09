import { useRef, useState } from 'react'
import BeforeAfter from '../components/BeforeAfter.jsx'
import { Icon } from '../components/icons.jsx'
import { downloadBlob } from '../lib/image.js'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8080'
const MASK_MARGIN = 0.25 // 탐지 박스 대비 마스크 타원 여유

// 비식별화 워크플로: 업로드 → 자동 탐지 → 항목 선택 → 제거·자연 복원 → 저장
export default function Privacy({ engine }) {
  const [imageFile, setImageFile] = useState(null)
  const [imageURL, setImageURL] = useState(null)
  const [natural, setNatural] = useState(null)
  const [dets, setDets] = useState(null) // [{type, box, score, on}]
  const [phase, setPhase] = useState('idle') // idle | detecting | ready | removing | done
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)

  const imgElRef = useRef(null)

  const reset = () => { setDets(null); setResult(null); setError(null); setPhase('idle') }

  const loadFile = async (file) => {
    if (!file || !file.type?.startsWith('image/')) return
    reset()
    setImageFile(file)
    const url = URL.createObjectURL(file)
    setImageURL(url)
    const img = new Image()
    img.onload = () => setNatural({ w: img.naturalWidth, h: img.naturalHeight })
    img.src = url
    runDetect(file)
  }

  const runDetect = async (file) => {
    setPhase('detecting'); setError(null)
    try {
      const form = new FormData()
      form.append('image', file, 'image.png')
      const res = await fetch(`${API_BASE}/api/detect`, { method: 'POST', body: form })
      if (!res.ok) {
        const msg = await res.json().then((d) => d.message).catch(() => null)
        throw new Error(msg ?? `서버 오류 (${res.status})`)
      }
      const data = await res.json()
      setDets(data.detections.map((d) => ({ ...d, on: true })))
      setPhase('ready')
    } catch (e) {
      setError(e.message ?? '자동 탐지에 실패했어요.')
      setDets([])
      setPhase('ready')
    }
  }

  const toggle = (i) => setDets((ds) => ds.map((d, j) => (j === i ? { ...d, on: !d.on } : d)))
  const setAll = (on) => setDets((ds) => ds.map((d) => ({ ...d, on })))
  const selected = (dets || []).filter((d) => d.on)

  const runRemove = async () => {
    if (!imageFile || !natural || !selected.length || phase === 'removing') return
    setPhase('removing'); setError(null)
    try {
      // 선택된 박스들을 타원 마스크로 렌더링
      const mc = document.createElement('canvas')
      mc.width = natural.w; mc.height = natural.h
      const ctx = mc.getContext('2d')
      ctx.fillStyle = 'black'
      ctx.fillRect(0, 0, mc.width, mc.height)
      ctx.fillStyle = 'white'
      for (const { box: [x, y, w, h] } of selected) {
        ctx.beginPath()
        ctx.ellipse(x + w / 2, y + h / 2, (w / 2) * (1 + MASK_MARGIN), (h / 2) * (1 + MASK_MARGIN), 0, 0, Math.PI * 2)
        ctx.fill()
      }
      const maskBlob = await new Promise((r) => mc.toBlob(r, 'image/png'))
      const form = new FormData()
      form.append('image', imageFile, 'image.png')
      form.append('mask', maskBlob, 'mask.png')
      const res = await fetch(`${API_BASE}/api/inpaint`, { method: 'POST', body: form })
      if (!res.ok) {
        const msg = await res.json().then((d) => d.message).catch(() => null)
        throw new Error(msg ?? `서버 오류 (${res.status})`)
      }
      const blob = await res.blob()
      const elapsed = Number(res.headers.get('X-Elapsed-Ms')) || null
      setResult({ url: URL.createObjectURL(blob), blob, elapsed })
      setPhase('done')
    } catch (e) {
      setError(e.message ?? '제거에 실패했어요.')
      setPhase('ready')
    }
  }

  const download = () => result && downloadBlob(result.blob, `jium-redacted-${Date.now()}.png`)
  const newImage = (file) => {
    if (!file || !file.type?.startsWith('image/')) return
    setImageURL(null); setNatural(null); loadFile(file)
  }

  return (
    <main className="studio privacy">
      <div className="studio-head">
        <div className="studio-title">
          <div className="tile tile-sm"><Icon.shield /></div>
          <div>
            <h2>개인정보 비식별화</h2>
            <p>얼굴을 자동으로 찾아 지우고, 모자이크 대신 배경을 자연스럽게 복원합니다.</p>
          </div>
        </div>
        <span className={`pill ${engine ? 'pill-ok' : 'pill-down'}`}>
          {engine ? `엔진: ${engine}` : '추론 서버 연결 안 됨'}
        </span>
      </div>

      <div className="studio-grid">
        <div
          className={`stage-wrap ${dragOver ? 'drag' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); newImage(e.dataTransfer.files[0]) }}
        >
          {!imageURL && (
            <label className="dropzone">
              <input type="file" accept="image/*" hidden onChange={(e) => newImage(e.target.files[0])} />
              <img className="dropzone-mascot" src="/mascot-select.png" alt="" />
              <p className="dropzone-title">비식별화할 사진을 올려주세요</p>
              <p className="dropzone-sub">현장 사진 · CCTV 캡처 · 사고 사진 — 업로드 즉시 자동 분석됩니다</p>
            </label>
          )}

          {imageURL && phase !== 'done' && (
            <div className="stage stage-result">
              <div className="pv-frame" style={natural ? { aspectRatio: `${natural.w} / ${natural.h}`, maxWidth: `${Math.round(560 * (natural.w / natural.h))}px` } : undefined}>
                <img ref={imgElRef} src={imageURL} alt="원본" />
                {natural && (dets || []).map((d, i) => {
                  const [x, y, w, h] = d.box
                  return (
                    <button
                      key={i}
                      className={`pv-box ${d.on ? 'on' : ''}`}
                      style={{
                        left: `${(x / natural.w) * 100}%`,
                        top: `${(y / natural.h) * 100}%`,
                        width: `${(w / natural.w) * 100}%`,
                        height: `${(h / natural.h) * 100}%`,
                      }}
                      onClick={() => toggle(i)}
                      title={d.on ? '클릭하면 제외' : '클릭하면 포함'}
                    >
                      <em>{d.on ? '제거' : '유지'}</em>
                    </button>
                  )
                })}
                {phase === 'detecting' && (
                  <div className="stage-busy">
                    <div className="scanline" />
                    <img src="/mascot-select.png" alt="" />
                    <span>개인정보를 찾는 중…</span>
                  </div>
                )}
                {phase === 'removing' && (
                  <div className="stage-busy">
                    <div className="scanline" />
                    <img src="/mascot-erase.png" alt="" />
                    <span>제거하고 배경을 복원하는 중…</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {phase === 'done' && result && (
            <div className="stage stage-result">
              <BeforeAfter
                before={imageURL}
                after={result.url}
                beforeLabel="원본"
                afterLabel="비식별 완료"
                initial={15}
                style={natural ? { aspectRatio: `${natural.w} / ${natural.h}`, maxWidth: `${Math.round(560 * (natural.w / natural.h))}px` } : undefined}
              />
            </div>
          )}
        </div>

        <aside className="toolbar">
          {phase === 'done' && result ? (
            <>
              <div className="tool-group">
                <img className="result-mascot" src="/mascot-perfect.png" alt="" />
                <span className="tool-label tool-label-done"><Icon.check /> 비식별 완료</span>
                <p className="result-desc">{selected.length}개 항목이 제거되고 배경이 복원됐습니다.</p>
                {result.elapsed && <p className="result-meta">처리 {(result.elapsed / 1000).toFixed(1)}초</p>}
              </div>
              <button className="btn btn-primary btn-block" onClick={download}><Icon.download /> 이미지 저장</button>
              <button className="btn btn-ghost btn-sm btn-block" onClick={() => { setResult(null); setPhase('ready') }}>다시 선택</button>
              <label className="btn btn-ghost btn-sm btn-block">새 사진<input type="file" accept="image/*" hidden onChange={(e) => newImage(e.target.files[0])} /></label>
            </>
          ) : (
            <>
              <div className="tool-group">
                <span className="tool-label">감지된 개인정보 {dets ? `· ${dets.length}건` : ''}</span>
                {!dets && <p className="result-desc">사진을 올리면 자동으로 분석합니다.</p>}
                {dets && !dets.length && <p className="result-desc">감지된 항목이 없습니다. 수동 편집은 스튜디오를 이용하세요.</p>}
                {dets && dets.length > 0 && (
                  <ul className="pv-list">
                    {dets.map((d, i) => (
                      <li key={i}>
                        <label>
                          <input type="checkbox" checked={d.on} onChange={() => toggle(i)} />
                          <span>얼굴 {i + 1}</span>
                          <em>{Math.round(d.score * 100)}%</em>
                        </label>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              {dets && dets.length > 1 && (
                <div className="btn-row">
                  <button className="btn btn-outline btn-sm" onClick={() => setAll(true)}>전체 선택</button>
                  <button className="btn btn-outline btn-sm" onClick={() => setAll(false)}>전체 해제</button>
                </div>
              )}
              {error && <p className="error">{error}</p>}
              <button
                className="btn btn-primary btn-block btn-run"
                onClick={runRemove}
                disabled={!selected.length || phase === 'detecting' || phase === 'removing'}
              >
                {phase === 'removing' ? '복원 중…' : `${selected.length || 0}개 제거·복원`}
              </button>
              {imageURL && <label className="btn btn-ghost btn-sm btn-block">다른 사진<input type="file" accept="image/*" hidden onChange={(e) => newImage(e.target.files[0])} /></label>}
            </>
          )}
        </aside>
      </div>
    </main>
  )
}
