import { useEffect, useRef, useState } from 'react'
import BeforeAfter from '../components/BeforeAfter.jsx'
import { Icon } from '../components/icons.jsx'
import { downloadBlob } from '../lib/image.js'
import { postForm } from '../lib/api.js'

const MASK_MARGIN = 0.25 // 탐지 박스 대비 마스크 타원 여유
const TYPE_LABEL = { face: '얼굴', plate: '번호판' }

// 항목별 전용 페이지 구성 — kind에 따라 탐지 결과를 필터하고 문구를 바꾼다
const KINDS = {
  face: {
    label: '얼굴',
    title: '얼굴 지우기',
    desc: '사진 속 모든 얼굴을 자동으로 찾아 지우고, 그 자리는 원래 배경으로 자연스럽게 메웁니다.',
    dropSub: '단체 사진 · 현장 사진 · CCTV 캡처 — 얼굴을 자동으로 찾아 표시합니다',
    sample: '/sample-face.png',
    beta: false,
  },
  plate: {
    label: '번호판',
    title: '차량 번호판 지우기',
    desc: '차량 번호판을 자동으로 찾아 지우고, 그 자리는 원래 배경으로 자연스럽게 메웁니다.',
    dropSub: '주차장 · 사고 · 거리 사진 — 차량 번호판을 자동으로 찾아 표시합니다',
    sample: null,
    beta: false,
  },
  text: {
    label: '텍스트 개인정보',
    title: '텍스트 개인정보 지우기',
    desc: '전화번호, 주민등록번호, 계좌·카드번호 같은 숫자 개인정보를 글자째 찾아 지웁니다.',
    dropSub: '전단지 · 서류 · 차량 광고 — 전화번호·계좌번호 등을 자동으로 찾아 표시합니다',
    sample: '/sample-text.png',
    beta: true,
  },
}

// 비식별화 워크플로: 업로드 → 자동 탐지 → 항목 선택 → 제거·자연 복원 → 저장
export default function Privacy({ engine, kind = 'face' }) {
  const meta = KINDS[kind]
  const [imageFile, setImageFile] = useState(null)
  const [imageURL, setImageURL] = useState(null)
  const [natural, setNatural] = useState(null)
  const [dets, setDets] = useState(null) // [{type, box, score, on}]
  const [phase, setPhase] = useState('idle') // idle | detecting | ready | removing | done
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [metrics, setMetrics] = useState({}) // {detectMs, restoreMs}

  const imgElRef = useRef(null)

  const reset = () => { setDets(null); setResult(null); setError(null); setPhase('idle'); setMetrics({}) }

  // 샘플 체험 (#/privacy/face?demo=1) — 번들된 예시 사진으로 흐름을 바로 보여준다
  const loadSample = async () => {
    if (!meta.sample) return
    try {
      const blob = await (await fetch(meta.sample)).blob()
      loadFile(new File([blob], 'sample.png', { type: 'image/png' }))
    } catch { setError('샘플 이미지를 불러오지 못했어요.') }
  }
  useEffect(() => {
    if (window.location.hash.includes('demo=1')) loadSample()
  }, []) // eslint-disable-line

  const loadFile = async (file) => {
    if (!file || !file.type?.startsWith('image/')) return
    reset()
    setImageFile(file)
    const url = URL.createObjectURL(file)
    setImageURL(url)
    const img = new Image()
    img.onload = () => {
      setNatural({ w: img.naturalWidth, h: img.naturalHeight })
    }
    img.src = url
    runDetect(file)
  }

  const runDetect = async (file) => {
    setPhase('detecting'); setError(null)
    const started = performance.now()
    try {
      const form = new FormData()
      form.append('image', file, 'image.png')
      form.append('targets', kind)
      const res = await postForm('/api/detect', form)
      const data = await res.json()
      const ms = performance.now() - started
      setMetrics((m) => ({ ...m, detectMs: ms }))
      const matched = data.detections.filter((d) => d.type === kind)
      setDets(matched.map((d) => ({ ...d, on: true })))
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
      for (const { type, box: [x, y, w, h] } of selected) {
        if (type === 'text') {
          // 글자는 각지게 — 타원은 획 끝을 남긴다
          const mx = (w * MASK_MARGIN) / 2, my = (h * MASK_MARGIN) / 2
          ctx.fillRect(x - mx, y - my, w + mx * 2, h + my * 2)
        } else {
          ctx.beginPath()
          ctx.ellipse(x + w / 2, y + h / 2, (w / 2) * (1 + MASK_MARGIN), (h / 2) * (1 + MASK_MARGIN), 0, 0, Math.PI * 2)
          ctx.fill()
        }
      }
      const maskBlob = await new Promise((r) => mc.toBlob(r, 'image/png'))
      const form = new FormData()
      form.append('image', imageFile, 'image.png')
      form.append('mask', maskBlob, 'mask.png')
      const res = await postForm('/api/inpaint', form)
      const blob = await res.blob()
      const elapsed = Number(res.headers.get('X-Elapsed-Ms')) || null
      const engineName = res.headers.get('X-Engine')
      setMetrics((m) => ({ ...m, restoreMs: elapsed, engineName }))
      setResult({ url: URL.createObjectURL(blob), blob, elapsed })
      setPhase('done')
    } catch (e) {
      setError(e.message ?? '제거에 실패했어요.')
      setPhase('ready')
    }
  }

  // 처리 진행을 3단계로 요약 — 로그 나열 대신 지금 어디까지 왔는지 한눈에 보여준다
  const detectError = Boolean(error) && (dets?.length ?? 0) === 0
  const removeError = Boolean(error) && (dets?.length ?? 0) > 0
  const steps = [
    { key: 'up', label: '사진 업로드', hint: natural ? `${natural.w}×${natural.h}` : '준비 중', state: 'done' },
    {
      key: 'find', label: `${meta.label} 자동 찾기`,
      hint: phase === 'detecting' ? '찾는 중…' : detectError ? '실패' : dets ? `${dets.length}건 찾음` : '대기',
      state: phase === 'detecting' ? 'active' : detectError ? 'error' : dets != null ? 'done' : 'wait',
    },
    {
      key: 'erase', label: '지우고 배경 복원',
      hint: phase === 'removing' ? '지우는 중…' : phase === 'done' && result ? `${selected.length}건 완료` : removeError ? '실패' : '대기',
      state: phase === 'removing' ? 'active' : phase === 'done' && result ? 'done' : removeError ? 'error' : 'wait',
    },
  ]

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
            <h2>{meta.title} {meta.beta && <span className="rtag">베타</span>}</h2>
            <p>{meta.desc}</p>
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
              <p className="dropzone-title">사진을 올리면 바로 찾아드려요</p>
              <p className="dropzone-sub">{meta.dropSub}</p>
              {meta.sample && (
                <span className="btn btn-outline btn-sm" onClick={(e) => { e.preventDefault(); loadSample() }}>샘플 사진으로 체험</span>
              )}
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
                    <span>지우고 배경을 메우는 중…</span>
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
                afterLabel="완료"
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
                <span className="tool-label tool-label-done"><Icon.check /> 다 지웠어요</span>
                <p className="result-desc">{selected.length}개 항목을 지우고 그 자리를 배경으로 복원했어요.</p>
                {result.elapsed && <p className="result-meta">처리 {(result.elapsed / 1000).toFixed(1)}초</p>}
              </div>
              <button className="btn btn-primary btn-block" onClick={download}><Icon.download /> 이미지 저장</button>
              <button className="btn btn-ghost btn-sm btn-block" onClick={() => { setResult(null); setPhase('ready') }}>다시 선택</button>
              <label className="btn btn-ghost btn-sm btn-block">새 사진<input type="file" accept="image/*" hidden onChange={(e) => newImage(e.target.files[0])} /></label>
            </>
          ) : (
            <>
              <div className="tool-group">
                <span className="tool-label">찾아낸 {meta.label} {dets ? `· ${dets.length}건` : ''}</span>
                {!dets && <p className="result-desc">사진을 올리면 자동으로 찾아드립니다.</p>}
                {dets && !dets.length && (
                  <p className="result-desc">
                    자동으로 찾은 {meta.label}이(가) 없어요. 다른 개인정보는
                    <a href="#/privacy"> 항목 선택</a>이나 <a href="#/studio">스튜디오</a>에서 지울 수 있어요.
                  </p>
                )}
                {dets && dets.length > 0 && (
                  <ul className="pv-list">
                    {dets.map((d, i) => (
                      <li key={i}>
                        <label>
                          <input type="checkbox" checked={d.on} onChange={() => toggle(i)} />
                          <span>{d.label ?? TYPE_LABEL[d.type] ?? d.type} {i + 1}</span>
                          <span className="pv-score" title={`신뢰도 ${Math.round(d.score * 100)}%`}>
                            <i style={{ width: `${Math.round(d.score * 100)}%` }} />
                          </span>
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
                {phase === 'removing' ? '지우는 중…' : `${selected.length || 0}개 지우기`}
              </button>
              {imageURL && <label className="btn btn-ghost btn-sm btn-block">다른 사진<input type="file" accept="image/*" hidden onChange={(e) => newImage(e.target.files[0])} /></label>}
            </>
          )}
        </aside>
      </div>

      {/* 처리 현황 — 진행 단계 · 요약 · 문제 안내를 사람이 읽기 좋게 */}
      {imageURL && (
        <div className="pv-status">
          <ol className="pv-steps">
            {steps.map((s, i) => (
              <li key={s.key} className={`pv-step pv-step-${s.state}`}>
                <span className="pv-step-dot">
                  {s.state === 'done' ? <Icon.check />
                    : s.state === 'active' ? <span className="pv-spin" />
                    : s.state === 'error' ? '!' : i + 1}
                </span>
                <span className="pv-step-body">
                  <b>{s.label}</b>
                  <small>{s.hint}</small>
                </span>
              </li>
            ))}
          </ol>

          {detectError || removeError ? (
            <div className="pv-note">
              <div>
                <b>{detectError ? '자동으로 찾지 못했어요' : '지우는 중 문제가 생겼어요'}</b>
                <p>
                  {detectError
                    ? '사진이 어둡거나 대상이 너무 작으면 못 찾을 수 있어요. 다른 사진으로 다시 시도하거나, 스튜디오에서 직접 칠해 지울 수 있어요.'
                    : '일시적인 문제일 수 있어요. 잠시 후 다시 시도해 주세요.'}
                </p>
              </div>
              <button
                className="btn btn-outline btn-sm"
                onClick={() => imageFile && (detectError ? runDetect(imageFile) : runRemove())}
              >
                다시 시도
              </button>
            </div>
          ) : (
            <div className="pv-facts">
              <span className="pv-fact"><b>{engine ?? '—'}</b>추론 엔진</span>
              <span className="pv-fact"><b>{metrics.detectMs ? `${(metrics.detectMs / 1000).toFixed(1)}초` : '—'}</b>찾는 데 걸린 시간</span>
              <span className="pv-fact"><b>{metrics.restoreMs ? `${(metrics.restoreMs / 1000).toFixed(1)}초` : '—'}</b>지우고 복원한 시간</span>
              {metrics.engineName && phase === 'done' && <span className="pv-fact"><b>{metrics.engineName}</b>복원 엔진</span>}
            </div>
          )}
        </div>
      )}
    </main>
  )
}
