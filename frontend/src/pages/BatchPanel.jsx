import { useRef, useState } from 'react'
import { Icon } from '../components/icons.jsx'
import { downloadBlob } from '../lib/image.js'
import { postForm } from '../lib/api.js'

const MAX_DISPLAY = 640

// 일괄 지우기: 첫 사진에 그린 마스크를 모든 사진에 적용해 순차 인페인팅.
// 같은 위치에 워터마크가 박힌 상품컷 정리 용도라 브러시만 제공한다.
export default function BatchPanel({ brushDefault = 26 }) {
  const [files, setFiles] = useState([])
  const [strokes, setStrokes] = useState([])
  const [brushSize, setBrushSize] = useState(brushDefault)
  const [progress, setProgress] = useState(null) // {done, total}
  const [results, setResults] = useState(null) // [{name, url, blob, error}]
  const [error, setError] = useState(null)

  const dispRef = useRef(null)
  const maskRef = useRef(null)
  const imgRef = useRef(null)
  const drawing = useRef(false)
  const cur = useRef(null)

  const loadFiles = (list) => {
    const imgs = [...list].filter((f) => f.type?.startsWith('image/'))
    if (!imgs.length) return
    setFiles(imgs); setStrokes([]); setResults(null); setError(null); setProgress(null)
    const url = URL.createObjectURL(imgs[0])
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      const disp = dispRef.current
      const s = Math.min(1, MAX_DISPLAY / Math.max(img.naturalWidth, img.naturalHeight))
      disp.width = Math.round(img.naturalWidth * s)
      disp.height = Math.round(img.naturalHeight * s)
      const mask = maskRef.current
      mask.width = img.naturalWidth
      mask.height = img.naturalHeight
      render([])
    }
    img.src = url
  }

  const render = (list) => {
    const disp = dispRef.current
    const mask = maskRef.current
    const img = imgRef.current
    if (!disp || !mask || !img) return
    const dctx = disp.getContext('2d')
    dctx.clearRect(0, 0, disp.width, disp.height)
    dctx.drawImage(img, 0, 0, disp.width, disp.height)
    const mctx = mask.getContext('2d')
    mctx.fillStyle = 'black'
    mctx.fillRect(0, 0, mask.width, mask.height)
    const scale = mask.width / disp.width
    dctx.strokeStyle = dctx.fillStyle = 'rgba(37,99,235,0.5)'
    mctx.strokeStyle = mctx.fillStyle = 'white'
    for (const s of list) {
      for (const [ctx, k] of [[dctx, 1], [mctx, scale]]) {
        ctx.lineWidth = s.size * k
        ctx.lineCap = ctx.lineJoin = 'round'
        ctx.beginPath()
        ctx.moveTo(s.points[0].x * k, s.points[0].y * k)
        for (const p of s.points) ctx.lineTo(p.x * k, p.y * k)
        ctx.stroke()
      }
    }
  }

  const pointFrom = (e) => {
    const r = dispRef.current.getBoundingClientRect()
    return {
      x: (e.clientX - r.left) * (dispRef.current.width / r.width),
      y: (e.clientY - r.top) * (dispRef.current.height / r.height),
    }
  }
  const onDown = (e) => {
    if (!imgRef.current || progress) return
    dispRef.current.setPointerCapture(e.pointerId)
    drawing.current = true
    cur.current = { size: brushSize, points: [pointFrom(e)] }
    render([...strokes, cur.current])
  }
  const onMove = (e) => {
    if (!drawing.current || !cur.current) return
    cur.current.points.push(pointFrom(e))
    render([...strokes, cur.current])
  }
  const onUp = () => {
    if (!drawing.current) return
    drawing.current = false
    if (cur.current?.points.length) setStrokes((s) => [...s, cur.current])
    cur.current = null
  }

  const runAll = async () => {
    if (!files.length || !strokes.length || progress) return
    setError(null)
    setProgress({ done: 0, total: files.length })
    const maskBlob = await new Promise((r) => maskRef.current.toBlob(r, 'image/png'))
    const out = []
    for (let i = 0; i < files.length; i++) {
      try {
        const form = new FormData()
        form.append('image', files[i], files[i].name)
        form.append('mask', maskBlob, 'mask.png') // 크기가 달라도 서버가 리사이즈해 적용
        const res = await postForm('/api/inpaint', form)
        const blob = await res.blob()
        out.push({ name: files[i].name, blob, url: URL.createObjectURL(blob) })
      } catch (e) {
        out.push({ name: files[i].name, error: e.message ?? '실패' })
      }
      setProgress({ done: i + 1, total: files.length })
    }
    setResults(out)
    setProgress(null)
  }

  const saveAll = async () => {
    for (const r of results.filter((r) => r.blob)) {
      downloadBlob(r.blob, `jium-${r.name.replace(/\.[^.]+$/, '')}.png`)
      await new Promise((r2) => setTimeout(r2, 350)) // 브라우저 다운로드 차단 방지
    }
  }

  const reset = () => { setFiles([]); setStrokes([]); setResults(null); setProgress(null); setError(null) }

  return (
    <div className="studio-grid">
      <div className="stage-wrap">
        {!files.length && (
          <label className="dropzone">
            <input type="file" accept="image/*" multiple hidden onChange={(e) => loadFiles(e.target.files)} />
            <img className="dropzone-mascot" src="/mascot-select.png" alt="" />
            <p className="dropzone-title">여러 장을 한 번에 선택하세요</p>
            <p className="dropzone-sub">같은 위치에 워터마크가 있는 사진들 · JPG · PNG</p>
          </label>
        )}

        {files.length > 0 && !results && (
          <div className="batch-stage">
            <div className="stage">
              <canvas
                ref={dispRef}
                className="stage-canvas batch-canvas"
                onPointerDown={onDown}
                onPointerMove={onMove}
                onPointerUp={onUp}
                onPointerLeave={onUp}
              />
              {!strokes.length && !progress && (
                <div className="stage-hint">첫 사진에서 지울 부분을 칠하세요 — 전체에 똑같이 적용됩니다</div>
              )}
              {progress && (
                <div className="stage-busy">
                  <div className="scanline" />
                  <img src="/mascot-erase.png" alt="" />
                  <span>{progress.done} / {progress.total}장 처리 중…</span>
                </div>
              )}
            </div>
            <div className="batch-thumbs">
              {files.map((f, i) => (
                <span key={f.name + i} className={i === 0 ? 'on' : ''}>
                  <img src={URL.createObjectURL(f)} alt={f.name} />
                  {i === 0 && <em>마스크 기준</em>}
                </span>
              ))}
            </div>
          </div>
        )}

        {results && (
          <div className="batch-results">
            {results.map((r) => (
              <figure key={r.name}>
                {r.url ? <img src={r.url} alt={r.name} /> : <div className="batch-fail">실패</div>}
                <figcaption>
                  <span title={r.name}>{r.name}</span>
                  {r.blob && (
                    <button onClick={() => downloadBlob(r.blob, `jium-${r.name.replace(/\.[^.]+$/, '')}.png`)}>
                      <Icon.download width="15" height="15" />
                    </button>
                  )}
                </figcaption>
              </figure>
            ))}
          </div>
        )}

        <canvas ref={maskRef} style={{ display: 'none' }} />
      </div>

      <aside className="toolbar">
        {!results ? (
          <>
            <div className="tool-group">
              <span className="tool-label">브러시 크기 · {brushSize}px</span>
              <input type="range" min="6" max="90" value={brushSize} onChange={(e) => setBrushSize(+e.target.value)} />
            </div>
            <div className="tool-group">
              <span className="tool-label">편집</span>
              <button className="btn btn-ghost btn-sm btn-block" onClick={() => setStrokes([])} disabled={!strokes.length}>마스크 다시 그리기</button>
            </div>
            {error && <p className="error">{error}</p>}
            <button className="btn btn-primary btn-block btn-run" onClick={runAll} disabled={!files.length || !strokes.length || !!progress}>
              {progress ? `${progress.done}/${progress.total} 처리 중…` : files.length ? `${files.length}장 일괄 지우기` : '일괄 지우기'}
            </button>
            <label className="btn btn-ghost btn-sm btn-block">
              사진 다시 선택
              <input type="file" accept="image/*" multiple hidden onChange={(e) => loadFiles(e.target.files)} />
            </label>
          </>
        ) : (
          <>
            <div className="tool-group">
              <img className="result-mascot" src="/mascot-perfect.png" alt="" />
              <span className="tool-label tool-label-done"><Icon.check /> {results.filter((r) => r.blob).length}장 완성</span>
              <p className="result-desc">개별 저장하거나 한 번에 내려받으세요.</p>
            </div>
            <button className="btn btn-primary btn-block" onClick={saveAll}><Icon.download /> 모두 저장</button>
            <button className="btn btn-ghost btn-sm btn-block" onClick={reset}>새로 시작</button>
          </>
        )}
      </aside>
    </div>
  )
}
