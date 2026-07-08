import { useCallback, useEffect, useRef, useState } from 'react'
import BeforeAfter from '../components/BeforeAfter.jsx'
import { account } from '../lib/account.js'
import { downloadBlob, postProcess } from '../lib/image.js'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8080'
const MAX_DISPLAY = 680

function drawPath(ctx, points, scale, color, size) {
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = size
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  if (points.length === 1) {
    const p = points[0]
    ctx.beginPath()
    ctx.arc(p.x * scale, p.y * scale, size / 2, 0, Math.PI * 2)
    ctx.fill()
    return
  }
  ctx.beginPath()
  ctx.moveTo(points[0].x * scale, points[0].y * scale)
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x * scale, points[i].y * scale)
  ctx.stroke()
}

export default function Editor({ acc, setAcc, requirePaywall, engine }) {
  const [imageFile, setImageFile] = useState(null)
  const [imageURL, setImageURL] = useState(null)
  const [strokes, setStrokes] = useState([])
  const [redo, setRedo] = useState([])
  const [tool, setTool] = useState('brush') // brush | erase
  const [brushSize, setBrushSize] = useState(30)
  const [busy, setBusy] = useState(false)
  const [resultURL, setResultURL] = useState(null)
  const [resultBlob, setResultBlob] = useState(null)
  const [resultHD, setResultHD] = useState(false)
  const [error, setError] = useState(null)

  const dispRef = useRef(null) // 화면 표시 (이미지 + 빨간 마스크)
  const overlayRef = useRef(document.createElement('canvas')) // 마스크 오버레이(표시 크기)
  const maskRef = useRef(null) // 서버 전송용 (원본 해상도, 흑백)
  const imgRef = useRef(null)
  const drawing = useRef(false)
  const cur = useRef(null)

  const renderAll = useCallback((list) => {
    const disp = dispRef.current
    const mask = maskRef.current
    const img = imgRef.current
    if (!disp || !mask || !img) return

    const overlay = overlayRef.current
    overlay.width = disp.width
    overlay.height = disp.height
    const octx = overlay.getContext('2d')

    const mctx = mask.getContext('2d')
    mctx.globalCompositeOperation = 'source-over'
    mctx.fillStyle = 'black'
    mctx.fillRect(0, 0, mask.width, mask.height)

    const scale = mask.width / disp.width
    for (const s of list) {
      octx.globalCompositeOperation = s.mode === 'erase' ? 'destination-out' : 'source-over'
      drawPath(octx, s.points, 1, 'rgba(255,70,70,0.72)', s.size)
      mctx.globalCompositeOperation = 'source-over'
      drawPath(mctx, s.points, scale, s.mode === 'erase' ? 'black' : 'white', s.size * scale)
    }

    const dctx = disp.getContext('2d')
    dctx.clearRect(0, 0, disp.width, disp.height)
    dctx.drawImage(img, 0, 0, disp.width, disp.height)
    dctx.drawImage(overlay, 0, 0)
  }, [])

  useEffect(() => {
    renderAll(strokes)
  }, [strokes, renderAll])

  const loadImage = (file) => {
    if (!file) return
    setImageFile(file)
    setStrokes([])
    setRedo([])
    setResultURL(null)
    setResultBlob(null)
    setError(null)

    const url = URL.createObjectURL(file)
    setImageURL(url)
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      const scale = Math.min(1, MAX_DISPLAY / Math.max(img.naturalWidth, img.naturalHeight))
      const disp = dispRef.current
      disp.width = Math.round(img.naturalWidth * scale)
      disp.height = Math.round(img.naturalHeight * scale)
      const mask = maskRef.current
      mask.width = img.naturalWidth
      mask.height = img.naturalHeight
      renderAll([])
    }
    img.src = url
  }

  const pointFrom = (e) => {
    const rect = dispRef.current.getBoundingClientRect()
    const sx = dispRef.current.width / rect.width
    const sy = dispRef.current.height / rect.height
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy }
  }

  const onDown = (e) => {
    if (!imgRef.current) return
    dispRef.current.setPointerCapture(e.pointerId)
    drawing.current = true
    cur.current = { mode: tool, size: brushSize, points: [pointFrom(e)] }
    renderAll([...strokes, cur.current])
  }
  const onMove = (e) => {
    if (!drawing.current) return
    cur.current.points.push(pointFrom(e))
    renderAll([...strokes, cur.current])
  }
  const onUp = () => {
    if (!drawing.current) return
    drawing.current = false
    if (cur.current) {
      setStrokes((s) => [...s, cur.current])
      setRedo([])
      cur.current = null
    }
  }

  const undo = useCallback(() => {
    setStrokes((s) => {
      if (!s.length) return s
      setRedo((r) => [...r, s[s.length - 1]])
      return s.slice(0, -1)
    })
  }, [])
  const redoAction = useCallback(() => {
    setRedo((r) => {
      if (!r.length) return r
      setStrokes((s) => [...s, r[r.length - 1]])
      return r.slice(0, -1)
    })
  }, [])
  const clearMask = () => {
    setStrokes([])
    setRedo([])
    setResultURL(null)
    setResultBlob(null)
  }

  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) redoAction()
        else undo()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redoAction])

  const run = async () => {
    if (!imageFile || busy || !strokes.length) return
    const decision = account.canRun(acc)
    if (!decision.ok) {
      requirePaywall()
      return
    }
    setBusy(true)
    setError(null)
    try {
      const maskBlob = await new Promise((r) => maskRef.current.toBlob(r, 'image/png'))
      const form = new FormData()
      form.append('image', imageFile)
      form.append('mask', maskBlob, 'mask.png')
      const res = await fetch(`${API_BASE}/api/inpaint`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`서버 오류 (${res.status})`)
      const raw = await res.blob()

      const hd = decision.mode === 'hd'
      const processed = await postProcess(raw, {
        maxDim: hd ? null : account.FREE_MAX_DIM,
        watermark: !hd,
      })
      setResultBlob(processed)
      setResultURL(URL.createObjectURL(processed))
      setResultHD(hd)
      setAcc(account.consume(acc, decision.mode))
    } catch (err) {
      setError(err.message ?? '요청에 실패했습니다.')
    } finally {
      setBusy(false)
    }
  }

  const download = () => {
    if (resultBlob) downloadBlob(resultBlob, `jium-${Date.now()}.png`)
  }

  const remaining = account.freeRemaining(acc)
  const canRun = account.canRun(acc)

  return (
    <main className="editor">
      <div className="editor-head">
        <h2>AI 지우개</h2>
        <div className="editor-status">
          {engine ? (
            <span className="pill pill-ok">엔진 연결됨 · {engine}</span>
          ) : (
            <span className="pill pill-down">추론 서버 연결 안 됨</span>
          )}
          {acc.plan === 'pro' ? (
            <span className="pill pill-pro">크레딧 {acc.credits}</span>
          ) : (
            <span className="pill">오늘 무료 {remaining}/{account.FREE_DAILY_LIMIT}</span>
          )}
        </div>
      </div>

      <div className="editor-grid">
        <div className="stage-wrap">
          {!imageURL && (
            <label className="dropzone">
              <input type="file" accept="image/*" onChange={(e) => loadImage(e.target.files[0])} hidden />
              <div className="dropzone-icon">🖼️</div>
              <p className="dropzone-title">사진을 선택하거나 끌어다 놓으세요</p>
              <p className="dropzone-sub">JPG · PNG · 최대 한 장</p>
            </label>
          )}

          {imageURL && !resultURL && (
            <div className="stage">
              <canvas
                ref={dispRef}
                className={`stage-canvas tool-${tool}`}
                onPointerDown={onDown}
                onPointerMove={onMove}
                onPointerUp={onUp}
                onPointerLeave={onUp}
              />
              {!strokes.length && <div className="stage-hint">지울 부분을 칠하세요</div>}
            </div>
          )}

          {resultURL && (
            <div className="stage">
              <BeforeAfter before={imageURL} after={resultURL} />
              {!resultHD && <div className="wm-note">무료 저장은 720p · 워터마크가 포함됩니다</div>}
            </div>
          )}
        </div>

        <aside className="toolbar">
          {!resultURL ? (
            <>
              <div className="tool-group">
                <span className="tool-label">도구</span>
                <div className="seg">
                  <button className={tool === 'brush' ? 'on' : ''} onClick={() => setTool('brush')}>
                    🖌️ 브러시
                  </button>
                  <button className={tool === 'erase' ? 'on' : ''} onClick={() => setTool('erase')}>
                    🧽 지우개
                  </button>
                </div>
              </div>

              <div className="tool-group">
                <span className="tool-label">브러시 크기 · {brushSize}px</span>
                <input
                  type="range" min="8" max="90" value={brushSize}
                  onChange={(e) => setBrushSize(Number(e.target.value))}
                />
              </div>

              <div className="tool-group">
                <span className="tool-label">편집</span>
                <div className="btn-row">
                  <button className="btn btn-outline btn-sm" onClick={undo} disabled={!strokes.length}>
                    ↶ 되돌리기
                  </button>
                  <button className="btn btn-outline btn-sm" onClick={redoAction} disabled={!redo.length}>
                    ↷ 다시
                  </button>
                </div>
                <button className="btn btn-ghost btn-sm btn-block" onClick={clearMask} disabled={!strokes.length}>
                  전체 지우기
                </button>
              </div>

              {error && <p className="error">{error}</p>}

              <button
                className="btn btn-primary btn-block btn-run"
                onClick={run}
                disabled={!imageURL || busy || !strokes.length}
              >
                {busy ? '지우는 중…' : canRun.ok ? '✨ 지우기 실행' : '무료 소진 · 업그레이드'}
              </button>
              <label className="btn btn-ghost btn-sm btn-block">
                다른 사진 올리기
                <input type="file" accept="image/*" onChange={(e) => loadImage(e.target.files[0])} hidden />
              </label>
            </>
          ) : (
            <>
              <div className="tool-group">
                <span className="tool-label">완성됐어요 🎉</span>
                <p className="result-desc">
                  {resultHD ? 'HD 원본으로 저장할 수 있어요.' : '무료 결과입니다. HD·워터마크 제거는 Pro에서.'}
                </p>
              </div>
              <button className="btn btn-primary btn-block" onClick={download}>
                ⬇ 이미지 저장
              </button>
              <button className="btn btn-outline btn-block" onClick={() => { setResultURL(null); setResultBlob(null) }}>
                더 지우기
              </button>
              <label className="btn btn-ghost btn-sm btn-block">
                새 사진 시작
                <input type="file" accept="image/*" onChange={(e) => loadImage(e.target.files[0])} hidden />
              </label>
            </>
          )}
        </aside>
      </div>

      <canvas ref={maskRef} style={{ display: 'none' }} />
    </main>
  )
}
