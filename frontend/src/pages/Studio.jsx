import { useCallback, useEffect, useRef, useState } from 'react'
import BeforeAfter from '../components/BeforeAfter.jsx'
import { Icon } from '../components/icons.jsx'
import { MODES, MODE_MAP } from '../lib/modes.js'
import { copyBlobToClipboard, downloadBlob } from '../lib/image.js'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8080'
const MAX_DISPLAY = 640
const SAMPLES = ['/demo-before.png', '/sample-street.png']

function stroke(ctx, points, scale, color, size) {
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = size
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  if (points.length === 1) {
    ctx.beginPath()
    ctx.arc(points[0].x * scale, points[0].y * scale, Math.max(0.5, size / 2), 0, Math.PI * 2)
    ctx.fill()
    return
  }
  ctx.beginPath()
  ctx.moveTo(points[0].x * scale, points[0].y * scale)
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x * scale, points[i].y * scale)
  ctx.stroke()
}

export default function Studio({ modeId, setModeId, engine, onError }) {
  const mode = MODE_MAP[modeId] ?? MODES[0]
  const isOutpaint = mode.special === 'outpaint'

  const [imageFile, setImageFile] = useState(null)
  const [imageURL, setImageURL] = useState(null)
  const [natural, setNatural] = useState(null) // {w,h}
  const [strokes, setStrokes] = useState([])
  const [redo, setRedo] = useState([])
  const [tool, setTool] = useState('brush')
  const [brushSize, setBrushSize] = useState(mode.brush ?? 30)
  const [busy, setBusy] = useState(false)
  const [smartBusy, setSmartBusy] = useState(false)
  const [result, setResult] = useState(null) // {url, blob}
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [cursor, setCursor] = useState(null) // {x,y,size} for brush preview
  const [expand, setExpand] = useState(30)

  const dispRef = useRef(null)
  const overlayRef = useRef(null)
  const maskRef = useRef(null)
  const imgRef = useRef(null)
  const drawing = useRef(false)
  const cur = useRef(null)

  if (!overlayRef.current) overlayRef.current = document.createElement('canvas')

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
    for (const s of list || []) {
      if (!s) continue
      if (s.mode === 'auto') {
        // 스마트 클릭 결과: 원본 해상도 마스크 + 파란 틴트 오버레이
        octx.globalCompositeOperation = 'source-over'
        octx.drawImage(s.tint, 0, 0, overlay.width, overlay.height)
        mctx.globalCompositeOperation = 'lighten'
        mctx.drawImage(s.m, 0, 0, mask.width, mask.height)
        mctx.globalCompositeOperation = 'source-over'
        continue
      }
      if (!s.points || !s.points.length) continue
      octx.globalCompositeOperation = s.mode === 'erase' ? 'destination-out' : 'source-over'
      stroke(octx, s.points, 1, 'rgba(37,99,235,0.5)', s.size)
      mctx.globalCompositeOperation = 'source-over'
      stroke(mctx, s.points, scale, s.mode === 'erase' ? 'black' : 'white', s.size * scale)
    }
    const dctx = disp.getContext('2d')
    dctx.clearRect(0, 0, disp.width, disp.height)
    dctx.drawImage(img, 0, 0, disp.width, disp.height)
    dctx.drawImage(overlay, 0, 0)
  }, [])

  useEffect(() => { renderAll(strokes) }, [strokes, renderAll])
  useEffect(() => { setBrushSize(mode.brush ?? 30) }, [modeId]) // eslint-disable-line

  const reset = () => {
    setStrokes([]); setRedo([]); setResult(null); setError(null)
  }

  const loadFile = (file) => {
    if (!file || !file.type?.startsWith('image/')) return
    reset()
    setImageFile(file)
    const url = URL.createObjectURL(file)
    setImageURL(url)
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      setNatural({ w: img.naturalWidth, h: img.naturalHeight })
      const disp = dispRef.current
      if (disp) {
        const s = Math.min(1, MAX_DISPLAY / Math.max(img.naturalWidth, img.naturalHeight))
        disp.width = Math.round(img.naturalWidth * s)
        disp.height = Math.round(img.naturalHeight * s)
        const mask = maskRef.current
        mask.width = img.naturalWidth
        mask.height = img.naturalHeight
        renderAll([])
      }
    }
    img.src = url
  }

  const loadURL = async (src) => {
    try {
      const blob = await (await fetch(src)).blob()
      loadFile(new File([blob], src.split('/').pop(), { type: blob.type }))
    } catch { setError('예시 이미지를 불러오지 못했어요.') }
  }

  // 클립보드 붙여넣기
  useEffect(() => {
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'))
      if (item) loadFile(item.getAsFile())
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [])

  const pointFrom = (e) => {
    const r = dispRef.current.getBoundingClientRect()
    return {
      x: (e.clientX - r.left) * (dispRef.current.width / r.width),
      y: (e.clientY - r.top) * (dispRef.current.height / r.height),
    }
  }
  const onDown = (e) => {
    if (!imgRef.current || isOutpaint) return
    if (tool === 'smart') {
      const p = pointFrom(e)
      runSmartSelect(p.x / dispRef.current.width, p.y / dispRef.current.height)
      return
    }
    dispRef.current.setPointerCapture(e.pointerId)
    drawing.current = true
    cur.current = { mode: tool, size: brushSize, points: [pointFrom(e)] }
    renderAll([...strokes, cur.current])
  }
  const onMove = (e) => {
    const r = dispRef.current?.getBoundingClientRect()
    if (r && tool !== 'smart') setCursor({ x: e.clientX, y: e.clientY, size: brushSize * (r.width / dispRef.current.width) })
    if (!drawing.current || !cur.current) return
    cur.current.points.push(pointFrom(e))
    renderAll([...strokes, cur.current])
  }
  const onUp = () => {
    if (!drawing.current) return
    drawing.current = false
    const s = cur.current
    cur.current = null
    if (s && s.points.length) { setStrokes((prev) => [...prev, s]); setRedo([]) }
  }

  const undo = useCallback(() => setStrokes((s) => {
    if (!s.length) return s
    setRedo((r) => [...r, s[s.length - 1]])
    return s.slice(0, -1)
  }), [])
  const redoAct = useCallback(() => setRedo((r) => {
    if (!r.length) return r
    setStrokes((s) => [...s, r[r.length - 1]])
    return r.slice(0, -1)
  }), [])

  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        e.shiftKey ? redoAct() : undo()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redoAct])

  // 클릭 지점의 객체를 서버(GrabCut)가 분리해 마스크 스트로크로 추가
  const runSmartSelect = async (nx, ny) => {
    if (!imageFile || smartBusy) return
    setSmartBusy(true); setError(null)
    try {
      const form = new FormData()
      form.append('image', imageFile, 'image.png')
      form.append('x', nx)
      form.append('y', ny)
      const res = await fetch(`${API_BASE}/api/segment`, { method: 'POST', body: form })
      if (!res.ok) {
        const msg = await res.json().then((d) => d.message).catch(() => null)
        throw new Error(msg ?? `서버 오류 (${res.status})`)
      }
      const bmp = await createImageBitmap(await res.blob())
      const m = document.createElement('canvas')
      m.width = bmp.width; m.height = bmp.height
      const mx = m.getContext('2d')
      mx.drawImage(bmp, 0, 0)
      // 흰색(객체) → 반투명 파랑, 검정 → 투명 틴트 생성
      const t = document.createElement('canvas')
      t.width = bmp.width; t.height = bmp.height
      const tx = t.getContext('2d')
      const src = mx.getImageData(0, 0, m.width, m.height)
      const out = tx.createImageData(m.width, m.height)
      for (let i = 0; i < src.data.length; i += 4) {
        if (src.data[i] > 127) {
          out.data[i] = 37; out.data[i + 1] = 99; out.data[i + 2] = 235; out.data[i + 3] = 128
        }
      }
      tx.putImageData(out, 0, 0)
      setStrokes((prev) => [...prev, { mode: 'auto', m, tint: t }])
      setRedo([])
    } catch (e) { setError(e.message ?? '영역 선택에 실패했어요.') } finally { setSmartBusy(false) }
  }

  const postInpaint = async (imageBlob, maskBlob) => {
    const form = new FormData()
    form.append('image', imageBlob, 'image.png')
    form.append('mask', maskBlob, 'mask.png')
    const res = await fetch(`${API_BASE}/api/inpaint`, { method: 'POST', body: form })
    if (!res.ok) {
      // 백엔드가 내려주는 표준 에러 JSON의 message를 그대로 보여준다
      const msg = await res.json().then((d) => d.message).catch(() => null)
      throw new Error(msg ?? `서버 오류 (${res.status})`)
    }
    const blob = await res.blob()
    const elapsed = Number(res.headers.get('X-Elapsed-Ms')) || null
    return { blob, elapsed }
  }

  const runRemoval = async () => {
    if (!imageFile || busy || !strokes.length) return
    setBusy(true); setError(null)
    try {
      const maskBlob = await new Promise((r) => maskRef.current.toBlob(r, 'image/png'))
      const out = await postInpaint(imageFile, maskBlob)
      setResult({ url: URL.createObjectURL(out.blob), blob: out.blob, elapsed: out.elapsed })
    } catch (e) { setError(e.message ?? '요청 실패') } finally { setBusy(false) }
  }

  const runOutpaint = async () => {
    if (!imageFile || busy || !natural) return
    setBusy(true); setError(null)
    try {
      const pct = expand / 100
      const px = Math.round(natural.w * pct)
      const py = Math.round(natural.h * pct)
      const W = natural.w + px * 2
      const H = natural.h + py * 2
      const ic = document.createElement('canvas'); ic.width = W; ic.height = H
      const ictx = ic.getContext('2d')
      // 가장자리 픽셀을 늘려 채운 뒤 원본을 얹음 (LaMa가 마스크 영역만 재생성)
      ictx.drawImage(imgRef.current, 0, 0, natural.w, natural.h, 0, 0, W, H)
      ictx.filter = 'blur(8px)'; ictx.drawImage(ic, 0, 0); ictx.filter = 'none'
      ictx.drawImage(imgRef.current, px, py, natural.w, natural.h)
      const mc = document.createElement('canvas'); mc.width = W; mc.height = H
      const mctx = mc.getContext('2d')
      mctx.fillStyle = 'white'; mctx.fillRect(0, 0, W, H)
      mctx.fillStyle = 'black'; mctx.fillRect(px, py, natural.w, natural.h)
      const imageBlob = await new Promise((r) => ic.toBlob(r, 'image/png'))
      const maskBlob = await new Promise((r) => mc.toBlob(r, 'image/png'))
      const out = await postInpaint(imageBlob, maskBlob)
      setResult({ url: URL.createObjectURL(out.blob), blob: out.blob, elapsed: out.elapsed, wide: true })
    } catch (e) { setError(e.message ?? '요청 실패') } finally { setBusy(false) }
  }

  const download = () => result && downloadBlob(result.blob, `jium-${mode.id}-${Date.now()}.png`)
  const copy = async () => {
    try { await copyBlobToClipboard(result.blob); setCopied(true); setTimeout(() => setCopied(false), 1600) }
    catch (e) { setError(e.message) }
  }
  const continueEdit = () => {
    // 결과를 새 입력으로 이어서 편집
    loadFile(new File([result.blob], 'continue.png', { type: 'image/png' }))
  }

  const ModeIcon = Icon[mode.icon]

  return (
    <main className="studio">
      <div className="studio-modes">
        {MODES.map((m) => {
          const I = Icon[m.icon]
          return (
            <button key={m.id} className={`mtab ${m.id === mode.id ? 'on' : ''}`} onClick={() => setModeId(m.id)}>
              <I width="18" height="18" />
              <span>{m.name}</span>
            </button>
          )
        })}
      </div>

      <div className="studio-head">
        <div className="studio-title">
          <div className="tile tile-sm"><ModeIcon /></div>
          <div>
            <h2>{mode.name}</h2>
            <p>{mode.hint}</p>
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
          onDrop={(e) => { e.preventDefault(); setDragOver(false); loadFile(e.dataTransfer.files[0]) }}
        >
          {!imageURL && (
            <label className="dropzone">
              <input type="file" accept="image/*" hidden onChange={(e) => loadFile(e.target.files[0])} />
              <img className="dropzone-mascot" src="/mascot-select.png" alt="" />
              <p className="dropzone-title">사진을 끌어다 놓거나 클릭해서 선택</p>
              <p className="dropzone-sub">붙여넣기(Ctrl+V)도 됩니다 · JPG · PNG</p>
              <div className="samples">
                <span>예시로 체험:</span>
                {SAMPLES.map((s) => (
                  <img key={s} src={s} alt="예시" onClick={(e) => { e.preventDefault(); loadURL(s) }} />
                ))}
              </div>
            </label>
          )}

          {imageURL && !result && !isOutpaint && (
            <div className="stage">
              <canvas
                ref={dispRef}
                className={`stage-canvas tool-${tool}`}
                onPointerDown={onDown}
                onPointerMove={onMove}
                onPointerUp={onUp}
                onPointerLeave={() => { onUp(); setCursor(null) }}
              />
              {smartBusy && <div className="stage-hint">영역을 찾는 중…</div>}
              {!smartBusy && !strokes.length && !busy && (
                <div className="stage-hint">{tool === 'smart' ? '지울 대상을 클릭하세요' : mode.hint}</div>
              )}
              {busy && (
                <div className="stage-busy">
                  <div className="scanline" />
                  <img src="/mascot-erase.png" alt="" />
                  <span>지우가 지우는 중…</span>
                </div>
              )}
            </div>
          )}

          {imageURL && !result && isOutpaint && (
            <div className="stage">
              <img className="stage-canvas" src={imageURL} alt="원본" style={{ outline: `${Math.round(expand / 3)}px solid var(--blue-tint-2)` }} />
              {!busy && <div className="stage-hint">바깥 파란 영역만큼 배경을 확장합니다</div>}
              {busy && (
                <div className="stage-busy">
                  <div className="scanline" />
                  <img src="/mascot-erase.png" alt="" />
                  <span>배경을 그려내는 중…</span>
                </div>
              )}
            </div>
          )}

          {result && (
            <div className="stage stage-result">
              {result.wide ? (
                <img className="stage-canvas" src={result.url} alt="결과" />
              ) : (
                <BeforeAfter
                  before={imageURL}
                  after={result.url}
                  beforeLabel="원본"
                  afterLabel="완료"
                  initial={15}
                  style={natural ? {
                    aspectRatio: `${natural.w} / ${natural.h}`,
                    maxWidth: `${Math.round(560 * (natural.w / natural.h))}px`,
                  } : undefined}
                />
              )}
            </div>
          )}

          {/* 캔버스 밖에서도 hidden 마스크 유지 */}
          <canvas ref={maskRef} style={{ display: 'none' }} />
        </div>

        <aside className="toolbar">
          {!result ? (
            isOutpaint ? (
              <>
                <div className="tool-group">
                  <span className="tool-label">확장 비율 · {expand}%</span>
                  <input type="range" min="10" max="60" value={expand} onChange={(e) => setExpand(+e.target.value)} />
                </div>
                {error && <p className="error">{error}</p>}
                <button className="btn btn-primary btn-block btn-run" onClick={runOutpaint} disabled={!imageURL || busy}>
                  {busy ? '확장 중…' : <><Icon.sparkle width="18" height="18" /> 배경 확장</>}
                </button>
                <label className="btn btn-ghost btn-sm btn-block">다른 사진<input type="file" accept="image/*" hidden onChange={(e) => loadFile(e.target.files[0])} /></label>
              </>
            ) : (
              <>
                <div className="tool-group">
                  <span className="tool-label">도구</span>
                  <div className="seg">
                    <button className={tool === 'smart' ? 'on' : ''} onClick={() => setTool('smart')}><Icon.wand width="16" height="16" /> 스마트</button>
                    <button className={tool === 'brush' ? 'on' : ''} onClick={() => setTool('brush')}><Icon.brush width="16" height="16" /> 브러시</button>
                    <button className={tool === 'erase' ? 'on' : ''} onClick={() => setTool('erase')}><Icon.eraser width="16" height="16" /> 지우개</button>
                  </div>
                  {tool === 'smart' && <p className="tool-tip">지울 대상을 클릭하면 AI가 영역을 자동으로 선택합니다.</p>}
                </div>
                <div className="tool-group">
                  <span className="tool-label">브러시 크기 · {brushSize}px</span>
                  <input type="range" min="6" max="90" value={brushSize} onChange={(e) => setBrushSize(+e.target.value)} />
                </div>
                <div className="tool-group">
                  <span className="tool-label">편집</span>
                  <div className="btn-row">
                    <button className="btn btn-outline btn-sm" onClick={undo} disabled={!strokes.length}><Icon.undo /> 되돌리기</button>
                    <button className="btn btn-outline btn-sm" onClick={redoAct} disabled={!redo.length}><Icon.redo /> 다시</button>
                  </div>
                  <button className="btn btn-ghost btn-sm btn-block" onClick={() => { setStrokes([]); setRedo([]) }} disabled={!strokes.length}>전체 지우기</button>
                </div>
                {error && <p className="error">{error}</p>}
                <button className="btn btn-primary btn-block btn-run" onClick={runRemoval} disabled={!imageURL || busy || !strokes.length}>
                  {busy ? '지우는 중…' : <><Icon.sparkle width="18" height="18" /> 지우기 실행</>}
                </button>
                <label className="btn btn-ghost btn-sm btn-block">다른 사진<input type="file" accept="image/*" hidden onChange={(e) => loadFile(e.target.files[0])} /></label>
              </>
            )
          ) : (
            <>
              <div className="tool-group">
                <img className="result-mascot" src="/mascot-perfect.png" alt="" />
                <span className="tool-label tool-label-done"><Icon.check /> 완성됐어요</span>
                <p className="result-desc">HD 원본으로 저장하거나 클립보드에 복사하세요.</p>
                {result.elapsed && (
                  <p className="result-meta">
                    {engine ? `${engine} 엔진 · ` : ''}처리 {(result.elapsed / 1000).toFixed(1)}초
                  </p>
                )}
              </div>
              <button className="btn btn-primary btn-block" onClick={download}><Icon.download /> 이미지 저장</button>
              <button className="btn btn-outline btn-block" onClick={copy}><Icon.copy /> {copied ? '복사됨!' : '클립보드 복사'}</button>
              {!isOutpaint && <button className="btn btn-outline btn-block" onClick={continueEdit}>결과 이어서 지우기</button>}
              <button className="btn btn-ghost btn-sm btn-block" onClick={() => setResult(null)}>다시 편집</button>
              <label className="btn btn-ghost btn-sm btn-block">새 사진<input type="file" accept="image/*" hidden onChange={(e) => loadFile(e.target.files[0])} /></label>
            </>
          )}
        </aside>
      </div>

      {cursor && !result && !isOutpaint && (
        <div className="brush-cursor" style={{ left: cursor.x, top: cursor.y, width: cursor.size, height: cursor.size }} />
      )}
    </main>
  )
}
