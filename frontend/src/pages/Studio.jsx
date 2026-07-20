import { useCallback, useEffect, useRef, useState } from 'react'
import BatchPanel from './BatchPanel.jsx'
import { Icon } from '../components/icons.jsx'
import ModelPicker from '../components/ModelPicker.jsx'
import StudioStage from '../components/StudioStage.jsx'
import StudioToolbar from '../components/StudioToolbar.jsx'
import UploadDropzone from '../components/UploadDropzone.jsx'
import { MODES, MODE_MAP } from '../lib/modes.js'
import { copyBlobToClipboard, downloadBlob } from '../lib/image.js'
import { bitmapToMaskAndTint, renderStrokes } from '../lib/maskCanvas.js'
import { postForm } from '../lib/api.js'

const MAX_DISPLAY = 640
const SAMPLES = ['/demo-before.png', '/sample-street.png']
const OUTPAINT_EDGE_BLUR = 8 // 가장자리 픽셀을 늘려 채울 때 흐림 정도

export default function Studio({ modeId, setModeId, engine, engines, setEngine, onError }) {
  const mode = MODE_MAP[modeId] ?? MODES[0]
  const isOutpaint = mode.special === 'outpaint'
  const isBatch = mode.special === 'batch'

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
    renderStrokes({
      display: dispRef.current,
      mask: maskRef.current,
      overlay: overlayRef.current,
      image: imgRef.current,
      strokes: list,
    })
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
      const res = await postForm('/api/segment', form)
      const bitmap = await createImageBitmap(await res.blob())
      const { m, tint } = bitmapToMaskAndTint(bitmap)
      setStrokes((prev) => [...prev, { mode: 'auto', m, tint }])
      setRedo([])
    } catch (e) { setError(e.message ?? '영역 선택에 실패했어요.') } finally { setSmartBusy(false) }
  }

  const postInpaint = async (imageBlob, maskBlob) => {
    const form = new FormData()
    form.append('image', imageBlob, 'image.png')
    form.append('mask', maskBlob, 'mask.png')
    if (engine) form.append('engine', engine)
    const res = await postForm('/api/inpaint', form)
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
      ictx.filter = `blur(${OUTPAINT_EDGE_BLUR}px)`; ictx.drawImage(ic, 0, 0); ictx.filter = 'none'
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
        {engines?.length
          ? <ModelPicker engines={engines} value={engine} onChange={setEngine} />
          : <span className="pill pill-down">추론 서버 연결 안 됨</span>}
      </div>

      {isBatch ? <BatchPanel brushDefault={mode.brush} /> : (
      <div className="studio-grid">
        <div
          className={`stage-wrap ${dragOver ? 'drag' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); loadFile(e.dataTransfer.files[0]) }}
        >
          {!imageURL && (
            <UploadDropzone
              title="사진을 끌어다 놓거나 클릭해서 선택"
              sub="붙여넣기(Ctrl+V)도 됩니다 · JPG · PNG"
              onPick={loadFile}
            >
              <div className="samples">
                <span>예시로 체험:</span>
                {SAMPLES.map((s) => (
                  <img key={s} src={s} alt="예시" onClick={(e) => { e.preventDefault(); loadURL(s) }} />
                ))}
              </div>
            </UploadDropzone>
          )}

          {imageURL && (
            <StudioStage
              imageURL={imageURL}
              result={result}
              isOutpaint={isOutpaint}
              natural={natural}
              expand={expand}
              busy={busy}
              canvasRef={dispRef}
              tool={tool}
              hint={mode.hint}
              smartBusy={smartBusy}
              hasStrokes={strokes.length > 0}
              handlers={{ onDown, onMove, onUp, onLeave: () => { onUp(); setCursor(null) } }}
            />
          )}

          {/* 캔버스 밖에서도 hidden 마스크 유지 */}
          <canvas ref={maskRef} style={{ display: 'none' }} />
        </div>

        <StudioToolbar
          result={result}
          isOutpaint={isOutpaint}
          resultActions={{
            engine, copied,
            onDownload: download, onCopy: copy, onContinue: continueEdit,
            onReEdit: () => setResult(null), onPick: loadFile,
          }}
          outpaint={{
            expand, setExpand, error, busy,
            canRun: Boolean(imageURL), onRun: runOutpaint, onPick: loadFile,
          }}
          brush={{
            tool, setTool, brushSize, setBrushSize, strokes, redo,
            onUndo: undo, onRedo: redoAct,
            onClear: () => { setStrokes([]); setRedo([]) },
            error, busy, canRun: Boolean(imageURL), onRun: runRemoval, onPick: loadFile,
          }}
        />
      </div>
      )}

      {cursor && !result && !isOutpaint && !isBatch && (
        <div className="brush-cursor" style={{ left: cursor.x, top: cursor.y, width: cursor.size, height: cursor.size }} />
      )}
    </main>
  )
}
