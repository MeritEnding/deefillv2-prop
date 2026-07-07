import { useEffect, useRef, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8080'

function App() {
  const [imageFile, setImageFile] = useState(null)
  const [imageURL, setImageURL] = useState(null)
  const [resultURL, setResultURL] = useState(null)
  const [maskPreviewURL, setMaskPreviewURL] = useState(null)
  const [brushSize, setBrushSize] = useState(24)
  const [isProcessing, setIsProcessing] = useState(false)
  const [engine, setEngine] = useState(null)
  const [error, setError] = useState(null)

  const displayCanvasRef = useRef(null)   // 화면 표시용 (이미지 + 빨간 마스크)
  const maskCanvasRef = useRef(null)      // 서버 전송용 (원본 해상도, 흑백)
  const imgRef = useRef(null)
  const drawingRef = useRef(false)
  const lastPointRef = useRef(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((res) => res.json())
      .then((data) => setEngine(data.engine ?? null))
      .catch(() => setEngine(null))
  }, [])

  const loadImage = (file) => {
    if (!file) return
    setImageFile(file)
    setResultURL(null)
    setMaskPreviewURL(null)
    setError(null)

    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      setImageURL(url)

      const display = displayCanvasRef.current
      const scale = Math.min(1, 640 / img.naturalWidth)
      display.width = Math.round(img.naturalWidth * scale)
      display.height = Math.round(img.naturalHeight * scale)
      display.getContext('2d').drawImage(img, 0, 0, display.width, display.height)

      const mask = maskCanvasRef.current
      mask.width = img.naturalWidth
      mask.height = img.naturalHeight
      const mctx = mask.getContext('2d')
      mctx.fillStyle = 'black'
      mctx.fillRect(0, 0, mask.width, mask.height)
    }
    img.src = url
  }

  const canvasPoint = (e) => {
    const rect = displayCanvasRef.current.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  const strokeTo = (point) => {
    const display = displayCanvasRef.current
    const mask = maskCanvasRef.current
    const from = lastPointRef.current ?? point

    const dctx = display.getContext('2d')
    dctx.strokeStyle = 'rgba(255, 60, 60, 0.75)'
    dctx.lineWidth = brushSize
    dctx.lineCap = 'round'
    dctx.beginPath()
    dctx.moveTo(from.x, from.y)
    dctx.lineTo(point.x, point.y)
    dctx.stroke()

    const scale = mask.width / display.width
    const mctx = mask.getContext('2d')
    mctx.strokeStyle = 'white'
    mctx.lineWidth = brushSize * scale
    mctx.lineCap = 'round'
    mctx.beginPath()
    mctx.moveTo(from.x * scale, from.y * scale)
    mctx.lineTo(point.x * scale, point.y * scale)
    mctx.stroke()

    lastPointRef.current = point
  }

  const handlePointerDown = (e) => {
    if (!imageURL) return
    drawingRef.current = true
    lastPointRef.current = null
    strokeTo(canvasPoint(e))
  }
  const handlePointerMove = (e) => {
    if (drawingRef.current) strokeTo(canvasPoint(e))
  }
  const handlePointerUp = () => {
    drawingRef.current = false
    lastPointRef.current = null
  }

  const clearMask = () => {
    const img = imgRef.current
    if (!img) return
    const display = displayCanvasRef.current
    display.getContext('2d').drawImage(img, 0, 0, display.width, display.height)
    const mask = maskCanvasRef.current
    const mctx = mask.getContext('2d')
    mctx.fillStyle = 'black'
    mctx.fillRect(0, 0, mask.width, mask.height)
    setResultURL(null)
    setMaskPreviewURL(null)
  }

  const runInpaint = async () => {
    if (!imageFile || isProcessing) return
    setIsProcessing(true)
    setError(null)
    try {
      const maskBlob = await new Promise((resolve) =>
        maskCanvasRef.current.toBlob(resolve, 'image/png'),
      )
      setMaskPreviewURL(URL.createObjectURL(maskBlob))

      const form = new FormData()
      form.append('image', imageFile)
      form.append('mask', maskBlob, 'mask.png')

      const res = await fetch(`${API_BASE}/api/inpaint`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`서버 오류 (${res.status})`)
      setResultURL(URL.createObjectURL(await res.blob()))
    } catch (err) {
      setError(err.message ?? '요청에 실패했습니다.')
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>DeepFillv2 이미지 인페인팅 시연</h1>
        <p className="subtitle">
          이미지를 올리고 지우고 싶은 영역을 칠한 뒤 복원 버튼을 누르세요.
          {engine && <span className="engine-badge">엔진: {engine}</span>}
          {engine === null && <span className="engine-badge down">추론 서버 연결 안 됨</span>}
        </p>
      </header>

      <section className="controls">
        <label className="file-label">
          이미지 선택
          <input type="file" accept="image/*" onChange={(e) => loadImage(e.target.files[0])} />
        </label>
        <label className="brush-label">
          브러시 {brushSize}px
          <input
            type="range" min="4" max="80" value={brushSize}
            onChange={(e) => setBrushSize(Number(e.target.value))}
          />
        </label>
        <button onClick={clearMask} disabled={!imageURL}>마스크 지우기</button>
        <button className="primary" onClick={runInpaint} disabled={!imageURL || isProcessing}>
          {isProcessing ? '복원 중…' : '인페인팅 실행'}
        </button>
      </section>

      {error && <p className="error">{error}</p>}

      <section className="workspace">
        <figure>
          <figcaption>입력 / 마스크 스케치</figcaption>
          <canvas
            ref={displayCanvasRef}
            className={imageURL ? 'canvas ready' : 'canvas'}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerUp}
          />
          {!imageURL && <p className="placeholder">이미지를 선택하면 여기에 표시됩니다.</p>}
        </figure>

        {maskPreviewURL && (
          <figure>
            <figcaption>전송된 마스크</figcaption>
            <img src={maskPreviewURL} alt="마스크" />
          </figure>
        )}

        {resultURL && (
          <figure>
            <figcaption>복원 결과</figcaption>
            <img src={resultURL} alt="인페인팅 결과" />
          </figure>
        )}
      </section>

      <canvas ref={maskCanvasRef} style={{ display: 'none' }} />
    </div>
  )
}

export default App
