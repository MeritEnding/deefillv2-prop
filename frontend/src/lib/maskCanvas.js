// 마스크 편집 캔버스 렌더링 헬퍼 — Studio에서 표시 캔버스(파란 오버레이)와
// 원본 해상도 마스크 캔버스(흑백)를 동시에 갱신할 때 쓴다.

export const STROKE_TINT = 'rgba(37,99,235,0.5)'

export function drawStroke(ctx, points, scale, color, size) {
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

// 스트로크 목록을 (표시 캔버스 + 마스크 캔버스)에 다시 그린다.
// 'auto' 스트로크는 스마트 클릭 결과로, 원본 해상도 마스크(m)와 파란 틴트(tint)를 가진다.
export function renderStrokes({ display, mask, overlay, image, strokes }) {
  if (!display || !mask || !image) return
  overlay.width = display.width
  overlay.height = display.height
  const overlayCtx = overlay.getContext('2d')
  const maskCtx = mask.getContext('2d')
  maskCtx.globalCompositeOperation = 'source-over'
  maskCtx.fillStyle = 'black'
  maskCtx.fillRect(0, 0, mask.width, mask.height)
  const scale = mask.width / display.width
  for (const s of strokes || []) {
    if (!s) continue
    if (s.mode === 'auto') {
      overlayCtx.globalCompositeOperation = 'source-over'
      overlayCtx.drawImage(s.tint, 0, 0, overlay.width, overlay.height)
      maskCtx.globalCompositeOperation = 'lighten'
      maskCtx.drawImage(s.m, 0, 0, mask.width, mask.height)
      maskCtx.globalCompositeOperation = 'source-over'
      continue
    }
    if (!s.points || !s.points.length) continue
    overlayCtx.globalCompositeOperation = s.mode === 'erase' ? 'destination-out' : 'source-over'
    drawStroke(overlayCtx, s.points, 1, STROKE_TINT, s.size)
    maskCtx.globalCompositeOperation = 'source-over'
    drawStroke(maskCtx, s.points, scale, s.mode === 'erase' ? 'black' : 'white', s.size * scale)
  }
  const displayCtx = display.getContext('2d')
  displayCtx.clearRect(0, 0, display.width, display.height)
  displayCtx.drawImage(image, 0, 0, display.width, display.height)
  displayCtx.drawImage(overlay, 0, 0)
}

// 서버가 준 흑백 마스크 비트맵을 (마스크 캔버스, 파란 틴트 캔버스) 쌍으로 변환한다.
export function bitmapToMaskAndTint(bitmap) {
  const m = document.createElement('canvas')
  m.width = bitmap.width
  m.height = bitmap.height
  const maskCtx = m.getContext('2d')
  maskCtx.drawImage(bitmap, 0, 0)

  const t = document.createElement('canvas')
  t.width = bitmap.width
  t.height = bitmap.height
  const tintCtx = t.getContext('2d')
  const src = maskCtx.getImageData(0, 0, m.width, m.height)
  const out = tintCtx.createImageData(m.width, m.height)
  for (let i = 0; i < src.data.length; i += 4) {
    if (src.data[i] > 127) {
      out.data[i] = 37; out.data[i + 1] = 99; out.data[i + 2] = 235; out.data[i + 3] = 128
    }
  }
  tintCtx.putImageData(out, 0, 0)
  return { m, tint: t }
}
