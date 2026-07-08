// 결과 이미지 후처리: 무료 플랜은 720p 다운스케일 + 워터마크, Pro는 원본 그대로.

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

export async function postProcess(resultBlob, { maxDim, watermark }) {
  const url = URL.createObjectURL(resultBlob)
  try {
    const img = await loadImage(url)
    let w = img.naturalWidth
    let h = img.naturalHeight
    if (maxDim && Math.max(w, h) > maxDim) {
      const s = maxDim / Math.max(w, h)
      w = Math.round(w * s)
      h = Math.round(h * s)
    }
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(img, 0, 0, w, h)

    if (watermark) {
      const fs = Math.max(16, Math.round(w * 0.032))
      ctx.font = `700 ${fs}px ${getComputedStyle(document.body).fontFamily}`
      ctx.textBaseline = 'bottom'
      const label = '지움 · jium.app'
      const pad = Math.round(fs * 0.6)
      const tw = ctx.measureText(label).width
      ctx.fillStyle = 'rgba(0,0,0,0.32)'
      ctx.fillRect(w - tw - pad * 2.4, h - fs - pad * 1.8, tw + pad * 2.4, fs + pad * 1.4)
      ctx.fillStyle = 'rgba(255,255,255,0.92)'
      ctx.fillText(label, w - tw - pad * 1.2, h - pad * 0.6)
    }

    return await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'))
  } finally {
    URL.revokeObjectURL(url)
  }
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
