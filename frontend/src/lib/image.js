// 이미지 유틸: 로드, 다운로드, 클립보드 복사.

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
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

export async function copyBlobToClipboard(blob) {
  if (!navigator.clipboard || !window.ClipboardItem) throw new Error('클립보드 미지원 브라우저')
  await navigator.clipboard.write([new window.ClipboardItem({ [blob.type]: blob })])
}

// 파일/Blob을 표시용 캔버스 크기에 맞춰 로드하고, 원본 해상도 정보를 함께 반환.
export async function fileToImage(file) {
  const url = URL.createObjectURL(file)
  const img = await loadImage(url)
  return { img, url }
}
