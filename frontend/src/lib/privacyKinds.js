// 개인정보 항목별 전용 페이지 구성 — kind에 따라 탐지 대상을 필터하고 문구를 바꾼다.
export const TYPE_LABEL = { face: '얼굴', plate: '번호판' }

export const KINDS = {
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

const MASK_MARGIN = 0.25 // 탐지 박스 대비 마스크 여유

// 선택된 탐지 박스들을 인페인팅 마스크 PNG blob으로 렌더링한다.
// 얼굴·번호판은 타원, 텍스트는 획 끝이 남지 않도록 사각형으로 그린다.
export function buildDetectionMask(natural, selected) {
  const canvas = document.createElement('canvas')
  canvas.width = natural.w
  canvas.height = natural.h
  const ctx = canvas.getContext('2d')
  ctx.fillStyle = 'black'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = 'white'
  for (const { type, box: [x, y, w, h] } of selected) {
    if (type === 'text') {
      const mx = (w * MASK_MARGIN) / 2, my = (h * MASK_MARGIN) / 2
      ctx.fillRect(x - mx, y - my, w + mx * 2, h + my * 2)
    } else {
      ctx.beginPath()
      ctx.ellipse(x + w / 2, y + h / 2, (w / 2) * (1 + MASK_MARGIN), (h / 2) * (1 + MASK_MARGIN), 0, 0, Math.PI * 2)
      ctx.fill()
    }
  }
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/png'))
}
