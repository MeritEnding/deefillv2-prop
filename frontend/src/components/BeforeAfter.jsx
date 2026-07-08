import { useRef, useState } from 'react'

// 드래그로 원본/결과를 비교하는 슬라이더. clip-path로 위 레이어를 잘라 보여준다.
export default function BeforeAfter({ before, after, beforeLabel = '원본', afterLabel = '지움 후' }) {
  const [pos, setPos] = useState(50)
  const wrapRef = useRef(null)
  const dragging = useRef(false)

  const move = (clientX) => {
    const rect = wrapRef.current.getBoundingClientRect()
    const p = ((clientX - rect.left) / rect.width) * 100
    setPos(Math.max(0, Math.min(100, p)))
  }

  return (
    <div
      ref={wrapRef}
      className="ba"
      onPointerDown={(e) => {
        dragging.current = true
        e.currentTarget.setPointerCapture(e.pointerId)
        move(e.clientX)
      }}
      onPointerMove={(e) => dragging.current && move(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
      onPointerLeave={() => (dragging.current = false)}
    >
      <img className="ba-img" src={after} alt={afterLabel} draggable="false" />
      <img
        className="ba-img ba-over"
        src={before}
        alt={beforeLabel}
        draggable="false"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      />
      <span className="ba-tag ba-tag-left" style={{ opacity: pos > 12 ? 1 : 0 }}>
        {beforeLabel}
      </span>
      <span className="ba-tag ba-tag-right" style={{ opacity: pos < 88 ? 1 : 0 }}>
        {afterLabel}
      </span>
      <div className="ba-handle" style={{ left: `${pos}%` }}>
        <div className="ba-knob">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M9 6 4 12l5 6M15 6l5 6-5 6" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </div>
    </div>
  )
}
