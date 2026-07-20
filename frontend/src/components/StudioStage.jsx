import BeforeAfter from './BeforeAfter.jsx'

function BusyOverlay({ message }) {
  return (
    <div className="stage-busy">
      <div className="scanline" />
      <img src="/mascot-erase.png" alt="" />
      <span>{message}</span>
    </div>
  )
}

function DrawingStage({ canvasRef, tool, handlers, hint, smartBusy, hasStrokes, busy }) {
  return (
    <div className="stage">
      <canvas
        ref={canvasRef}
        className={`stage-canvas tool-${tool}`}
        onPointerDown={handlers.onDown}
        onPointerMove={handlers.onMove}
        onPointerUp={handlers.onUp}
        onPointerLeave={handlers.onLeave}
      />
      {smartBusy && <div className="stage-hint">영역을 찾는 중…</div>}
      {!smartBusy && !hasStrokes && !busy && (
        <div className="stage-hint">{tool === 'smart' ? '지울 대상을 클릭하세요' : hint}</div>
      )}
      {busy && <BusyOverlay message="지우가 지우는 중…" />}
    </div>
  )
}

function OutpaintStage({ imageURL, expand, busy }) {
  return (
    <div className="stage">
      <img className="stage-canvas" src={imageURL} alt="원본" style={{ outline: `${Math.round(expand / 3)}px solid var(--blue-tint-2)` }} />
      {!busy && <div className="stage-hint">바깥 파란 영역만큼 배경을 확장합니다</div>}
      {busy && <BusyOverlay message="배경을 그려내는 중…" />}
    </div>
  )
}

function ResultStage({ imageURL, result, natural }) {
  return (
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
  )
}

// 스튜디오 중앙 작업 영역 — 상태(그리기·확장·결과)에 맞는 스테이지를 렌더링한다.
export default function StudioStage(props) {
  const { imageURL, result, isOutpaint } = props
  if (result) return <ResultStage imageURL={imageURL} result={result} natural={props.natural} />
  if (isOutpaint) return <OutpaintStage imageURL={imageURL} expand={props.expand} busy={props.busy} />
  return (
    <DrawingStage
      canvasRef={props.canvasRef}
      tool={props.tool}
      handlers={props.handlers}
      hint={props.hint}
      smartBusy={props.smartBusy}
      hasStrokes={props.hasStrokes}
      busy={props.busy}
    />
  )
}
