import { Icon } from './icons.jsx'

function FilePickButton({ label, onPick, className = 'btn btn-ghost btn-sm btn-block' }) {
  return (
    <label className={className}>
      {label}
      <input type="file" accept="image/*" hidden onChange={(e) => onPick(e.target.files[0])} />
    </label>
  )
}

function ResultActions({ result, engine, isOutpaint, copied, onDownload, onCopy, onContinue, onReEdit, onPick }) {
  return (
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
      <button className="btn btn-primary btn-block" onClick={onDownload}><Icon.download /> 이미지 저장</button>
      <button className="btn btn-outline btn-block" onClick={onCopy}><Icon.copy /> {copied ? '복사됨!' : '클립보드 복사'}</button>
      {!isOutpaint && <button className="btn btn-outline btn-block" onClick={onContinue}>결과 이어서 지우기</button>}
      <button className="btn btn-ghost btn-sm btn-block" onClick={onReEdit}>다시 편집</button>
      <FilePickButton label="새 사진" onPick={onPick} />
    </>
  )
}

function OutpaintControls({ expand, setExpand, error, busy, canRun, onRun, onPick }) {
  return (
    <>
      <div className="tool-group">
        <span className="tool-label">확장 비율 · {expand}%</span>
        <input type="range" min="10" max="60" value={expand} onChange={(e) => setExpand(+e.target.value)} />
      </div>
      {error && <p className="error">{error}</p>}
      <button className="btn btn-primary btn-block btn-run" onClick={onRun} disabled={!canRun || busy}>
        {busy ? '확장 중…' : <><Icon.sparkle width="18" height="18" /> 배경 확장</>}
      </button>
      <FilePickButton label="다른 사진" onPick={onPick} />
    </>
  )
}

function BrushControls({
  tool, setTool, brushSize, setBrushSize, strokes, redo,
  onUndo, onRedo, onClear, error, busy, canRun, onRun, onPick,
}) {
  return (
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
          <button className="btn btn-outline btn-sm" onClick={onUndo} disabled={!strokes.length}><Icon.undo /> 되돌리기</button>
          <button className="btn btn-outline btn-sm" onClick={onRedo} disabled={!redo.length}><Icon.redo /> 다시</button>
        </div>
        <button className="btn btn-ghost btn-sm btn-block" onClick={onClear} disabled={!strokes.length}>전체 지우기</button>
      </div>
      {error && <p className="error">{error}</p>}
      <button className="btn btn-primary btn-block btn-run" onClick={onRun} disabled={!canRun || busy || !strokes.length}>
        {busy ? '지우는 중…' : <><Icon.sparkle width="18" height="18" /> 지우기 실행</>}
      </button>
      <FilePickButton label="다른 사진" onPick={onPick} />
    </>
  )
}

// 스튜디오 우측 도구 패널 — 결과 화면이면 저장/복사, 아니면 모드별 편집 도구를 보여준다.
export default function StudioToolbar({ result, isOutpaint, outpaint, brush, resultActions }) {
  if (result) return <aside className="toolbar"><ResultActions result={result} {...resultActions} isOutpaint={isOutpaint} /></aside>
  return (
    <aside className="toolbar">
      {isOutpaint ? <OutpaintControls {...outpaint} /> : <BrushControls {...brush} />}
    </aside>
  )
}
