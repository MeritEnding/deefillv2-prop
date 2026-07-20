import { Icon } from './icons.jsx'
import { TYPE_LABEL } from '../lib/privacyKinds.js'

function FilePickButton({ label, onPick }) {
  return (
    <label className="btn btn-ghost btn-sm btn-block">
      {label}
      <input type="file" accept="image/*" hidden onChange={(e) => onPick(e.target.files[0])} />
    </label>
  )
}

function DoneActions({ result, selectedCount, onDownload, onReselect, onPick }) {
  return (
    <>
      <div className="tool-group">
        <img className="result-mascot" src="/mascot-perfect.png" alt="" />
        <span className="tool-label tool-label-done"><Icon.check /> 다 지웠어요</span>
        <p className="result-desc">{selectedCount}개 항목을 지우고 그 자리를 배경으로 복원했어요.</p>
        {result.elapsed && <p className="result-meta">처리 {(result.elapsed / 1000).toFixed(1)}초</p>}
      </div>
      <button className="btn btn-primary btn-block" onClick={onDownload}><Icon.download /> 이미지 저장</button>
      <button className="btn btn-ghost btn-sm btn-block" onClick={onReselect}>다시 선택</button>
      <FilePickButton label="새 사진" onPick={onPick} />
    </>
  )
}

function DetectionList({ dets, label, onToggle }) {
  if (!dets) return <p className="result-desc">사진을 올리면 자동으로 찾아드립니다.</p>
  if (!dets.length) {
    return (
      <p className="result-desc">
        자동으로 찾은 {label}이(가) 없어요. 다른 개인정보는
        <a href="#/privacy"> 항목 선택</a>이나 <a href="#/studio">스튜디오</a>에서 지울 수 있어요.
      </p>
    )
  }
  return (
    <ul className="pv-list">
      {dets.map((d, i) => (
        <li key={i}>
          <label>
            <input type="checkbox" checked={d.on} onChange={() => onToggle(i)} />
            <span>{d.label ?? TYPE_LABEL[d.type] ?? d.type} {i + 1}</span>
            <span className="pv-score" title={`신뢰도 ${Math.round(d.score * 100)}%`}>
              <i style={{ width: `${Math.round(d.score * 100)}%` }} />
            </span>
            <em>{Math.round(d.score * 100)}%</em>
          </label>
        </li>
      ))}
    </ul>
  )
}

// 비식별화 우측 패널 — 완료 화면이면 저장, 아니면 탐지 목록·선택·실행을 보여준다.
export default function PrivacyToolbar({
  done, result, meta, dets, selectedCount, phase, error, hasImage,
  onToggle, onSetAll, onRun, onDownload, onReselect, onPick,
}) {
  if (done && result) {
    return (
      <aside className="toolbar">
        <DoneActions
          result={result}
          selectedCount={selectedCount}
          onDownload={onDownload}
          onReselect={onReselect}
          onPick={onPick}
        />
      </aside>
    )
  }
  return (
    <aside className="toolbar">
      <div className="tool-group">
        <span className="tool-label">찾아낸 {meta.label} {dets ? `· ${dets.length}건` : ''}</span>
        <DetectionList dets={dets} label={meta.label} onToggle={onToggle} />
      </div>
      {dets && dets.length > 1 && (
        <div className="btn-row">
          <button className="btn btn-outline btn-sm" onClick={() => onSetAll(true)}>전체 선택</button>
          <button className="btn btn-outline btn-sm" onClick={() => onSetAll(false)}>전체 해제</button>
        </div>
      )}
      {error && <p className="error">{error}</p>}
      <button
        className="btn btn-primary btn-block btn-run"
        onClick={onRun}
        disabled={!selectedCount || phase === 'detecting' || phase === 'removing'}
      >
        {phase === 'removing' ? '지우는 중…' : `${selectedCount || 0}개 지우기`}
      </button>
      {hasImage && <FilePickButton label="다른 사진" onPick={onPick} />}
    </aside>
  )
}
