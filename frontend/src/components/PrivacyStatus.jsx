import { Icon } from './icons.jsx'

function StepDot({ state, index }) {
  if (state === 'done') return <Icon.check />
  if (state === 'active') return <span className="pv-spin" />
  if (state === 'error') return '!'
  return index + 1
}

function ErrorNote({ detectError, onRetry }) {
  return (
    <div className="pv-note">
      <div>
        <b>{detectError ? '자동으로 찾지 못했어요' : '지우는 중 문제가 생겼어요'}</b>
        <p>
          {detectError
            ? '사진이 어둡거나 대상이 너무 작으면 못 찾을 수 있어요. 다른 사진으로 다시 시도하거나, 스튜디오에서 직접 칠해 지울 수 있어요.'
            : '일시적인 문제일 수 있어요. 잠시 후 다시 시도해 주세요.'}
        </p>
      </div>
      <button className="btn btn-outline btn-sm" onClick={onRetry}>다시 시도</button>
    </div>
  )
}

function Facts({ engine, metrics, done }) {
  const seconds = (ms) => (ms ? `${(ms / 1000).toFixed(1)}초` : '—')
  return (
    <div className="pv-facts">
      <span className="pv-fact"><b>{engine ?? '—'}</b>추론 엔진</span>
      <span className="pv-fact"><b>{seconds(metrics.detectMs)}</b>찾는 데 걸린 시간</span>
      <span className="pv-fact"><b>{seconds(metrics.restoreMs)}</b>지우고 복원한 시간</span>
      {metrics.engineName && done && <span className="pv-fact"><b>{metrics.engineName}</b>복원 엔진</span>}
    </div>
  )
}

// 처리 현황 — 진행 단계 · 요약 · 문제 안내를 사람이 읽기 좋게 보여준다.
export default function PrivacyStatus({ steps, detectError, removeError, onRetry, engine, metrics, done }) {
  return (
    <div className="pv-status">
      <ol className="pv-steps">
        {steps.map((s, i) => (
          <li key={s.key} className={`pv-step pv-step-${s.state}`}>
            <span className="pv-step-dot"><StepDot state={s.state} index={i} /></span>
            <span className="pv-step-body">
              <b>{s.label}</b>
              <small>{s.hint}</small>
            </span>
          </li>
        ))}
      </ol>

      {detectError || removeError
        ? <ErrorNote detectError={detectError} onRetry={onRetry} />
        : <Facts engine={engine} metrics={metrics} done={done} />}
    </div>
  )
}
