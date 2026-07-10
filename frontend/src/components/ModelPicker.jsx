// 인페인팅 모델(버전) 선택기 — 여러 엔진 중 사용할 모델을 사용자가 고른다.
// /api/engines 카탈로그를 받아 드롭다운으로 보여주고, 사용 불가 모델은 비활성 처리한다.
export default function ModelPicker({ engines, value, onChange }) {
  if (!engines?.length) return null
  const current = engines.find((e) => e.id === value)
  return (
    <label className="model-picker" title={current?.desc ?? '인페인팅에 사용할 모델을 선택하세요'}>
      <span className="model-picker-cap">모델</span>
      <select
        className="model-picker-select"
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
      >
        {engines.map((e) => (
          <option key={e.id} value={e.id} disabled={!e.available}>
            {e.label} · {e.version}{e.available ? '' : ' (사용 불가)'}
          </option>
        ))}
      </select>
    </label>
  )
}
