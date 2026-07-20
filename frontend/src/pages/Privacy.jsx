import { useEffect, useRef, useState } from 'react'
import BeforeAfter from '../components/BeforeAfter.jsx'
import { Icon } from '../components/icons.jsx'
import ModelPicker from '../components/ModelPicker.jsx'
import PrivacyStatus from '../components/PrivacyStatus.jsx'
import PrivacyToolbar from '../components/PrivacyToolbar.jsx'
import UploadDropzone from '../components/UploadDropzone.jsx'
import { downloadBlob } from '../lib/image.js'
import { postForm } from '../lib/api.js'
import { KINDS, buildDetectionMask } from '../lib/privacyKinds.js'

// 비식별화 워크플로: 업로드 → 자동 탐지 → 항목 선택 → 제거·자연 복원 → 저장
export default function Privacy({ engine, engines, setEngine, kind = 'face' }) {
  const meta = KINDS[kind]
  const [imageFile, setImageFile] = useState(null)
  const [imageURL, setImageURL] = useState(null)
  const [natural, setNatural] = useState(null)
  const [dets, setDets] = useState(null) // [{type, box, score, on}]
  const [phase, setPhase] = useState('idle') // idle | detecting | ready | removing | done
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [metrics, setMetrics] = useState({}) // {detectMs, restoreMs}

  const imgElRef = useRef(null)

  const reset = () => { setDets(null); setResult(null); setError(null); setPhase('idle'); setMetrics({}) }

  // 샘플 체험 (#/privacy/face?demo=1) — 번들된 예시 사진으로 흐름을 바로 보여준다
  const loadSample = async () => {
    if (!meta.sample) return
    try {
      const blob = await (await fetch(meta.sample)).blob()
      loadFile(new File([blob], 'sample.png', { type: 'image/png' }))
    } catch { setError('샘플 이미지를 불러오지 못했어요.') }
  }
  useEffect(() => {
    if (window.location.hash.includes('demo=1')) loadSample()
  }, []) // eslint-disable-line

  const loadFile = async (file) => {
    if (!file || !file.type?.startsWith('image/')) return
    reset()
    setImageFile(file)
    const url = URL.createObjectURL(file)
    setImageURL(url)
    const img = new Image()
    img.onload = () => {
      setNatural({ w: img.naturalWidth, h: img.naturalHeight })
    }
    img.src = url
    runDetect(file)
  }

  const runDetect = async (file) => {
    setPhase('detecting'); setError(null)
    const started = performance.now()
    try {
      const form = new FormData()
      form.append('image', file, 'image.png')
      form.append('targets', kind)
      const res = await postForm('/api/detect', form)
      const data = await res.json()
      const ms = performance.now() - started
      setMetrics((m) => ({ ...m, detectMs: ms }))
      const matched = data.detections.filter((d) => d.type === kind)
      setDets(matched.map((d) => ({ ...d, on: true })))
      setPhase('ready')
    } catch (e) {
      setError(e.message ?? '자동 탐지에 실패했어요.')
      setDets([])
      setPhase('ready')
    }
  }

  const toggle = (i) => setDets((ds) => ds.map((d, j) => (j === i ? { ...d, on: !d.on } : d)))
  const setAll = (on) => setDets((ds) => ds.map((d) => ({ ...d, on })))
  const selected = (dets || []).filter((d) => d.on)

  const runRemove = async () => {
    if (!imageFile || !natural || !selected.length || phase === 'removing') return
    setPhase('removing'); setError(null)
    try {
      const maskBlob = await buildDetectionMask(natural, selected)
      const form = new FormData()
      form.append('image', imageFile, 'image.png')
      form.append('mask', maskBlob, 'mask.png')
      if (engine) form.append('engine', engine)
      const res = await postForm('/api/inpaint', form)
      const blob = await res.blob()
      const elapsed = Number(res.headers.get('X-Elapsed-Ms')) || null
      const engineName = res.headers.get('X-Engine')
      setMetrics((m) => ({ ...m, restoreMs: elapsed, engineName }))
      setResult({ url: URL.createObjectURL(blob), blob, elapsed })
      setPhase('done')
    } catch (e) {
      setError(e.message ?? '제거에 실패했어요.')
      setPhase('ready')
    }
  }

  // 처리 진행을 3단계로 요약 — 로그 나열 대신 지금 어디까지 왔는지 한눈에 보여준다
  const detectError = Boolean(error) && (dets?.length ?? 0) === 0
  const removeError = Boolean(error) && (dets?.length ?? 0) > 0
  const steps = [
    { key: 'up', label: '사진 업로드', hint: natural ? `${natural.w}×${natural.h}` : '준비 중', state: 'done' },
    {
      key: 'find', label: `${meta.label} 자동 찾기`,
      hint: phase === 'detecting' ? '찾는 중…' : detectError ? '실패' : dets ? `${dets.length}건 찾음` : '대기',
      state: phase === 'detecting' ? 'active' : detectError ? 'error' : dets != null ? 'done' : 'wait',
    },
    {
      key: 'erase', label: '지우고 배경 복원',
      hint: phase === 'removing' ? '지우는 중…' : phase === 'done' && result ? `${selected.length}건 완료` : removeError ? '실패' : '대기',
      state: phase === 'removing' ? 'active' : phase === 'done' && result ? 'done' : removeError ? 'error' : 'wait',
    },
  ]

  const download = () => result && downloadBlob(result.blob, `jium-redacted-${Date.now()}.png`)
  const newImage = (file) => {
    if (!file || !file.type?.startsWith('image/')) return
    setImageURL(null); setNatural(null); loadFile(file)
  }

  const frameStyle = natural
    ? { aspectRatio: `${natural.w} / ${natural.h}`, maxWidth: `${Math.round(560 * (natural.w / natural.h))}px` }
    : undefined

  return (
    <main className="studio privacy">
      <div className="studio-head">
        <div className="studio-title">
          <div className="tile tile-sm"><Icon.shield /></div>
          <div>
            <h2>{meta.title} {meta.beta && <span className="rtag">베타</span>}</h2>
            <p>{meta.desc}</p>
          </div>
        </div>
        {engines?.length
          ? <ModelPicker engines={engines} value={engine} onChange={setEngine} />
          : <span className="pill pill-down">추론 서버 연결 안 됨</span>}
      </div>

      <div className="studio-grid">
        <div
          className={`stage-wrap ${dragOver ? 'drag' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); newImage(e.dataTransfer.files[0]) }}
        >
          {!imageURL && (
            <UploadDropzone title="사진을 올리면 바로 찾아드려요" sub={meta.dropSub} onPick={newImage}>
              {meta.sample && (
                <span className="btn btn-outline btn-sm" onClick={(e) => { e.preventDefault(); loadSample() }}>샘플 사진으로 체험</span>
              )}
            </UploadDropzone>
          )}

          {imageURL && phase !== 'done' && (
            <div className="stage stage-result">
              <div className="pv-frame" style={frameStyle}>
                <img ref={imgElRef} src={imageURL} alt="원본" />
                {natural && (dets || []).map((d, i) => {
                  const [x, y, w, h] = d.box
                  return (
                    <button
                      key={i}
                      className={`pv-box ${d.on ? 'on' : ''}`}
                      style={{
                        left: `${(x / natural.w) * 100}%`,
                        top: `${(y / natural.h) * 100}%`,
                        width: `${(w / natural.w) * 100}%`,
                        height: `${(h / natural.h) * 100}%`,
                      }}
                      onClick={() => toggle(i)}
                      title={d.on ? '클릭하면 제외' : '클릭하면 포함'}
                    >
                      <em>{d.on ? '제거' : '유지'}</em>
                    </button>
                  )
                })}
                {phase === 'detecting' && (
                  <div className="stage-busy">
                    <div className="scanline" />
                    <img src="/mascot-select.png" alt="" />
                    <span>개인정보를 찾는 중…</span>
                  </div>
                )}
                {phase === 'removing' && (
                  <div className="stage-busy">
                    <div className="scanline" />
                    <img src="/mascot-erase.png" alt="" />
                    <span>지우고 배경을 메우는 중…</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {phase === 'done' && result && (
            <div className="stage stage-result">
              <BeforeAfter
                before={imageURL}
                after={result.url}
                beforeLabel="원본"
                afterLabel="완료"
                initial={15}
                style={frameStyle}
              />
            </div>
          )}
        </div>

        <PrivacyToolbar
          done={phase === 'done'}
          result={result}
          meta={meta}
          dets={dets}
          selectedCount={selected.length}
          phase={phase}
          error={error}
          hasImage={Boolean(imageURL)}
          onToggle={toggle}
          onSetAll={setAll}
          onRun={runRemove}
          onDownload={download}
          onReselect={() => { setResult(null); setPhase('ready') }}
          onPick={newImage}
        />
      </div>

      {imageURL && (
        <PrivacyStatus
          steps={steps}
          detectError={detectError}
          removeError={removeError}
          onRetry={() => imageFile && (detectError ? runDetect(imageFile) : runRemove())}
          engine={engine}
          metrics={metrics}
          done={phase === 'done'}
        />
      )}
    </main>
  )
}
