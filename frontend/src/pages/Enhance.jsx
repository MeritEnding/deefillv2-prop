import { useEffect, useState } from 'react'
import BeforeAfter from '../components/BeforeAfter.jsx'
import { Icon } from '../components/icons.jsx'
import { downloadBlob } from '../lib/image.js'
import { postForm } from '../lib/api.js'

// 화질 복원 모드 — 업로드하면 자동 처리하고 전후를 비교해 보여준다
export const ENHANCE_MODES = [
  {
    id: 'face',
    label: '얼굴 복원',
    icon: 'sparkle',
    endpoint: '/api/restore',
    countHeader: 'X-Restored-Count',
    desc: '흐리거나 저화질인 얼굴을 AI가 또렷하게 되살립니다. 오래된 사진, 저장하며 뭉개진 사진에 쓰세요.',
    dropSub: '오래된 가족사진 · 저화질 캡처 — 얼굴을 찾아 자동으로 선명하게 복원합니다',
    sample: '/sample-restore.png',
    done: ({ count }) => (count > 0 ? `얼굴 ${count}개를 복원했어요.` : '복원할 얼굴을 찾지 못해 원본 그대로예요.'),
  },
  {
    id: 'upscale',
    label: '고화질 확대',
    icon: 'expand',
    endpoint: '/api/upscale',
    countHeader: null,
    scalable: true,
    desc: '작은 사진을 초해상도 AI로 2배·4배 키웁니다. 그냥 늘리는 것보다 결이 살아있어요.',
    dropSub: '작은 프로필 · 오래된 스캔 — 픽셀을 채워 넣으며 선명하게 키웁니다',
    sample: '/sample-face.png',
    done: ({ scale, natural }) =>
      natural ? `${scale}배로 키웠어요 — ${natural.w}×${natural.h} → ${natural.w * scale}×${natural.h * scale}` : `${scale}배로 키웠어요.`,
  },
]

export default function Enhance({ engine, modeId = 'face', navigate }) {
  const mode = ENHANCE_MODES.find((m) => m.id === modeId) ?? ENHANCE_MODES[0]
  const [imageFile, setImageFile] = useState(null)
  const [imageURL, setImageURL] = useState(null)
  const [natural, setNatural] = useState(null)
  const [phase, setPhase] = useState('idle') // idle | working | done
  const [result, setResult] = useState(null) // {url, blob, count, elapsed}
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [scale, setScale] = useState(2)

  // 샘플 체험 (#/enhance?demo=1)
  const loadSample = async () => {
    if (!mode.sample) return
    try {
      const blob = await (await fetch(mode.sample)).blob()
      loadFile(new File([blob], 'sample.png', { type: 'image/png' }))
    } catch { setError('샘플 이미지를 불러오지 못했어요.') }
  }
  useEffect(() => {
    if (window.location.hash.includes('demo=1')) loadSample()
  }, []) // eslint-disable-line

  const loadFile = async (file) => {
    if (!file || !file.type?.startsWith('image/')) return
    setResult(null); setError(null)
    setImageFile(file)
    const url = URL.createObjectURL(file)
    setImageURL(url)
    const img = new Image()
    img.onload = () => setNatural({ w: img.naturalWidth, h: img.naturalHeight })
    img.src = url
    run(file)
  }

  const run = async (file, useScale = scale) => {
    setPhase('working'); setError(null)
    try {
      const form = new FormData()
      form.append('image', file, 'image.png')
      if (mode.scalable) form.append('scale', useScale)
      const res = await postForm(mode.endpoint, form)
      const blob = await res.blob()
      setResult({
        url: URL.createObjectURL(blob),
        blob,
        count: mode.countHeader ? Number(res.headers.get(mode.countHeader)) || 0 : 0,
        scale: useScale,
        elapsed: Number(res.headers.get('X-Elapsed-Ms')) || null,
      })
      setPhase('done')
    } catch (e) {
      setError(e.message ?? '처리에 실패했어요.')
      setPhase('idle')
    }
  }

  const changeScale = (s) => {
    setScale(s)
    if (imageFile) run(imageFile, s)
  }

  const download = () => result && downloadBlob(result.blob, `jium-enhance-${Date.now()}.png`)
  const newImage = (file) => {
    if (!file || !file.type?.startsWith('image/')) return
    setImageURL(null); setNatural(null); loadFile(file)
  }
  const frameStyle = natural
    ? { aspectRatio: `${natural.w} / ${natural.h}`, maxWidth: `${Math.round(560 * (natural.w / natural.h))}px` }
    : undefined
  const ModeIcon = Icon[mode.icon] ?? Icon.sparkle

  return (
    <main className="studio privacy">
      <div className="studio-head">
        <div className="studio-title">
          <div className="tile tile-sm"><ModeIcon /></div>
          <div>
            <h2>{mode.label}</h2>
            <p>{mode.desc}</p>
          </div>
        </div>
        <span className={`pill ${engine ? 'pill-ok' : 'pill-down'}`}>
          {engine ? `엔진: ${engine}` : '추론 서버 연결 안 됨'}
        </span>
      </div>

      <div className="enhance-tabs">
        {ENHANCE_MODES.map((m) => (
          <button
            key={m.id}
            className={`etab ${m.id === mode.id ? 'on' : ''}`}
            onClick={() => navigate?.(`/enhance?mode=${m.id}`)}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="studio-grid">
        <div
          className={`stage-wrap ${dragOver ? 'drag' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); newImage(e.dataTransfer.files[0]) }}
        >
          {!imageURL && (
            <label className="dropzone">
              <input type="file" accept="image/*" hidden onChange={(e) => newImage(e.target.files[0])} />
              <img className="dropzone-mascot" src="/mascot-restore.png" alt="" />
              <p className="dropzone-title">사진을 올리면 바로 살려드려요</p>
              <p className="dropzone-sub">{mode.dropSub}</p>
              {mode.sample && (
                <span className="btn btn-outline btn-sm" onClick={(e) => { e.preventDefault(); loadSample() }}>샘플 사진으로 체험</span>
              )}
            </label>
          )}

          {imageURL && phase === 'working' && (
            <div className="stage stage-result">
              <div className="pv-frame" style={frameStyle}>
                <img src={imageURL} alt="원본" />
                <div className="stage-busy">
                  <div className="scanline" />
                  <img src="/mascot-restore.png" alt="" />
                  <span>되살리는 중…</span>
                </div>
              </div>
            </div>
          )}

          {phase === 'done' && result && (
            <div className="stage stage-result">
              <BeforeAfter before={imageURL} after={result.url} beforeLabel="원본" afterLabel={mode.scalable ? '확대' : '복원'} initial={50} style={frameStyle} />
            </div>
          )}
        </div>

        <aside className="toolbar">
          {phase === 'done' && result ? (
            <>
              <div className="tool-group">
                <img className="result-mascot" src="/mascot-perfect.png" alt="" />
                <span className="tool-label tool-label-done"><Icon.check /> 다 됐어요</span>
                <p className="result-desc">{mode.done({ count: result.count, scale: result.scale, natural })}</p>
                {result.elapsed && <p className="result-meta">처리 {(result.elapsed / 1000).toFixed(1)}초</p>}
              </div>
              {mode.scalable && (
                <div className="btn-row">
                  {[2, 4].map((s) => (
                    <button key={s} className={`btn btn-sm ${scale === s ? 'btn-dark' : 'btn-outline'}`} onClick={() => changeScale(s)}>{s}배</button>
                  ))}
                </div>
              )}
              <button className="btn btn-primary btn-block" onClick={download}><Icon.download /> 이미지 저장</button>
              <label className="btn btn-ghost btn-sm btn-block">새 사진<input type="file" accept="image/*" hidden onChange={(e) => newImage(e.target.files[0])} /></label>
            </>
          ) : (
            <>
              <div className="tool-group">
                <span className="tool-label">{mode.label}</span>
                <p className="result-desc">
                  {phase === 'working' ? '처리 중이에요 — 잠시만요.' : '사진을 올리면 자동으로 처리됩니다.'}
                </p>
              </div>
              {mode.scalable && (
                <div className="btn-row">
                  {[2, 4].map((s) => (
                    <button key={s} className={`btn btn-sm ${scale === s ? 'btn-dark' : 'btn-outline'}`} onClick={() => changeScale(s)}>{s}배</button>
                  ))}
                </div>
              )}
              {error && <p className="error">{error}</p>}
              {imageURL && <label className="btn btn-ghost btn-sm btn-block">다른 사진<input type="file" accept="image/*" hidden onChange={(e) => newImage(e.target.files[0])} /></label>}
            </>
          )}
        </aside>
      </div>
    </main>
  )
}
