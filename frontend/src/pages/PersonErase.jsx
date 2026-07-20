import { useEffect, useState } from 'react'
import BeforeAfter from '../components/BeforeAfter.jsx'
import { Icon } from '../components/icons.jsx'
import { downloadBlob } from '../lib/image.js'
import { postForm } from '../lib/api.js'

const MIN_COVERAGE = 0.005 // 이보다 작으면 "사람 없음"으로 안내

// 마스크 PNG(255=사람)를 파란 반투명 하이라이트 dataURL로 변환
async function maskToTint(maskBlob) {
  const bmp = await createImageBitmap(maskBlob)
  const canvas = document.createElement('canvas')
  canvas.width = bmp.width
  canvas.height = bmp.height
  const ctx = canvas.getContext('2d')
  ctx.drawImage(bmp, 0, 0)
  const src = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const out = ctx.createImageData(canvas.width, canvas.height)
  for (let i = 0; i < src.data.length; i += 4) {
    if (src.data[i] > 127) {
      out.data[i] = 37; out.data[i + 1] = 99; out.data[i + 2] = 235; out.data[i + 3] = 128
    }
  }
  ctx.putImageData(out, 0, 0)
  return canvas.toDataURL()
}

// 사람 전체 지우기: 업로드 → 실루엣 세그멘테이션 → 하이라이트 확인 → 제거·배경 복원
export default function PersonErase({ engine }) {
  const [imageFile, setImageFile] = useState(null)
  const [imageURL, setImageURL] = useState(null)
  const [natural, setNatural] = useState(null)
  const [phase, setPhase] = useState('idle') // idle | finding | ready | removing | done
  const [maskBlob, setMaskBlob] = useState(null)
  const [tintURL, setTintURL] = useState(null)
  const [coverage, setCoverage] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)

  // 샘플 체험 (#/privacy/person?demo=1)
  const loadSample = async () => {
    try {
      const blob = await (await fetch('/sample-face.png')).blob()
      loadFile(new File([blob], 'sample.png', { type: 'image/png' }))
    } catch { setError('샘플 이미지를 불러오지 못했어요.') }
  }
  useEffect(() => {
    if (window.location.hash.includes('demo=1')) loadSample()
  }, []) // eslint-disable-line

  const loadFile = async (file) => {
    if (!file || !file.type?.startsWith('image/')) return
    setMaskBlob(null); setTintURL(null); setResult(null); setError(null); setCoverage(0)
    setImageFile(file)
    const url = URL.createObjectURL(file)
    setImageURL(url)
    const img = new Image()
    img.onload = () => setNatural({ w: img.naturalWidth, h: img.naturalHeight })
    img.src = url
    runSegment(file)
  }

  const runSegment = async (file) => {
    setPhase('finding'); setError(null)
    try {
      const form = new FormData()
      form.append('image', file, 'image.png')
      const res = await postForm('/api/segment-people', form)
      const cov = Number(res.headers.get('X-Person-Coverage')) || 0
      const blob = await res.blob()
      setCoverage(cov)
      if (cov >= MIN_COVERAGE) {
        setMaskBlob(blob)
        setTintURL(await maskToTint(blob))
      }
      setPhase('ready')
    } catch (e) {
      setError(e.message ?? '사람 영역을 찾지 못했어요.')
      setPhase('ready')
    }
  }

  const runRemove = async () => {
    if (!imageFile || !maskBlob || phase === 'removing') return
    setPhase('removing'); setError(null)
    try {
      const form = new FormData()
      form.append('image', imageFile, 'image.png')
      form.append('mask', maskBlob, 'mask.png')
      if (engine) form.append('engine', engine)
      const res = await postForm('/api/inpaint', form)
      const blob = await res.blob()
      const elapsed = Number(res.headers.get('X-Elapsed-Ms')) || null
      setResult({ url: URL.createObjectURL(blob), blob, elapsed })
      setPhase('done')
    } catch (e) {
      setError(e.message ?? '제거에 실패했어요.')
      setPhase('ready')
    }
  }

  const download = () => result && downloadBlob(result.blob, `jium-person-${Date.now()}.png`)
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
          <div className="tile tile-sm"><Icon.person /></div>
          <div>
            <h2>사람 전체 지우기</h2>
            <p>사람 실루엣을 통째로 찾아 지우고, 그 자리는 원래 배경으로 자연스럽게 메웁니다.</p>
          </div>
        </div>
        <span className={`pill ${engine ? 'pill-ok' : 'pill-down'}`}>
          {engine ? `엔진: ${engine}` : '추론 서버 연결 안 됨'}
        </span>
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
              <img className="dropzone-mascot" src="/mascot-select.png" alt="" />
              <p className="dropzone-title">사진을 올리면 사람을 찾아드려요</p>
              <p className="dropzone-sub">매물 사진 · 현장 사진 속 행인 — 실루엣 전체를 자동으로 잡아 표시합니다</p>
              <span className="btn btn-outline btn-sm" onClick={(e) => { e.preventDefault(); loadSample() }}>샘플 사진으로 체험</span>
            </label>
          )}

          {imageURL && phase !== 'done' && (
            <div className="stage stage-result">
              <div className="pv-frame" style={frameStyle}>
                <img src={imageURL} alt="원본" />
                {tintURL && <img className="pv-tint" src={tintURL} alt="사람 영역" />}
                {(phase === 'finding' || phase === 'removing') && (
                  <div className="stage-busy">
                    <div className="scanline" />
                    <img src={phase === 'finding' ? '/mascot-select.png' : '/mascot-erase.png'} alt="" />
                    <span>{phase === 'finding' ? '사람 실루엣을 찾는 중…' : '지우고 배경을 메우는 중…'}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {phase === 'done' && result && (
            <div className="stage stage-result">
              <BeforeAfter before={imageURL} after={result.url} beforeLabel="원본" afterLabel="완료" initial={15} style={frameStyle} />
            </div>
          )}
        </div>

        <aside className="toolbar">
          {phase === 'done' && result ? (
            <>
              <div className="tool-group">
                <img className="result-mascot" src="/mascot-perfect.png" alt="" />
                <span className="tool-label tool-label-done"><Icon.check /> 다 지웠어요</span>
                <p className="result-desc">사람 영역을 지우고 그 자리를 배경으로 복원했어요.</p>
                {result.elapsed && <p className="result-meta">처리 {(result.elapsed / 1000).toFixed(1)}초</p>}
              </div>
              <button className="btn btn-primary btn-block" onClick={download}><Icon.download /> 이미지 저장</button>
              <label className="btn btn-ghost btn-sm btn-block">새 사진<input type="file" accept="image/*" hidden onChange={(e) => newImage(e.target.files[0])} /></label>
            </>
          ) : (
            <>
              <div className="tool-group">
                <span className="tool-label">찾아낸 사람 영역</span>
                {!imageURL && <p className="result-desc">사진을 올리면 자동으로 찾아드립니다.</p>}
                {imageURL && phase === 'ready' && maskBlob && (
                  <p className="result-desc">파랗게 칠해진 부분이 지워집니다 — 화면의 <strong>{Math.round(coverage * 100)}%</strong></p>
                )}
                {imageURL && phase === 'ready' && !maskBlob && !error && (
                  <p className="result-desc">
                    사람을 찾지 못했어요. 다른 대상은 <a href="#/privacy">항목 선택</a>이나 <a href="#/studio">스튜디오</a>에서 지울 수 있어요.
                  </p>
                )}
              </div>
              {error && <p className="error">{error}</p>}
              <button
                className="btn btn-primary btn-block btn-run"
                onClick={runRemove}
                disabled={!maskBlob || phase !== 'ready'}
              >
                {phase === 'removing' ? '지우는 중…' : '사람 지우기'}
              </button>
              {imageURL && <label className="btn btn-ghost btn-sm btn-block">다른 사진<input type="file" accept="image/*" hidden onChange={(e) => newImage(e.target.files[0])} /></label>}
            </>
          )}
        </aside>
      </div>
    </main>
  )
}
