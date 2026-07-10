import { useEffect, useState } from 'react'

const STORE_KEY = 'jium-community-posts'

// 데모용 시드 글 — 실제 글은 브라우저(localStorage)에만 저장된다
const SEED = [
  { id: 1, name: '마케터K', title: '상품컷 40장 워터마크 10분 만에 정리했어요', body: '일괄 지우기에 40장 올리고 첫 장에만 칠했는데 전부 처리됐습니다. 포토샵 배치 액션보다 훨씬 간단하네요.', at: '2026-07-09 20:14', likes: 12 },
  { id: 2, name: '안전관리자', title: '현장 사진 보고용으로 쓰고 있습니다', body: '작업자 얼굴이 자동으로 잡혀서 체크만 하면 끝. 모자이크가 아니라서 보고서가 깔끔합니다. 번호판 인식도 기대 중.', at: '2026-07-09 18:02', likes: 9 },
  { id: 3, name: '지우팬', title: '스마트 클릭 진짜 편하네요', body: '전봇대 클릭 한 번에 영역을 딱 잡아줘서 놀랐어요. 애매하게 잡히면 붓으로 조금만 보정하면 됩니다.', at: '2026-07-08 22:41', likes: 7 },
]

export default function Community() {
  const [posts, setPosts] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) ?? SEED } catch { return SEED }
  })
  const [open, setOpen] = useState(null)
  const [form, setForm] = useState({ name: '', title: '', body: '' })

  useEffect(() => { localStorage.setItem(STORE_KEY, JSON.stringify(posts)) }, [posts])

  const submit = (e) => {
    e.preventDefault()
    if (!form.title.trim() || !form.body.trim()) return
    const post = {
      id: Date.now(),
      name: form.name.trim() || '익명',
      title: form.title.trim(),
      body: form.body.trim(),
      at: new Date().toLocaleString('ko-KR', { hour12: false }).replace(/\. /g, '-').replace('.', ''),
      likes: 0,
    }
    setPosts((p) => [post, ...p])
    setForm({ name: '', title: '', body: '' })
    setOpen(post.id)
  }

  const like = (id) => setPosts((p) => p.map((x) => (x.id === id ? { ...x, likes: x.likes + 1 } : x)))

  return (
    <main className="studio page-narrow">
      <div className="section-head">
        <h2 className="sec-title">Community <i>|</i> <small>커뮤니티</small></h2>
        <p className="section-sub">사용 후기와 활용법을 나눠주세요. 글은 이 브라우저에만 저장됩니다.</p>
      </div>

      <form className="comm-form" onSubmit={submit}>
        <div className="comm-form-row">
          <input value={form.name} maxLength={12} placeholder="닉네임 (선택)" onChange={(e) => setForm({ ...form, name: e.target.value })} />
          <input value={form.title} maxLength={60} placeholder="제목" onChange={(e) => setForm({ ...form, title: e.target.value })} required />
        </div>
        <textarea value={form.body} maxLength={600} rows={3} placeholder="내용을 적어주세요" onChange={(e) => setForm({ ...form, body: e.target.value })} required />
        <button className="btn btn-dark btn-sm" disabled={!form.title.trim() || !form.body.trim()}>글 올리기</button>
      </form>

      <ul className="notice-list">
        {posts.map((p) => (
          <li key={p.id} className={open === p.id ? 'open' : ''}>
            <button className="notice-row" onClick={() => setOpen(open === p.id ? null : p.id)}>
              <span className="rtag">{p.name}</span>
              <strong>{p.title}</strong>
              <time>{p.at}</time>
            </button>
            {open === p.id && (
              <div className="notice-body">
                <p>{p.body}</p>
                <button className="comm-like" onClick={() => like(p.id)}>공감 {p.likes}</button>
              </div>
            )}
          </li>
        ))}
      </ul>
      <p className="research-note">데모 게시판입니다 — 글은 서버가 아니라 사용 중인 브라우저에만 저장됩니다. 시드 글은 예시입니다.</p>
    </main>
  )
}
