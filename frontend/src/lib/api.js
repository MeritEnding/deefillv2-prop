// 백엔드 API 클라이언트 — 서버 주소와 에러 처리를 한 곳에서 관리한다.
// 컴포넌트에서는 fetch를 직접 쓰지 말고 이 모듈을 통한다 (docs/CODE_STYLE.md).

export const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8080'

// 실패 응답이면 백엔드 표준 에러 JSON의 message로 Error를 던진다.
async function ensureOk(res) {
  if (res.ok) return res
  const msg = await res.json().then((d) => d.message).catch(() => null)
  throw new Error(msg ?? `서버 오류 (${res.status})`)
}

/** GET 후 JSON 파싱. 실패 시 서버 message로 Error. */
export async function getJSON(path, options) {
  const res = await ensureOk(await fetch(`${API_BASE}${path}`, options))
  return res.json()
}

/** FormData POST. 응답 본문(blob·json)과 헤더를 쓸 수 있게 Response를 그대로 반환. */
export async function postForm(path, form, options) {
  return ensureOk(await fetch(`${API_BASE}${path}`, { method: 'POST', body: form, ...options }))
}
