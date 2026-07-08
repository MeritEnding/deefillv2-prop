// 무료/Pro 플랜과 크레딧 상태를 localStorage로 관리한다. (데모용 모의 결제)
const KEY = 'jium.account.v1'
const FREE_DAILY_LIMIT = 3
const FREE_MAX_DIM = 720 // 무료는 720p로 다운스케일 + 워터마크

function today() {
  return new Date().toISOString().slice(0, 10)
}

function defaults() {
  return { plan: 'free', credits: 0, freeUsedDate: today(), freeUsed: 0 }
}

export function loadAccount() {
  let acc
  try {
    acc = { ...defaults(), ...JSON.parse(localStorage.getItem(KEY) || '{}') }
  } catch {
    acc = defaults()
  }
  if (acc.freeUsedDate !== today()) {
    acc.freeUsedDate = today()
    acc.freeUsed = 0
  }
  return acc
}

export function saveAccount(acc) {
  localStorage.setItem(KEY, JSON.stringify(acc))
  return acc
}

export const account = {
  FREE_DAILY_LIMIT,
  FREE_MAX_DIM,

  freeRemaining(acc) {
    return Math.max(0, FREE_DAILY_LIMIT - acc.freeUsed)
  },

  // 이번 실행이 크레딧을 쓸지(HD·워터마크 없음) 무료 슬롯을 쓸지 판단
  canRun(acc) {
    if (acc.plan === 'pro' && acc.credits > 0) return { ok: true, mode: 'hd' }
    if (this.freeRemaining(acc) > 0) return { ok: true, mode: 'free' }
    if (acc.credits > 0) return { ok: true, mode: 'hd' }
    return { ok: false, mode: null }
  },

  consume(acc, mode) {
    const next = { ...acc }
    if (mode === 'hd') next.credits = Math.max(0, next.credits - 1)
    else next.freeUsed = next.freeUsed + 1
    return saveAccount(next)
  },

  addCredits(acc, n) {
    return saveAccount({ ...acc, credits: acc.credits + n, plan: 'pro' })
  },

  reset(acc) {
    return saveAccount({ ...defaults() })
  },
}

// 크레딧 팩 (모의 결제)
export const PLANS = [
  {
    id: 'free',
    name: 'Free',
    price: 0,
    unit: '',
    tagline: '가볍게 써보기',
    features: ['하루 3장 무료', '최대 720p 저장', '워터마크 포함', '표준 처리 속도'],
    cta: '무료로 시작',
    highlight: false,
  },
  {
    id: 'pro20',
    name: 'Pro 20',
    price: 4900,
    unit: '원',
    credits: 20,
    tagline: '가장 인기 있는 선택',
    features: ['크레딧 20장', 'HD 원본 저장', '워터마크 없음', '우선 처리', '상업적 이용 허용'],
    cta: '20장 구매',
    highlight: true,
  },
  {
    id: 'pro100',
    name: 'Pro 100',
    price: 19000,
    unit: '원',
    credits: 100,
    tagline: '자주 쓰는 셀러·크리에이터',
    features: ['크레딧 100장', 'HD 원본 저장', '워터마크 없음', '우선 처리', 'API 액세스(베타)'],
    cta: '100장 구매',
    highlight: false,
  },
]
