import { PLANS } from '../lib/account.js'

export default function Pricing({ acc, onBuy, navigate }) {
  return (
    <main className="pricing">
      <div className="section-head" style={{ paddingTop: 56 }}>
        <span className="eyebrow">요금제</span>
        <h2>필요한 만큼만, 크레딧으로.</h2>
        <p className="section-sub">
          한 장 지울 때마다 크레딧 1장. 남은 크레딧은 사라지지 않습니다.
        </p>
      </div>

      <div className="plans">
        {PLANS.map((p) => (
          <article className={`plan ${p.highlight ? 'plan-hi' : ''}`} key={p.id}>
            {p.highlight && <span className="plan-badge">인기</span>}
            <h3 className="plan-name">{p.name}</h3>
            <p className="plan-tag">{p.tagline}</p>
            <div className="plan-price">
              {p.price === 0 ? (
                <span className="plan-free">₩0</span>
              ) : (
                <>
                  <span className="plan-amt">₩{p.price.toLocaleString()}</span>
                  <span className="plan-unit"> / {p.credits}크레딧</span>
                </>
              )}
            </div>
            <ul className="plan-feats">
              {p.features.map((f) => (
                <li key={f}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path d="m5 12 5 5L20 7" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  {f}
                </li>
              ))}
            </ul>
            {p.id === 'free' ? (
              <button className="btn btn-ghost btn-block" onClick={() => navigate('/editor')}>
                {p.cta}
              </button>
            ) : (
              <button
                className={`btn btn-block ${p.highlight ? 'btn-primary' : 'btn-outline'}`}
                onClick={() => onBuy(p)}
              >
                {p.cta}
              </button>
            )}
          </article>
        ))}
      </div>

      <p className="pricing-foot">
        · 데모용 모의 결제입니다. 실제 청구는 발생하지 않습니다.
        {acc.plan === 'pro' && ` 현재 보유 크레딧: ${acc.credits}장`}
      </p>
    </main>
  )
}
