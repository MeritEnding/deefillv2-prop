import { PLANS } from '../lib/account.js'

// 무료 한도 소진 시 뜨는 업그레이드 모달.
export default function Paywall({ onClose, onBuy }) {
  const packs = PLANS.filter((p) => p.id !== 'free')
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <button className="modal-x" onClick={onClose} aria-label="닫기">
          ✕
        </button>
        <span className="eyebrow">오늘의 무료 사용을 모두 썼어요</span>
        <h3 className="modal-title">Pro로 계속 지우기</h3>
        <p className="modal-sub">
          크레딧을 충전하면 <b>HD 원본</b>을 <b>워터마크 없이</b> 저장할 수 있어요.
        </p>
        <div className="modal-packs">
          {packs.map((p) => (
            <button
              key={p.id}
              className={`pack ${p.highlight ? 'pack-hi' : ''}`}
              onClick={() => onBuy(p)}
            >
              <div className="pack-top">
                <span className="pack-name">{p.name}</span>
                {p.highlight && <span className="pack-badge">인기</span>}
              </div>
              <div className="pack-price">₩{p.price.toLocaleString()}</div>
              <div className="pack-sub">
                크레딧 {p.credits}장 · 장당 ₩{Math.round(p.price / p.credits)}
              </div>
            </button>
          ))}
        </div>
        <p className="modal-foot">데모용 모의 결제 · 실제 청구 없음</p>
      </div>
    </div>
  )
}
