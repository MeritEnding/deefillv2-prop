# LHNet Frontend

지움(LHNet) 웹 앱 — React + Vite.

## 화면
- `/#/` — 랜딩 (비식별 플랫폼 소개, 복원 갤러리, 연구 지표, 연동 API)
- `/#/privacy` — **비식별화 워크플로** (메인): 업로드 → 얼굴 자동 탐지 → 항목 선택 → 제거·자연 복원
- `/#/studio` — 수동 스튜디오: 객체 지우개(스마트 클릭·브러시), 워터마크 제거, 사진 복원, 일괄 지우기, 배경 확장 등

## 실행
```bash
npm install
npm run dev      # 5173 — 백엔드(lhnet-backend)가 8080에 떠 있어야 함
```

환경변수: `VITE_API_BASE` (기본 `http://localhost:8080`)
