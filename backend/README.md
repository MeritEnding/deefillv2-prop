# LHNet Backend

지움(LHNet) 플랫폼의 Spring Boot API 게이트웨이.

## 엔드포인트
- `POST /api/inpaint` — 이미지+마스크 인페인팅 프록시 (`X-Engine`, `X-Elapsed-Ms` 헤더)
- `POST /api/detect` — 얼굴 자동 탐지 프록시
- `POST /api/segment` — 클릭 좌표 기반 객체 분리 프록시
- `POST /api/v1/redact` — 기업 연동용 원콜 비식별화 (`X-API-Key` 인증, `X-Redacted-Count`)
- `GET /api/v1/usage` — 키별 사용량 조회
- `GET /api/health` — 추론 서비스 상태·엔진 이름

## 실행
```bash
./mvnw spring-boot:run   # 8080 — 추론 서비스(lhnet-model)가 8000에 떠 있어야 함
```

환경변수
- `API_KEYS` — 공개 API 키 목록(콤마 구분), 기본 `jium-demo-key`
