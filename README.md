# lash-tryon (가칭)

속눈썹 샵의 디자인이나 마음에 드는 레퍼런스 사진을 **내 눈에 입혀보고** 예약 여부를 결정하게 해주는 서비스.

- 소비자: 셀카 + 레퍼런스 → 내 눈에 적용한 결과 확인
- 샵: 자기 디자인을 올려두면 고객이 방문 전에 입혀보고 예약으로 넘어옴 (B2B2C)

## 문서

| 문서 | 내용 |
|---|---|
| [01-concept](docs/01-concept.md) | 서비스 구조, 가설, 유저 플로우, 양면 구조 |
| [02-research](docs/02-research.md) | 국내·해외·일본·대만 경쟁/시장 리서치 |
| [03-mvp](docs/03-mvp.md) | MVP 기능 범위, 뺄 것, 기술 고려사항 |
| [04-marketing](docs/04-marketing.md) | 초기 마케팅 계획 (채널, 콘텐츠, 바이럴 루프, 지표) |
| [05-costs-roadmap](docs/05-costs-roadmap.md) | 예상 비용, 단계별 로드맵 |
| [06-decisions](docs/06-decisions.md) | 결정 로그 + 열린 질문 |
| [ideas/](ideas/) | 러프 아이디어 원문 + 리뷰 기록 (날짜별) |

## 지금 단계

**Phase 0 — 품질 검증.** 셀카 + 레퍼런스로 납득되는 합성이 나오는지 먼저 확인한다. 이게 안 나오면 나머지는 의미 없음.

## 스파이크

- `spikes/a-live-ar/` — 방식 A 실시간 파라미터 렌더링 데모. MediaPipe Face Landmarker + 2D 캔버스 가닥 렌더링. 컬 J/C/D, 구간별 길이 5개, 볼륨 3단계, 프리셋 3종.
  - 레퍼런스 사진 → 래시 맵 번역: `server.py`의 `/api/lashmap`이 OpenAI 비전(gpt-4.1, json_schema)으로 컬·구간 길이·볼륨·스타일 JSON을 뽑아 바로 적용.
  - 실행: `python3 spikes/a-live-ar/server.py` → http://localhost:8787 (Chrome/Safari, 카메라 허용). 키는 `spikes/a-live-ar/.env`의 `OPENAI_API_KEY` 하나로 번역·합성 다 됨. `GEMINI_API_KEY`는 비교용 선택.
  - 사진 업로드 모드도 됨 (두 번째 Choose File)
