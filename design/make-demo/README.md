# Make 데모 (GPT · Figma Make 산출물, 2026-10-30 동기화)

원본: `~/Documents/ChatGPT/Development/lash-pro-make/`. 영상(68MB)·얼굴 모델은 제외하고 화면 HTML, 인수인계 문서, 오프라인 렌더러, 속눈썹 PNG·제품 카드 이미지만 가져옴.

- `index.html` — 6화면 고객 플로우 데모 (랜딩 → 스튜디오 룩 → 카메라 → 라이브 → 보정 중 → 결과). 라이브는 미리 렌더한 영상.
- `render_tracked_lashes.py` — 오프라인 렌더러. 우리 앱에 가져온 것: 랜드마크 EMA(0.68/0.32), 눈 구멍(aperture) 복원으로 가림 처리, 알파 가장자리 블러, 따뜻한 차콜 톤.
- 실제 앱(`spikes/a-live-ar/`)은 이 플로우를 기준으로 재구성 (docs/07-flow.md).
