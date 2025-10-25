# 시스템 아키텍처 개요

## 개념 흐름
- 엣지 레이어(Raspberry Pi, Arduino)가 생체신호를 수집하고 10초마다 요약 특징을 생성
- AI 서버는 품질 검증과 상태 추론을 수행한 뒤 목표 BPM과 태그를 반환
- (선택) LLM 재랭커가 추천 후보 곡을 BPM 기반으로 재정렬
- 플레이백 서비스는 목표 BPM과 상태를 받아 음악 재생을 트리거
- 모든 주요 이벤트는 `data/` 하위 로그 파일에 회전 저장

## 서비스 포트
- AI Server: `http://localhost:8000`
- LLM Recommender: `http://localhost:8010`
- Playback Server: `http://localhost:8020`

## 데이터 계약 요약
- Pi → AI `/infer`: 생체 특징(JSON) → 상태/확신도/목표 BPM/태그
- AI → LLM `/rerank`: 상태, 목표 BPM, 후보 곡 목록 → 재정렬된 ID 목록
- Pi → Playback `/set_target`: 목표 BPM과 상태 → 재생 모드 및 성공 여부

## 구성 및 로그
- `.env` 파일에서 각 서비스 URL과 API 키를 로드하며, 값이 없으면 안전하게 기본 동작으로 폴백
- 각 FastAPI 앱은 콘솔 + 회전 파일 로깅을 수행하며 로그는 `data/` 디렉터리에 저장
- `scripts/run_dev.bat`는 세 개의 서비스를 개별 콘솔에서 동시에 실행하도록 구성

## 실행 순서
1. `config/.env.example`을 복사해 `.env`를 생성하고 필요한 값을 채움
2. `scripts\run_dev.bat`을 실행해 AI, LLM, Playback 서버를 띄움
3. `edge\pi\pi_agent.py`를 실행하면 10초 주기로 시뮬레이션 데이터가 흐르고 엔드투엔드 시나리오를 확인 가능

## 미래 확장
- `ai-server/models/state_lgbm.pkl` 경로에 LightGBM 모델을 배치해 추론 로직을 대체 예정
- LLM 재랭커는 OpenAI/Anthropic API 호출을 통해 분위기/장르 기반 정렬을 지원 예정
- Playback 서비스는 Spotify Connect, YouTube API, 로컬 라이브러리 재생으로 확장될 예정

## Spotify 연동 흐름
- Playback 서비스는 `/login` → `/callback` OAuth 플로우로 Spotify 토큰을 확보하고 `playback/.tokens/spotify.json`에 저장한다.
- 상태별 추천 파라미터는 `playback/config/state_params.yaml`에서 로드되며, 필요 시 `/config/reload` 엔드포인트로 런타임에 갱신할 수 있다.
- `/set_target` 호출 시 상태·BPM에 따라 Spotify Recommendations API를 조회하고, LLM 재랭커(옵션)와 쿨다운 필터를 거쳐 Connect 디바이스(`SPOTIFY_DEVICE_NAME`)로 재생을 시작한다.
- 재생 결과는 `data/playback_log.csv`에 기록되며, Spotify 디바이스를 찾지 못하거나 재생이 실패하면 로컬 폴백 모드로 응답한다.
=======