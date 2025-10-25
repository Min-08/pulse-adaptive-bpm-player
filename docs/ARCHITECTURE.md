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
- Playback 서비스는 Spotify Connect 강화 및 로컬 라이브러리 재생 품질 개선을 목표로 한다

## Spotify 연동 흐름
- Playback 서비스는 `/login` → `/callback` OAuth 플로우로 Spotify 토큰을 확보하고 `playback/.tokens/spotify.json`에 저장한다.
- 상태별 추천 파라미터는 `playback/config/state_params.yaml`에서 로드되며, 필요 시 `/config/reload` 엔드포인트로 런타임에 갱신할 수 있다.
- `/set_target` 호출 시 상태·BPM에 따라 Spotify Recommendations API를 조회하고, LLM 재랭커(옵션)와 쿨다운 필터를 거쳐 Connect 디바이스(`SPOTIFY_DEVICE_NAME`)로 재생을 시작한다.
- 재생 결과는 `data/playback_log.csv`에 기록되며, Spotify 디바이스를 찾지 못하거나 재생이 실패하면 로컬 폴백 모드로 응답한다.

## 로컬 백업 음악 흐름
- `LOCAL_MUSIC_DIR`에는 `library.csv`와 MP3 파일이 저장되며, CSV의 `track_name`·`file_name`·실제 파일이 모두 일치해야 재생 후보로 인정된다.
- Playback 서비스는 CSV를 로드하여 상태 태그와 BPM을 기준으로 후보를 선별하고, Spotify 실패 또는 `PLAYBACK_MODE=local`일 때 로컬 엔진으로 곡을 재생한다.
- 로컬 재생 엔진은 MP3 존재 여부를 재확인한 뒤, `playsound`(사용 가능 시) 또는 운영체제 기본 플레이어를 호출하여 실제 음악을 실행한다.

## AI 설정 및 정책 관리
- `ai-server/config/model.yaml`에 SQI 하한, RMSSD/HR 기반 규칙 임계치, 상태 유지 최소 시간(히스테리시스), BPM 스무딩 계수(α), 상태별 기본 BPM을 선언한다.
- AI 서버는 `/config/reload` 호출 시 YAML을 재로딩하고, `ai-server/models/state_lgbm.pkl`이 존재하면 LightGBM 모델 추론을 시도하며 실패 시 규칙 기반으로 폴백한다.
- 디바이스별 최근 상태·BPM을 메모리에 유지하여 최소 유지 시간 이전에는 상태 전환을 막고, `alpha` 기반 지수 스무딩으로 BPM을 점진적으로 변경한다.

## Spotify 파라미터 조정 가이드
- `playback/config/state_params.yaml`에서 상태별 장르, 에너지·밸런스 범위, 템포 기본값을 수정하면 코드 수정 없이 추천 동작을 조정할 수 있다.
- `/config/reload` 엔드포인트를 호출하면 YAML과 로컬 라이브러리를 다시 읽어 변경 사항이 즉시 반영된다.
- LLM 재랭커를 사용하려면 `.env`에 `LLM_RECOMMENDER_URL`을 지정하고, 응답이 없을 경우 템포 차이 기반 규칙 정렬로 자동 폴백한다.

## 센서 입력 및 시뮬레이터
- `edge/pi/pi_agent.py`는 `PI_SENSOR_MODE=serial` 설정 시 아두이노에서 출력되는 JSON 라인을 실시간으로 읽어 AI 서버에 전달한다.
- JSON에는 `device_id`와 `features`가 포함되며, `features`에는 `hr_mean`, `rmssd`, `sdnn`, `sqi` 등 AI 서버 요구 필드가 모두 존재해야 한다.
- 시리얼 입력이 없거나 파싱에 실패하면 합성 데이터 생성기로 즉시 폴백하여 데이터 스트림이 끊기지 않는다.

## LLM 기반 후보 생성·재탐색
- Playback 서버는 OpenAI ChatGPT API(`OPENAI_API_KEY`)를 이용해 상태·BPM에 맞는 곡 제목과 아티스트 목록을 JSON으로 받는다.
- 1차 후보가 Spotify에서 재생되지 않으면 `OPENAI_SMALL_MODEL`을 사용해 재검색 프롬프트를 보내고, 그래도 실패하면 로컬 예비 곡으로 전환한다.
- 호출 개수와 시간 제한은 `.env`의 `PRIMARY_LLM_TRACK_LIMIT`, `RETRY_LLM_TRACK_LIMIT`, `OPENAI_TIMEOUT`으로 조정 가능하다.

## 개인화 BPM 정책
- AI 서버는 `personalization` 설정에 기반한 LinUCB 컨텍스추얼 밴딧을 통해 상태/시간대/심박 특징을 컨텍스트로 사용해 BPM 보정을 학습한다.
- 학습된 파라미터는 `data/bandit_state.json`에 저장되어 재시작 후에도 누적 학습이 유지되며, 최소 샘플(`min_reward_samples`)이 쌓이기 전에는 기본 BPM 정책을 사용한다.
- 보상은 최근 RMSSD와 상태에 기반하여 계산되며, `load_config`·`/config/reload` 호출 시 밴딧 설정도 함께 갱신된다.
