# Pulse Adaptive BPM Player

Biofeedback 기반으로 음악 BPM을 적응적으로 조절하는 MVP 아키텍처.

## 구성 요약

```
[Arduino Sensors] → [Raspberry Pi Gateway]
↓
(Local SQI Filter)
↓
[AI Server / Cloud]
├ Quality Analysis (Rule 기반 SQI)
├ Stress/Focus Classification (Rule 기반)
└ BPM Decision (Rule 기반)
↓
[LLM Music Selector]
└ Spotify / YouTube API → Music Playback
```

```
pulse-adaptive-bpm-player/
├─ .gitignore
├─ config/
│  └─ .env.example          # 엔드포인트·API 키 템플릿
├─ data/
│  └─ .gitkeep              # 수집 로그/샘플 CSV 저장
├─ docs/
│  └─ ARCHITECTURE.md       # 시스템/데이터 계약 요약
├─ edge/
│  ├─ arduino/
│  │  └─ placeholder.ino    # 센서 스케치 자리표시자
│  └─ pi/
│     ├─ requirements.txt   # Pi 에이전트 의존성
│     └─ pi_agent.py        # 10초 요약→AI 서버 호출→재생지시
├─ ai-server/
│  ├─ requirements.txt      # FastAPI + 규칙 기반 추론
│  ├─ main.py               # /infer: SQI/상태/BPM 결정
│  └─ models/
│     └─ .gitkeep           # 모델 파일 배치 경로
├─ llm-recommender/
│  ├─ requirements.txt
│  └─ llm_router.py         # /rerank: 후보 재정렬(초기 규칙 기반)
├─ playback/
│  ├─ requirements.txt
│  ├─ config/
│  │  └─ state_params.yaml  # 상태별 Spotify 추천 파라미터
│  └─ server.py             # /set_target: Spotify 재생 트리거
└─ scripts/
   └─ run_dev.bat           # 윈도우 로컬 실행(3 서비스 기동)
```

## 로컬 실행 방법 (Windows)
1. Python 3.11 가상환경을 생성하고 `ai-server`, `llm-recommender`, `playback`, `edge/pi` 각 디렉터리에서 `pip install -r requirements.txt`를 실행한다.
2. `config/.env.example`을 복사해 `config/.env`를 만들고 필요한 값을 채운다.
3. PowerShell 또는 명령 프롬프트에서 `scripts\run_dev.bat`을 실행한다. 세 개의 콘솔 창이 열리며 AI/LLM/Playback 서버가 각각 8000/8010/8020 포트에서 기동된다.
4. 브라우저로 `http://localhost:8020/login`에 접속해 Spotify 계정을 한 번 승인한다. 토큰은 `playback/.tokens/spotify.json`에 저장된다.
5. librespot/raspotify 등 Spotify Connect 디바이스를 `.env`의 `SPOTIFY_DEVICE_NAME`과 동일한 이름으로 실행한다.
6. 별도 콘솔에서 `python edge\pi\pi_agent.py`를 실행하면 10초 주기로 합성 생체 특징이 전송되고, 추론 결과가 Playback 서비스에 전달되는 것을 확인할 수 있다.

## 현재 구현 현황 및 TODO
- AI 서버: 규칙 기반으로 상태/BPM을 반환하며, 실제 모델(`ai-server/models/state_lgbm.pkl`)을 로드하도록 확장 필요
- LLM 재랭커: 기본 BPM 차이 기준 정렬만 제공하며, 추후 LLM API 연동이 필요
- Playback 서비스: Spotify OAuth·추천·디바이스 탐색·재생을 지원하며, YouTube/로컬 재생 폴백 로직은 아직 TODO
- Edge Pi 에이전트: 합성 데이터를 주기적으로 전송하며, 실제 센서 연동(시리얼/BLE) 구현이 남아 있음

1. Python 3.11 가상환경을 생성하고 `ai-server`, `llm-recommender`, `playback`, `edge/pi` 각 디렉터리에서 `pip install -r requirements.txt` 실행.
2. `config/.env.example`을 복사해 `config/.env`를 만들고 필요한 값을 채움.
3. PowerShell/명령 프롬프트에서 `scripts\run_dev.bat` 실행 → AI/LLM/Playback 서버가 각각 8000/8010/8020 포트에서 기동됨.
4. 브라우저에서 `http://localhost:8020/login` 접속해 **Spotify 계정 승인**. 토큰은 `playback/.tokens/spotify.json`에 저장됨.
5. `librespot/raspotify` 등 **Spotify Connect 디바이스**를 `.env`의 `SPOTIFY_DEVICE_NAME`과 동일한 이름으로 실행.
6. 별도 콘솔에서 `python edge\pi\pi_agent.py` 실행 → 10초 주기로 합성 생체 특징 전송 및 추론 결과가 Playback 서비스로 전달되는지 확인.

## 현재 구현 현황 및 TODO

* **AI 서버**: 규칙 기반으로 상태/BPM을 반환. 추후 실제 모델(`ai-server/models/state_lgbm.pkl`) 로딩으로 확장.
* **LLM 재랭커**: 현재 BPM 차이 기준 기본 정렬. 추후 LLM API 연동으로 분위기/장르 기반 재랭킹.
* **Playback 서비스**: Spotify OAuth·추천·디바이스 탐색·재생 지원. YouTube/로컬 재생 폴백 로직은 TODO.
* **Edge Pi 에이전트**: 합성 데이터 주기 전송. 실제 센서 연동(시리얼/BLE) 미구현.

## 향후 연동 TODO

* `ai-server/models/state_lgbm.pkl` 모델 로딩으로 규칙 기반 추론 대체.
* `llm-recommender`에서 OpenAI/Anthropic LLM 호출을 통해 분위기·장르 기반 재랭킹 구현.
* `playback` 서비스에 Spotify Connect 외 YouTube API, 로컬 파일 재생 로직 추가.