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
│  └─ server.py             # /set_target: 재생 트리거(모킹)
└─ scripts/
   └─ run_dev.bat           # 윈도우 로컬 실행(3 서비스 기동)
```

## 로컬 실행 방법 (Windows)
1. Python 3.11 가상환경을 생성하고 `ai-server`, `llm-recommender`, `playback`, `edge/pi` 각 디렉터리에서 `pip install -r requirements.txt`를 실행한다.
2. `config/.env.example`을 복사해 `config/.env`를 만들고 필요한 값을 채운다.
3. PowerShell 또는 명령 프롬프트에서 `scripts\run_dev.bat`을 실행한다. 세 개의 콘솔 창이 열리며 AI/LLM/Playback 서버가 각각 8000/8010/8020 포트에서 기동된다.
4. 별도 콘솔에서 `python edge\pi\pi_agent.py`를 실행하면 10초 주기로 합성 생체 특징이 전송되고, 추론 결과가 Playback 서비스에 전달되는 것을 확인할 수 있다.

## 향후 연동 TODO
- `ai-server/models/state_lgbm.pkl` 모델을 로딩해 규칙 기반 추론을 대체
- `llm-recommender`에서 OpenAI/Anthropic LLM을 호출해 분위기·장르 기반 재랭킹 구현
- `playback` 서비스에서 Spotify Connect, YouTube API, 로컬 파일 재생 로직 추가
