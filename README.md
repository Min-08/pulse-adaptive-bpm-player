[Arduino Sensors] → [Raspberry Pi Gateway]
↓
(Local SQI Filter)
↓
[AI Server / Cloud]
├ Quality Analysis (RandomForest)
├ Stress/Focus Classification (LightGBM)
└ BPM Decision (Rule / Bandit)
↓
[LLM Music Selector]
└ Spotify / YouTube API → Music Playback

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
│  │  └─ pulse_adaptive_bpm_player.ino  # 센서 스케치(초안)
│  └─ pi/
│     ├─ requirements.txt   # Pi 에이전트 의존성
│     └─ pi_agent.py        # 10초 요약→AI 서버 호출→재생지시
├─ ai-server/
│  ├─ requirements.txt      # FastAPI + ML
│  ├─ main.py               # /infer: SQI/상태/BPM 결정
│  └─ models/
│     └─ .gitkeep           # model.pkl 배치 경로
├─ llm-recommender/
│  ├─ requirements.txt
│  └─ llm_router.py         # /rerank: 후보 재정렬(초기 규칙 기반)
├─ playback/
│  ├─ requirements.txt
│  └─ server.py             # /set_target: Spotify/로컬 재생 컨트롤
└─ scripts/
   └─ run_dev.bat           # 윈도우 로컬 실행(3 서비스 기동)
