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
