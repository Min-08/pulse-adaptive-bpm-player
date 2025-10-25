@echo off
echo Starting all services...

start cmd /k "cd ai-server && python -m venv .venv && call .venv\Scripts\activate && pip install -r requirements.txt && uvicorn main:app --reload --port 8000"
start cmd /k "cd llm-recommender && python -m venv .venv && call .venv\Scripts\activate && pip install -r requirements.txt && uvicorn llm_router:app --reload --port 8010"
start cmd /k "cd playback && python -m venv .venv && call .venv\Scripts\activate && pip install -r requirements.txt && uvicorn server:app --reload --port 8020"

echo All services launched.
pause
