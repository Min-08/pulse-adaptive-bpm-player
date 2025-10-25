@echo off
setlocal ENABLEDELAYEDEXPANSION

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.."
set "PROJECT_ROOT=%CD%"

echo [run_dev] 프로젝트 루트: %PROJECT_ROOT%
echo [run_dev] .env 로드 시도 중...
if exist "%PROJECT_ROOT%\config\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /r "^[^#].*=" "%PROJECT_ROOT%\config\.env"`) do (
        if not "%%A"=="" (
            set "%%A=%%B"
        )
    )
) else (
    echo [run_dev] config\.env 파일이 없어 기본값으로 진행합니다.
)

echo [run_dev] AI Server, LLM Recommender, Playback 서비스를 각각 새 콘솔에서 시작합니다.
start "AI Server" cmd /k "cd /d %PROJECT_ROOT%\ai-server && python -m uvicorn main:app --host 0.0.0.0 --port 8000"
start "LLM Recommender" cmd /k "cd /d %PROJECT_ROOT%\llm-recommender && python -m uvicorn llm_router:app --host 0.0.0.0 --port 8010"
start "Playback Server" cmd /k "cd /d %PROJECT_ROOT%\playback && python -m uvicorn server:app --host 0.0.0.0 --port 8020"

echo [run_dev] 모든 서비스 실행 명령을 전송했습니다.
popd
endlocal
