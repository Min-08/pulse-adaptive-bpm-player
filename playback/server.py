"""음악 재생 트리거용 FastAPI 서버 (현재는 모킹)."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

ENV_PATH = find_dotenv()
if ENV_PATH:
    load_dotenv(ENV_PATH)
else:
    load_dotenv()


def configure_logger() -> logging.Logger:
    """콘솔 및 회전 파일 로그를 설정한다."""
    logger = logging.getLogger("playback")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    os.makedirs("data", exist_ok=True)
    file_handler = RotatingFileHandler(
        "data/playback.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


LOGGER = configure_logger()


class PlaybackRequest(BaseModel):
    bpm_target: int = Field(..., ge=0)
    state: str = Field(..., min_length=1)


class PlaybackResponse(BaseModel):
    ok: bool
    mode: str
    track_uri: Optional[str] = None


def determine_mode() -> str:
    """환경 변수 기반 재생 모드를 반환한다."""
    mode = os.getenv("PLAYBACK_MODE", "noop").lower()
    if mode not in {"spotify", "youtube", "local"}:
        return "noop"
    return mode


def mock_select_track(bpm_target: int, state: str) -> Optional[str]:
    """간단한 규칙으로 가상 트랙 URI를 만든다."""
    if bpm_target <= 0:
        return None
    return f"mock://{state.lower()}-{bpm_target}"


def trigger_playback(mode: str, request: PlaybackRequest) -> Optional[str]:
    """재생 트리거 로직 (현재는 모킹)."""
    LOGGER.info("[Playback] 상태: %s / 목표 BPM: %d / 모드: %s", request.state, request.bpm_target, mode)
    if mode == "spotify":
        # TODO: Spotify 기기 검색 및 재생 제어 로직을 구현한다.
        pass
    elif mode == "youtube":
        # TODO: YouTube API 연동으로 재생을 제어한다.
        pass
    elif mode == "local":
        # TODO: 로컬 MP3 디렉터리를 탐색해 재생하도록 구현한다.
        pass
    return mock_select_track(request.bpm_target, request.state)


app = FastAPI(title="Playback Server", version="0.1.0")


@app.post("/set_target", response_model=PlaybackResponse)
def set_target(request: PlaybackRequest) -> PlaybackResponse:
    """목표 BPM과 상태를 받아 재생을 트리거한다."""
    mode = determine_mode()
    track_uri = trigger_playback(mode, request)
    response = PlaybackResponse(ok=True, mode=mode, track_uri=track_uri)
    LOGGER.info("플레이백 응답: %s", response.model_dump())
    return response


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """상태 확인 엔드포인트."""
    return {"status": "ok"}
