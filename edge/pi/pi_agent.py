"""Raspberry Pi 에이전트 시뮬레이터."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

import numpy as np
import requests
from dotenv import load_dotenv, find_dotenv


def load_configuration() -> Dict[str, str]:
    """`.env` 파일을 로드하고 필수 URL을 반환한다."""
    env_path = find_dotenv()
    if env_path:
        load_dotenv(env_path)
    else:
        # TODO: 실제 배포 시 명시적인 경로를 지정하거나 예외 처리 로직 강화 필요
        load_dotenv()

    return {
        "ai_url": os.getenv("AI_SERVER_URL", "http://localhost:8000/infer"),
        "playback_url": os.getenv("PLAYBACK_SERVER_URL", "http://localhost:8020"),
    }


def configure_logger() -> logging.Logger:
    """콘솔 및 파일로 로깅하도록 설정한다."""
    logger = logging.getLogger("pi_agent")
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
        "data/pi_agent.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def generate_synthetic_features() -> Dict[str, Any]:
    """랜덤 노이즈를 포함한 합성 생체 특징을 생성한다."""
    global SESSION_SECONDS
    base_hr = random.uniform(60, 95)
    rmssd = random.uniform(15, 45)
    sdnn = rmssd * random.uniform(1.2, 1.8)
    pnn50 = max(0.0, min(0.5, np.clip(random.gauss(0.2, 0.1), 0.0, 0.6)))
    hr_std = random.uniform(2.5, 6.0)
    hr_z = random.uniform(-1.0, 1.0)
    acc_rms = abs(random.gauss(0.01, 0.005))
    sqi = max(0.0, min(1.0, random.uniform(0.3, 0.95)))
    time_of_day = random.choice(["morning", "afternoon", "evening", "night"])

    session_minutes = SESSION_SECONDS // 60
    SESSION_SECONDS += 10
    return {
        "device_id": "pi-01",
        "features": {
            "hr_mean": round(base_hr, 1),
            "hr_std": round(hr_std, 1),
            "rmssd": round(rmssd, 1),
            "sdnn": round(sdnn, 1),
            "pnn50": round(pnn50, 2),
            "hr_z": round(hr_z, 2),
            "acc_rms": round(acc_rms, 3),
            "sqi": round(sqi, 2),
            "time_of_day": time_of_day,
            "session_min": int(session_minutes),
        },
    }


def post_json(url: str, payload: Dict[str, Any], timeout: int = 10) -> Optional[Dict[str, Any]]:
    """POST 요청을 보내고 JSON 응답을 반환한다."""
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:  # pragma: no cover - 네트워크 의존
        LOGGER.warning("요청 실패: %s", exc)
    except json.JSONDecodeError as exc:  # pragma: no cover - 외부 응답 의존
        LOGGER.warning("JSON 파싱 실패: %s", exc)
    return None


SESSION_SECONDS = 0
LOGGER = configure_logger()


def main() -> None:
    """10초 주기로 합성 데이터를 AI 서버에 전송하고 플레이백을 호출한다."""
    config = load_configuration()
    ai_url = config["ai_url"]
    playback_url = config["playback_url"].rstrip("/") + "/set_target"

    LOGGER.info("Pi 에이전트 시뮬레이터 시작: AI=%s, Playback=%s", ai_url, playback_url)
    last_state: Optional[Dict[str, Any]] = None

    while True:
        payload = generate_synthetic_features()
        LOGGER.info("생성한 특징 전송: %s", payload)

        ai_response = post_json(ai_url, payload)
        if ai_response is None:
            LOGGER.warning("AI 서버 응답이 없어 직전 상태를 재사용한다")
            ai_response = last_state
        else:
            last_state = ai_response

        if ai_response is None:
            LOGGER.warning("사용 가능한 상태가 없어 다음 주기로 건너뜀")
        else:
            playback_payload = {
                "bpm_target": ai_response.get("bpm_target"),
                "state": ai_response.get("state"),
            }
            LOGGER.info("플레이백 호출: %s", playback_payload)
            playback_response = post_json(playback_url, playback_payload)
            LOGGER.info("플레이백 응답: %s", playback_response)

        LOGGER.info("다음 사이클까지 10초 대기")
        time.sleep(10)


if __name__ == "__main__":
    main()