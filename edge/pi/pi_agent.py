"""Raspberry Pi 에이전트 시뮬레이터."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
from dotenv import load_dotenv, find_dotenv

try:
    import serial  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - pyserial 미설치 대비
    serial = None  # type: ignore[assignment]


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
        "device_id": os.getenv("PI_DEVICE_ID", "pi-01"),
        "sensor_mode": os.getenv("PI_SENSOR_MODE", "synthetic"),
        "serial_port": os.getenv("PI_SERIAL_PORT", "/dev/ttyACM0"),
        "serial_baudrate": int(os.getenv("PI_SERIAL_BAUDRATE", "115200")),
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


SESSION_START = time.time()
LOGGER = configure_logger()


def infer_time_of_day() -> str:
    """현재 시각 기준 시간대를 추정한다."""

    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


class SyntheticFeatureSource:
    """합성 데이터 기반 특징 생성기."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self.elapsed_seconds = 0

    def read(self) -> Dict[str, Any]:
        base_hr = random.uniform(60, 95)
        rmssd = random.uniform(15, 45)
        sdnn = rmssd * random.uniform(1.2, 1.8)
        pnn50 = max(0.0, min(0.5, np.clip(random.gauss(0.2, 0.1), 0.0, 0.6)))
        hr_std = random.uniform(2.5, 6.0)
        hr_z = random.uniform(-1.0, 1.0)
        acc_rms = abs(random.gauss(0.01, 0.005))
        sqi = max(0.0, min(1.0, random.uniform(0.3, 0.95)))
        session_minutes = self.elapsed_seconds // 60
        self.elapsed_seconds += 10
        return {
            "device_id": self.device_id,
            "features": {
                "hr_mean": round(base_hr, 1),
                "hr_std": round(hr_std, 1),
                "rmssd": round(rmssd, 1),
                "sdnn": round(sdnn, 1),
                "pnn50": round(pnn50, 2),
                "hr_z": round(hr_z, 2),
                "acc_rms": round(acc_rms, 3),
                "sqi": round(sqi, 2),
                "time_of_day": infer_time_of_day(),
                "session_min": int(session_minutes),
            },
        }


class SerialFeatureSource:
    """시리얼 포트에서 전달되는 아두이노 요약 데이터를 읽는다."""

    def __init__(self, port: str, baudrate: int, device_id: str) -> None:
        self.port = port
        self.baudrate = baudrate
        self.device_id = device_id
        self._serial: Optional[Any] = None
        self._connect()

    @property
    def ready(self) -> bool:
        return self._serial is not None

    def _connect(self) -> None:
        if serial is None:
            LOGGER.warning("pyserial이 설치되어 있지 않아 합성 데이터로 대체합니다.")
            return
        try:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=1)
            LOGGER.info("시리얼 포트 연결 성공: %s", self.port)
        except Exception as exc:  # pragma: no cover - 하드웨어 의존
            LOGGER.error("시리얼 포트 연결 실패: %s", exc)
            self._serial = None

    def _normalize_payload(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        features = payload.get("features") or payload
        required = {"hr_mean", "hr_std", "rmssd", "sdnn", "pnn50", "hr_z", "acc_rms", "sqi"}
        if not required.issubset(features.keys()):
            LOGGER.warning("시리얼 데이터에 필수 키가 부족해 무시합니다: %s", features)
            return None
        if "session_min" not in features:
            elapsed = int((time.time() - SESSION_START) / 60)
            features["session_min"] = elapsed
        if "time_of_day" not in features:
            features["time_of_day"] = infer_time_of_day()
        return {
            "device_id": payload.get("device_id", self.device_id),
            "features": features,
        }

    def read(self) -> Optional[Dict[str, Any]]:
        if self._serial is None:
            return None
        try:
            raw = self._serial.readline().decode("utf-8").strip()
        except Exception as exc:  # pragma: no cover - 하드웨어 의존
            LOGGER.warning("시리얼 포트 읽기 실패: %s", exc)
            return None
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.warning("시리얼 데이터가 JSON 형식이 아닙니다: %s", raw)
            return None
        return self._normalize_payload(payload)


def prepare_sources(config: Dict[str, Any]) -> Tuple[Any, SyntheticFeatureSource]:
    device_id = config.get("device_id", "pi-01")
    synthetic = SyntheticFeatureSource(device_id)
    mode = config.get("sensor_mode", "synthetic").lower()
    if mode == "serial":
        port = config.get("serial_port", "/dev/ttyACM0")
        baudrate = int(config.get("serial_baudrate", 115200))
        serial_source = SerialFeatureSource(port, baudrate, device_id)
        if serial_source.ready:
            return serial_source, synthetic
        LOGGER.warning("시리얼 데이터 원본을 사용할 수 없어 합성 데이터를 사용합니다.")
    return synthetic, synthetic


def main() -> None:
    """10초 주기로 합성 데이터를 AI 서버에 전송하고 플레이백을 호출한다."""
    config = load_configuration()
    ai_url = config["ai_url"]
    playback_url = config["playback_url"].rstrip("/") + "/set_target"
    device_id = config.get("device_id", "pi-01")
    primary_source, synthetic_source = prepare_sources(config)

    LOGGER.info(
        "Pi 에이전트 시작: device=%s, AI=%s, Playback=%s, sensor_mode=%s",
        device_id,
        ai_url,
        playback_url,
        config.get("sensor_mode", "synthetic"),
    )
    last_state: Optional[Dict[str, Any]] = None

    while True:
        payload = primary_source.read()
        if payload is None:
            if primary_source is not synthetic_source:
                LOGGER.warning("센서 데이터를 읽지 못해 합성 데이터를 사용합니다")
            payload = synthetic_source.read()
        LOGGER.info("전송할 특징: %s", payload)

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