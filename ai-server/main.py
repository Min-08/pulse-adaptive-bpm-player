"""생체신호 기반 상태 추론 FastAPI 서버."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field

try:
    from joblib import load as joblib_load
except Exception:  # pragma: no cover - joblib 미설치 대비
    joblib_load = None

"""생체신호 기반 상태 추론용 FastAPI 서버."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import List, Tuple

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

# 환경 변수 로드
ENV_PATH = find_dotenv()
if ENV_PATH:
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
Path("data").mkdir(exist_ok=True)
logger.add(
    "data/ai_server.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    rotation="1 MB",
    retention=5,
    encoding="utf-8",
    enqueue=True,
)

CONFIG_PATH = Path("ai-server/config/model.yaml")
MODEL_PATH = Path("ai-server/models/state_lgbm.pkl")
DEFAULT_CONFIG: Dict[str, Any] = {
    "sqi": {"min_valid": 0.5},
    "rules": {
        "rmssd_low": 20,
        "rmssd_high": 35,
        "hr_flow_min": 90,
        "hr_flow_max": 110,
        "hr_z_high": 0.7,
    },
    "hysteresis": {"min_hold_seconds": 120},
    "smoothing": {"alpha": 0.15},
    "bpm_policy": {
        "HighStress": 75,
        "Focus": 100,
        "Flow": 120,
        "LowFocus": 125,
    },
}
CONFIG_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "data": DEFAULT_CONFIG}
MODEL = None
MODEL_AVAILABLE = False

def configure_logger() -> logging.Logger:
    """콘솔 및 회전 파일 핸들러를 갖는 로거를 생성한다."""
    logger = logging.getLogger("ai_server")
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
        "data/ai_server.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


LOGGER = configure_logger()

class FeatureSet(BaseModel):
    hr_mean: float = Field(..., description="최근 10초 평균 심박수")
    hr_std: float = Field(..., ge=0, description="심박수 표준편차")
    rmssd: float = Field(..., ge=0, description="RMSSD 값")
    sdnn: float = Field(..., ge=0, description="SDNN 값")
    pnn50: float = Field(..., ge=0, le=1, description="pNN50 비율")
    hr_z: float = Field(..., description="심박수 Z-점수")
    acc_rms: float = Field(..., ge=0, description="가속도 RMS")
    sqi: float = Field(..., ge=0, le=1, description="Signal Quality Index")
    time_of_day: str = Field(..., description="시간대 라벨")
    session_min: int = Field(0, ge=0, description="세션 진행 시간(분)")

class InferenceRequest(BaseModel):
    device_id: str = Field(..., min_length=1, description="디바이스 식별자")
    features: FeatureSet


class InferenceResponse(BaseModel):
    state: str
    confidence: float = Field(..., ge=0, le=1)
    bpm_target: Optional[int] = Field(default=None, ge=0)
    tags: list[str]


@dataclass
class DeviceState:
    last_state: str
    last_state_change: float
    last_bpm: float
    last_update: float


DEVICE_REGISTRY: Dict[str, DeviceState] = {}


def load_config(force: bool = False) -> Dict[str, Any]:
    if not force and CONFIG_CACHE["data"] is not None:
        return CONFIG_CACHE["data"]
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        CONFIG_CACHE.update({"loaded_at": time.time(), "data": data})
        logger.info("AI 설정을 로드했습니다: {}", CONFIG_PATH)
        return data
    except FileNotFoundError:
        logger.warning("model.yaml을 찾지 못해 기본값으로 동작합니다: {}", CONFIG_PATH)
    except yaml.YAMLError as exc:
        logger.warning("model.yaml 파싱 실패로 기본값을 사용합니다: {}", exc)
    CONFIG_CACHE.update({"loaded_at": time.time(), "data": DEFAULT_CONFIG})
    return DEFAULT_CONFIG


def load_model() -> None:
    global MODEL, MODEL_AVAILABLE
    if not MODEL_PATH.exists() or joblib_load is None:
        logger.info("LightGBM 모델이 없어 규칙 기반으로 동작합니다.")
        MODEL = None
        MODEL_AVAILABLE = False
        return
    try:
        MODEL = joblib_load(MODEL_PATH)
        MODEL_AVAILABLE = True
        logger.info("LightGBM 모델을 로드했습니다: {}", MODEL_PATH)
    except Exception as exc:  # pragma: no cover - 모델 로딩 실패 대비
        logger.error("모델 로드 실패로 규칙 기반으로 폴백합니다: {}", exc)
        MODEL = None
        MODEL_AVAILABLE = False


def quality_gate(features: FeatureSet, config: Dict[str, Any]) -> bool:
    min_sqi = float(config.get("sqi", {}).get("min_valid", 0.5))
    return features.sqi >= min_sqi


def rule_based_state(features: FeatureSet, config: Dict[str, Any]) -> str:
    rules = config.get("rules", {})
    rmssd_low = float(rules.get("rmssd_low", 20))
    rmssd_high = float(rules.get("rmssd_high", 35))
    hr_flow_min = float(rules.get("hr_flow_min", 90))
    hr_flow_max = float(rules.get("hr_flow_max", 110))
    hr_z_high = float(rules.get("hr_z_high", 0.7))

    if features.rmssd < rmssd_low or (features.hr_z > hr_z_high and features.rmssd < rmssd_low + 2):
        return "HighStress"
    if features.rmssd > rmssd_high and features.hr_z < 0.3:
        return "LowFocus"
    if hr_flow_min <= features.hr_mean <= hr_flow_max:
        return "Flow"
    return "Focus"


NUMERIC_FEATURES = [
    "hr_mean",
    "hr_std",
    "rmssd",
    "sdnn",
    "pnn50",
    "hr_z",
    "acc_rms",
    "sqi",
    "session_min",
]


def model_inference(features: FeatureSet) -> Optional[tuple[str, float]]:
    if not MODEL_AVAILABLE or MODEL is None:
        return None
    feature_dict = {name: getattr(features, name) for name in NUMERIC_FEATURES}
    frame = pd.DataFrame([feature_dict], columns=NUMERIC_FEATURES)
    try:
        prediction = MODEL.predict(frame)[0]
        confidence = 0.8
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(frame)[0]
            confidence = float(np.max(proba))
        return str(prediction), confidence
    except Exception as exc:  # pragma: no cover - 모델 예측 실패 대비
        logger.error("모델 추론 중 오류가 발생해 규칙 기반으로 폴백합니다: {}", exc)
        return None


def apply_hysteresis(device_id: str, new_state: str, config: Dict[str, Any]) -> str:
    hysteresis_cfg = config.get("hysteresis", {})
    min_hold = float(hysteresis_cfg.get("min_hold_seconds", 120))
    now = time.time()
    current = DEVICE_REGISTRY.get(device_id)
    if current is None:
        DEVICE_REGISTRY[device_id] = DeviceState(new_state, now, 0.0, now)
        return new_state
    if current.last_state == new_state:
        current.last_update = now
        return new_state
    elapsed = now - current.last_state_change
    if elapsed < min_hold:
        current.last_update = now
        return current.last_state
    current.last_state = new_state
    current.last_state_change = now
    current.last_update = now
    return new_state


def smooth_bpm(device_id: str, target_state: str, config: Dict[str, Any]) -> int:
    smoothing_cfg = config.get("smoothing", {})
    alpha = float(smoothing_cfg.get("alpha", 0.15))
    bpm_policy = config.get("bpm_policy", {})
    base_bpm = float(bpm_policy.get(target_state, bpm_policy.get("Focus", 100)))
    now = time.time()
    current = DEVICE_REGISTRY.get(device_id)
    if current is None:
        current = DeviceState(target_state, now, base_bpm, now)
        DEVICE_REGISTRY[device_id] = current
        return int(round(base_bpm))
    if current.last_state != target_state:
        current.last_state = target_state
        current.last_state_change = now
    current.last_update = now
    if current.last_bpm == 0.0:
        current.last_bpm = base_bpm
    smoothed = current.last_bpm + alpha * (base_bpm - current.last_bpm)
    current.last_bpm = smoothed
    return int(round(smoothed))


def log_session_row(device_id: str, state: str, bpm: Optional[int]) -> None:
    path = Path("data/session_log.csv")
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fp:
        row = [
            datetime.utcnow().isoformat(),
            device_id,
            state,
            bpm if bpm is not None else "",
        ]
        if is_new:
            fp.write("ts,device_id,state,bpm\n")
        fp.write(",".join(map(str, row)) + "\n")


app = FastAPI(title="Pulse Adaptive AI Server", version="1.1.0")


@app.on_event("startup")
def startup() -> None:
    load_config(force=True)
    load_model()
    bpm_target: int = Field(..., ge=0)
    tags: List[str]


def load_state_model() -> None:
    """ML 모델 로드 자리."""
    # TODO: ai-server/models/state_lgbm.pkl 경로에서 LightGBM 모델을 로드하도록 구현 예정
    return None


def apply_quality_rules(features: FeatureSet) -> Tuple[bool, str]:
    """신호 품질을 평가한다."""
    if features.sqi < 0.5:
        return False, "신호 품질이 낮아 추론을 건너뜀"
    return True, "신호 품질 양호"


def rule_based_inference(features: FeatureSet) -> Tuple[str, int, List[str]]:
    """규칙 기반 상태 분류와 목표 BPM 산출."""
    if features.rmssd < 20:
        return "HighStress", 75, ["calm"]
    if features.rmssd > 35:
        return "LowFocus", 125, ["energize"]
    if 90 <= features.hr_mean <= 110:
        return "Flow", 120, ["steady"]
    return "Focus", 100, ["balance"]


load_state_model()

app = FastAPI(title="Pulse Adaptive AI Server", version="0.1.0")

@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    config = load_config()
    features = request.features
    device_id = request.device_id
    logger.info("수신한 특징: device=%s, 데이터=%s", device_id, features.model_dump())

    if not quality_gate(features, config):
        logger.warning("신호 품질이 기준에 미달하여 Invalid 응답을 반환합니다")
        log_session_row(device_id, "Invalid", None)
        return InferenceResponse(state="Invalid", confidence=0.2, bpm_target=None, tags=["auto", "v1"])

    model_result = model_inference(features)
    if model_result is None:
        state = rule_based_state(features, config)
        confidence = 0.8
    else:
        state, confidence = model_result

    stable_state = apply_hysteresis(device_id, state, config)
    bpm_target = smooth_bpm(device_id, stable_state, config)
    log_session_row(device_id, stable_state, bpm_target)

    response = InferenceResponse(
        state=stable_state,
        confidence=float(confidence),
        bpm_target=int(bpm_target),
        tags=["auto", "v1"],
    )
    logger.info("추론 결과: %s", response.model_dump())
    return response


@app.get("/config/reload")
def reload_config() -> Dict[str, Any]:
    load_config(force=True)
    load_model()
    return {"reloaded": True, "model_loaded": MODEL_AVAILABLE}


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}

    """생체 특징을 받아 상태, 목표 BPM, 태그를 반환한다."""
    LOGGER.info("수신한 요청: device_id=%s, features=%s", request.device_id, request.features.model_dump())

    is_quality_ok, quality_msg = apply_quality_rules(request.features)
    if not is_quality_ok:
        LOGGER.warning("품질 검사 실패: %s", quality_msg)
        return InferenceResponse(
            state="Invalid",
            confidence=0.3,
            bpm_target=0,
            tags=["low_quality"],
        )

    state, bpm_target, tags = rule_based_inference(request.features)
    confidence = 0.8  # TODO: 모델 연동 시 동적으로 계산하도록 개선

    response = InferenceResponse(
        state=state,
        confidence=confidence,
        bpm_target=bpm_target,
        tags=tags,
    )
    LOGGER.info("추론 결과: %s", response.model_dump())
    return response


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """상태 확인용 엔드포인트."""
    return {"status": "ok"}