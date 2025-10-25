"""생체신호 기반 상태 추론 FastAPI 서버."""

from __future__ import annotations

import json
import pickle
import sys
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from math import cos, pi, sin
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas 미설치 대비
    pd = None  # type: ignore[assignment]

try:
    from joblib import load as joblib_load
except Exception:  # pragma: no cover - joblib 미설치 대비
    joblib_load = None


ENV_PATH = find_dotenv()
if ENV_PATH:
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)
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
DEFAULT_CONFIG: Dict[str, Any] = {
    "sqi": {
        "min_valid": 0.5,
        "ml_floor": 0.4,
        "ml_enabled": True,
        "ml_model_path": "ai-server/models/sqi_rf.pkl",
    },
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
    "model": {
        "path": "ai-server/models/state_lgbm.pkl",
        "enabled": True,
    },
    "personalization": {
        "enabled": True,
        "alpha": 0.3,
        "actions": [-10, -5, 0, 5, 10],
        "persistence_path": "data/bandit_state.json",
        "min_reward_samples": 3,
    },
}
CONFIG_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "data": deepcopy(DEFAULT_CONFIG)}
STATE_MODEL = None
STATE_MODEL_AVAILABLE = False
SQI_MODEL = None
SQI_MODEL_AVAILABLE = False
BANDIT_MANAGER = None

STATE_ORDER = ["HighStress", "Focus", "Flow", "LowFocus"]
CONTEXT_DIMENSION = 11
TIME_OF_DAY_MAP = {
    "morning": 8,
    "afternoon": 13,
    "evening": 18,
    "night": 23,
}


class FeatureSet(BaseModel):
    """Pi 에이전트가 전달하는 생체 특징 스키마."""

    hr_mean: float = Field(..., description="최근 10초 평균 심박수")
    hr_std: float = Field(..., ge=0, description="심박수 표준편차")
    rmssd: float = Field(..., ge=0, description="RMSSD 값")
    sdnn: float = Field(..., ge=0, description="SDNN 값")
    pnn50: float = Field(..., ge=0, le=1, description="pNN50 비율")
    hr_z: float = Field(..., description="심박수 Z-점수")
    acc_rms: float = Field(..., ge=0, description="가속도 RMS")
    sqi: float = Field(..., ge=0, le=1, description="로컬 SQI")
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
    sqi: Optional[float] = Field(default=None, ge=0, le=1)


@dataclass
class DeviceState:
    last_state: str
    last_state_change: float
    last_bpm: float
    last_update: float
    bandit_context: Optional[np.ndarray] = None
    bandit_action: Optional[float] = None
    last_rmssd: Optional[float] = None
    last_features: Optional[FeatureSet] = None
    reward_samples: int = 0


DEVICE_REGISTRY: Dict[str, DeviceState] = {}

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

SQI_FEATURES = [
    "acc_rms",
    "hr_std",
    "rmssd",
    "sdnn",
    "pnn50",
]


class LinUCBAgent:
    """LinUCB 컨텍스추얼 밴딧 구현."""

    def __init__(self, actions: List[float], alpha: float, dimension: int, state: Optional[Dict[str, Any]] = None) -> None:
        self.actions = [float(action) for action in actions]
        self.alpha = float(alpha)
        self.dimension = int(dimension)
        self.A: Dict[float, np.ndarray] = {
            action: np.eye(self.dimension, dtype=float) for action in self.actions
        }
        self.b: Dict[float, np.ndarray] = {
            action: np.zeros((self.dimension,), dtype=float) for action in self.actions
        }
        if state:
            self._restore(state)

    def _restore(self, payload: Dict[str, Any]) -> None:
        matrices = payload.get("A", {})
        vectors = payload.get("b", {})
        for key, value in matrices.items():
            action = float(key)
            if action in self.A:
                matrix = np.array(value, dtype=float)
                if matrix.shape == (self.dimension, self.dimension):
                    self.A[action] = matrix
        for key, value in vectors.items():
            action = float(key)
            if action in self.b:
                vector = np.array(value, dtype=float)
                if vector.shape == (self.dimension,):
                    self.b[action] = vector

    def select(self, context: np.ndarray) -> float:
        best_action = self.actions[0]
        best_score = float("-inf")
        for action in self.actions:
            A = self.A[action]
            b = self.b[action]
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(A)
            theta = A_inv @ b
            exploitation = float(theta @ context)
            exploration = float(self.alpha * np.sqrt(context @ A_inv @ context))
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def update(self, action: float, context: np.ndarray, reward: float) -> None:
        if action not in self.A:
            return
        context = context.reshape(self.dimension)
        self.A[action] = self.A[action] + np.outer(context, context)
        self.b[action] = self.b[action] + reward * context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "A": {str(action): matrix.tolist() for action, matrix in self.A.items()},
            "b": {str(action): vector.tolist() for action, vector in self.b.items()},
        }


class BanditManager:
    """디바이스별 LinUCB 에이전트를 관리한다."""

    def __init__(self, path: Path, actions: List[float], alpha: float, dimension: int) -> None:
        self.path = path
        self.actions = actions
        self.alpha = alpha
        self.dimension = dimension
        self._lock = threading.Lock()
        self._agents: Dict[str, LinUCBAgent] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception as exc:  # pragma: no cover - 파일 파싱 실패 대비
            logger.warning("밴딧 상태 파일을 읽는 중 문제가 발생해 초기화합니다: {}", exc)
            return
        meta = payload.get("meta", {})
        if int(meta.get("dimension", self.dimension)) != self.dimension:
            logger.warning("밴딧 상태의 차원이 달라 초기화합니다: %s", meta)
            return
        stored_actions = [float(x) for x in meta.get("actions", self.actions)]
        if stored_actions != self.actions:
            logger.warning("밴딧 행동 구성이 달라 기존 상태를 무시합니다.")
            return
        agents = payload.get("agents", {})
        for device_id, state in agents.items():
            self._agents[device_id] = LinUCBAgent(self.actions, self.alpha, self.dimension, state)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "meta": {
                    "actions": self.actions,
                    "alpha": self.alpha,
                    "dimension": self.dimension,
                },
                "agents": {device: agent.to_dict() for device, agent in self._agents.items()},
            }
            with self.path.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)

    def get_agent(self, device_id: str) -> LinUCBAgent:
        with self._lock:
            agent = self._agents.get(device_id)
            if agent is None:
                agent = LinUCBAgent(self.actions, self.alpha, self.dimension)
                self._agents[device_id] = agent
            return agent

    def update(self, device_id: str, action: float, context: np.ndarray, reward: float) -> None:
        agent = self.get_agent(device_id)
        agent.update(action, context, reward)

    def select(self, device_id: str, context: np.ndarray) -> float:
        agent = self.get_agent(device_id)
        return agent.select(context)


def load_bandit_manager(config: Dict[str, Any]) -> Optional[BanditManager]:
    """설정에 따라 밴딧 매니저를 초기화하거나 비활성화한다."""

    global BANDIT_MANAGER
    personal_cfg = config.get("personalization", {})
    if not personal_cfg.get("enabled", False):
        BANDIT_MANAGER = None
        return None
    actions = [float(x) for x in personal_cfg.get("actions", [-10, -5, 0, 5, 10])]
    if not actions:
        BANDIT_MANAGER = None
        return None
    alpha = float(personal_cfg.get("alpha", 0.3))
    path = Path(personal_cfg.get("persistence_path", "data/bandit_state.json"))
    if BANDIT_MANAGER is None or (
        BANDIT_MANAGER.actions != actions
        or not np.isclose(BANDIT_MANAGER.alpha, alpha)
        or BANDIT_MANAGER.dimension != CONTEXT_DIMENSION
        or BANDIT_MANAGER.path != path
    ):
        BANDIT_MANAGER = BanditManager(path, actions, alpha, CONTEXT_DIMENSION)
    return BANDIT_MANAGER


def _time_of_day_encoding(label: str) -> Tuple[float, float]:
    hour = TIME_OF_DAY_MAP.get(label, 12)
    angle = 2 * pi * (hour % 24) / 24.0
    return sin(angle), cos(angle)


def build_context_vector(features: FeatureSet, state: str) -> np.ndarray:
    """LinUCB 컨텍스트 벡터를 구성한다."""

    state = state if state in STATE_ORDER else "Focus"
    sin_val, cos_val = _time_of_day_encoding(features.time_of_day)
    hr_mean_norm = np.clip(features.hr_mean, 40.0, 160.0) / 160.0
    hr_z_norm = np.clip(features.hr_z / 3.0, -1.0, 1.0)
    rmssd_norm = np.clip(features.rmssd, 0.0, 80.0) / 80.0
    session_norm = np.clip(features.session_min, 0, 240) / 240.0

    context: List[float] = [
        1.0,
        hr_mean_norm,
        hr_z_norm,
        rmssd_norm,
        session_norm,
        sin_val,
        cos_val,
    ]
    for label in STATE_ORDER:
        context.append(1.0 if state == label else 0.0)
    return np.array(context, dtype=float)


def compute_reward(previous_rmssd: Optional[float], current_rmssd: float, state: str) -> float:
    """RMSSD 변화량과 상태를 기반으로 보상을 계산한다."""

    base = np.clip(current_rmssd, 0.0, 80.0) / 50.0
    delta = 0.0
    if previous_rmssd is not None:
        delta = (current_rmssd - previous_rmssd) / 15.0
    state_bonus = 0.0
    if state == "Flow":
        state_bonus = 0.2
    elif state == "HighStress":
        state_bonus = -0.3
    reward = base + delta + state_bonus
    return float(np.clip(reward, -1.0, 2.5))


def determine_base_bpm(state: str, config: Dict[str, Any]) -> float:
    bpm_policy = config.get("bpm_policy", {})
    return float(bpm_policy.get(state, bpm_policy.get("Focus", 100)))


def personalize_bpm(
    device_id: str,
    state: str,
    features: FeatureSet,
    base_bpm: float,
    config: Dict[str, Any],
) -> Tuple[float, float]:
    """컨텍스추얼 밴딧 기반 개인화 보정을 적용한다."""

    manager = load_bandit_manager(config)
    if manager is None:
        return base_bpm, 0.0
    device_state = DEVICE_REGISTRY.get(device_id)
    if device_state is None:
        now = time.time()
        device_state = DeviceState(state, now, base_bpm, now)
        DEVICE_REGISTRY[device_id] = device_state
    context = build_context_vector(features, state)
    personal_cfg = config.get("personalization", {})
    min_samples = int(personal_cfg.get("min_reward_samples", 0))
    if (
        device_state.bandit_action is not None
        and device_state.bandit_context is not None
        and device_state.last_rmssd is not None
    ):
        reward = compute_reward(device_state.last_rmssd, features.rmssd, state)
        manager.update(device_id, float(device_state.bandit_action), device_state.bandit_context, reward)
        logger.info("밴딧 보상 업데이트: device=%s, reward=%.3f", device_id, reward)
        device_state.reward_samples += 1
    if device_state.reward_samples < min_samples:
        selected_action = 0.0
    else:
        selected_action = manager.select(device_id, context)
    device_state.bandit_action = selected_action
    device_state.bandit_context = context
    device_state.last_rmssd = features.rmssd
    try:
        manager.save()
    except Exception as exc:  # pragma: no cover - 파일 쓰기 실패 대비
        logger.warning("밴딧 상태 저장 중 오류가 발생했습니다: {}", exc)
    personalized = base_bpm + selected_action
    return float(personalized), float(selected_action)

def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """재귀적으로 딕셔너리를 병합한다."""

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base.get(key, {}), value)
        else:
            base[key] = value
    return base


def load_config(force: bool = False) -> Dict[str, Any]:
    """YAML 구성을 로드하고 캐시에 저장한다."""

    if not force and CONFIG_CACHE["data"] is not None:
        return CONFIG_CACHE["data"]
    global BANDIT_MANAGER
    merged = deepcopy(DEFAULT_CONFIG)
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        merged = deep_update(merged, data)
        logger.info("AI 설정을 로드했습니다: {}", CONFIG_PATH)
    except FileNotFoundError:
        logger.warning("model.yaml을 찾지 못해 기본값으로 동작합니다: {}", CONFIG_PATH)
    except yaml.YAMLError as exc:
        logger.warning("model.yaml 파싱 중 오류가 발생해 기본값을 사용합니다: {}", exc)
    CONFIG_CACHE.update({"loaded_at": time.time(), "data": merged})
    BANDIT_MANAGER = None
    return merged


def _load_serialized(path: Path) -> Any:
    """joblib이 없을 때 pickle로 대체 로딩한다."""

    if joblib_load is not None:
        return joblib_load(path)
    with path.open("rb") as fp:
        return pickle.load(fp)


def load_models(config: Optional[Dict[str, Any]] = None) -> None:
    """LightGBM 상태 모델과 SQI 모델을 로드한다."""

    global STATE_MODEL, STATE_MODEL_AVAILABLE, SQI_MODEL, SQI_MODEL_AVAILABLE
    cfg = config or load_config()

    model_cfg = cfg.get("model", {})
    model_enabled = bool(model_cfg.get("enabled", True))
    model_path = Path(model_cfg.get("path", "ai-server/models/state_lgbm.pkl"))
    if model_enabled and model_path.exists():
        try:
            STATE_MODEL = _load_serialized(model_path)
            STATE_MODEL_AVAILABLE = True
            logger.info("상태 분류 LightGBM 모델을 로드했습니다: {}", model_path)
        except Exception as exc:  # pragma: no cover - 모델 로딩 실패 대비
            logger.error("LightGBM 모델 로딩 실패로 규칙 기반으로 폴백합니다: {}", exc)
            STATE_MODEL = None
            STATE_MODEL_AVAILABLE = False
    else:
        if model_enabled:
            logger.warning("LightGBM 모델 파일을 찾을 수 없어 규칙 기반으로 동작합니다: {}", model_path)
        else:
            logger.info("설정에 따라 LightGBM 사용이 비활성화되었습니다.")
        STATE_MODEL = None
        STATE_MODEL_AVAILABLE = False

    sqi_cfg = cfg.get("sqi", {})
    sqi_enabled = bool(sqi_cfg.get("ml_enabled", False))
    sqi_path = Path(sqi_cfg.get("ml_model_path", "ai-server/models/sqi_rf.pkl"))
    if sqi_enabled and sqi_path.exists():
        try:
            SQI_MODEL = _load_serialized(sqi_path)
            SQI_MODEL_AVAILABLE = True
            logger.info("SQI RandomForest 모델을 로드했습니다: {}", sqi_path)
        except Exception as exc:  # pragma: no cover - 모델 로딩 실패 대비
            logger.error("SQI 모델 로딩 실패로 로컬 SQI만 사용합니다: {}", exc)
            SQI_MODEL = None
            SQI_MODEL_AVAILABLE = False
    else:
        if sqi_enabled:
            logger.warning("SQI 모델 파일이 없어 로컬 SQI만 사용합니다: {}", sqi_path)
        else:
            logger.info("설정에 따라 SQI ML 사용이 비활성화되었습니다.")
        SQI_MODEL = None
        SQI_MODEL_AVAILABLE = False


def rule_based_state(features: FeatureSet, config: Dict[str, Any]) -> str:
    """규칙 기반 상태 분류 로직."""

    rules = config.get("rules", {})
    rmssd_low = float(rules.get("rmssd_low", 20))
    rmssd_high = float(rules.get("rmssd_high", 35))
    hr_flow_min = float(rules.get("hr_flow_min", 90))
    hr_flow_max = float(rules.get("hr_flow_max", 110))
    hr_z_high = float(rules.get("hr_z_high", 0.7))

    if features.rmssd < rmssd_low or (
        features.hr_z > hr_z_high and features.rmssd < rmssd_low + 2
    ):
        return "HighStress"
    if features.rmssd > rmssd_high and features.hr_z < 0.3:
        return "LowFocus"
    if hr_flow_min <= features.hr_mean <= hr_flow_max:
        return "Flow"
    return "Focus"


def run_sqi_model(features: FeatureSet) -> Optional[float]:
    """SQI RandomForest 모델 추론."""

    if not SQI_MODEL_AVAILABLE or SQI_MODEL is None:
        return None
    sample = np.array([[getattr(features, name, 0.0) for name in SQI_FEATURES]])
    try:
        if hasattr(SQI_MODEL, "predict_proba"):
            proba = SQI_MODEL.predict_proba(sample)
            if proba.shape[1] > 1:
                return float(np.clip(proba[0][1], 0.0, 1.0))
            return float(np.clip(proba[0][0], 0.0, 1.0))
        prediction = SQI_MODEL.predict(sample)
        return float(np.clip(prediction[0], 0.0, 1.0))
    except Exception as exc:  # pragma: no cover - 모델 예측 실패 대비
        logger.error("SQI 모델 추론 실패로 로컬 SQI만 사용합니다: {}", exc)
        return None


def evaluate_quality(features: FeatureSet, config: Dict[str, Any]) -> Tuple[bool, float, Optional[float]]:
    """로컬 SQI와 ML SQI를 결합해 최종 품질을 산출한다."""

    sqi_cfg = config.get("sqi", {})
    min_valid = float(sqi_cfg.get("min_valid", 0.5))
    ml_floor = float(sqi_cfg.get("ml_floor", 0.0))
    sqi_local = float(np.clip(features.sqi, 0.0, 1.0))
    sqi_ml = run_sqi_model(features) if bool(sqi_cfg.get("ml_enabled", False)) else None
    effective_sqi = sqi_local
    if sqi_ml is not None:
        effective_sqi = min(effective_sqi, sqi_ml)
    if sqi_ml is not None and effective_sqi < ml_floor:
        logger.warning("ML SQI 결과가 임계값 이하입니다: %.3f", effective_sqi)
    is_valid = effective_sqi >= min_valid
    return is_valid, effective_sqi, sqi_ml


def model_inference(features: FeatureSet, effective_sqi: float) -> Optional[Tuple[str, float]]:
    """LightGBM 모델 기반 상태 추론."""

    if not STATE_MODEL_AVAILABLE or STATE_MODEL is None:
        return None
    feature_dict = {name: getattr(features, name) for name in NUMERIC_FEATURES}
    feature_dict["sqi"] = effective_sqi
    if pd is not None:
        frame: Any = pd.DataFrame([feature_dict], columns=NUMERIC_FEATURES)
    else:
        frame = [feature_dict]
    try:
        prediction = STATE_MODEL.predict(frame)[0]
        confidence = 0.8
        if hasattr(STATE_MODEL, "predict_proba"):
            proba = STATE_MODEL.predict_proba(frame)[0]
            confidence = float(np.max(proba))
        return str(prediction), confidence
    except Exception as exc:  # pragma: no cover - 모델 예측 실패 대비
        logger.error("LightGBM 추론 실패로 규칙 기반으로 폴백합니다: {}", exc)
        return None


def apply_hysteresis(device_id: str, new_state: str, config: Dict[str, Any]) -> str:
    """상태가 너무 자주 바뀌지 않도록 히스테리시스를 적용한다."""

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
    if now - current.last_state_change < min_hold:
        current.last_update = now
        return current.last_state
    current.last_state = new_state
    current.last_state_change = now
    current.last_update = now
    return new_state


def smooth_bpm(device_id: str, target_state: str, desired_bpm: float, config: Dict[str, Any]) -> int:
    """개인화된 목표 BPM을 스무딩하여 점진적으로 변경한다."""

    smoothing_cfg = config.get("smoothing", {})
    alpha = float(smoothing_cfg.get("alpha", 0.15))
    now = time.time()
    current = DEVICE_REGISTRY.get(device_id)
    if current is None:
        current = DeviceState(target_state, now, desired_bpm, now)
        DEVICE_REGISTRY[device_id] = current
        return int(round(desired_bpm))
    if current.last_state != target_state:
        current.last_state = target_state
        current.last_state_change = now
    current.last_update = now
    if current.last_bpm == 0.0:
        current.last_bpm = desired_bpm
    smoothed = current.last_bpm + alpha * (desired_bpm - current.last_bpm)
    current.last_bpm = smoothed
    return int(round(smoothed))


def log_session_row(
    device_id: str,
    state: str,
    bpm: Optional[int],
    confidence: float,
    sqi_value: float,
    sqi_ml: Optional[float],
) -> None:
    """세션 로그 CSV에 추론 결과를 기록한다."""

    path = Path("data/session_log.csv")
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fp:
        if is_new:
            fp.write("ts,device_id,state,bpm,confidence,sqi,sqi_ml\n")
        row = [
            datetime.utcnow().isoformat(),
            device_id,
            state,
            bpm if bpm is not None else "",
            f"{confidence:.4f}",
            f"{sqi_value:.4f}",
            f"{sqi_ml:.4f}" if sqi_ml is not None else "",
        ]
        fp.write(",".join(map(str, row)) + "\n")


app = FastAPI(title="Pulse Adaptive AI Server", version="1.2.0")


@app.on_event("startup")
def startup() -> None:
    """애플리케이션 기동 시 설정과 모델을 로드한다."""

    config = load_config(force=True)
    load_models(config)


@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    """생체 특징을 받아 상태·BPM을 산출한다."""

    config = load_config()
    features = request.features
    device_id = request.device_id
    logger.info("수신한 특징: device=%s, 데이터=%s", device_id, features.model_dump())

    is_valid, effective_sqi, sqi_ml = evaluate_quality(features, config)
    if not is_valid:
        logger.warning("신호 품질이 기준에 미달하여 Invalid 응답을 반환합니다")
        log_session_row(device_id, "Invalid", None, 0.2, effective_sqi, sqi_ml)
        return InferenceResponse(
            state="Invalid",
            confidence=0.2,
            bpm_target=None,
            tags=["auto", "v3"],
            sqi=effective_sqi,
        )

    model_result = model_inference(features, effective_sqi)
    if model_result is None:
        state = rule_based_state(features, config)
        confidence = 0.8
    else:
        state, confidence = model_result

    stable_state = apply_hysteresis(device_id, state, config)
    base_bpm = determine_base_bpm(stable_state, config)
    personalized_bpm, action = personalize_bpm(
        device_id,
        stable_state,
        features,
        base_bpm,
        config,
    )
    logger.info(
        "BPM 정책 적용: 상태=%s, 기본=%.1f, 개인화 조정=%.1f (행동=%.1f)",
        stable_state,
        base_bpm,
        personalized_bpm,
        action,
    )
    bpm_target = smooth_bpm(device_id, stable_state, personalized_bpm, config)
    log_session_row(device_id, stable_state, bpm_target, confidence, effective_sqi, sqi_ml)
    current_state = DEVICE_REGISTRY.get(device_id)
    if current_state is not None:
        current_state.last_features = features

    response = InferenceResponse(
        state=stable_state,
        confidence=float(confidence),
        bpm_target=int(bpm_target),
        tags=["auto", "v3"],
        sqi=effective_sqi,
    )
    logger.info("추론 결과: %s", response.model_dump())
    return response


@app.get("/config/reload")
def reload_config() -> Dict[str, Any]:
    """설정과 모델을 재로딩한다."""

    config = load_config(force=True)
    load_models(config)
    return {
        "reloaded": True,
        "state_model_loaded": STATE_MODEL_AVAILABLE,
        "sqi_model_loaded": SQI_MODEL_AVAILABLE,
    }


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    """상태 확인용 엔드포인트."""

    return {"status": "ok"}

