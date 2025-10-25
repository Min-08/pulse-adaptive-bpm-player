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


class InferenceRequest(BaseModel):
    device_id: str = Field(..., min_length=1, description="디바이스 식별자")
    features: FeatureSet


class InferenceResponse(BaseModel):
    state: str
    confidence: float = Field(..., ge=0, le=1)
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
