"""LLM 기반 재랭킹을 위한 FastAPI 서버 (현재는 규칙 기반)."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import List

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
    """콘솔 및 회전 로그 파일을 설정한다."""
    logger = logging.getLogger("llm_recommender")
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
        "data/llm_recommender.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


LOGGER = configure_logger()


class Candidate(BaseModel):
    id: str = Field(..., min_length=1)
    bpm: float


class RerankRequest(BaseModel):
    state: str
    bpm_target: float
    candidates: List[Candidate]


class RerankResponse(BaseModel):
    ranked_ids: List[str]
    source: str


app = FastAPI(title="LLM Recommender", version="0.1.0")


def deterministic_rerank(bpm_target: float, candidates: List[Candidate]) -> List[str]:
    """목표 BPM과의 차이를 기준으로 후보를 정렬한다."""
    sorted_candidates = sorted(
        candidates,
        key=lambda c: abs(c.bpm - bpm_target),
    )
    return [candidate.id for candidate in sorted_candidates[:3]]


@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest) -> RerankResponse:
    """후보 곡 목록을 재정렬한다."""
    LOGGER.info("수신한 재랭킹 요청: state=%s, bpm_target=%s, 후보 수=%d", request.state, request.bpm_target, len(request.candidates))

    if not request.candidates:
        LOGGER.warning("후보 목록이 비어 있어 빈 결과를 반환")
        return RerankResponse(ranked_ids=[], source="rule-fallback")

    # TODO: LLM 기반 재랭킹 추가 - OpenAI/Anthropic API 호출 로직으로 대체 예정

    ranked_ids = deterministic_rerank(request.bpm_target, request.candidates)
    response = RerankResponse(ranked_ids=ranked_ids, source="rule-fallback")
    LOGGER.info("재랭킹 결과: %s", response.model_dump())
    return response


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """상태 확인용 엔드포인트."""
    return {"status": "ok"}
