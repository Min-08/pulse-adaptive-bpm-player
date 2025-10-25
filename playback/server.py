"""Spotify 및 로컬 백업 재생을 담당하는 FastAPI 서버."""

from __future__ import annotations

import base64
import csv
import json
import logging
import os
import secrets
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests
import yaml
from cachetools import TTLCache
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

try:
    from playsound import playsound as playsound_sync
except Exception:  # pragma: no cover - 모듈 미설치 환경 대비
    playsound_sync = None

# ------------------------------------------------------------------
# 환경 변수 로드
# ------------------------------------------------------------------
ENV_PATH = find_dotenv()
if ENV_PATH:
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

# ------------------------------------------------------------------
# 로거 설정
# ------------------------------------------------------------------
def configure_logger() -> logging.Logger:
    """콘솔 및 회전 파일 로그를 설정한다."""
    logger = logging.getLogger("playback")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

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

# ------------------------------------------------------------------
# 전역 상수/상태
# ------------------------------------------------------------------
TOKEN_PATH = Path("playback/.tokens/spotify.json")
TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

STATE_CONFIG_PATH = Path(os.getenv("STATE_PARAMS_PATH", "playback/config/state_params.yaml"))
STATE_CONFIG_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "data": None}
STATE_CONFIG_LOCK = threading.Lock()
DEFAULT_STATE_PARAMS: Dict[str, Any] = {
    "Focus": {
        "tempo": {"min": 95, "target": 105, "max": 115},
        "energy": {"min": 0.3, "max": 0.6},
        "valence": {"min": 0.4, "max": 0.7},
        "genres": ["lo-fi", "focus", "study", "chill"],
    }
}

RECOMMENDATION_CACHE: TTLCache = TTLCache(maxsize=64, ttl=900)
COOLDOWN_SECONDS = 1800
PLAYED_TRACKS: Dict[str, float] = {}
PLAYED_ARTISTS: Dict[str, float] = {}
OAUTH_STATE: Dict[str, float] = OrderedDict()
OAUTH_STATE_TTL = 600

# ------------------------------------------------------------------
# 모델/예외/스키마
# ------------------------------------------------------------------
class SpotifyAuthError(RuntimeError):
    """Spotify 인증 관련 예외."""

class PlaybackRequest(BaseModel):
    bpm_target: int = Field(..., ge=0)
    state: str = Field(..., min_length=1)

class PlaybackResponse(BaseModel):
    ok: bool
    mode: str
    track_uri: Optional[str] = None
    device: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class TrackCandidate:
    """재생 후보 트랙 정보."""
    uri: str
    track_id: str
    name: str
    artists: List[str]
    tempo: Optional[float]
    source: str

class LocalLibraryError(RuntimeError):
    """로컬 음악 라이브러리 관련 예외."""

@dataclass
class LocalTrack:
    """로컬 라이브러리에 저장된 트랙 정보."""
    track_name: str
    artist: str
    bpm: Optional[int]
    states: List[str]
    file_name: str
    file_path: Path

# ------------------------------------------------------------------
# 토큰 저장/갱신
# ------------------------------------------------------------------
class SpotifyTokenStore:
    """Spotify 토큰 저장 및 갱신을 담당한다."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as fp:
                    self._data = json.load(fp)
            except json.JSONDecodeError:
                LOGGER.warning("토큰 파일을 파싱하지 못해 초기화합니다: %s", self.path)
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        with self.path.open("w", encoding="utf-8") as fp:
            json.dump(self._data, fp, ensure_ascii=False, indent=2)

    def update_tokens(self, tokens: Dict[str, Any]) -> None:
        with self._lock:
            expires_in = int(tokens.get("expires_in", 3600))
            expires_at = int(time.time()) + expires_in
            self._data.update(
                {
                    "access_token": tokens.get("access_token"),
                    "refresh_token": tokens.get("refresh_token") or self._data.get("refresh_token"),
                    "expires_at": expires_at,
                }
            )
            self.save()
            LOGGER.info("Spotify 토큰 정보를 갱신했습니다 (만료 시각: %s)", expires_at)

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            if self.path.exists():
                self.path.unlink()

    def get_refresh_token(self) -> str:
        refresh_token = self._data.get("refresh_token")
        if not refresh_token:
            raise SpotifyAuthError("Spotify 인증이 필요합니다. /login을 먼저 호출하세요.")
        return refresh_token

    def get_access_token(self) -> str:
        with self._lock:
            access_token = self._data.get("access_token")
            expires_at = self._data.get("expires_at", 0)
        now = int(time.time())
        if access_token and now < expires_at - 60:
            return access_token
        refresh_token = self.get_refresh_token()
        tokens = refresh_access_token(refresh_token)
        self.update_tokens(tokens)
        return self._data.get("access_token", "")

TOKEN_STORE = SpotifyTokenStore(TOKEN_PATH)

# ------------------------------------------------------------------
# 로컬 라이브러리/재생
# ------------------------------------------------------------------
class LocalMusicLibrary:
    """CSV 기반 로컬 음악 라이브러리를 관리한다."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.tracks_dir = self.root_dir / "tracks"
        self.tracks_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.root_dir / "library.csv"
        self._tracks: List[LocalTrack] = []
        self._lock = threading.Lock()
        self.reload()

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(ch.lower() for ch in value if ch.isalnum())

    @staticmethod
    def _parse_states(raw: str) -> List[str]:
        if not raw:
            return []
        token = raw.replace(",", ";")
        states = [p.strip().casefold() for p in token.split(";") if p.strip()]
        return states

    def _resolve_file_path(self, file_name: str) -> Path:
        candidate = Path(file_name)
        if candidate.is_absolute():
            return candidate
        return self.tracks_dir / candidate.name

    def _load_row(self, row: Dict[str, str]) -> LocalTrack:
        track_name = (row.get("track_name") or "").strip()
        artist = (row.get("artist") or "").strip()
        bpm_value = (row.get("bpm") or "").strip()
        file_name = (row.get("file_name") or "").strip()
        states_raw = (row.get("states") or "").strip()
        if not track_name or not file_name:
            raise LocalLibraryError("track_name 또는 file_name 값이 비어 있어 로컬 곡을 무시합니다.")
        file_path = self._resolve_file_path(file_name)
        if file_path.suffix.lower() != ".mp3":
            raise LocalLibraryError(f"MP3 파일이 아니어서 제외되었습니다: {file_path}")
        if not file_path.exists():
            raise LocalLibraryError(f"로컬 음악 파일을 찾을 수 없습니다: {file_path}")
        if self._normalize(track_name) != self._normalize(file_path.stem):
            raise LocalLibraryError(
                f"곡명과 파일명이 일치하지 않아 제외되었습니다: {track_name} != {file_path.stem}"
            )
        bpm: Optional[int] = None
        if bpm_value:
            try:
                bpm = int(float(bpm_value))
            except ValueError:
                LOGGER.warning("BPM 값을 정수로 변환하지 못해 None으로 처리합니다: %s", bpm_value)
        states = self._parse_states(states_raw)
        return LocalTrack(
            track_name=track_name,
            artist=artist,
            bpm=bpm,
            states=states,
            file_name=file_name,
            file_path=file_path,
        )

    def reload(self) -> int:
        with self._lock:
            self._tracks = []
            if not self.csv_path.exists():
                LOGGER.warning("로컬 라이브러리 CSV를 찾을 수 없어 빈 라이브러리로 동작합니다: %s", self.csv_path)
                return 0
            with self.csv_path.open("r", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    try:
                        track = self._load_row(row)
                    except LocalLibraryError as exc:
                        LOGGER.warning("로컬 트랙을 무시합니다: %s", exc)
                        continue
                    except Exception as exc:  # pragma: no cover
                        LOGGER.warning("로컬 트랙 로드 중 알 수 없는 오류: %s", exc)
                        continue
                    self._tracks.append(track)
            LOGGER.info("로컬 음악 라이브러리 로드 완료 (트랙 수: %d)", len(self._tracks))
            return len(self._tracks)

    def pick_track(self, state: str, bpm_target: int) -> LocalTrack:
        with self._lock:
            if not self._tracks:
                raise LocalLibraryError("로컬 라이브러리에 재생 가능한 곡이 없습니다.")
            normalized_state = (state or "").casefold()
            candidates = [
                t for t in self._tracks if not t.states or normalized_state in t.states
            ] or list(self._tracks)

        def score(track: LocalTrack) -> float:
            if bpm_target and track.bpm:
                return abs(track.bpm - bpm_target)
            if track.bpm is not None:
                return abs(track.bpm - 100)
            return 9999.0

        ordered = sorted(candidates, key=score)
        for track in ordered:
            if not track.file_path.exists():
                LOGGER.warning("선택 후보 파일이 사라져 건너뜁니다: %s", track.file_path)
                continue
            return track
        raise LocalLibraryError("조건에 맞는 로컬 음악을 찾지 못했습니다.")

    def count(self) -> int:
        with self._lock:
            return len(self._tracks)

class LocalPlaybackEngine:
    """로컬 MP3 재생을 담당하는 엔진."""

    def __init__(self, library: LocalMusicLibrary) -> None:
        self.library = library

    def play(self, track: LocalTrack) -> bool:
        if not track.file_path.exists():
            LOGGER.error("로컬 재생 파일이 존재하지 않습니다: %s", track.file_path)
            return False

        if playsound_sync is not None:
            def _runner() -> None:
                try:
                    playsound_sync(str(track.file_path))
                except Exception as exc:  # pragma: no cover
                    LOGGER.error("playsound 재생 중 오류: %s", exc)
            threading.Thread(target=_runner, daemon=True).start()
            return True

        try:
            if os.name == "nt":
                os.startfile(str(track.file_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(track.file_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(["xdg-open", str(track.file_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as exc:  # pragma: no cover
            LOGGER.error("OS 기본 플레이어 실행 실패: %s", exc)
            return False

LOCAL_LIBRARY = LocalMusicLibrary(Path(os.getenv("LOCAL_MUSIC_DIR", "playback/local_playlist")))
LOCAL_PLAYBACK_ENGINE = LocalPlaybackEngine(LOCAL_LIBRARY)

# ------------------------------------------------------------------
# 유틸 함수들
# ------------------------------------------------------------------
def cleanup_oauth_state() -> None:
    """만료된 OAuth state 값을 정리한다."""
    now = time.time()
    for key in list(OAUTH_STATE.keys()):
        if now - OAUTH_STATE[key] > OAUTH_STATE_TTL:
            OAUTH_STATE.pop(key, None)

def determine_mode() -> str:
    """환경 변수 기반 재생 모드를 반환한다."""
    mode = os.getenv("PLAYBACK_MODE", "spotify").lower()
    if mode not in {"spotify", "local"}:
        return "spotify"
    return mode

def load_state_params(force: bool = False) -> Dict[str, Any]:
    """상태별 Spotify 파라미터 구성을 로드한다."""
    with STATE_CONFIG_LOCK:
        if not force and STATE_CONFIG_CACHE["data"] is not None:
            return STATE_CONFIG_CACHE["data"]
        try:
            with STATE_CONFIG_PATH.open("r", encoding="utf-8") as fp:
                config = yaml.safe_load(fp) or {}
            STATE_CONFIG_CACHE.update({"loaded_at": time.time(), "data": config})
            LOGGER.info("상태 파라미터 구성을 로드했습니다: %s", STATE_CONFIG_PATH)
            return config
        except FileNotFoundError:
            LOGGER.warning("state_params.yaml 파일을 찾을 수 없어 기본값으로 동작합니다: %s", STATE_CONFIG_PATH)
        except yaml.YAMLError as exc:
            LOGGER.warning("state_params.yaml 파싱 중 오류가 발생해 기본값으로 동작합니다: %s", exc)
        STATE_CONFIG_CACHE.update({"loaded_at": time.time(), "data": DEFAULT_STATE_PARAMS})
        return DEFAULT_STATE_PARAMS

def reload_state_params() -> Dict[str, Any]:
    """외부 요청으로 구성 파일을 재로딩한다."""
    config = load_state_params(force=True)
    LOCAL_LIBRARY.reload()
    return config

def get_params_for_state(state: str, bpm_target: Optional[int], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """상태와 목표 BPM에 따른 추천 파라미터를 반환한다."""
    normalized_state = state.strip() if state else ""
    base = cfg.get(normalized_state)
    if not base:
        LOGGER.warning("정의되지 않은 상태(%s)로 요청되어 Focus 기본값을 사용합니다.", state)
        base = cfg.get("Focus", DEFAULT_STATE_PARAMS["Focus"])

    tempo_cfg = base.get("tempo", {})
    if bpm_target and bpm_target > 0:
        target = int(bpm_target)
        tempo_min = max(50, target - 10)
        tempo_max = min(200, target + 10)
    else:
        target = int(tempo_cfg.get("target", 105))
        tempo_min = int(tempo_cfg.get("min", max(50, target - 10)))
        tempo_max = int(tempo_cfg.get("max", min(200, target + 10)))

    energy_cfg = base.get("energy", {})
    valence_cfg = base.get("valence", {})
    genres = base.get("genres", DEFAULT_STATE_PARAMS["Focus"].get("genres", []))
    market = os.getenv("SPOTIFY_MARKET", "KR")

    return {
        "seed_genres": ",".join(genres[:5]) if genres else "",
        "target_tempo": target,
        "min_tempo": tempo_min,
        "max_tempo": tempo_max,
        "min_energy": energy_cfg.get("min", 0.3),
        "max_energy": energy_cfg.get("max", 0.6),
        "min_valence": valence_cfg.get("min", 0.4),
        "max_valence": valence_cfg.get("max", 0.7),
        "limit": 25,
        "market": market,
    }

def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """Refresh Token을 사용해 Access Token을 갱신한다."""
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SpotifyAuthError("Spotify 클라이언트 자격 증명이 설정되어 있지 않습니다.")
    token_url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {"Authorization": f"Basic {auth_header}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    response = requests.post(token_url, data=data, headers=headers, timeout=10)
    if response.status_code != 200:
        LOGGER.error("Spotify 토큰 갱신 실패: %s", response.text)
        raise SpotifyAuthError("Spotify 토큰 갱신에 실패했습니다. /login을 다시 실행하세요.")
    return response.json()

def exchange_code_for_token(code: str) -> Dict[str, Any]:
    """인가 코드를 토큰으로 교환한다."""
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
    if not all([client_id, client_secret, redirect_uri]):
        raise SpotifyAuthError("Spotify OAuth 설정이 완전하지 않습니다. .env를 확인하세요.")
    token_url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {"Authorization": f"Basic {auth_header}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri}
    response = requests.post(token_url, data=data, headers=headers, timeout=10)
    if response.status_code != 200:
        LOGGER.error("Spotify 인가 코드 교환 실패: %s", response.text)
        raise SpotifyAuthError("Spotify 토큰 발급에 실패했습니다. 다시 로그인하세요.")
    return response.json()

def request_spotify_api(method: str, url: str, token: str, **kwargs: Any) -> requests.Response:
    """Spotify Web API 요청을 공통 처리한다."""
    headers = kwargs.pop("headers", {})
    headers.update({"Authorization": f"Bearer {token}"})
    try:
        response = requests.request(method, url, headers=headers, timeout=10, **kwargs)
    except requests.RequestException as exc:
        LOGGER.error("Spotify API 요청 중 예외가 발생했습니다: %s", exc)
        raise HTTPException(status_code=502, detail="Spotify API 호출에 실패했습니다.") from exc
    return response

def fetch_devices(token: str) -> List[Dict[str, Any]]:
    """Spotify 기기 목록을 가져온다."""
    response = request_spotify_api("GET", "https://api.spotify.com/v1/me/player/devices", token)
    if response.status_code != 200:
        LOGGER.error("디바이스 조회 실패: %s", response.text)
        raise HTTPException(status_code=502, detail="Spotify 디바이스 정보를 가져오지 못했습니다.")
    data = response.json()
    return data.get("devices", [])

def find_device(devices: List[Dict[str, Any]], expected_name: str) -> Optional[Dict[str, Any]]:
    """지정된 이름과 일치하는 디바이스를 찾는다."""
    target = expected_name.casefold()
    for device in devices:
        if device.get("name", "").casefold() == target:
            return device
    return None

def fetch_audio_features(token: str, track_ids: List[str]) -> Dict[str, float]:
    """트랙의 오디오 피쳐를 조회해 템포를 얻는다."""
    if not track_ids:
        return {}
    features: Dict[str, float] = {}
    for i in range(0, len(track_ids), 100):
        chunk = track_ids[i : i + 100]
        params = {"ids": ",".join(chunk)}
        response = request_spotify_api("GET", "https://api.spotify.com/v1/audio-features", token, params=params)
        if response.status_code != 200:
            LOGGER.warning("오디오 피쳐 조회 실패: %s", response.text)
            continue
        data = response.json()
        for item in data.get("audio_features", []) or []:
            if item and item.get("id") and item.get("tempo") is not None:
                features[item["id"]] = item["tempo"]
    return features

def fetch_recommendations(token: str, params: Dict[str, Any]) -> List[TrackCandidate]:
    """Spotify 추천 API를 호출한다."""
    query = {k: v for k, v in params.items() if v not in (None, "")}
    response = request_spotify_api("GET", "https://api.spotify.com/v1/recommendations", token, params=query)
    if response.status_code != 200:
        LOGGER.error("추천 곡 조회 실패: %s", response.text)
        raise HTTPException(status_code=502, detail="Spotify 추천 곡을 가져오지 못했습니다.")
    data = response.json()
    tracks = data.get("tracks", [])
    if not tracks:
        return []
    track_ids = [track.get("id") for track in tracks if track.get("id")]
    tempos = fetch_audio_features(token, track_ids) if track_ids else {}
    candidates: List[TrackCandidate] = []
    for track in tracks:
        track_id = track.get("id")
        if not track_id:
            continue
        uri = track.get("uri")
        name = track.get("name", "")
        artists = [artist.get("name", "") for artist in track.get("artists", [])]
        tempo = tempos.get(track_id)
        candidates.append(
            TrackCandidate(
                uri=uri,
                track_id=track_id,
                name=name,
                artists=artists,
                tempo=tempo,
                source="spotify_recommendations",
            )
        )
    return candidates

def call_llm_reranker(state: str, bpm_target: int, candidates: List[TrackCandidate]) -> Optional[List[str]]:
    """LLM 재랭커 서비스 호출."""
    url = os.getenv("LLM_RECOMMENDER_URL")
    if not url:
        return None
    payload = {
        "state": state,
        "bpm_target": bpm_target,
        "candidates": [{"id": c.uri, "bpm": c.tempo} for c in candidates],
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        ranked_ids = data.get("ranked_ids")
        if isinstance(ranked_ids, list):
            return ranked_ids
    except requests.RequestException as exc:
        LOGGER.warning("LLM 재랭킹 호출 실패: %s", exc)
    return None

def rule_based_rerank(bpm_target: int, candidates: List[TrackCandidate]) -> List[TrackCandidate]:
    """템포 차이에 기반한 기본 재랭킹."""
    return sorted(
        candidates,
        key=lambda c: abs(c.tempo - bpm_target) if c.tempo is not None else 9999,
    )

def apply_cooldown_filter(candidates: List[TrackCandidate]) -> List[TrackCandidate]:
    """최근 재생한 트랙/아티스트를 제외한다."""
    now = time.time()
    for cache in (PLAYED_TRACKS, PLAYED_ARTISTS):
        for key in list(cache.keys()):
            if now - cache[key] > COOLDOWN_SECONDS:
                cache.pop(key, None)
    filtered: List[TrackCandidate] = []
    for candidate in candidates:
        if PLAYED_TRACKS.get(candidate.track_id):
            continue
        if any(PLAYED_ARTISTS.get(artist) for artist in candidate.artists):
            continue
        filtered.append(candidate)
    return filtered

def attempt_playback(token: str, device_id: str, candidate: TrackCandidate) -> bool:
    """후보 트랙 재생을 시도한다."""
    payload = {"uris": [candidate.uri]}
    params = {"device_id": device_id}
    try:
        response = request_spotify_api(
            "PUT", "https://api.spotify.com/v1/me/player/play", token, params=params, json=payload
        )
    except HTTPException as exc:
        LOGGER.warning("트랙 재생 요청 중 예외 발생(%s): %s", candidate.uri, exc.detail)
        return False
    if response.status_code in (200, 202, 204):
        return True
    LOGGER.warning("트랙 재생 실패(%s): %s", candidate.uri, response.text)
    return False

def log_playback_event(
    state: str,
    bpm_target: int,
    device: str,
    candidate: TrackCandidate,
    source: str,
    notes: str = "",
) -> None:
    """재생 결과를 CSV로 기록한다."""
    os.makedirs("data", exist_ok=True)
    path = Path("data/playback_log.csv")
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        if is_new:
            writer.writerow(["ts", "state", "bpm_target", "device", "track_uri", "source", "notes"])
        timestamp = datetime.utcnow().isoformat()
        writer.writerow([timestamp, state, bpm_target, device, candidate.uri, source, notes])

def record_cooldown(candidate: TrackCandidate) -> None:
    """성공적으로 재생된 트랙과 아티스트를 쿨다운에 등록한다."""
    now = time.time()
    PLAYED_TRACKS[candidate.track_id] = now
    for artist in candidate.artists:
        if artist:
            PLAYED_ARTISTS[artist] = now

def handle_local_playback(request: PlaybackRequest, source_label: str) -> PlaybackResponse:
    """로컬 라이브러리에서 곡을 선택해 재생한다."""
    try:
        track = LOCAL_LIBRARY.pick_track(request.state, request.bpm_target)
    except LocalLibraryError as exc:
        LOGGER.error("로컬 트랙 선택 실패: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    success = LOCAL_PLAYBACK_ENGINE.play(track)
    if not success:
        LOGGER.error("로컬 음악 재생에 실패했습니다: %s", track.file_path)
        raise HTTPException(
            status_code=500,
            detail="로컬 음악 재생 엔진에서 오류가 발생했습니다. 로그를 확인하세요.",
        )

    candidate = TrackCandidate(
        uri=f"local://{track.file_name}",
        track_id=track.file_name,
        name=track.track_name,
        artists=[track.artist] if track.artist else [],
        tempo=track.bpm,
        source=source_label,
    )
    record_cooldown(candidate)
    artist_label = track.artist or "알 수 없음"
    notes = f"로컬 재생 곡: {track.track_name} / 아티스트: {artist_label}".strip()
    log_playback_event(
        request.state,
        request.bpm_target,
        "local_files",
        candidate,
        source_label,
        notes=notes,
    )
    response = PlaybackResponse(
        ok=True,
        mode="local",
        track_uri=candidate.uri,
        device="local_files",
        source=source_label,
        notes=notes,
    )
    LOGGER.info("로컬 음악 재생 성공: %s", response.model_dump())
    return response

def build_login_url(state: str) -> str:
    """Spotify 인가 URL을 생성한다."""
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
    if not client_id or not redirect_uri:
        raise HTTPException(status_code=500, detail="Spotify OAuth 설정이 누락되었습니다.")
    scope = "user-read-playback-state user-modify-playback-state user-read-currently-playing"
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "show_dialog": "false",
    }
    return f"https://accounts.spotify.com/authorize?{urlencode(params)}"

# ------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------
app = FastAPI(title="Playback Server", version="1.1.0")

@app.get("/health")
def healthcheck() -> Dict[str, str]:
    """상태 확인 엔드포인트."""
    return {"status": "ok"}

@app.get("/login")
def login() -> RedirectResponse:
    """Spotify OAuth 로그인을 시작한다."""
    cleanup_oauth_state()
    state = secrets.token_urlsafe(16)
    OAUTH_STATE[state] = time.time()
    login_url = build_login_url(state)
    LOGGER.info("Spotify 로그인 리다이렉트: %s", login_url)
    return RedirectResponse(login_url)

@app.get("/callback")
def callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, str]:
    """Spotify OAuth 콜백을 처리한다."""
    cleanup_oauth_state()
    if error:
        LOGGER.error("Spotify 인증 중 오류 발생: %s", error)
        raise HTTPException(status_code=400, detail=f"Spotify 인증 실패: {error}")
    if not code or not state:
        raise HTTPException(status_code=400, detail="code 또는 state 값이 없습니다.")
    if state not in OAUTH_STATE:
        raise HTTPException(status_code=400, detail="알 수 없는 state 값입니다. 다시 시도하세요.")
    try:
        tokens = exchange_code_for_token(code)
    except SpotifyAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    TOKEN_STORE.update_tokens(tokens)
    OAUTH_STATE.pop(state, None)
    return {"message": "Spotify 인증이 완료되었습니다."}

@app.get("/devices")
def list_devices() -> Dict[str, Any]:
    """현재 Spotify 계정의 디바이스 목록을 반환한다."""
    try:
        token = TOKEN_STORE.get_access_token()
    except SpotifyAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    devices = fetch_devices(token)
    return {"devices": devices}

@app.post("/config/reload")
def reload_config() -> Dict[str, Any]:
    """상태별 추천 파라미터/로컬 라이브러리 구성을 재로딩한다."""
    config = reload_state_params()
    return {
        "loaded": True,
        "states": list(config.keys()),
        "local_tracks": LOCAL_LIBRARY.count(),
    }

def handle_local_mode(request: PlaybackRequest) -> PlaybackResponse:
    """환경 설정이 로컬 모드일 때 로컬 음악을 재생한다."""
    return handle_local_playback(request, "local_mode")

@app.post("/set_target", response_model=PlaybackResponse)
def set_target(request: PlaybackRequest) -> PlaybackResponse:
    """목표 BPM과 상태를 받아 Spotify 재생을 트리거한다."""
    mode = determine_mode()
    if mode != "spotify":
        return handle_local_mode(request)

    try:
        token = TOKEN_STORE.get_access_token()
    except SpotifyAuthError as exc:
        LOGGER.error("Spotify 토큰 확보 실패: %s", exc)
        raise HTTPException(status_code=401, detail=str(exc))

    config = load_state_params()
    params = get_params_for_state(request.state, request.bpm_target, config)

    cache_key = (request.state.lower(), request.bpm_target)
    if cache_key in RECOMMENDATION_CACHE:
        candidates = RECOMMENDATION_CACHE[cache_key]
        LOGGER.info("추천 후보를 캐시에서 불러왔습니다: %s", cache_key)
    else:
        candidates = fetch_recommendations(token, params)
        if not candidates:
            raise HTTPException(status_code=502, detail="Spotify 추천 결과가 비어 있습니다.")
        RECOMMENDATION_CACHE[cache_key] = candidates

    ranked_ids = call_llm_reranker(request.state, request.bpm_target, candidates)
    if ranked_ids:
        id_to_candidate = {c.uri: c for c in candidates}
        ranked_uri_set = {uri for uri in ranked_ids if uri in id_to_candidate}
        ordered = [id_to_candidate[uri] for uri in ranked_ids if uri in id_to_candidate]
        remaining = [c for c in candidates if c.uri not in ranked_uri_set]
        ordered.extend(remaining)
        ordered_candidates = ordered
        source = "llm_reranked"
    else:
        ordered_candidates = rule_based_rerank(request.bpm_target, candidates)
        source = "rule_sort"

    filtered_candidates = apply_cooldown_filter(ordered_candidates)
    if not filtered_candidates:
        LOGGER.warning("쿨다운 필터로 사용 가능한 후보가 없어 전체 후보 사용")
        filtered_candidates = ordered_candidates

    expected_device_name = os.getenv("SPOTIFY_DEVICE_NAME", "")
    if not expected_device_name:
        raise HTTPException(status_code=400, detail="SPOTIFY_DEVICE_NAME 값을 .env에 설정하세요.")
    devices = fetch_devices(token)
    device = find_device(devices, expected_device_name)
    if not device:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Spotify 디바이스 '{expected_device_name}'을 찾지 못했습니다. "
                "librespot/raspotify 실행 상태를 확인하세요."
            ),
        )

    device_id = device.get("id")
    if not device_id:
        raise HTTPException(status_code=502, detail="선택한 디바이스에 ID가 없습니다.")

    selected: Optional[TrackCandidate] = None
    for candidate in filtered_candidates:
        if attempt_playback(token, device_id, candidate):
            selected = candidate
            record_cooldown(candidate)
            log_playback_event(
                request.state,
                request.bpm_target,
                expected_device_name,
                candidate,
                source,
                notes=f"재생 곡: {candidate.name} / 아티스트: {', '.join(candidate.artists)}",
            )
            break

    if selected is None:
        LOGGER.error("모든 Spotify 후보 재생에 실패해 로컬 폴백을 시도합니다.")
        try:
            return handle_local_playback(request, "spotify_fallback")
        except HTTPException as exc:
            LOGGER.error("로컬 폴백도 실패했습니다: %s", exc.detail)
            raise HTTPException(
                status_code=502,
                detail="Spotify와 로컬 재생 모두 실패했습니다. 로그를 확인하세요.",
            ) from exc

    response = PlaybackResponse(
        ok=True,
        mode="spotify",
        track_uri=selected.uri,
        device=expected_device_name,
        source=source,
        notes=f"재생 곡: {selected.name} / 아티스트: {', '.join(selected.artists)}",
    )
    LOGGER.info("Spotify 재생 성공: %s", response.model_dump())
    return response
