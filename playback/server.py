"""Spotify 및 로컬 백업 재생을 담당하는 FastAPI 서버."""

from __future__ import annotations

import base64
import csv
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests
import yaml
from cachetools import TTLCache
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from loguru import logger
from pydantic import BaseModel, Field

try:
    from playsound import playsound as playsound_sync
except Exception:  # pragma: no cover - 모듈 부재 시 대비
    playsound_sync = None

ENV_PATH = find_dotenv()
if ENV_PATH:
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
os.makedirs("data", exist_ok=True)
logger.add(
    "data/playback.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    rotation="1 MB",
    retention=5,
    encoding="utf-8",
    enqueue=True,
)

TOKEN_PATH = Path("playback/.tokens/spotify.json")
TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
STATE_PARAMS_PATH = Path(os.getenv("STATE_PARAMS_PATH", "playback/config/state_params.yaml"))
LOCAL_MUSIC_DIR = Path(os.getenv("LOCAL_MUSIC_DIR", "playback/local_playlist"))
STATE_CONFIG_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "data": None}
STATE_CONFIG_LOCK = threading.Lock()
RECOMMENDATION_CACHE: TTLCache = TTLCache(maxsize=64, ttl=900)
COOLDOWN_SECONDS = 1800
PLAYED_TRACKS: Dict[str, float] = {}
PLAYED_ARTISTS: Dict[str, float] = {}
OAUTH_STATE: Dict[str, float] = {}
OAUTH_STATE_TTL = 600


@dataclass
class TrackCandidate:
    """Spotify 추천 후보 트랙 정보."""

    uri: str
    track_id: str
    name: str
    artists: List[str]
    tempo: Optional[float]
    source: str


@dataclass
class LocalTrack:
    """로컬 플레이리스트에 등록된 곡 정보."""

    track_name: str
    artist: str
    bpm: Optional[float]
    states: List[str]
    file_name: str
    file_path: Path


class SpotifyAuthError(RuntimeError):
    """Spotify 인증 문제를 나타내는 예외."""


class PlaybackRequest(BaseModel):
    bpm_target: Optional[int] = Field(None, ge=0)
    state: str = Field(..., min_length=1)


class PlaybackResponse(BaseModel):
    ok: bool
    mode: str
    track_uri: Optional[str] = None
    device: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None


class SpotifyTokenStore:
    """Spotify 토큰을 파일에 저장하고 자동 갱신한다."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as fp:
                    self._data = json.load(fp)
            except json.JSONDecodeError:
                logger.warning("저장된 Spotify 토큰을 읽지 못했습니다. 초기화합니다: {}", self.path)
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as fp:
            json.dump(self._data, fp, ensure_ascii=False, indent=2)

    def update(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            expires_in = int(payload.get("expires_in", 3600))
            expires_at = int(time.time()) + expires_in
            self._data.update(
                {
                    "access_token": payload.get("access_token"),
                    "refresh_token": payload.get("refresh_token") or self._data.get("refresh_token"),
                    "expires_at": expires_at,
                }
            )
            self._save()
            logger.info("Spotify 토큰 정보를 갱신했습니다 (만료 시각: {})", expires_at)

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            if self.path.exists():
                self.path.unlink()

    def require_refresh_token(self) -> str:
        refresh_token = self._data.get("refresh_token")
        if not refresh_token:
            raise SpotifyAuthError("Spotify 인증이 필요합니다. /login을 먼저 수행하세요.")
        return refresh_token

    def get_access_token(self) -> str:
        with self._lock:
            access_token = self._data.get("access_token")
            expires_at = self._data.get("expires_at", 0)
        now = int(time.time())
        if access_token and now < expires_at - 60:
            return access_token
        refresh_token = self.require_refresh_token()
        tokens = refresh_access_token(refresh_token)
        self.update(tokens)
        return self._data.get("access_token", "")


TOKEN_STORE = SpotifyTokenStore(TOKEN_PATH)


class LocalLibrary:
    """로컬 음악 라이브러리를 관리한다."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.csv_path = root_dir / "library.csv"
        self.tracks_dir = root_dir / "tracks"
        self._lock = threading.Lock()
        self._tracks: List[LocalTrack] = []
        self.load()

    def load(self) -> None:
        with self._lock:
            self._tracks = []
            if not self.csv_path.exists():
                logger.warning("로컬 라이브러리 CSV가 존재하지 않습니다: {}", self.csv_path)
                return
            try:
                with self.csv_path.open("r", encoding="utf-8") as fp:
                    reader = csv.DictReader(fp)
                    for row in reader:
                        file_name = (row.get("file_name") or "").strip()
                        track_name = (row.get("track_name") or "").strip()
                        artist = (row.get("artist") or "").strip()
                        bpm_value = row.get("bpm")
                        bpm = float(bpm_value) if bpm_value else None
                        states_raw = row.get("states") or ""
                        states = [token.strip().lower() for token in states_raw.replace(",", ";").split(";") if token.strip()]
                        file_path = self.tracks_dir / file_name
                        if not track_name or not file_name:
                            logger.warning("CSV 행에 필수 정보가 없어 건너뜁니다: %s", row)
                            continue
                        if not file_path.exists():
                            logger.warning("MP3 파일을 찾을 수 없어 건너뜁니다: %s", file_path)
                            continue
                        self._tracks.append(
                            LocalTrack(
                                track_name=track_name,
                                artist=artist,
                                bpm=bpm,
                                states=states,
                                file_name=file_name,
                                file_path=file_path,
                            )
                        )
            except Exception as exc:  # pragma: no cover - 파일 파싱 예외 대비
                logger.error("로컬 라이브러리 로드 중 오류: {}", exc)
                self._tracks = []

    def available_tracks(self) -> List[LocalTrack]:
        with self._lock:
            return list(self._tracks)

    def pick_tracks(self, state: str, bpm_target: Optional[int]) -> List[LocalTrack]:
        state_key = state.lower()
        candidates = []
        for track in self.available_tracks():
            if state_key in track.states or not track.states:
                candidates.append(track)
        if bpm_target is not None:
            candidates.sort(key=lambda t: abs((t.bpm or bpm_target) - bpm_target))
        return candidates

    def play(self, track: LocalTrack) -> bool:
        logger.info("로컬 곡 재생 시도: {} - {}", track.artist, track.track_name)
        if not track.file_path.exists():
            logger.error("로컬 곡 파일이 존재하지 않아 재생할 수 없습니다: {}", track.file_path)
            return False
        if playsound_sync is not None:
            try:
                threading.Thread(target=playsound_sync, args=(str(track.file_path),), daemon=True).start()
                return True
            except Exception as exc:  # pragma: no cover
                logger.warning("playsound 재생 실패, 시스템 기본 플레이어 시도: {}", exc)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(track.file_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(track.file_path)])
            else:
                subprocess.Popen(["xdg-open", str(track.file_path)])
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("시스템 플레이어 호출 실패: {}", exc)
            return False


LOCAL_LIBRARY = LocalLibrary(LOCAL_MUSIC_DIR)


def cleanup_oauth_state() -> None:
    now = time.time()
    expired = [key for key, ts in OAUTH_STATE.items() if now - ts > OAUTH_STATE_TTL]
    for key in expired:
        OAUTH_STATE.pop(key, None)


def load_state_params(force: bool = False) -> Dict[str, Any]:
    with STATE_CONFIG_LOCK:
        if not force and STATE_CONFIG_CACHE["data"] is not None:
            return STATE_CONFIG_CACHE["data"]
        try:
            with STATE_PARAMS_PATH.open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
            STATE_CONFIG_CACHE.update({"loaded_at": time.time(), "data": data})
            logger.info("상태별 Spotify 파라미터 구성을 로드했습니다: {}", STATE_PARAMS_PATH)
            return data
        except FileNotFoundError:
            logger.warning("state_params.yaml 파일을 찾을 수 없어 기본값 없이 동작합니다: {}", STATE_PARAMS_PATH)
        except yaml.YAMLError as exc:
            logger.warning("state_params.yaml 파싱 실패: {}", exc)
        STATE_CONFIG_CACHE.update({"loaded_at": time.time(), "data": {}})
        return {}


def get_params_for_state(state: str, bpm_target: Optional[int], cfg: Dict[str, Any]) -> Dict[str, Any]:
    normalized = state.strip()
    base = cfg.get(normalized) or cfg.get("Focus") or {}
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
    genres = base.get("genres", ["lo-fi", "focus", "study", "chill"])
    market = os.getenv("SPOTIFY_MARKET", "KR")
    return {
        "seed_genres": ",".join(genres[:5]),
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


def build_login_url(state: str) -> str:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
    if not client_id or not redirect_uri:
        raise HTTPException(status_code=500, detail="Spotify OAuth 환경 변수가 설정되지 않았습니다.")
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


def exchange_code_for_token(code: str) -> Dict[str, Any]:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
    if not all([client_id, client_secret, redirect_uri]):
        raise SpotifyAuthError("Spotify OAuth 설정이 누락되었습니다. .env를 확인하세요.")
    token_url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    response = requests.post(token_url, data=data, headers=headers, timeout=10)
    if response.status_code != 200:
        logger.error("Spotify 인가 코드 교환 실패: {}", response.text)
        raise SpotifyAuthError("Spotify 토큰 발급에 실패했습니다. 다시 로그인하세요.")
    return response.json()


def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not all([client_id, client_secret]):
        raise SpotifyAuthError("Spotify 클라이언트 자격 증명이 없습니다. .env를 확인하세요.")
    token_url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    response = requests.post(token_url, data=data, headers=headers, timeout=10)
    if response.status_code != 200:
        logger.error("Spotify 토큰 갱신 실패: {}", response.text)
        raise SpotifyAuthError("Spotify 토큰 갱신에 실패했습니다. /login을 다시 실행하세요.")
    return response.json()


def request_spotify_api(method: str, url: str, token: str, **kwargs: Any) -> requests.Response:
    headers = kwargs.pop("headers", {})
    headers.update({"Authorization": f"Bearer {token}"})
    try:
        response = requests.request(method, url, headers=headers, timeout=10, **kwargs)
    except requests.RequestException as exc:
        logger.error("Spotify API 요청 실패: {}", exc)
        raise HTTPException(status_code=502, detail="Spotify API 호출에 실패했습니다.") from exc
    return response


def fetch_devices(token: str) -> List[Dict[str, Any]]:
    response = request_spotify_api("GET", "https://api.spotify.com/v1/me/player/devices", token)
    if response.status_code != 200:
        logger.error("디바이스 조회 실패: {}", response.text)
        raise HTTPException(status_code=502, detail="Spotify 디바이스 정보를 가져오지 못했습니다.")
    return response.json().get("devices", [])


def find_device(devices: List[Dict[str, Any]], expected_name: str) -> Optional[Dict[str, Any]]:
    target = expected_name.casefold()
    for device in devices:
        name = (device.get("name") or "").casefold()
        if name == target:
            return device
    return None


def fetch_audio_features(token: str, track_ids: List[str]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for chunk_start in range(0, len(track_ids), 100):
        chunk = track_ids[chunk_start : chunk_start + 100]
        params = {"ids": ",".join(chunk)}
        response = request_spotify_api("GET", "https://api.spotify.com/v1/audio-features", token, params=params)
        if response.status_code != 200:
            logger.warning("오디오 피쳐 조회 실패: {}", response.text)
            continue
        for item in response.json().get("audio_features", []) or []:
            if item and item.get("id") and item.get("tempo") is not None:
                results[item["id"]] = item["tempo"]
    return results


def fetch_recommendations(token: str, params: Dict[str, Any]) -> List[TrackCandidate]:
    query = {k: v for k, v in params.items() if v not in (None, "")}
    response = request_spotify_api("GET", "https://api.spotify.com/v1/recommendations", token, params=query)
    if response.status_code != 200:
        logger.error("추천 곡 조회 실패: {}", response.text)
        raise HTTPException(status_code=502, detail="Spotify 추천 결과를 가져오지 못했습니다.")
    payload = response.json()
    tracks = payload.get("tracks", [])
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


def call_llm_reranker(state: str, bpm_target: Optional[int], candidates: List[TrackCandidate]) -> Optional[List[str]]:
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
            logger.info("LLM 재랭킹 결과를 수신했습니다 (%d개)", len(ranked_ids))
            return ranked_ids
    except requests.RequestException as exc:
        logger.warning("LLM 재랭킹 호출 실패: {}", exc)
    return None


def rule_based_rerank(bpm_target: Optional[int], candidates: List[TrackCandidate]) -> List[TrackCandidate]:
    if bpm_target is None:
        return candidates
    return sorted(
        candidates,
        key=lambda c: abs(c.tempo - bpm_target) if c.tempo is not None else 9999,
    )


def apply_cooldown_filter(candidates: List[TrackCandidate]) -> List[TrackCandidate]:
    now = time.time()
    for cache in (PLAYED_TRACKS, PLAYED_ARTISTS):
        expired = [key for key, ts in cache.items() if now - ts > COOLDOWN_SECONDS]
        for key in expired:
            cache.pop(key, None)
    filtered: List[TrackCandidate] = []
    for candidate in candidates:
        if PLAYED_TRACKS.get(candidate.track_id):
            continue
        if any(PLAYED_ARTISTS.get(artist) for artist in candidate.artists if artist):
            continue
        filtered.append(candidate)
    return filtered


def record_cooldown(candidate: TrackCandidate) -> None:
    now = time.time()
    PLAYED_TRACKS[candidate.track_id] = now
    for artist in candidate.artists:
        if artist:
            PLAYED_ARTISTS[artist] = now


def attempt_spotify_playback(token: str, device_id: str, candidate: TrackCandidate) -> bool:
    payload = {"uris": [candidate.uri]}
    params = {"device_id": device_id}
    response = request_spotify_api(
        "PUT",
        "https://api.spotify.com/v1/me/player/play",
        token,
        params=params,
        json=payload,
    )
    if response.status_code in (200, 202, 204):
        return True
    logger.warning("Spotify 재생 실패(%s): %s", candidate.uri, response.text)
    return False


def log_playback_event(
    mode: str,
    state: str,
    bpm_target: Optional[int],
    device: Optional[str],
    track_ref: str,
    source: str,
    notes: str = "",
) -> None:
    path = Path("data/playback_log.csv")
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        if is_new:
            writer.writerow(["ts", "mode", "state", "bpm_target", "device", "track_uri", "source", "notes"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            mode,
            state,
            bpm_target if bpm_target is not None else "",
            device or "",
            track_ref,
            source,
            notes,
        ])


def determine_mode() -> str:
    mode = os.getenv("PLAYBACK_MODE", "spotify").lower()
    if mode not in {"spotify", "local"}:
        return "spotify"
    return mode


def handle_local_playback(request: PlaybackRequest, source: str = "local") -> PlaybackResponse:
    candidates = LOCAL_LIBRARY.pick_tracks(request.state, request.bpm_target)
    if not candidates:
        logger.error("로컬 라이브러리에 사용할 곡이 없습니다. library.csv를 확인하세요.")
        raise HTTPException(status_code=500, detail="로컬 음악 라이브러리에 곡이 없습니다.")
    for track in candidates:
        if LOCAL_LIBRARY.play(track):
            notes = f"재생 곡: {track.track_name} / 아티스트: {track.artist}"
            log_playback_event("local", request.state, request.bpm_target, "local_player", track.file_name, source, notes)
            logger.info("로컬 곡 재생 성공: %s", notes)
            return PlaybackResponse(
                ok=True,
                mode="local",
                track_uri=track.file_name,
                device="local_player",
                source=source,
                notes=notes,
            )
    logger.error("모든 로컬 곡 재생에 실패했습니다.")
    raise HTTPException(status_code=500, detail="로컬 음악 재생에 실패했습니다.")


app = FastAPI(title="Playback Server", version="1.1.0")


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/login")
def login() -> RedirectResponse:
    cleanup_oauth_state()
    state_token = secrets.token_urlsafe(16)
    OAUTH_STATE[state_token] = time.time()
    login_url = build_login_url(state_token)
    logger.info("Spotify 로그인 페이지로 리다이렉트합니다")
    return RedirectResponse(login_url)


@app.get("/callback")
def callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None) -> Dict[str, str]:
    cleanup_oauth_state()
    if error:
        logger.error("Spotify 인증 오류: {}", error)
        raise HTTPException(status_code=400, detail=f"Spotify 인증 실패: {error}")
    if not code or not state:
        raise HTTPException(status_code=400, detail="code/state 파라미터가 누락되었습니다.")
    if state not in OAUTH_STATE:
        raise HTTPException(status_code=400, detail="만료되었거나 알 수 없는 state 값입니다. 다시 시도하세요.")
    try:
        tokens = exchange_code_for_token(code)
        TOKEN_STORE.update(tokens)
    except SpotifyAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        OAUTH_STATE.pop(state, None)
    return {"message": "Spotify 인증이 완료되었습니다."}


@app.get("/devices")
def list_devices() -> Dict[str, Any]:
    try:
        token = TOKEN_STORE.get_access_token()
    except SpotifyAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    devices = fetch_devices(token)
    expected = os.getenv("SPOTIFY_DEVICE_NAME", "")
    for device in devices:
        device["is_target"] = device.get("name", "").casefold() == expected.casefold()
    return {"devices": devices, "target": expected}


@app.post("/config/reload")
def reload_config() -> Dict[str, Any]:
    config = load_state_params(force=True)
    LOCAL_LIBRARY.load()
    return {
        "loaded": True,
        "states": list(config.keys()),
        "local_tracks": len(LOCAL_LIBRARY.available_tracks()),
    }


@app.post("/set_target", response_model=PlaybackResponse)
def set_target(request: PlaybackRequest) -> PlaybackResponse:
    mode = determine_mode()
    if mode == "local":
        logger.info("PLAYBACK_MODE=local 설정으로 로컬 곡을 재생합니다")
        return handle_local_playback(request, source="local_mode")

    try:
        token = TOKEN_STORE.get_access_token()
    except SpotifyAuthError as exc:
        logger.error("Spotify 토큰 확보 실패: {}", exc)
        raise HTTPException(status_code=401, detail=str(exc))

    config = load_state_params()
    params = get_params_for_state(request.state, request.bpm_target, config)

    cache_key = (request.state.lower(), request.bpm_target)
    if cache_key in RECOMMENDATION_CACHE:
        candidates = RECOMMENDATION_CACHE[cache_key]
        logger.info("추천 후보를 캐시에서 사용합니다: %s", cache_key)
    else:
        candidates = fetch_recommendations(token, params)
        if not candidates:
            logger.warning("Spotify 추천 결과가 비어 있어 로컬 폴백을 시도합니다")
            return handle_local_playback(request, source="spotify_empty")
        RECOMMENDATION_CACHE[cache_key] = candidates

    ranked_ids = call_llm_reranker(request.state, request.bpm_target, candidates)
    if ranked_ids:
        mapping = {candidate.uri: candidate for candidate in candidates}
        ordered = [mapping[uri] for uri in ranked_ids if uri in mapping]
        ordered.extend([candidate for candidate in candidates if candidate.uri not in mapping])
        ordered_candidates = ordered
        source = "llm_reranked"
    else:
        ordered_candidates = rule_based_rerank(request.bpm_target, candidates)
        source = "rule_sort"

    filtered_candidates = apply_cooldown_filter(ordered_candidates)
    if not filtered_candidates:
        logger.warning("쿨다운으로 사용 가능한 Spotify 후보가 없습니다. 전체 후보를 사용합니다.")
        filtered_candidates = ordered_candidates

    expected_device = os.getenv("SPOTIFY_DEVICE_NAME", "")
    if not expected_device:
        raise HTTPException(status_code=422, detail="SPOTIFY_DEVICE_NAME 값을 .env에 설정하세요.")
    devices = fetch_devices(token)
    device = find_device(devices, expected_device)
    if not device:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Spotify 디바이스 '{expected_device}'을 찾지 못했습니다. librespot/raspotify 실행 여부와 이름을 확인하세요."
            ),
        )
    device_id = device.get("id")
    if not device_id:
        raise HTTPException(status_code=502, detail="선택한 디바이스에서 ID를 확인할 수 없습니다.")

    selected: Optional[TrackCandidate] = None
    for candidate in filtered_candidates:
        if attempt_spotify_playback(token, device_id, candidate):
            selected = candidate
            record_cooldown(candidate)
            notes = f"재생 곡: {candidate.name} / 아티스트: {', '.join(candidate.artists)}"
            log_playback_event("spotify", request.state, request.bpm_target, expected_device, candidate.uri, source, notes)
            logger.info("Spotify 재생 성공: {}", notes)
            break

    if selected is None:
        logger.error("모든 Spotify 재생 시도가 실패했습니다. 로컬 폴백을 진행합니다.")
        return handle_local_playback(request, source="spotify_fallback")

    return PlaybackResponse(
        ok=True,
        mode="spotify",
        track_uri=selected.uri,
        device=expected_device,
        source=source,
        notes=f"재생 곡: {selected.name} / 아티스트: {', '.join(selected.artists)}",
    )


load_state_params()

