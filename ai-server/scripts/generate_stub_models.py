"""네트워크 없이 사용할 수 있는 간이 ML 모델을 생성해 저장한다."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Iterable, List


STATE_ORDER = ["HighStress", "Focus", "Flow", "LowFocus"]


def _ensure_records(X: Iterable) -> List[dict]:
    """pandas DataFrame 또는 리스트 입력을 공통 형태로 변환한다."""

    if hasattr(X, "to_dict"):
        return list(X.to_dict(orient="records"))  # type: ignore[arg-type]
    if isinstance(X, list):
        return X
    return [dict(item) for item in X]  # type: ignore[arg-type]


class SimpleStateModel:
    """LightGBM 대체용 간이 상태 분류기."""

    classes_ = STATE_ORDER

    def predict(self, X: Iterable) -> List[str]:
        outputs = []
        for row in _ensure_records(X):
            outputs.append(self._predict_one(row))
        return outputs

    def predict_proba(self, X: Iterable) -> List[List[float]]:
        prob_list = []
        for row in _ensure_records(X):
            logits = self._compute_logits(row)
            exps = [math.exp(v) for v in logits]
            denom = sum(exps)
            probs = [v / denom for v in exps]
            prob_list.append(probs)
        return prob_list

    def _predict_one(self, row: dict) -> str:
        probs = self._compute_state_scores(row)
        return max(probs, key=probs.get)

    def _compute_logits(self, row: dict) -> List[float]:
        scores = self._compute_state_scores(row)
        return [scores[label] for label in STATE_ORDER]

    def _compute_state_scores(self, row: dict) -> dict:
        rmssd = float(row.get("rmssd", 0.0))
        hr_mean = float(row.get("hr_mean", 0.0))
        hr_z = float(row.get("hr_z", 0.0))
        acc_rms = float(row.get("acc_rms", 0.0))
        session_min = float(row.get("session_min", 0.0))

        scores = {
            "HighStress": -abs(rmssd - 18) - 0.5 * max(0.0, hr_z - 0.6),
            "Focus": -abs(rmssd - 28) - 0.2 * abs(hr_mean - 100),
            "Flow": -abs(hr_mean - 118) - 0.3 * abs(rmssd - 32),
            "LowFocus": -abs(rmssd - 40) - 0.2 * abs(hr_z + 0.2),
        }
        if acc_rms > 0.04:
            scores["HighStress"] += 0.4
        if session_min > 60:
            scores["LowFocus"] += 0.2
        return scores


class SimpleSQIModel:
    """RandomForest 대체용 간이 SQI 분류기."""

    classes_ = [0, 1]

    def predict(self, X: Iterable) -> List[float]:
        return [float(probs[1] >= 0.5) for probs in self.predict_proba(X)]

    def predict_proba(self, X: Iterable) -> List[List[float]]:
        prob_list = []
        for row in _ensure_records(X):
            prob = self._score(row)
            prob_list.append([1.0 - prob, prob])
        return prob_list

    def _score(self, row: dict) -> float:
        acc_rms = float(row.get("acc_rms", 0.0))
        hr_std = float(row.get("hr_std", 0.0))
        rmssd = float(row.get("rmssd", 0.0))
        sdnn = float(row.get("sdnn", 0.0))
        pnn50 = float(row.get("pnn50", 0.0))

        quality = 0.9
        quality -= min(0.6, acc_rms * 8)
        quality -= min(0.3, max(0.0, hr_std - 6) * 0.05)
        quality += min(0.1, max(0.0, rmssd - 25) * 0.002)
        quality += min(0.05, max(0.0, pnn50 - 0.2))
        quality -= min(0.2, max(0.0, 30 - sdnn) * 0.005)
        return max(0.0, min(1.0, quality))


def save_models() -> None:
    base_dir = Path(__file__).resolve().parents[1] / "models"
    base_dir.mkdir(parents=True, exist_ok=True)
    state_path = base_dir / "state_lgbm.pkl"
    sqi_path = base_dir / "sqi_rf.pkl"
    with state_path.open("wb") as fp:
        pickle.dump(SimpleStateModel(), fp)
    with sqi_path.open("wb") as fp:
        pickle.dump(SimpleSQIModel(), fp)
    print(f"모델 파일 저장 완료: {state_path}, {sqi_path}")


if __name__ == "__main__":
    save_models()

