"""Стратегии оценки риска возврата — паттерн Strategy.

Скоринг-сервис получает стратегию через конфиг (SCORING_STRATEGY), что
позволяет A/B-сравнивать модели и безопасно откатываться на правила.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import CAT_FEATURES, FEATURE_NAMES


class ScoringStrategy(ABC):
    name = "abstract"
    version = "0"

    @abstractmethod
    def score(self, features: dict) -> float:
        """Вероятность того, что возврат — злоупотребление, [0..1]."""

    def score_batch(self, rows: list[dict]) -> np.ndarray:
        return np.array([self.score(r) for r in rows])


class RuleBasedStrategy(ScoringStrategy):
    """Baseline, как в типичном «ручном» антифроде: аддитивный балл по правилам."""

    name = "rules"
    version = "rules-1"

    def score(self, f: dict) -> float:
        s = 0.0
        s += 0.35 * (f["return_rate"] > 0.5)
        s += 0.20 * (f["share_fast_returns"] > 0.5)
        s += 0.25 * (f["n_not_received_claims"] >= 1)
        s += 0.15 * (f["share_damaged"] > 0.5)
        s += 0.10 * (f["ret_is_high_value"] == 1)
        s += 0.10 * (f["n_addresses"] >= 3)
        return min(s, 1.0)


class CatBoostStrategy(ScoringStrategy):
    name = "catboost"

    def __init__(self, model_path: str | Path) -> None:
        from catboost import CatBoostClassifier

        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))
        meta = Path(model_path).with_suffix(".json")
        self.version = json.loads(meta.read_text())["version"] if meta.exists() else "catboost"

    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)[FEATURE_NAMES]
        for c in CAT_FEATURES:
            df[c] = df[c].astype(str)
        return df

    def score(self, features: dict) -> float:
        # онлайн-путь: без pandas — список значений в порядке признаков (в ~10 раз быстрее DataFrame)
        row = [str(features[n]) if n in CAT_FEATURES else features[n] for n in FEATURE_NAMES]
        return float(self.model.predict([row], prediction_type="Probability")[0][1])

    def score_batch(self, rows: list[dict]) -> np.ndarray:
        return self.model.predict_proba(self._frame(rows))[:, 1]

    def explain(self, features: dict, top: int = 5) -> list[dict]:
        from catboost import Pool

        df = self._frame([features])
        shap = self.model.get_feature_importance(Pool(df, cat_features=CAT_FEATURES), type="ShapValues")[0][:-1]
        idx = np.argsort(-np.abs(shap))[:top]
        return [{"feature": FEATURE_NAMES[i], "value": features[FEATURE_NAMES[i]], "shap": round(float(shap[i]), 4)} for i in idx]


class BlendStrategy(ScoringStrategy):
    """Взвешенная смесь стратегий (например, модель + правила на период прогрева)."""

    name = "blend"

    def __init__(self, parts: list[tuple[ScoringStrategy, float]]) -> None:
        self.parts = parts
        self.version = "+".join(f"{p.version}*{w}" for p, w in parts)

    def score(self, f: dict) -> float:
        return sum(w * p.score(f) for p, w in self.parts) / sum(w for _, w in self.parts)


def make_strategy(name: str, model_path: str | None = None) -> ScoringStrategy:
    if name == "rules":
        return RuleBasedStrategy()
    if name == "catboost":
        return CatBoostStrategy(model_path or "models/catboost_returns.cbm")
    if name == "blend":
        return BlendStrategy([(CatBoostStrategy(model_path or "models/catboost_returns.cbm"), 0.8), (RuleBasedStrategy(), 0.2)])
    raise ValueError(f"unknown strategy {name}")
