"""Классификаторы обращений — паттерн Strategy (+ Adapter для модели Hugging Face).

Все стратегии возвращают Prediction(category, category_conf, priority, priority_conf, probs).
* KeywordStrategy  — baseline на ключевых словах (как «ручные» правила в helpdesk);
* TfidfStrategy    — TF-IDF (слова + символьные n-граммы) + логистическая регрессия, по модели на задачу;
* HFTransformerStrategy — BERT-подобная модель (transformers): общий энкодер и две головы
  (категория, приоритет). Обучается скриптом ticketflow.train_hf; можно загрузить
  предобученный чекпойнт (например, cointegrated/rubert-tiny2) через HF_MODEL.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..datagen import CATEGORIES, PRIORITIES


@dataclass
class Prediction:
    category: str
    category_conf: float
    priority: str
    priority_conf: float
    model: str

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class ClassifierStrategy(ABC):
    name = "abstract"

    @abstractmethod
    def predict_proba(self, texts: list[str], tiers: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает (P(category) [n,7], P(priority) [n,4])."""

    def predict(self, text: str, tier: str = "free") -> Prediction:
        pc, pp = self.predict_proba([text], [tier])
        c, p = int(pc[0].argmax()), int(pp[0].argmax())
        return Prediction(CATEGORIES[c], float(pc[0, c]), PRIORITIES[p], float(pp[0, p]), self.name)


KEYWORDS = {
    "security": ["взлом", "фишинг", "уязвим", "утечк", "чужие устройства", "незнакомого ip", "не я"],
    "refund": ["возврат", "верните", "вернуть деньги", "вернут оплату"],
    "account_access": ["войти", "пароль", "смс с кодом", "доступ", "двухфактор", "учётную запись", "email в профиле"],
    "delivery": ["заказ", "курьер", "посылк", "доставк", "трек", "пункт выдачи"],
    "billing": ["списал", "оплат", "счёт", "чек", "подписк", "тариф", "акт сверки"],
    "feature_request": ["добавьте", "предлагаю", "было бы удобно", "хотелось бы", "сделайте", "не хватает"],
    "technical": ["ошибка", "вылетает", "не работает", "тормозит", "белый экран", "api", "пропали", "грузится"],
}
URGENT_WORDS = ["срочно", "горит", "третий раз", "работа стоит", "не может работать", "лежит весь", "ни у кого", "в суд"]


class KeywordStrategy(ClassifierStrategy):
    name = "keywords"

    def predict_proba(self, texts, tiers=None):
        pc = np.full((len(texts), len(CATEGORIES)), 0.01)
        pp = np.zeros((len(texts), len(PRIORITIES)))
        for i, t in enumerate(texts):
            low = t.lower()
            for k, c in enumerate(CATEGORIES):
                pc[i, k] += sum(w in low for w in KEYWORDS[c]) * (1.5 if c == "security" else 1.0)  # риск важнее
            if pc[i].max() <= 0.01:
                pc[i, CATEGORIES.index("technical")] += 0.5
            urg = sum(w in low for w in URGENT_WORDS)
            cat = CATEGORIES[int(pc[i].argmax())]
            lvl = 2 if cat == "security" else 0 if cat == "feature_request" else 1
            lvl = min(lvl + urg + ((tiers or ["free"] * len(texts))[i] == "enterprise"), 3)
            pp[i, lvl] = 1.0
        return pc / pc.sum(1, keepdims=True), pp


def _with_tier(texts, tiers):
    tiers = tiers or ["free"] * len(texts)
    return [f"__tier_{tr}__ {t}" for t, tr in zip(texts, tiers)]


class TfidfStrategy(ClassifierStrategy):
    name = "tfidf_logreg"

    def __init__(self, path: str | Path | None = None, models: dict | None = None) -> None:
        import joblib

        m = models or joblib.load(path)
        self.vec, self.cat, self.pri = m["vec"], m["cat"], m["pri"]

    @staticmethod
    def fit(texts, tiers, y_cat, y_pri) -> dict:
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion

        vec = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                                  max_features=60000))])
        X = vec.fit_transform(_with_tier(texts, tiers))
        _ = hstack
        cat = LogisticRegression(max_iter=3000, C=8).fit(X, [CATEGORIES.index(c) for c in y_cat])
        pri = LogisticRegression(max_iter=3000, C=4).fit(X, [PRIORITIES.index(p) for p in y_pri])
        return {"vec": vec, "cat": cat, "pri": pri}

    def predict_proba(self, texts, tiers=None):
        X = self.vec.transform(_with_tier(texts, tiers))
        return self.cat.predict_proba(X), self.pri.predict_proba(X)


class HFTransformerStrategy(ClassifierStrategy):
    """Адаптер к модели transformers с двумя головами (см. ticketflow.classify.hf_model)."""

    name = "hf_transformer"

    def __init__(self, path: str | Path = "models/hf_ticket_model") -> None:
        import torch
        from transformers import PreTrainedTokenizerFast

        from .hf_model import TicketModel

        self.torch = torch
        self.tok = PreTrainedTokenizerFast.from_pretrained(str(path))
        self.model = TicketModel.load(path).eval()
        self.meta = json.loads((Path(path) / "meta.json").read_text())

    def predict_proba(self, texts, tiers=None):
        enc = self.tok(_with_tier(texts, tiers), padding=True, truncation=True, max_length=96, return_tensors="pt")
        with self.torch.no_grad():
            lc, lp = self.model(enc["input_ids"], enc["attention_mask"])
        return self.torch.softmax(lc, -1).numpy(), self.torch.softmax(lp, -1).numpy()


class EnsembleStrategy(ClassifierStrategy):
    """Композиция стратегий: взвешенное среднее вероятностей, веса отдельно для категории и приоритета.

    Модель хорошо работает на знакомых формулировках, правила — устойчивее на новых;
    ансамбль сохраняет точность модели и не «проваливается» на новых формулировках.
    """

    name = "ensemble"

    def __init__(self, parts: list[tuple[ClassifierStrategy, float, float]]) -> None:
        self.parts = parts  # (strategy, w_category, w_priority)

    def predict_proba(self, texts, tiers=None):
        pc = pp = 0
        for st, wc, wp in self.parts:
            a, b = st.predict_proba(texts, tiers)
            pc, pp = pc + wc * a, pp + wp * b
        return pc / sum(w for _, w, _ in self.parts), pp / sum(w for _, _, w in self.parts)


def make_strategy(name: str, path: str | None = None) -> ClassifierStrategy:
    if name == "ensemble":
        return EnsembleStrategy([(HFTransformerStrategy(path or "models/hf_ticket_model"), 0.5, 1.0),
                                 (KeywordStrategy(), 0.5, 0.0)])
    if name == "keywords":
        return KeywordStrategy()
    if name == "tfidf_logreg":
        return TfidfStrategy(path or "models/tfidf_logreg.joblib")
    if name == "hf_transformer":
        return HFTransformerStrategy(path or "models/hf_ticket_model")
    raise ValueError(f"unknown classifier {name}")
