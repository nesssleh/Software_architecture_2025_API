"""Эмбеддинги текста — Strategy + Adapter.

* SpacyVectorEmbedder       — предобученные векторы ru_core_news_md (300d), среднее по токенам
                              без стоп-слов; работает офлайн, ~40 МБ.
* SentenceTransformerEmbedder — sentence-transformers (например, intfloat/multilingual-e5-small),
                              используется в docker-compose при наличии доступа к HF Hub.
* TfidfEmbedder             — baseline: TF-IDF по символьным n-граммам (разреженный → плотный SVD).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


class Embedder(ABC):
    name = "abstract"
    dim = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Возвращает L2-нормированную матрицу [n, dim]."""


@lru_cache(maxsize=2)
def _spacy(model: str):
    import spacy

    return spacy.load(model, disable=["parser", "ner"])


class SpacyVectorEmbedder(Embedder):
    name = "spacy-ru-md"
    dim = 300

    def __init__(self, model: str = "ru_core_news_md") -> None:
        self.nlp = _spacy(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=64)):
            vecs = [t.vector for t in doc if t.has_vector and not t.is_stop and not t.is_punct]
            if vecs:
                out[i] = np.mean(vecs, 0)
        return l2norm(out)

    def lemmas(self, text: str) -> list[str]:
        return [t.lemma_.lower() for t in self.nlp(text)]


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str = "intfloat/multilingual-e5-small") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = model

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


class TfidfEmbedder(Embedder):
    name = "tfidf-char"

    def __init__(self, corpus: list[str] | None = None) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        self.vec.fit(corpus or ["init"])

    def embed(self, texts: list[str]) -> np.ndarray:
        return l2norm(self.vec.transform(texts).toarray().astype(np.float32))


def make_embedder() -> Embedder:
    kind = os.getenv("EMBEDDER", "spacy")
    if kind == "st":
        return SentenceTransformerEmbedder(os.getenv("ST_MODEL", "intfloat/multilingual-e5-small"))
    return SpacyVectorEmbedder()
