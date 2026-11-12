"""Сборка фасада из переменных окружения (единое место конфигурации)."""
from __future__ import annotations

import os

from .embedding.embedders import SpacyVectorEmbedder, make_embedder
from .explain.explainers import LLMExplainer, TemplateExplainer
from .extraction.extractors import GazetteerExtractor, HybridExtractor, LLMExtractor, SemanticExtractor
from .facade import SkillMatchFacade
from .llm import NullLLM, make_llm
from .storage.repository import DocumentRepository, make_repository


def build_facade(repo: DocumentRepository | None = None) -> SkillMatchFacade:
    emb = make_embedder()
    llm = make_llm()
    lemm = emb.lemmas if isinstance(emb, SpacyVectorEmbedder) and os.getenv("LEMMATIZE", "0") == "1" else None
    extractor = HybridExtractor(GazetteerExtractor(lemm), SemanticExtractor(emb),
                                None if isinstance(llm, NullLLM) else LLMExtractor(llm))
    explainer = TemplateExplainer() if isinstance(llm, NullLLM) else LLMExplainer(llm)
    return SkillMatchFacade(repo or make_repository(), extractor, emb, explainer)
