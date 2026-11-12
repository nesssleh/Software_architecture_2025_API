"""Извлечение навыков — паттерн Strategy.

* GazetteerExtractor — словарь синонимов таксономии + лемматизация (spaCy), точный и быстрый;
* SemanticExtractor  — для предложений без явных упоминаний: близость эмбеддинга предложения
                       к описаниям навыков (ловит «неявные» формулировки опыта);
* LLMExtractor       — LLM со structured output (JSON) + жёсткая валидация по таксономии
                       и проверка, что evidence действительно есть в тексте (контроль галлюцинаций);
* HybridExtractor    — объединение стратегий, LLM опционален и падает в fallback при недоступности.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..embedding.embedders import Embedder
from ..llm import LLMClient, LLMUnavailable
from ..taxonomy import SKILLS, alias_index

log = logging.getLogger(__name__)
_SENT = re.compile(r"(?<=[.!?;])\s+")
_ALIAS_HIT = re.compile(r"(?<![\w+#/])(" + "|".join(re.escape(a) for a in sorted(alias_index(), key=len, reverse=True))
                        + r")(?![\w+#])", re.I)


@dataclass
class ExtractedSkill:
    name: str
    category: str
    evidence: str
    method: str
    confidence: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class SkillExtractor(ABC):
    name = "abstract"

    @abstractmethod
    def extract(self, text: str) -> list[ExtractedSkill]: ...


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


class GazetteerExtractor(SkillExtractor):
    name = "gazetteer"

    def __init__(self, lemmatizer=None) -> None:
        self.idx = alias_index()
        self.stems: list[tuple[re.Pattern, str]] = []
        alts = sorted(self.idx, key=len, reverse=True)
        self.rx = re.compile(r"(?<![\w+#/])(" + "|".join(re.escape(a) for a in alts) + r")(?![\w+#])", re.I)
        # русская морфология без лемматизатора: «питоне», «постгресом», «кафкой» -> основа + окончание
        for a in alts:
            if re.fullmatch(r"[а-яё]{4,}", a):
                stem = a[:-1] if a[-1] in "аяоеиыйь" else a
                self.stems.append((re.compile(r"(?<![\w])" + re.escape(stem) + r"[а-яё]{0,3}(?![\w])", re.I), self.idx[a]))
        self.lemmatizer = lemmatizer  # SpacyVectorEmbedder.lemmas — «постгресом» -> «постгрес»

    def extract(self, text: str) -> list[ExtractedSkill]:
        found: dict[str, ExtractedSkill] = {}
        for s in sentences(text):
            cands = [s]
            if self.lemmatizer:
                cands.append(" ".join(self.lemmatizer(s)))
            for c in cands:
                for m in self.rx.finditer(c):
                    name = self.idx[m.group(1).lower()]
                    if name not in found:
                        found[name] = ExtractedSkill(name, SKILLS[name].category, s, self.name, 1.0)
            for rx, name in self.stems:
                if name not in found and rx.search(s):
                    found[name] = ExtractedSkill(name, SKILLS[name].category, s, self.name, 0.95)
        return list(found.values())


class SemanticExtractor(SkillExtractor):
    name = "semantic"

    def __init__(self, embedder: Embedder, threshold: float = 0.6, margin: float = 0.02) -> None:
        self.embedder, self.threshold, self.margin = embedder, threshold, margin
        self.names = list(SKILLS)
        self.mat = embedder.embed([f"{n}. {SKILLS[n].description}" for n in self.names])

    def extract(self, text: str, skip: set[str] | None = None) -> list[ExtractedSkill]:
        sents = [s for s in sentences(text) if not _ALIAS_HIT.search(s)]  # только предложения без явных упоминаний
        if not sents:
            return []
        sims = self.embedder.embed(sents) @ self.mat.T
        out: dict[str, ExtractedSkill] = {}
        for i, s in enumerate(sents):
            order = np.argsort(-sims[i])
            best, second = sims[i, order[0]], sims[i, order[1]]
            name = self.names[order[0]]
            if best >= self.threshold and best - second >= self.margin and name not in (skip or set()):
                if name not in out or out[name].confidence < best:
                    out[name] = ExtractedSkill(name, SKILLS[name].category, s, self.name, round(float(best), 3))
        return list(out.values())


EXTRACT_SYSTEM = """Ты извлекаешь профессиональные навыки из текста резюме или вакансии.
Используй ТОЛЬКО навыки из списка (каноническое написание): {skills}.
Для каждого навыка укажи evidence — точную цитату из текста (5-15 слов), подтверждающую навык.
Формат: {{"skills": [{{"name": "...", "evidence": "..."}}]}}"""


class LLMExtractor(SkillExtractor):
    name = "llm"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract(self, text: str) -> list[ExtractedSkill]:
        data = self.llm.complete_json(EXTRACT_SYSTEM.format(skills=", ".join(SKILLS)), text[:6000])
        out, low = {}, text.lower()
        for item in data.get("skills", []):
            name, ev = item.get("name"), (item.get("evidence") or "").strip()
            if name not in SKILLS:          # навык вне таксономии — галлюцинация, отбрасываем
                log.info("llm hallucinated skill %r dropped", name)
                continue
            if not ev or ev.lower()[:40] not in low:  # цитаты нет в тексте — снижаем доверие
                conf = 0.5
            else:
                conf = 0.9
            out.setdefault(name, ExtractedSkill(name, SKILLS[name].category, ev, self.name, conf))
        return [s for s in out.values() if s.confidence >= 0.5]


class HybridExtractor(SkillExtractor):
    name = "hybrid"

    def __init__(self, gazetteer: GazetteerExtractor, semantic: SemanticExtractor | None = None,
                 llm: LLMExtractor | None = None) -> None:
        self.g, self.s, self.l = gazetteer, semantic, llm

    def extract(self, text: str) -> list[ExtractedSkill]:
        res = {e.name: e for e in self.g.extract(text)}
        if self.l is not None:
            try:
                for e in self.l.extract(text):
                    res.setdefault(e.name, e)
            except (LLMUnavailable, ValueError) as err:  # LLM недоступна / мусорный ответ — работаем без неё
                log.warning("LLM extractor fallback: %s", err)
        if self.s is not None:
            for e in self.s.extract(text, skip=set(res)):
                res.setdefault(e.name, e)
        return sorted(res.values(), key=lambda e: e.name)
