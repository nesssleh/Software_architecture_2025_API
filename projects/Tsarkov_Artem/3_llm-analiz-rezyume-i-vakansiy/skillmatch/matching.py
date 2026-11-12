"""Сопоставление навыков кандидата и требований вакансии.

Для каждого требования ищется лучший навык кандидата:
  exact (1.0)  — тот же канонический навык;
  related (0.6)— связанный по таксономии (PostgreSQL ↔ MySQL, PyTorch ↔ TensorFlow);
  semantic (cos·0.5) — близкий по эмбеддингу описания навыка (cos ≥ 0.75).
Итоговый скор = 0.8 · покрытие must-have + 0.2 · покрытие nice-to-have
(если nice-to-have нет — только must-have), плюс семантическая близость документов
для ранжирования кандидатов с одинаковым покрытием.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .embedding.embedders import Embedder
from .taxonomy import SKILLS, related


@dataclass
class RequirementMatch:
    requirement: str
    matched_by: str | None
    kind: str  # exact | related | semantic | missing
    score: float


@dataclass
class MatchResult:
    score: float
    must_coverage: float
    nice_coverage: float
    doc_similarity: float
    matches: list[RequirementMatch] = field(default_factory=list)

    @property
    def missing_must(self) -> list[str]:
        return [m.requirement for m in self.matches if m.kind == "missing" and m.requirement in self._must]

    _must: set = field(default_factory=set, repr=False)

    def as_dict(self) -> dict:
        return {"score": round(self.score, 4), "must_coverage": round(self.must_coverage, 4),
                "nice_coverage": round(self.nice_coverage, 4), "doc_similarity": round(self.doc_similarity, 4),
                "missing_must": self.missing_must,
                "matches": [m.__dict__ for m in self.matches]}


class SkillMatcher:
    def __init__(self, embedder: Embedder, semantic_min: float = 0.75) -> None:
        self.embedder = embedder
        self.names = list(SKILLS)
        emb = embedder.embed([f"{n}. {SKILLS[n].description}" for n in self.names])
        self.sim = emb @ emb.T
        self.pos = {n: i for i, n in enumerate(self.names)}
        self.semantic_min = semantic_min

    def _best(self, req: str, have: set[str]) -> RequirementMatch:
        if req in have:
            return RequirementMatch(req, req, "exact", 1.0)
        rel = [h for h in have if related(req, h)]
        if rel:
            return RequirementMatch(req, rel[0], "related", 0.6)
        if have and req in self.pos:
            cand = [(self.sim[self.pos[req], self.pos[h]], h) for h in have if h in self.pos]
            if cand:
                s, h = max(cand)
                if s >= self.semantic_min:
                    return RequirementMatch(req, h, "semantic", round(float(s) * 0.5, 3))
        return RequirementMatch(req, None, "missing", 0.0)

    def match(self, resume_skills: set[str], must: list[str], nice: list[str] | None = None,
              doc_similarity: float = 0.0) -> MatchResult:
        nice = nice or []
        mm = [self._best(r, resume_skills) for r in must]
        nm = [self._best(r, resume_skills) for r in nice]
        mc = float(np.mean([m.score for m in mm])) if mm else 1.0
        nc = float(np.mean([m.score for m in nm])) if nm else 0.0
        score = 0.8 * mc + 0.2 * nc if nm else mc
        res = MatchResult(score=0.9 * score + 0.1 * max(doc_similarity, 0), must_coverage=mc, nice_coverage=nc,
                          doc_similarity=doc_similarity, matches=mm + nm)
        res._must = set(must)
        return res
