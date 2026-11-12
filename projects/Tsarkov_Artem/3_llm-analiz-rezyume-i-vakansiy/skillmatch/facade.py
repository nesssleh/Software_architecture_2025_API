"""SkillMatchFacade — единая точка входа (паттерн Facade) и конвейер обработки документа (Pipeline).

Конвейер документа: parse → split_title → extract skills → split must/nice → embed → save.
API и Kafka-воркер используют только фасад и не знают о стратегиях/хранилищах.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Callable

from .embedding.embedders import Embedder
from .explain.explainers import Explainer
from .extraction.extractors import SkillExtractor, sentences
from .matching import SkillMatcher
from .parsing.parsers import split_title
from .storage.repository import Document, DocumentRepository

_NICE = re.compile(r"(будет плюсом|желательно|плюсом будет|nice to have|приветствуется)", re.I)

Stage = Callable[[dict], dict]

try:
    from prometheus_client import Counter, Histogram

    DOCS = Counter("skillmatch_documents_ingested_total", "Обработанные документы", ["type"])
    SKILLS_FOUND = Counter("skillmatch_skills_extracted_total", "Извлечённые навыки", ["method"])
    MATCH_SCORE = Histogram("skillmatch_match_score", "Скор сопоставления", buckets=[i / 10 for i in range(1, 11)])
except ImportError:  # pragma: no cover
    DOCS = SKILLS_FOUND = MATCH_SCORE = None


class DocumentPipeline:
    def __init__(self, extractor: SkillExtractor, embedder: Embedder) -> None:
        self.extractor, self.embedder = extractor, embedder
        self.stages: list[tuple[str, Stage]] = [("split_title", self._split), ("extract", self._extract),
                                                ("requirements", self._requirements), ("embed", self._embed)]

    def run(self, ctx: dict) -> dict:
        ctx["timings_ms"] = {}
        for name, st in self.stages:
            t = time.perf_counter()
            ctx = st(ctx)
            ctx["timings_ms"][name] = round((time.perf_counter() - t) * 1000, 2)
        return ctx

    def _split(self, ctx):
        ctx["title"], ctx["body"] = split_title(ctx["text"])
        return ctx

    def _extract(self, ctx):
        ctx["skills"] = [e.as_dict() for e in self.extractor.extract(ctx["body"])]
        return ctx

    def _requirements(self, ctx):
        """Для вакансий: навык из предложения с маркером «будет плюсом» — nice-to-have, иначе must-have."""
        if ctx["type"] != "vacancy":
            ctx["must"], ctx["nice"] = [], []
            return ctx
        nice_sents = {s for s in sentences(ctx["body"]) if _NICE.search(s)}
        ctx["nice"] = sorted({s["name"] for s in ctx["skills"] if s["evidence"] in nice_sents})
        ctx["must"] = sorted({s["name"] for s in ctx["skills"]} - set(ctx["nice"]))
        return ctx

    def _embed(self, ctx):
        ctx["embedding"] = self.embedder.embed([ctx["body"]])[0]
        return ctx


class SkillMatchFacade:
    def __init__(self, repo: DocumentRepository, extractor: SkillExtractor, embedder: Embedder,
                 explainer: Explainer) -> None:
        self.repo, self.embedder, self.explainer = repo, embedder, explainer
        self.pipeline = DocumentPipeline(extractor, embedder)
        self.matcher = SkillMatcher(embedder)

    @staticmethod
    def doc_id(doc_type: str, text: str) -> str:
        return f"{doc_type[:3]}-{hashlib.sha1(text.encode()).hexdigest()[:12]}"

    def ingest(self, doc_type: str, text: str, doc_id: str | None = None) -> Document:
        if doc_type not in ("resume", "vacancy"):
            raise ValueError("type must be resume or vacancy")
        if len(text.strip()) < 20:
            raise ValueError("document is too short")
        ctx = self.pipeline.run({"type": doc_type, "text": text})
        doc = Document(doc_id or self.doc_id(doc_type, text), doc_type, ctx["title"], text, ctx["skills"],
                       ctx["must"], ctx["nice"], ctx["embedding"])
        self.repo.save(doc)
        if DOCS is not None:
            DOCS.labels(doc_type).inc()
            for s in doc.skills:
                SKILLS_FOUND.labels(s["method"]).inc()
        return doc

    def match(self, resume_id: str, vacancy_id: str, explain: bool = True) -> dict:
        r, v = self.repo.get(resume_id), self.repo.get(vacancy_id)
        if not r or not v:
            raise KeyError("resume or vacancy not found")
        sim = float(r.embedding @ v.embedding) if r.embedding is not None and v.embedding is not None else 0.0
        res = self.matcher.match(r.skill_names, v.must_have, v.nice_to_have, sim)
        out = {"resume_id": r.id, "vacancy_id": v.id, **res.as_dict()}
        if MATCH_SCORE is not None:
            MATCH_SCORE.observe(res.score)
        if explain:
            out["explanation"] = self.explainer.explain(v.title or v.id, res)
        return out

    def candidates(self, vacancy_id: str, k: int = 10, prefilter: int = 200) -> list[dict]:
        """Гибридный поиск: кандидаты из инвертированного индекса навыков ∪ ANN по эмбеддингу (pgvector),
        затем переранжирование скором сопоставления навыков."""
        v = self.repo.get(vacancy_id)
        if not v:
            raise KeyError("vacancy not found")
        ann = dict(self.repo.nearest(v.embedding, "resume", prefilter // 4))
        ids = list(dict.fromkeys(self.repo.by_skills(v.must_have + v.nice_to_have, "resume", prefilter) + list(ann)))
        docs = self.repo.get_many(ids)
        out = []
        for rid in ids:
            r = docs.get(rid)
            if r is None:
                continue
            sim = ann.get(rid, float(r.embedding @ v.embedding) if r.embedding is not None else 0.0)
            res = self.matcher.match(r.skill_names, v.must_have, v.nice_to_have, sim)
            out.append({"resume_id": rid, "title": r.title, "score": round(res.score, 4),
                        "must_coverage": round(res.must_coverage, 4), "missing_must": res.missing_must})
        return sorted(out, key=lambda x: -x["score"])[:k]
