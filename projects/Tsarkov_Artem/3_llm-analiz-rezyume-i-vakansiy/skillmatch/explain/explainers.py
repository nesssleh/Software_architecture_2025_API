"""Объяснение результата сопоставления — Strategy.

* TemplateExplainer — детерминированный текст из фактов матчинга (всегда доступен);
* LLMExplainer      — LLM формулирует рекомендации кандидату, но получает только
  факты (списки совпавших/отсутствующих навыков). Ответ проверяется: если LLM
  упомянула навык из таксономии, которого нет в фактах, — ответ отклоняется и
  используется шаблон (контроль галлюцинаций).
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod

from ..llm import LLMClient, LLMUnavailable
from ..matching import MatchResult
from ..taxonomy import SKILLS

log = logging.getLogger(__name__)


class Explainer(ABC):
    name = "abstract"

    @abstractmethod
    def explain(self, vacancy_title: str, res: MatchResult) -> dict: ...


def _facts(res: MatchResult) -> dict:
    f = {"exact": [], "partial": [], "missing_must": [], "missing_nice": []}
    for m in res.matches:
        if m.kind == "exact":
            f["exact"].append(m.requirement)
        elif m.kind in ("related", "semantic"):
            f["partial"].append(f"{m.requirement} (есть {m.matched_by})")
        elif m.requirement in res._must:
            f["missing_must"].append(m.requirement)
        else:
            f["missing_nice"].append(m.requirement)
    return f


class TemplateExplainer(Explainer):
    name = "template"

    def explain(self, vacancy_title: str, res: MatchResult) -> dict:
        f = _facts(res)
        verdict = "подходит" if res.must_coverage >= 0.8 else "частично подходит" if res.must_coverage >= 0.5 else "не подходит"
        parts = [f"Кандидат {verdict} на позицию «{vacancy_title}» (оценка {res.score:.0%}, "
                 f"покрытие обязательных требований {res.must_coverage:.0%})."]
        if f["exact"]:
            parts.append("Совпадают: " + ", ".join(f["exact"]) + ".")
        if f["partial"]:
            parts.append("Частично закрыто смежным опытом: " + "; ".join(f["partial"]) + ".")
        if f["missing_must"]:
            parts.append("Не хватает обязательных компетенций: " + ", ".join(f["missing_must"]) + ".")
        if f["missing_nice"]:
            parts.append("Было бы плюсом: " + ", ".join(f["missing_nice"]) + ".")
        cat_hint = {"language": "решить несколько практических задач и сделать pet-проект",
                    "database": "спроектировать схему и написать запросы в учебном проекте",
                    "devops": "развернуть свой сервис с использованием инструмента",
                    "ml": "пройти курс и выполнить проект с публикацией кода"}
        recs = []
        for s in f["missing_must"][:3]:
            if s in SKILLS:
                base = next((x for x in SKILLS[s].related if x in res_skills), None) if (res_skills := {m.matched_by for m in res.matches if m.matched_by}) else None
                tip = f" — опыт с {base} поможет быстрее" if base else ""
                recs.append(f"Освоить {s}: {cat_hint.get(SKILLS[s].category, 'получить практический опыт на реальной задаче')}{tip}.")
        return {"verdict": verdict, "summary": " ".join(parts), "recommendations": recs, "facts": f,
                "explainer": self.name}


SYSTEM = """Ты — карьерный консультант. Тебе дан JSON с фактами сопоставления резюме и вакансии.
Опирайся ТОЛЬКО на эти факты, не придумывай навыки и опыт.
Верни JSON: {"verdict": "подходит|частично подходит|не подходит", "summary": "2-3 предложения",
"recommendations": ["конкретный шаг по закрытию пробела", ...]}"""


class LLMExplainer(Explainer):
    name = "llm"

    def __init__(self, llm: LLMClient, fallback: Explainer | None = None) -> None:
        self.llm, self.fallback = llm, fallback or TemplateExplainer()

    def explain(self, vacancy_title: str, res: MatchResult) -> dict:
        facts = _facts(res)
        allowed = {re.sub(r" \(.*", "", s) for v in facts.values() for s in v} | \
                  {m.matched_by for m in res.matches if m.matched_by}
        try:
            out = self.llm.complete_json(SYSTEM, json.dumps({"vacancy": vacancy_title, "score": round(res.score, 2),
                                                              "must_coverage": round(res.must_coverage, 2), **facts},
                                                             ensure_ascii=False))
            text = (out.get("summary", "") + " " + " ".join(out.get("recommendations", []))).lower()
            mentioned = {s for s in SKILLS if re.search(r"(?<![\w+#])" + re.escape(s.lower()) + r"(?![\w+#])", text)}
            if mentioned - allowed:
                raise ValueError(f"hallucinated skills: {sorted(mentioned - allowed)}")
            return {**out, "facts": facts, "explainer": self.name}
        except (LLMUnavailable, ValueError, KeyError) as e:
            log.warning("LLM explainer fallback: %s", e)
            return {**self.fallback.explain(vacancy_title, res), "fallback_reason": str(e)[:200]}
