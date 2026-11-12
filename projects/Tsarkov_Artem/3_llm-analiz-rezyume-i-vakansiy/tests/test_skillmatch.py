import json

import pytest
from fastapi.testclient import TestClient

from skillmatch.embedding.embedders import SpacyVectorEmbedder
from skillmatch.explain.explainers import LLMExplainer, TemplateExplainer
from skillmatch.extraction.extractors import GazetteerExtractor, HybridExtractor, LLMExtractor, SemanticExtractor
from skillmatch.facade import SkillMatchFacade
from skillmatch.kafka_io import InMemoryBroker
from skillmatch.llm import LLMClient, NullLLM
from skillmatch.matching import SkillMatcher
from skillmatch.parsing.parsers import parse, split_title
from skillmatch.storage.repository import InMemoryRepository
from services.extractor_worker.worker import run as run_worker

VAC = ("Вакансия: Python Backend Developer. Требуется опыт с Python. Обязательно знание FastAPI. "
       "Опыт работы с постгрес от 2 лет. Ожидаем уверенное владение Docker. Будет плюсом: Apache Kafka.")
RES = ("Резюме. Желаемая должность: Python Backend Developer. Пишу на питоне 4 года. Стек: Django, MySQL. "
       "Упаковывал сервисы в образы и описывал окружение через compose-файлы.")


class FakeLLM(LLMClient):
    name = "fake"

    def __init__(self, payload: dict):
        self.payload = payload

    def complete(self, system, user, max_tokens=800):
        return "Конечно! " + json.dumps(self.payload, ensure_ascii=False)


@pytest.fixture(scope="module")
def emb():
    return SpacyVectorEmbedder()


@pytest.fixture(scope="module")
def facade(emb):
    return SkillMatchFacade(InMemoryRepository(), HybridExtractor(GazetteerExtractor(), SemanticExtractor(emb)), emb,
                            TemplateExplainer())


def test_split_title():
    t, body = split_title(VAC)
    assert t == "Python Backend Developer" and body.startswith("Требуется")


def test_gazetteer_aliases():
    names = {e.name for e in GazetteerExtractor().extract(split_title(VAC)[1])}
    assert {"Python", "FastAPI", "PostgreSQL", "Docker", "Apache Kafka"} <= names


def test_llm_extractor_drops_hallucinations():
    llm = FakeLLM({"skills": [{"name": "Python", "evidence": "Требуется опыт с Python"},
                              {"name": "Quantum Blockchain", "evidence": "нет такого"}]})
    names = {e.name for e in LLMExtractor(llm).extract(VAC)}
    assert names == {"Python"}


def test_hybrid_survives_llm_failure(emb):
    ex = HybridExtractor(GazetteerExtractor(), None, LLMExtractor(NullLLM()))
    assert "Python" in {e.name for e in ex.extract(VAC)}


def test_vacancy_must_nice_split(facade):
    v = facade.ingest("vacancy", VAC, "vac-1")
    assert "Apache Kafka" in v.nice_to_have and "Apache Kafka" not in v.must_have
    assert {"Python", "FastAPI", "PostgreSQL", "Docker"} <= set(v.must_have)


def test_match_related_and_missing(facade):
    facade.ingest("vacancy", VAC, "vac-1")
    facade.ingest("resume", RES, "res-1")
    out = facade.match("res-1", "vac-1")
    kinds = {m["requirement"]: m["kind"] for m in out["matches"]}
    assert kinds["Python"] == "exact"
    assert kinds["FastAPI"] == "related"        # есть Django
    assert kinds["PostgreSQL"] == "related"     # есть MySQL
    assert "Не хватает" in out["explanation"]["summary"] or out["must_coverage"] > 0.8
    assert 0 <= out["score"] <= 1


def test_llm_explainer_hallucination_guard(facade, emb):
    facade.ingest("vacancy", VAC, "vac-1")
    facade.ingest("resume", RES, "res-1")
    res = SkillMatcher(emb).match({"Python"}, ["Python", "Docker"])
    bad = LLMExplainer(FakeLLM({"verdict": "подходит", "summary": "Отличный опыт с Kubernetes.", "recommendations": []}))
    out = bad.explain("Backend", res)
    assert out["explainer"] == "template" and "hallucinated" in out["fallback_reason"]
    good = LLMExplainer(FakeLLM({"verdict": "частично подходит", "summary": "Есть Python, не хватает Docker.",
                                 "recommendations": ["Изучить Docker"]}))
    assert good.explain("Backend", res)["explainer"] == "llm"


def test_candidates(facade):
    facade.ingest("vacancy", VAC, "vac-1")
    facade.ingest("resume", RES, "res-1")
    facade.ingest("resume", "Резюме. Желаемая должность: Бухгалтер. Excel, 1С, отчётность и коммуникация с налоговой.", "res-2")
    c = facade.candidates("vac-1", k=5)
    assert c[0]["resume_id"] == "res-1"


def test_parse_formats():
    assert parse("привет мир".encode(), "a.txt") == "привет мир"
    with pytest.raises(ValueError):
        parse(b"x", "a.exe")


def test_kafka_worker(facade):
    broker = InMemoryBroker()
    p = broker.producer()
    p.send("documents.ingested", key="res-9", value={"id": "res-9", "type": "resume", "text": RES})
    p.send("documents.ingested", key="bad", value={"id": "bad", "type": "memo", "text": "коротко"})
    n = run_worker(facade, broker.consumer("documents.ingested", group_id="skill-extractor"), p, once=True)
    assert n == 1 and facade.repo.get("res-9") is not None
    assert len(broker.messages("documents.processed")) == 1
    assert len(broker.messages("documents.ingested.dlq")) == 1


def test_api():
    from services.api import app as m

    m.facade.cache_clear()
    c = TestClient(m.app)
    assert c.get("/health").json()["status"] == "ok"
    vid = c.post("/v1/documents", json={"type": "vacancy", "text": VAC}).json()["id"]
    rid = c.post("/v1/documents", json={"type": "resume", "text": RES}).json()["id"]
    r = c.post("/v1/match", json={"resume_id": rid, "vacancy_id": vid})
    assert r.status_code == 200 and "explanation" in r.json()
    assert c.post("/v1/match", json={"resume_id": "x", "vacancy_id": vid}).status_code == 404
    assert c.post("/v1/documents", json={"type": "memo", "text": VAC}).status_code == 422
    assert c.get(f"/v1/vacancies/{vid}/candidates").json()["candidates"][0]["resume_id"] == rid
    up = c.post("/v1/documents/upload", data={"type": "resume"}, files={"file": ("cv.txt", RES.encode(), "text/plain")})
    assert up.status_code == 201 and up.json()["skills"]
