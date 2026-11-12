"""E2E против живого стека: API → Kafka documents.ingested → extractor-worker ×2 → PostgreSQL/pgvector → API."""
import os
import time
import uuid

import httpx

httpx = httpx.Client(timeout=60)  # под нагрузкой стенда ответы бывают медленными

from .helpers import TopicWatcher, kafka_admin_state, live, wait_for

pytestmark = live
API = os.getenv("API_URL", "http://api:8000")
U = uuid.uuid4().hex[:6]
VAC = ("Вакансия: Python Backend Developer. Требуется опыт с Python. Обязательно знание FastAPI. "
       "Опыт работы с PostgreSQL от 2 лет. Ожидаем уверенное владение Docker. Будет плюсом: Apache Kafka.")
RES = (f"Резюме. Желаемая должность: Python Backend Developer. Пишу на питоне 4 года ({U}). Стек: Django, MySQL, Docker. "
       "Упаковывал сервисы в образы и описывал окружение через compose-файлы.")


def test_api_healthy():
    r = httpx.get(f"{API}/health", timeout=10)
    print(r.json())
    assert r.status_code == 200


def test_kafka_topics_and_consumer_groups():
    topics, groups = kafka_admin_state()
    print("topics:", sorted(t for t in topics if not t.startswith("__")), "groups:", sorted(groups))
    assert {"documents.ingested", "documents.processed"} <= topics
    assert "skill-extractor" in groups


def test_async_ingest_through_kafka_and_match():
    w = TopicWatcher("documents.processed")
    t0 = time.time()
    rid = f"e2e-res-{U}"
    r = httpx.post(f"{API}/v1/documents?mode=async", json={"type": "resume", "text": RES, "id": rid})
    assert r.status_code == 201 and r.json()["status"] == "queued"
    part, off, msg = w.find(lambda v: v.get("id") == rid, timeout=90)
    w.close()
    print(f"documents.processed partition={part} offset={off} за {time.time() - t0:.1f} с: {msg}")
    doc = wait_for(lambda: httpx.get(f"{API}/v1/documents/{rid}").status_code == 200, 30, 1, "резюме в pgvector")
    assert doc
    vid = httpx.post(f"{API}/v1/documents", json={"type": "vacancy", "text": VAC, "id": f"e2e-vac-{U}"}).json()["id"]
    m = httpx.post(f"{API}/v1/match", json={"resume_id": rid, "vacancy_id": vid}).json()
    print({k: m[k] for k in ("score", "must_coverage", "missing_must")}, m["explanation"]["summary"])
    kinds = {x["requirement"]: x["kind"] for x in m["matches"]}
    assert kinds["Python"] == "exact" and kinds["FastAPI"] == "related"
    cands = httpx.get(f"{API}/v1/vacancies/{vid}/candidates?k=5").json()["candidates"]
    assert any(c["resume_id"] == rid for c in cands)


def test_errors():
    assert httpx.post(f"{API}/v1/documents", json={"type": "memo", "text": VAC}).status_code == 422
    assert httpx.post(f"{API}/v1/match", json={"resume_id": "nope", "vacancy_id": "nope"}).status_code == 404
