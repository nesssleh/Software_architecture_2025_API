"""E2E против живого стека: API → Kafka → classifier-service ×2 → routing-service → PostgreSQL/Redis/Elasticsearch."""
import os
import time
import uuid

import httpx

httpx = httpx.Client(timeout=60)  # под нагрузкой стенда ответы бывают медленными

from .helpers import TopicWatcher, kafka_admin_state, live, wait_for

pytestmark = live
API = os.getenv("API_URL", "http://api:8000")


def test_api_healthy():
    r = httpx.get(f"{API}/health", timeout=10)
    print(r.json())
    assert r.status_code == 200


def test_kafka_topics_and_consumer_groups():
    topics, groups = kafka_admin_state()
    print("topics:", sorted(t for t in topics if not t.startswith("__")), "groups:", sorted(groups))
    assert {"tickets.created", "tickets.classified", "tickets.routed"} <= topics
    assert {"ticket-classifier", "ticket-router"} <= groups


def test_async_ticket_through_kafka():
    w = TopicWatcher("tickets.routed")
    tid = f"E2E-{uuid.uuid4().hex[:8]}"
    t0 = time.time()
    r = httpx.post(f"{API}/v1/tickets?mode=async", json={"ticket_id": tid, "text": "Не приходит смс с кодом для входа в аккаунт",
                                                          "customer_tier": "pro"})
    assert r.status_code == 201 and r.json()["status"] == "queued"
    part, off, msg = w.find(lambda v: v["ticket"]["ticket_id"] == tid, timeout=90)
    w.close()
    print(f"tickets.routed partition={part} offset={off} за {time.time() - t0:.1f} с: trace={msg['trace']}")
    assert msg["ticket"]["team"] in ("identity_team", "l1_triage")
    t = wait_for(lambda: httpx.get(f"{API}/v1/tickets/{tid}").json() if httpx.get(f"{API}/v1/tickets/{tid}").status_code == 200 else None,
                 30, 1, "тикет и аудит в PostgreSQL")
    print(t["ticket"]["team"], [a["command"] for a in t["audit"]])
    assert t["audit"]


def test_security_rule_undo_and_search():
    r = httpx.post(f"{API}/v1/tickets", json={"text": "Взломали аккаунт, сменили пароль и почту, вижу чужие устройства"})
    assert r.status_code == 201
    t = r.json()
    print(t["trace"], t["ticket"]["team"])
    assert t["ticket"]["team"] == "security_team" and t["trace"][0] == "security"
    u = httpx.post(f"{API}/v1/tickets/{t['ticket']['ticket_id']}/undo-routing").json()
    assert u["undone_commands"] >= 1 and u["ticket"]["team"] is None
    hits = wait_for(lambda: httpx.get(f"{API}/v1/search", params={"q": "взломали аккаунт"}).json()["hits"], 30, 1,
                    "поиск в Elasticsearch")
    print(hits[0])
    assert httpx.post(f"{API}/v1/tickets", json={"text": "hi"}).status_code == 422
