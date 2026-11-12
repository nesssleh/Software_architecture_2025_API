"""Интеграционные тесты пайплайна на InMemoryBroker + SQLite (без внешних сервисов)."""
import json

from fastapi.testclient import TestClient

from returnguard.datagen import generate
from returnguard.scoring.strategies import make_strategy
from returnguard.storage.repositories import CaseRepository, InMemoryCustomerRepo, SQLiteAnalyticsRepo
from returnguard.streaming.kafka_io import InMemoryBroker, run_consumer_loop
from services.feature_service.service import FeatureService
from services.scoring_service.service import ScoringService


def _pipeline(tmp_path, n=150):
    broker = InMemoryBroker()
    prod = broker.producer()
    events, labels, _ = generate(n_customers=n, months=3, seed=1)
    for ev in events:
        prod.send("orders.events", key=ev["customer_id"], value=json.dumps(ev))
    cust, an, cases = InMemoryCustomerRepo(), SQLiteAnalyticsRepo(), CaseRepository(f"sqlite:///{tmp_path}/c.db")
    fs = FeatureService(cust, an, prod)
    ss = ScoringService(make_strategy("catboost"), cases, an, prod)
    run_consumer_loop(broker.consumer("orders.events", group_id="feature-builder"), fs.handle, prod, "dlq", once=True)
    run_consumer_loop(broker.consumer("returns.to_score", group_id="return-scorer"), ss.handle, prod, "dlq", once=True)
    return broker, events, labels, cust, an, cases, fs, ss


def test_end_to_end(tmp_path):
    broker, events, labels, cust, an, cases, fs, ss = _pipeline(tmp_path)
    n_ret = sum(e["type"] == "return.requested" for e in events)
    assert len(broker.messages("returns.to_score")) == n_ret
    scored = [m.json() for m in broker.messages("returns.scored")]
    assert len(scored) == n_ret and all(s["decision"] in ("approve", "review", "block") for s in scored)
    assert broker.lag("feature-builder", "orders.events") == 0
    # кейсы открыты ровно для review/block
    assert sum(cases.counts().values()) == sum(s["decision"] != "approve" for s in scored)
    assert len(broker.messages("fraud.alerts")) == sum(s["decision"] == "block" for s in scored)
    assert an.daily_summary()


def test_redelivery_is_idempotent(tmp_path):
    broker, events, *_rest, fs, ss = _pipeline(tmp_path, n=40)
    before = len(broker.messages("returns.to_score"))
    # другая consumer group перечитывает топик с начала — дубликаты отбрасываются по event_id
    run_consumer_loop(broker.consumer("orders.events", group_id="replay"), fs.handle, broker.producer(), "dlq", once=True)
    assert len(broker.messages("returns.to_score")) == before


def test_poison_message_to_dlq(tmp_path):
    broker, *_rest, fs, ss = _pipeline(tmp_path, n=10)
    broker.producer().send("orders.events", key="X", value='{"type": "order.created"}')
    run_consumer_loop(broker.consumer("orders.events", group_id="feature-builder"), fs.handle, broker.producer(),
                      "orders.events.dlq", once=True)
    assert len(broker.messages("orders.events.dlq")) == 1


def test_api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api.db")
    from services.investigation_api import app as m

    m.deps.cache_clear()
    c = TestClient(m.app)
    h = {"X-API-Token": "dev-token"}
    assert c.get("/health").status_code == 200
    body = {"return_id": "R1", "customer_id": "C404", "order_id": "O1", "ts": "2025-03-04T10:00:00+00:00",
            "amount": 120.0, "reason": "size", "items": [{"sku": "A", "category": "apparel", "price": 120.0}]}
    assert c.post("/v1/score", json=body).status_code == 401
    r = c.post("/v1/score", json=body, headers=h)
    assert r.status_code == 200 and r.json()["decision"] in ("approve", "review", "block")
    assert c.post("/v1/score", json=body | {"reason": "bored"}, headers=h).status_code == 422
    assert c.get("/v1/customers/C404/risk", headers=h).status_code == 404
    cid = m.deps()["cases"].open_case("R9", "C9", 0.95, "block", ["score=0.95"], 999.0)
    assert c.get("/v1/cases", headers=h).json()["items"][0]["case_id"] == cid
    r = c.post(f"/v1/cases/{cid}/decision", json={"verdict": "fraud", "comment": "подтверждено"}, headers=h)
    assert r.json()["status"] == "closed"


def test_ingest_api():
    from services.order_ingest import app as ing

    c = TestClient(ing.app)
    r = c.post("/v1/carriers/boxberry", json={"Status": "Выдано", "Date": "03.03.2025 10:00", "OrderId": "O1", "ClientId": "C1"})
    assert r.json()["accepted"]
    assert c.post("/v1/carriers/dhl", json={}).status_code == 404
    bad = {"type": "return.requested", "ts": "2025-01-01T00:00:00", "customer_id": "C", "order_id": "O"}
    assert c.post("/v1/events", json=bad).status_code == 422
