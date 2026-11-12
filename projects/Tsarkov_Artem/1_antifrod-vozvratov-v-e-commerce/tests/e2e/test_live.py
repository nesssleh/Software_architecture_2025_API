"""E2E против живого стека: order-ingest → Kafka → feature-service → scoring-service → PostgreSQL/ClickHouse → API."""
import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx

httpx = httpx.Client(timeout=60)  # под нагрузкой стенда ответы бывают медленными
import pytest

from .helpers import TopicWatcher, kafka_admin_state, live, wait_for

pytestmark = live
API, ING = os.getenv("API_URL", "http://investigation-api:8000"), os.getenv("INGEST_URL", "http://order-ingest:8001")
H = {"X-API-Token": "dev-token"}
CID = f"E2E-{uuid.uuid4().hex[:8]}"


def ts(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def test_services_healthy():
    for url in (API, ING):
        r = httpx.get(f"{url}/health", timeout=10)
        print(url, r.json())
        assert r.status_code == 200


def test_kafka_topics_and_consumer_groups():
    topics, groups = kafka_admin_state()
    print("topics:", sorted(t for t in topics if not t.startswith("__")))
    print("groups:", sorted(groups))
    assert {"orders.events", "returns.to_score", "returns.scored", "fraud.alerts"} <= topics
    assert {"feature-builder", "return-scorer"} <= groups


@pytest.fixture(scope="module")
def scored():
    """Клиент с тремя «не получил» по вручённым заказам: событие проходит весь асинхронный конвейер."""
    w = TopicWatcher("returns.scored")
    t0 = time.time()
    rid = None
    for i in range(4):
        oid = f"{CID}-O{i}"
        items = [{"sku": "ELE1", "category": "electronics", "price": 900.0}]
        httpx.post(f"{ING}/v1/events", json={"type": "order.created", "ts": ts(30 - i), "customer_id": CID, "order_id": oid,
                                              "items": items, "amount": 900.0, "payment_method": "card", "address_id": "A1",
                                              "signup_ts": ts(60)}).raise_for_status()
        httpx.post(f"{ING}/v1/events", json={"type": "order.delivered", "ts": ts(28 - i), "customer_id": CID,
                                              "order_id": oid}).raise_for_status()
        rid = f"{CID}-R{i}"
        httpx.post(f"{ING}/v1/events", json={"type": "return.requested", "ts": ts(27 - i), "customer_id": CID, "order_id": oid,
                                              "return_id": rid, "items": items, "amount": 900.0, "reason": "not_received",
                                              "refund_method": "wallet"}).raise_for_status()
    part, off, msg = w.find(lambda v: v.get("return_id") == rid, timeout=90)
    w.close()
    print(f"returns.scored partition={part} offset={off} за {time.time() - t0:.1f} с: {msg}")
    return msg


def test_pipeline_end_to_end(scored):
    assert scored["customer_id"] == CID
    assert scored["decision"] == "block" and scored["decided_by"] == "hard_rule"  # 3 прошлых «не получил»


def test_case_opened_in_postgres(scored):
    items = wait_for(lambda: [c for c in httpx.get(f"{API}/v1/cases", params={"customer_id": CID}, headers=H).json()["items"]], 30, 1, "кейс в PostgreSQL")
    print(items[0])


def test_events_in_clickhouse_timeline(scored):
    ev = wait_for(lambda: len(httpx.get(f"{API}/v1/customers/{CID}/timeline", headers=H).json()["events"]) >= 12,
                  60, 1, "12 событий клиента в ClickHouse")
    assert ev


def test_sync_score_and_errors():
    body = {"return_id": "E2E-S", "customer_id": CID, "order_id": "O", "ts": ts(0), "amount": 500.0, "reason": "size",
            "items": [{"sku": "A", "category": "apparel", "price": 500.0}]}
    r = httpx.post(f"{API}/v1/score", json=body, headers=H)
    print(r.json())
    assert r.status_code == 200
    assert httpx.post(f"{API}/v1/score", json=body).status_code == 401
    assert httpx.post(f"{API}/v1/score", json=body | {"reason": "bored"}, headers=H).status_code == 422
    assert httpx.get(f"{API}/v1/analytics/daily", headers=H).status_code == 200
