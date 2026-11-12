"""Feature service.

Consumer group `feature-builder` читает `orders.events` (key = customer_id, поэтому
все события клиента упорядочены внутри одной партиции):
  * обновляет онлайн-состояние клиента в Redis;
  * пишет сырые события в ClickHouse (order_events);
  * на return.requested считает признаки и публикует сообщение в `returns.to_score`.
"""
from __future__ import annotations

import json
import logging

from returnguard.features import FeatureBuilder
from returnguard.storage.repositories import (AnalyticsRepository, CustomerFeatureRepository, make_analytics_repo,
                                              make_customer_repo)
from returnguard.streaming.kafka_io import EventProducer, Message, make_consumer, make_producer, run_consumer_loop

log = logging.getLogger("feature-service")
TOPIC_IN, TOPIC_OUT, DLQ, GROUP = "orders.events", "returns.to_score", "orders.events.dlq", "feature-builder"


def event_id(ev: dict) -> str:
    return ev.get("return_id") or f"{ev['type']}:{ev['order_id']}"


class FeatureService:
    def __init__(self, customers: CustomerFeatureRepository, analytics: AnalyticsRepository, producer: EventProducer) -> None:
        self.customers, self.analytics, self.producer = customers, analytics, producer
        self.fb = FeatureBuilder()

    def handle(self, msg: Message) -> None:
        ev = msg.json()
        eid = event_id(ev)
        if self.customers.seen_event(eid):  # повторная доставка — пропускаем
            return
        st = self.customers.get(ev["customer_id"])
        if ev["type"] == "return.requested":
            feats = self.fb.features(st, ev)
            self.producer.send(TOPIC_OUT, key=ev["customer_id"], value=json.dumps({
                "return_id": ev["return_id"], "customer_id": ev["customer_id"], "order_id": ev["order_id"],
                "ts": ev["ts"], "amount": ev["amount"], "features": feats}, ensure_ascii=False))
        self.fb.apply(st, ev)
        self.customers.save(st)
        self.analytics.insert_events([{**ev, "event_id": eid}])


def run(service: FeatureService | None = None, consumer=None, once: bool = False) -> int:
    producer = make_producer()
    service = service or FeatureService(make_customer_repo(), make_analytics_repo(), producer)
    consumer = consumer or make_consumer(TOPIC_IN, group_id=GROUP)
    return run_consumer_loop(consumer, service.handle, service.producer, DLQ, once=once)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from returnguard.obs import start_metrics_server

    start_metrics_server()
    run()
