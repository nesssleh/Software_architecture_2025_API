"""Kafka-воркеры.

classifier-service  (group ticket-classifier): tickets.created    → tickets.classified
routing-service     (group ticket-router)    : tickets.classified → tickets.routed (+ incidents.detected)
Ключ сообщений = ticket_id. Ошибки после 3 попыток → <topic>.dlq.

python -m services.workers classifier|router
"""
from __future__ import annotations

import json
import logging
import sys

from ticketflow.kafka_io import EventProducer, Message, make_consumer, make_producer, run_consumer_loop
from ticketflow.routing.commands import Ticket
from ticketflow.service import ClassificationService, RoutingService, build

FIELDS = set(Ticket.__dataclass_fields__)


class ClassifyHandler:
    topic_in, group = "tickets.created", "ticket-classifier"

    def __init__(self, svc: ClassificationService, producer: EventProducer) -> None:
        self.svc, self.producer = svc, producer

    def __call__(self, msg: Message) -> None:
        t = self.svc.classify(Ticket(**{k: v for k, v in msg.json().items() if k in FIELDS}))
        self.producer.send("tickets.classified", key=t.ticket_id, value=json.dumps(t.as_dict(), ensure_ascii=False))

    def handle_batch(self, msgs: list[Message]) -> None:
        """Батчевый инференс: одна прогонка модели на всю пачку сообщений из poll()."""
        tickets = [Ticket(**{k: v for k, v in m.json().items() if k in FIELDS}) for m in msgs]
        for t in self.svc.classify_batch(tickets):
            self.producer.send("tickets.classified", key=t.ticket_id, value=json.dumps(t.as_dict(), ensure_ascii=False))


class RouteHandlerWorker:
    topic_in, group = "tickets.classified", "ticket-router"

    def __init__(self, svc: RoutingService) -> None:
        self.svc = svc

    def __call__(self, msg: Message) -> None:
        self.svc.route(Ticket(**{k: v for k, v in msg.json().items() if k in FIELDS}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ticketflow.obs import start_metrics_server

    start_metrics_server()
    prod = make_producer()
    cls, rt = build(prod)
    h = ClassifyHandler(cls, prod) if sys.argv[1] == "classifier" else RouteHandlerWorker(rt)
    run_consumer_loop(make_consumer(h.topic_in, group_id=h.group), h, prod, h.topic_in + ".dlq")
