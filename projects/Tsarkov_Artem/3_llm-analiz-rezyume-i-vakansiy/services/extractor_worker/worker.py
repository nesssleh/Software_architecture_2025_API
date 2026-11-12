"""Extractor worker (consumer group `skill-extractor`).

documents.ingested → [parse/extract/embed через фасад] → PostgreSQL/pgvector → documents.processed
Ошибки после 3 попыток → documents.ingested.dlq. Ключ сообщения = id документа (идемпотентный upsert).
"""
from __future__ import annotations

import json
import logging

from skillmatch.facade import SkillMatchFacade
from skillmatch.factory import build_facade
from skillmatch.kafka_io import EventProducer, Message, make_consumer, make_producer, run_consumer_loop

GROUP, TOPIC_IN, TOPIC_OUT = "skill-extractor", "documents.ingested", "documents.processed"


class ExtractHandler:
    def __init__(self, facade: SkillMatchFacade, producer: EventProducer) -> None:
        self.facade, self.producer = facade, producer

    def __call__(self, msg: Message) -> None:
        ev = msg.json()
        doc = self.facade.ingest(ev["type"], ev["text"], ev["id"])
        self.producer.send(TOPIC_OUT, key=doc.id, value=json.dumps(
            {"id": doc.id, "type": doc.type, "skills": sorted(doc.skill_names), "must_have": doc.must_have},
            ensure_ascii=False))


def run(facade=None, consumer=None, producer=None, once=False) -> int:
    producer = producer or make_producer()
    h = ExtractHandler(facade or build_facade(), producer)
    return run_consumer_loop(consumer or make_consumer(TOPIC_IN, group_id=GROUP), h, producer, TOPIC_IN + ".dlq", once=once)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from skillmatch.obs import start_metrics_server

    start_metrics_server()
    run()
