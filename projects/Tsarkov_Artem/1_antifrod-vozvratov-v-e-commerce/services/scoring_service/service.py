"""Scoring service.

Consumer group `return-scorer` читает `returns.to_score`, прогоняет возврат через
цепочку обработчиков (Chain of Responsibility) с выбранной стратегией (Strategy)
и уведомляет подписчиков (Observer): кейсы в PostgreSQL, алерты в `fraud.alerts`,
метрики. Итог любого решения публикуется в `returns.scored`.
"""
from __future__ import annotations

import json
import logging
import os

from returnguard.alerts import CaseOpener, DecisionPublisher, KafkaAlertPublisher, MetricsCollector
from returnguard.scoring.chain import ReturnContext, build_chain
from returnguard.scoring.strategies import ScoringStrategy, make_strategy
from returnguard.storage.repositories import AnalyticsRepository, CaseRepository, make_analytics_repo
from returnguard.streaming.kafka_io import EventProducer, Message, make_consumer, make_producer, run_consumer_loop

log = logging.getLogger("scoring-service")
TOPIC_IN, TOPIC_OUT, DLQ, GROUP = "returns.to_score", "returns.scored", "returns.to_score.dlq", "return-scorer"


class ScoringService:
    def __init__(self, strategy: ScoringStrategy, cases: CaseRepository, analytics: AnalyticsRepository,
                 producer: EventProducer, block: float = 0.9, review: float = 0.6,
                 allowlist: set[str] | None = None) -> None:
        self.strategy = strategy
        self.chain = build_chain(strategy, allowlist, block, review)
        self.analytics, self.producer = analytics, producer
        self.publisher = DecisionPublisher()
        self.metrics = MetricsCollector()
        for l in (CaseOpener(cases), KafkaAlertPublisher(producer), self.metrics):
            self.publisher.subscribe(l)

    def score(self, req: dict) -> ReturnContext:
        ctx = self.chain.handle(ReturnContext(req["return_id"], req["customer_id"], req["features"], req["amount"]))
        self.publisher.publish(ctx)
        return ctx

    def handle(self, msg: Message) -> None:
        req = msg.json()
        ctx = self.score(req)
        out = {"return_id": ctx.return_id, "customer_id": ctx.customer_id, "ts": req["ts"], "score": ctx.score,
               "decision": ctx.decision, "decided_by": ctx.decided_by, "reasons": ctx.reasons,
               "model_version": self.strategy.version, "amount": ctx.amount}
        self.producer.send(TOPIC_OUT, key=ctx.customer_id, value=json.dumps(out, ensure_ascii=False))
        self.analytics.insert_scores([out])


def build_from_env() -> ScoringService:
    return ScoringService(make_strategy(os.getenv("SCORING_STRATEGY", "catboost"), os.getenv("MODEL_PATH")),
                          CaseRepository(), make_analytics_repo(), make_producer(),
                          float(os.getenv("BLOCK_THRESHOLD", 0.9)), float(os.getenv("REVIEW_THRESHOLD", 0.6)))


def run(service: ScoringService | None = None, consumer=None, once: bool = False) -> int:
    service = service or build_from_env()
    consumer = consumer or make_consumer(TOPIC_IN, group_id=GROUP)
    return run_consumer_loop(consumer, service.handle, service.producer, DLQ, once=once)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from returnguard.obs import start_metrics_server

    start_metrics_server()
    run()
