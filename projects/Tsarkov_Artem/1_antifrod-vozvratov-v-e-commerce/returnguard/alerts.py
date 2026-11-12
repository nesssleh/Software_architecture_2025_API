"""Реакция на решения скоринга — паттерн Observer.

ScoringService (subject) уведомляет подписчиков о каждом решении:
  * CaseOpener        — открывает кейс расследования в PostgreSQL для review/block;
  * KafkaAlertPublisher — публикует high-risk решение в fraud.alerts;
  * MetricsCollector  — счётчики решений (для /metrics и дашборда).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter

from prometheus_client import Counter as PCounter, Histogram

from .scoring.chain import ReturnContext

DECISIONS = PCounter("returnguard_decisions_total", "Решения по возвратам", ["decision", "decided_by"])
SCORES = Histogram("returnguard_risk_score", "Распределение риск-скоров", buckets=[i / 10 for i in range(1, 11)])
BLOCKED_AMOUNT = PCounter("returnguard_blocked_amount_total", "Сумма заблокированных возвратов")


class DecisionListener(ABC):
    @abstractmethod
    def on_decision(self, ctx: ReturnContext) -> None: ...


class DecisionPublisher:
    def __init__(self) -> None:
        self.listeners: list[DecisionListener] = []

    def subscribe(self, l: DecisionListener) -> None:
        self.listeners.append(l)

    def publish(self, ctx: ReturnContext) -> None:
        for l in self.listeners:
            l.on_decision(ctx)


class CaseOpener(DecisionListener):
    def __init__(self, cases) -> None:
        self.cases = cases

    def on_decision(self, ctx: ReturnContext) -> None:
        if ctx.decision in ("review", "block"):
            self.cases.open_case(ctx.return_id, ctx.customer_id, ctx.score or 0.0, ctx.decision, ctx.reasons, ctx.amount)


class KafkaAlertPublisher(DecisionListener):
    def __init__(self, producer, topic: str = "fraud.alerts") -> None:
        self.producer, self.topic = producer, topic

    def on_decision(self, ctx: ReturnContext) -> None:
        if ctx.decision == "block":
            self.producer.send(self.topic, key=ctx.customer_id, value=json.dumps({
                "return_id": ctx.return_id, "customer_id": ctx.customer_id, "score": ctx.score,
                "reasons": ctx.reasons, "amount": ctx.amount}, ensure_ascii=False))


class MetricsCollector(DecisionListener):
    def __init__(self) -> None:
        self.decisions: Counter = Counter()
        self.by_handler: Counter = Counter()

    def on_decision(self, ctx: ReturnContext) -> None:
        self.decisions[ctx.decision] += 1
        self.by_handler[ctx.decided_by] += 1
        DECISIONS.labels(ctx.decision, ctx.decided_by).inc()
        SCORES.observe(ctx.score or 0.0)
        if ctx.decision == "block":
            BLOCKED_AMOUNT.inc(ctx.amount)
