"""Наблюдатели за событиями тикетов — паттерн Observer.

TicketEventBus публикует события (classified, routed, undone). Подписчики:
  * SearchIndexer   — индексирует тикет в Elasticsearch (поиск похожих, аналитика);
  * SlaMonitor      — считает дедлайн реакции по приоритету и копит нарушения;
  * KafkaPublisher  — отправляет tickets.routed / incidents.detected;
  * MetricsListener — счётчики по командам/категориям для дашборда.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime, timedelta

try:
    from prometheus_client import Counter as PCounter

    ROUTED = PCounter("ticketflow_routed_total", "Маршрутизированные тикеты", ["team", "category", "priority"])
    RULES = PCounter("ticketflow_rules_fired_total", "Сработавшие правила цепочки", ["rule"])
except ImportError:  # pragma: no cover
    ROUTED = RULES = None

SLA_MINUTES = {"critical": 15, "high": 60, "medium": 240, "low": 1440}


class TicketListener(ABC):
    @abstractmethod
    def on_event(self, event: str, ticket: dict, meta: dict) -> None: ...


class TicketEventBus:
    def __init__(self) -> None:
        self.listeners: list[TicketListener] = []

    def subscribe(self, l: TicketListener) -> None:
        self.listeners.append(l)

    def publish(self, event: str, ticket: dict, meta: dict | None = None) -> None:
        for l in self.listeners:
            l.on_event(event, ticket, meta or {})


class SearchIndexer(TicketListener):
    def __init__(self, search) -> None:
        self.search = search

    def on_event(self, event, ticket, meta):
        if event in ("routed", "undone"):
            self.search.index(ticket)


class SlaMonitor(TicketListener):
    def __init__(self) -> None:
        self.deadlines: dict[str, str] = {}

    def on_event(self, event, ticket, meta):
        if event == "routed" and ticket.get("priority"):
            created = datetime.fromisoformat(ticket["created_at"])
            self.deadlines[ticket["ticket_id"]] = (created + timedelta(minutes=SLA_MINUTES[ticket["priority"]])).isoformat()


class KafkaPublisher(TicketListener):
    def __init__(self, producer) -> None:
        self.producer = producer

    def on_event(self, event, ticket, meta):
        if event == "routed":
            self.producer.send("tickets.routed", key=ticket["ticket_id"],
                               value=json.dumps({"ticket": ticket, **meta}, ensure_ascii=False))
            if "mass_incident" in ticket.get("tags", []):
                self.producer.send("incidents.detected", key=ticket["incident_id"],
                                   value=json.dumps({"incident_id": ticket["incident_id"], "ticket_id": ticket["ticket_id"]}))


class MetricsListener(TicketListener):
    def __init__(self) -> None:
        self.by_team, self.by_rule, self.events = Counter(), Counter(), Counter()

    def on_event(self, event, ticket, meta):
        self.events[event] += 1
        if event == "routed":
            self.by_team[ticket.get("team")] += 1
            for r in meta.get("trace", []):
                self.by_rule[r] += 1
            if ROUTED is not None:
                ROUTED.labels(str(ticket.get("team")), str(ticket.get("category")), str(ticket.get("priority"))).inc()
                for r in meta.get("trace", []):
                    RULES.labels(r).inc()
