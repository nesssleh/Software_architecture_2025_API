"""Прикладная логика: классификация и маршрутизация тикета (используется и API, и Kafka-воркерами)."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from .classify.strategies import ClassifierStrategy, make_strategy
from .events import KafkaPublisher, MetricsListener, SearchIndexer, SlaMonitor, TicketEventBus
from .kafka_io import EventProducer, make_producer
from .routing.chain import CounterStore, RouteHandler, RoutingContext, build_router
from .routing.commands import CommandInvoker, Ticket
from .storage.repositories import SearchRepository, TicketRepository, make_search


SNAPSHOT_FIELDS = ("team", "priority", "tags", "incident_id", "status")


class ClassificationService:
    def __init__(self, strategy: ClassifierStrategy) -> None:
        self.strategy = strategy

    def classify(self, t: Ticket) -> Ticket:
        p = self.strategy.predict(t.text, t.customer_tier)
        t.category, t.category_conf, t.priority, t.priority_conf, t.model = (p.category, p.category_conf, p.priority,
                                                                             p.priority_conf, p.model)
        t.status = "classified"
        return t

    def classify_batch(self, tickets: list[Ticket]) -> list[Ticket]:
        from .datagen import CATEGORIES, PRIORITIES

        pc, pp = self.strategy.predict_proba([t.text for t in tickets], [t.customer_tier for t in tickets])
        for t, c, p in zip(tickets, pc, pp):
            ci, pi = int(c.argmax()), int(p.argmax())
            t.category, t.category_conf = CATEGORIES[ci], float(c[ci])
            t.priority, t.priority_conf = PRIORITIES[pi], float(p[pi])
            t.model, t.status = self.strategy.name, "classified"
        return tickets


class RoutingService:
    def __init__(self, repo: TicketRepository, search: SearchRepository, producer: EventProducer,
                 router: RouteHandler | None = None) -> None:
        self.repo, self.search = repo, search
        self.router = router or build_router(CounterStore(_redis()), float(os.getenv("MIN_CONFIDENCE", 0.55)))
        self.invoker = CommandInvoker()
        self.bus = TicketEventBus()
        self.sla, self.metrics = SlaMonitor(), MetricsListener()
        for l in (SearchIndexer(search), self.sla, KafkaPublisher(producer), self.metrics):
            self.bus.subscribe(l)

    def route(self, t: Ticket) -> dict:
        ctx = self.router.handle(RoutingContext(t))
        before = {k: (list(v) if isinstance(v, list) else v) for k, v in t.as_dict().items() if k in SNAPSHOT_FIELDS}
        n0 = len(self.invoker.audit)
        self.invoker.run(t, ctx.commands)
        self.repo.upsert(t.as_dict())
        # снимок «до маршрутизации» в журнале: отмена работает в любом экземпляре API, а не только в том, где исполнялись команды
        self.repo.add_audit([{"ticket_id": t.ticket_id, "at": self.invoker.audit[n0]["at"] if len(self.invoker.audit) > n0 else "",
                              "actor": "router", "command": "snapshot", "before": before}] + self.invoker.audit[n0:])
        self.bus.publish("routed", t.as_dict(), {"trace": ctx.trace, "commands": [c.describe() for c in ctx.commands]})
        return {"ticket": t.as_dict(), "trace": ctx.trace, "commands": [c.describe() for c in ctx.commands],
                "sla_deadline": self.sla.deadlines.get(t.ticket_id)}

    def undo(self, ticket_id: str) -> dict | None:
        row = self.repo.get(ticket_id)
        if not row:
            return None
        t = Ticket(**{k: row[k] for k in Ticket.__dataclass_fields__ if k in row})
        n0 = len(self.invoker.audit)
        undone = self.invoker.undo_all(t)
        if not undone:  # команды исполнялись в другом процессе — восстанавливаем состояние по снимку из журнала
            audit = self.repo.audit(ticket_id)
            snaps = [i for i, a in enumerate(audit) if a["command"] == "snapshot"]
            if snaps and not any(a["command"].startswith("undo") for a in audit[snaps[-1]:]):
                for k, v in audit[snaps[-1]]["before"].items():
                    setattr(t, k, v)
                undone = sum(1 for a in audit[snaps[-1] + 1:] if not a["command"].startswith("undo"))
                self.invoker.audit.append({"ticket_id": ticket_id, "actor": "operator", "command": "undo_routing",
                                           "at": datetime.now(timezone.utc).isoformat()})
        t.status = "needs_triage" if undone else t.status
        self.repo.upsert(t.as_dict())
        self.repo.add_audit(self.invoker.audit[n0:])
        self.bus.publish("undone", t.as_dict())
        return {"ticket": t.as_dict(), "undone_commands": undone}


def _redis():
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    import redis

    return redis.Redis.from_url(url)


def build(producer: EventProducer | None = None) -> tuple[ClassificationService, RoutingService]:
    cls = ClassificationService(make_strategy(os.getenv("CLASSIFIER", "ensemble"), os.getenv("MODEL_PATH")))
    return cls, RoutingService(TicketRepository(), make_search(), producer or make_producer())
