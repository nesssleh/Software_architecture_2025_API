"""TicketFlow API.

POST /v1/tickets?mode=async|sync      — новое обращение (async: в Kafka tickets.created)
POST /v1/classify                      — предпросмотр классификации без сохранения
GET  /v1/tickets/{id}                  — тикет + журнал команд
POST /v1/tickets/{id}/undo-routing     — отменить авто-маршрутизацию (Command.undo)
GET  /v1/search?q=&team=               — поиск похожих обращений (Elasticsearch)
GET  /v1/stats                         — распределение по командам/приоритетам/категориям
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

from ticketflow.obs import instrument
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ticketflow.kafka_io import make_producer
from ticketflow.routing.commands import Ticket
from ticketflow.service import build

app = FastAPI(title="TicketFlow — классификация и маршрутизация обращений", version="1.0.0")
instrument(app)


@lru_cache(maxsize=1)
def producer():
    """Один Kafka-producer на процесс (создание producer'а на каждый запрос — это сотни миллисекунд)."""
    return make_producer()


@lru_cache(maxsize=1)
def services():
    return build()


class TicketIn(BaseModel):
    text: str = Field(min_length=5, max_length=10_000)
    customer_tier: Literal["free", "pro", "enterprise"] = "free"
    channel: Literal["email", "chat", "web", "app"] = "web"
    ticket_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "classifier": services()[0].strategy.name}


@app.post("/v1/classify")
def classify(body: TicketIn):
    return services()[0].strategy.predict(body.text, body.customer_tier).as_dict()


@app.post("/v1/tickets", status_code=201)
def create(body: TicketIn, mode: str = Query("sync", pattern="^(sync|async)$")):
    t = Ticket(ticket_id=body.ticket_id or f"T-{uuid.uuid4().hex[:10]}", text=body.text, customer_tier=body.customer_tier,
               channel=body.channel, created_at=datetime.now(timezone.utc).isoformat())
    if mode == "async":
        producer().send("tickets.created", key=t.ticket_id, value=t.as_dict())
        return {"ticket_id": t.ticket_id, "status": "queued"}
    cls, rt = services()
    return rt.route(cls.classify(t))


@app.get("/v1/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    rt = services()[1]
    t = rt.repo.get(ticket_id)
    if not t:
        raise HTTPException(404, "ticket not found")
    return {"ticket": t, "audit": rt.repo.audit(ticket_id), "sla_deadline": rt.sla.deadlines.get(ticket_id)}


@app.post("/v1/tickets/{ticket_id}/undo-routing")
def undo(ticket_id: str):
    r = services()[1].undo(ticket_id)
    if r is None:
        raise HTTPException(404, "ticket not found")
    return r


@app.get("/v1/search")
def search(q: str = Query(min_length=2), team: str | None = None, k: int = Query(10, le=50)):
    return {"query": q, "hits": services()[1].search.search(q, k, team)}


@app.get("/v1/stats")
def stats():
    rt = services()[1]
    return {**rt.repo.stats(), "rules_fired": dict(rt.metrics.by_rule)}
