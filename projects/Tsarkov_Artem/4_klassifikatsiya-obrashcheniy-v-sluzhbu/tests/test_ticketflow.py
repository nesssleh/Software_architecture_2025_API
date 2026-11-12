import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ticketflow.classify.strategies import KeywordStrategy, make_strategy
from ticketflow.datagen import CATEGORIES, generate
from ticketflow.kafka_io import InMemoryBroker, run_consumer_loop
from ticketflow.routing.chain import CounterStore, build_router, RoutingContext
from ticketflow.routing.commands import AddTagCommand, AssignTeamCommand, CommandInvoker, SetPriorityCommand, Ticket
from ticketflow.service import ClassificationService, RoutingService
from ticketflow.storage.repositories import InMemorySearch, TicketRepository
from services.workers import ClassifyHandler, RouteHandlerWorker

HF = Path("models/hf_ticket_model")


def _t(text, cat, conf=0.9, pr="medium", tier="free", tid="T1"):
    return Ticket(tid, text, tier, category=cat, category_conf=conf, priority=pr)


def test_datagen_labels():
    data = generate(300)
    assert {d["category"] for d in data} == set(CATEGORIES)
    assert all(d["team"] for d in data)


def test_commands_undo():
    t = _t("x", "billing")
    inv = CommandInvoker()
    inv.run(t, [AssignTeamCommand("billing_team", "r"), SetPriorityCommand("high", "r"), AddTagCommand("vip")])
    assert (t.team, t.priority, t.tags) == ("billing_team", "high", ["vip"])
    assert inv.undo_all(t) == 3
    assert (t.team, t.priority, t.tags) == (None, "medium", [])
    assert [a["command"] for a in inv.audit][-1] == "undo_assign_team"


def test_chain_security_overrides_model():
    ctx = build_router().handle(RoutingContext(_t("Похоже на фишинг, пришло письмо", "billing")))
    teams = [c.team for c in ctx.commands if isinstance(c, AssignTeamCommand)]
    assert teams == ["security_team"] and ctx.trace[0] == "security"


def test_chain_low_confidence_to_triage():
    ctx = build_router().handle(RoutingContext(_t("что-то непонятное", "billing", conf=0.3)))
    assert [c.team for c in ctx.commands if isinstance(c, AssignTeamCommand)] == ["l1_triage"]


def test_chain_vip_and_category():
    ctx = build_router().handle(RoutingContext(_t("Не пришёл чек", "billing", tier="enterprise", pr="low")))
    names = [c.name for c in ctx.commands]
    assert "set_priority" in names and ctx.trace == ["vip", "category"]


def test_outage_detection():
    clock = iter(range(0, 1000, 10))
    router = build_router(CounterStore(), outage_threshold=5, clock=lambda: next(clock))
    traces = [router.handle(RoutingContext(_t("Ошибка 500", "technical", tid=f"T{i}"))).trace for i in range(6)]
    assert "outage" not in traces[3] and "outage" in traces[5]


def test_keyword_baseline():
    p = KeywordStrategy().predict("Взломали аккаунт, сменили пароль")
    assert p.category == "security"


def test_pipeline_via_kafka(tmp_path):
    broker = InMemoryBroker()
    prod = broker.producer()
    cls = ClassificationService(KeywordStrategy())
    rt = RoutingService(TicketRepository(f"sqlite:///{tmp_path}/t.db"), InMemorySearch(), prod)
    for d in generate(50, seed=11):
        prod.send("tickets.created", key=d["ticket_id"], value=json.dumps(d, ensure_ascii=False))
    prod.send("tickets.created", key="bad", value="{}")
    h1, h2 = ClassifyHandler(cls, prod), RouteHandlerWorker(rt)
    run_consumer_loop(broker.consumer(h1.topic_in, group_id=h1.group), h1, prod, "tickets.created.dlq", once=True)
    run_consumer_loop(broker.consumer(h2.topic_in, group_id=h2.group), h2, prod, "tickets.classified.dlq", once=True)
    assert len(broker.messages("tickets.routed")) == 50
    assert len(broker.messages("tickets.created.dlq")) == 1
    assert sum(rt.repo.stats()["by_team"].values()) == 50
    assert rt.search.search("оплата")
    assert rt.repo.audit(broker.messages("tickets.routed")[0].key)


@pytest.mark.skipif(not HF.exists(), reason="HF model not trained")
def test_hf_strategy():
    p = make_strategy("hf_transformer", str(HF)).predict("Не приходит смс с кодом для входа")
    assert p.category == "account_access" and 0 <= p.category_conf <= 1


@pytest.mark.skipif(not HF.exists(), reason="HF model not trained")
def test_api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api.db")
    from services.intake_api import app as m

    m.services.cache_clear()
    c = TestClient(m.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/v1/tickets", json={"text": "СРОЧНО! Списали деньги дважды за подписку Pro", "customer_tier": "pro"})
    assert r.status_code == 201
    tid = r.json()["ticket"]["ticket_id"]
    assert r.json()["ticket"]["team"] in ("billing_team", "l1_triage")
    got = c.get(f"/v1/tickets/{tid}").json()
    assert got["audit"] and got["sla_deadline"]
    u = c.post(f"/v1/tickets/{tid}/undo-routing").json()
    assert u["undone_commands"] >= 1 and u["ticket"]["team"] is None
    assert c.get("/v1/search", params={"q": "списали деньги"}).json()["hits"]
    assert c.post("/v1/tickets", json={"text": "hi"}).status_code == 422
    assert c.get("/v1/tickets/nope").status_code == 404
