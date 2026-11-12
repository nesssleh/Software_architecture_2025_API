"""Интеграционные тесты: simulator -> Kafka(in-memory) -> consumer -> БД -> API."""
import json

from fastapi.testclient import TestClient

from drone_rl.storage import EpisodeRepository
from drone_rl.telemetry.kafka_io import InMemoryBroker
from services.simulator.run import simulate
from services.telemetry_consumer import consumer as consumer_mod


def test_pipeline_simulator_to_db(tmp_path):
    broker = InMemoryBroker()
    stats = simulate("heuristic", 5, "mixed", None, producer=broker.producer())
    assert len(stats.episodes) == 5
    msgs = broker.messages("drone.episodes")
    assert len(msgs) == 5 and json.loads(msgs[0].value)["payload"]["policy"] == "heuristic"
    # все шаги эпизода — в одной партиции (key = episode_id)
    steps = broker.messages("drone.telemetry")
    by_ep = {}
    for m in steps:
        by_ep.setdefault(m.key, set()).add(m.partition)
    assert all(len(p) == 1 for p in by_ep.values())

    repo = EpisodeRepository(f"sqlite:///{tmp_path}/t.db")
    n = consumer_mod.run(repo, broker.consumer("drone.episodes", group_id="episode-writer"), broker.producer(), once=True)
    assert n == 5 and repo.count() == 5
    # повторная доставка не создаёт дублей (идемпотентность)
    consumer_mod.run(repo, broker.consumer("drone.episodes", group_id="replay"), broker.producer(), once=True)
    assert repo.count() == 5
    assert broker.lag("episode-writer", "drone.episodes") == 0


def test_bad_message_goes_to_dlq(tmp_path):
    broker = InMemoryBroker()
    broker.producer().send("drone.episodes", "{not json", key="x")
    repo = EpisodeRepository(f"sqlite:///{tmp_path}/t.db")
    consumer_mod.run(repo, broker.consumer("drone.episodes", group_id="g"), broker.producer(), once=True)
    assert len(broker.messages("drone.episodes.dlq")) == 1


def test_api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api.db")
    from services.inference import app as app_mod

    app_mod.repo.cache_clear()
    c = TestClient(app_mod.app)
    assert c.get("/health").json()["status"] == "ok"
    assert "heuristic" in c.get("/v1/policies").json()["policies"]
    r = c.post("/v1/act", json={"observation": [0.0] * 17, "policy": "heuristic"})
    assert r.status_code == 200 and 0 <= r.json()["action"] <= 4
    assert c.post("/v1/act", json={"observation": [0.0] * 3}).status_code == 422
    r = c.post("/v1/episodes/run", json={"policy": "heuristic", "scenario": "open", "seed": 1, "render": True})
    assert r.status_code == 200 and r.json()["frame_png_base64"]
    assert c.post("/v1/episodes/run", json={"policy": "heuristic", "scenario": "moon"}).status_code == 400
    assert c.get("/v1/stats").json()["total_episodes"] == 1
