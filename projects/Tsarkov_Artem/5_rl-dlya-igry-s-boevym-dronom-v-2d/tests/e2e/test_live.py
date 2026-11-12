"""E2E против живого стека: simulator → Kafka (drone.telemetry / drone.episodes) → telemetry-consumer → PostgreSQL → API."""
import os
from collections import defaultdict

import httpx

httpx = httpx.Client(timeout=60)  # под нагрузкой стенда ответы бывают медленными

from .helpers import TopicWatcher, kafka_admin_state, live, wait_for

pytestmark = live
API = os.getenv("API_URL", "http://inference-api:8000")


def test_api_healthy():
    r = httpx.get(f"{API}/health", timeout=10)
    print(r.json())
    assert r.status_code == 200 and r.json()["model_loaded"]


def test_kafka_topics_and_consumer_groups():
    topics, groups = kafka_admin_state()
    print("topics:", sorted(t for t in topics if not t.startswith("__")), "groups:", sorted(groups))
    assert {"drone.telemetry", "drone.episodes"} <= topics
    assert "episode-writer" in groups


def test_simulator_streams_telemetry_ordered_by_episode():
    w = TopicWatcher("drone.telemetry")
    parts = defaultdict(set)
    seen = []

    def collect(v):
        seen.append(v)
        parts[v["episode_id"]].add(v["step"])
        return len(seen) >= 200
    w.find(collect, timeout=120)
    w.close()
    print(f"получено {len(seen)} сообщений телеметрии по {len(parts)} эпизодам")
    assert len(parts) >= 2


def test_episode_results_reach_postgres():
    st = wait_for(lambda: (s := httpx.get(f"{API}/v1/stats").json()) and s["total_episodes"] >= 10 and s, 120, 2,
                  "итоги эпизодов от симулятора в PostgreSQL")
    for row in st["by_policy"]:
        print(row)
    assert {r["policy"] for r in st["by_policy"]} >= {"ppo", "heuristic"}


def test_policy_inference():
    r = httpx.post(f"{API}/v1/act", json={"observation": [0.0] * 17, "policy": "ppo"})
    assert r.status_code == 200 and 0 <= r.json()["action"] <= 4
    e = httpx.post(f"{API}/v1/episodes/run", json={"policy": "ppo", "scenario": "forest", "seed": 3}).json()
    print({k: e[k] for k in ("scenario", "outcome", "steps", "return", "hits", "shots")})
    assert e["outcome"] == "target_destroyed"
    assert httpx.post(f"{API}/v1/act", json={"observation": [0.0] * 3}).status_code == 422
