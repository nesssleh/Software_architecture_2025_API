"""Simulator service: гоняет эпизоды выбранной политикой и публикует телеметрию в Kafka.

Топики:
  drone.telemetry  — шаги эпизода (key = episode_id), sample_every шагов
  drone.episodes   — итог эпизода (key = episode_id)
"""
from __future__ import annotations

import argparse
import json
import os

from drone_rl.env.drone_env import DroneCombatEnv
from drone_rl.policies.strategies import make_policy
from drone_rl.telemetry.events import KafkaTelemetryObserver, StatsObserver, TelemetryEvent, TelemetryObserver, TelemetrySubject
from drone_rl.telemetry.kafka_io import make_producer


class PolicyTagObserver(TelemetryObserver):
    """Добавляет имя политики в итог эпизода перед отправкой дальше (декорирует observer)."""

    def __init__(self, inner: TelemetryObserver, policy: str) -> None:
        self.inner, self.policy = inner, policy

    def notify(self, event: TelemetryEvent) -> None:
        if event.type == "episode_end":
            event.payload = {**event.payload, "policy": self.policy}
        self.inner.notify(event)


def simulate(policy_name: str, episodes: int, scenario: str, model_path: str | None, producer=None,
             sample_every: int = 5, seed0: int = 0) -> StatsObserver:
    subject = TelemetrySubject()
    stats = StatsObserver(policy_name)
    subject.attach(stats)
    subject.attach(PolicyTagObserver(KafkaTelemetryObserver(producer or make_producer(), sample_every=sample_every), policy_name))
    env = DroneCombatEnv(scenario=scenario, telemetry=subject)
    policy = make_policy(policy_name, model_path)
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(policy.act(obs, env))
            done = term or trunc
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.getenv("POLICY", "ppo"), help="ppo | heuristic | random | all")
    ap.add_argument("--episodes", type=int, default=int(os.getenv("EPISODES", 50)))
    ap.add_argument("--scenario", default=os.getenv("SCENARIO", "mixed"))
    ap.add_argument("--model", default=os.getenv("MODEL_PATH", "models/ppo_drone_best.zip"))
    ap.add_argument("--loop", action="store_true", default=os.getenv("LOOP", "0") == "1",
                    help="бесконечно гонять эпизоды пачками (для демонстрации и дашборда)")
    a = ap.parse_args()
    if a.loop:
        from drone_rl.obs import start_metrics_server

        start_metrics_server()
    producer = make_producer()
    policies = ["ppo", "heuristic", "random"] if a.policy == "all" else [a.policy]
    batch = 0
    while True:
        for pol in policies:
            stats = simulate(pol, a.episodes, a.scenario, a.model, producer=producer, seed0=batch * 10_000)
            producer.flush()
            succ = sum(e["success"] for e in stats.episodes) / max(len(stats.episodes), 1)
            print(json.dumps({"batch": batch, "policy": pol, "episodes": len(stats.episodes), "success_rate": succ},
                             ensure_ascii=False), flush=True)
        batch += 1
        if not a.loop:
            break


if __name__ == "__main__":
    main()
