"""Telemetry consumer: читает drone.episodes (consumer group episode-writer) и пишет в PostgreSQL.

Идемпотентность: upsert по episode_id — повторная доставка (at-least-once)
не создаёт дублей. Невалидные сообщения после 3 попыток уходят в drone.episodes.dlq.
"""
from __future__ import annotations

import logging

from drone_rl.storage import EpisodeRepository
from drone_rl.telemetry.kafka_io import EventConsumer, EventProducer, Message, make_consumer, make_producer, run_consumer_loop

GROUP = "episode-writer"


class EpisodeHandler:
    def __init__(self, repo: EpisodeRepository) -> None:
        self.repo = repo

    def __call__(self, msg: Message) -> None:
        ev = msg.json()
        s = ev["payload"]
        policy = s.get("policy", "unknown")
        # симулятор гоняет все политики на одинаковых seed (честное сравнение) → id эпизода совпадают, ключ включает политику
        self.repo.upsert(policy, f"{policy}:{ev['episode_id']}", s)


def run(repo: EpisodeRepository | None = None, consumer: EventConsumer | None = None,
        producer: EventProducer | None = None, once: bool = False) -> int:
    repo = repo or EpisodeRepository()
    consumer = consumer or make_consumer("drone.episodes", group_id=GROUP)
    return run_consumer_loop(consumer, EpisodeHandler(repo), producer or make_producer(), "drone.episodes.dlq", once=once)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from drone_rl.obs import start_metrics_server

    start_metrics_server()
    run()
