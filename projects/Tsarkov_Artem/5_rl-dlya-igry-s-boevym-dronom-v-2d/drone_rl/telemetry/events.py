"""Телеметрия — паттерн Observer.

Среда (Subject) публикует события step/episode_end. Наблюдатели подписываются
и обрабатывают их: KafkaTelemetryObserver отправляет в Kafka, MemoryObserver
копит для тестов, StatsObserver считает агрегаты. Среда ничего не знает о
том, куда уходят данные.
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram

    EPISODES = Counter("drone_episodes_total", "Эпизоды", ["policy", "scenario", "outcome"])
    RETURNS = Histogram("drone_episode_return", "Return эпизода", ["policy"], buckets=(-10, -5, 0, 5, 10, 15, 18, 20))
    STEPS = Counter("drone_steps_total", "Шаги симуляции", ["action"])
except ImportError:  # pragma: no cover
    EPISODES = RETURNS = STEPS = None


@dataclass
class TelemetryEvent:
    type: str  # "step" | "episode_end"
    episode_id: str
    step: int
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class TelemetryObserver(ABC):
    @abstractmethod
    def notify(self, event: TelemetryEvent) -> None: ...


class TelemetrySubject:
    def __init__(self) -> None:
        self._observers: list[TelemetryObserver] = []

    def attach(self, obs: TelemetryObserver) -> None:
        self._observers.append(obs)

    def detach(self, obs: TelemetryObserver) -> None:
        self._observers.remove(obs)

    def emit(self, event: TelemetryEvent) -> None:
        for o in self._observers:
            try:
                o.notify(event)
            except Exception:  # наблюдатель не должен ронять симуляцию
                log.exception("telemetry observer failed")


class MemoryObserver(TelemetryObserver):
    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def notify(self, event: TelemetryEvent) -> None:
        self.events.append(event)


class StatsObserver(TelemetryObserver):
    def __init__(self, policy: str = "unknown") -> None:
        self.policy = policy
        self.episodes: list[dict] = []
        self.action_counts: dict[str, int] = defaultdict(int)

    def notify(self, event: TelemetryEvent) -> None:
        if event.type == "step":
            self.action_counts[event.payload.get("action", "?")] += 1
            if STEPS is not None:
                STEPS.labels(event.payload.get("action", "?")).inc()
        elif event.type == "episode_end":
            self.episodes.append(event.payload)
            if EPISODES is not None:
                p = event.payload
                EPISODES.labels(p.get("policy", self.policy), p.get("scenario", "?"), p.get("outcome", "?")).inc()
                RETURNS.labels(p.get("policy", self.policy)).observe(p.get("return", 0.0))


class KafkaTelemetryObserver(TelemetryObserver):
    """Отправляет step-события в топик drone.telemetry, итоги эпизода — в drone.episodes.

    Ключ сообщения = episode_id, поэтому все события эпизода попадают в одну
    партицию и читаются по порядку.
    """

    def __init__(self, producer, step_topic: str = "drone.telemetry",
                 episode_topic: str = "drone.episodes", sample_every: int = 1) -> None:
        self.producer = producer
        self.step_topic = step_topic
        self.episode_topic = episode_topic
        self.sample_every = sample_every

    def notify(self, event: TelemetryEvent) -> None:
        if event.type == "step" and event.step % self.sample_every:
            return
        topic = self.episode_topic if event.type == "episode_end" else self.step_topic
        self.producer.send(topic, key=event.episode_id, value=event.to_json())
