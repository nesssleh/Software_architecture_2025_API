"""Общие помощники e2e-тестов против живого docker-compose стека."""
from __future__ import annotations

import json
import os
import time
import uuid

import pytest

LIVE = os.getenv("E2E_LIVE") == "1"
live = pytest.mark.skipif(not LIVE, reason="e2e: запускается только внутри docker-сети (docker compose --profile e2e run e2e)")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")


def wait_for(fn, timeout: float = 60, every: float = 1.0, what: str = "condition"):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            last = fn()
            if last:
                print(f"  ✓ {what}: {time.time() - t0:.1f} с")
                return last
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(every)
    raise AssertionError(f"timeout waiting for {what}; last={last!r}")


def kafka_admin_state():
    from kafka import KafkaAdminClient

    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP, client_id="e2e")
    try:
        if hasattr(admin, "list_groups"):          # kafka-python >= 3.0
            groups = {g["group_id"] for g in admin.list_groups()}
        else:                                       # kafka-python 2.x
            groups = {g[0] for g in admin.list_consumer_groups()}
        return set(admin.list_topics()), groups
    finally:
        admin.close()


class TopicWatcher:
    """Подписка на топик ДО отправки события: ловит только новые сообщения."""

    def __init__(self, topic: str):
        from kafka import KafkaConsumer

        self.c = KafkaConsumer(topic, bootstrap_servers=BOOTSTRAP, group_id=f"e2e-{uuid.uuid4().hex[:8]}",
                               auto_offset_reset="latest", enable_auto_commit=False,
                               key_deserializer=lambda k: k.decode() if k else None,
                               value_deserializer=lambda v: v.decode())
        wait_for(lambda: self._assigned(), 30, 0.2, f"назначение партиций {topic}")

    def _assigned(self):
        self.c.poll(timeout_ms=200)
        return bool(self.c.assignment())

    def find(self, pred, timeout: float = 60):
        found = []

        def step():
            for recs in self.c.poll(timeout_ms=500).values():
                for r in recs:
                    v = json.loads(r.value)
                    if pred(v):
                        found.append((r.partition, r.offset, v))
            return found[0] if found else None
        return wait_for(step, timeout, 0.1, "сообщение в Kafka")

    def close(self):
        self.c.close()
