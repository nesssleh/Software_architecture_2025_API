"""Тонкая обёртка над Kafka (паттерн Adapter).

Код сервисов работает с интерфейсами EventProducer / EventConsumer.
Реализации:
  * KafkaProducerAdapter / KafkaConsumerAdapter — реальный Kafka (kafka-python);
  * InMemoryBroker — брокер в памяти с той же семантикой (топики, партиции по
    ключу, consumer groups с независимыми offset'ами, DLQ). Используется в
    unit/integration тестах и в локальном демо без Docker.

Выбор реализации — по переменной окружения KAFKA_BOOTSTRAP.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)

try:  # метрики Prometheus (если библиотека установлена)
    from prometheus_client import Counter, Histogram

    MSG_PROCESSED = Counter("kafka_messages_processed_total", "Успешно обработанные сообщения", ["topic", "group"])
    MSG_FAILED = Counter("kafka_messages_failed_total", "Неудачные попытки обработки", ["topic", "group"])
    MSG_DLQ = Counter("kafka_messages_dlq_total", "Сообщения, отправленные в DLQ", ["topic", "group"])
    MSG_LATENCY = Histogram("kafka_message_processing_seconds", "Время обработки сообщения", ["topic", "group"],
                            buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10))
except ImportError:  # pragma: no cover
    MSG_PROCESSED = MSG_FAILED = MSG_DLQ = MSG_LATENCY = None


@dataclass
class Message:
    topic: str
    partition: int
    offset: int
    key: str | None
    value: str

    def json(self) -> Any:
        return json.loads(self.value)


class EventProducer(ABC):
    @abstractmethod
    def send(self, topic: str, value: str | dict, key: str | None = None) -> None: ...

    def flush(self) -> None:  # noqa: B027
        pass


class EventConsumer(ABC):
    @abstractmethod
    def poll(self, max_records: int = 100, timeout_ms: int = 500) -> list[Message]: ...

    @abstractmethod
    def commit(self) -> None: ...


# --------------------------------------------------------------------------- in-memory
class InMemoryBroker:
    """Потокобезопасный брокер в памяти, повторяющий ключевую семантику Kafka."""

    def __init__(self, partitions: int = 3) -> None:
        self.partitions = partitions
        self._log: dict[str, list[list[Message]]] = {}
        self._committed: dict[tuple[str, str, int], int] = defaultdict(int)
        self._lock = threading.RLock()

    def _topic(self, topic: str) -> list[list[Message]]:
        if topic not in self._log:
            self._log[topic] = [[] for _ in range(self.partitions)]
        return self._log[topic]

    def append(self, topic: str, key: str | None, value: str) -> Message:
        with self._lock:
            parts = self._topic(topic)
            p = (zlib.crc32(key.encode()) % self.partitions) if key else sum(map(len, parts)) % self.partitions
            msg = Message(topic, p, len(parts[p]), key, value)
            parts[p].append(msg)
            return msg

    def fetch(self, group: str, topics: Iterable[str], positions: dict, max_records: int) -> list[Message]:
        out: list[Message] = []
        with self._lock:
            for t in topics:
                for p, plog in enumerate(self._topic(t)):
                    pos = positions.setdefault((t, p), self._committed[(group, t, p)])
                    chunk = plog[pos: pos + max_records - len(out)]
                    out.extend(chunk)
                    positions[(t, p)] = pos + len(chunk)
                    if len(out) >= max_records:
                        return out
        return out

    def commit(self, group: str, positions: dict) -> None:
        with self._lock:
            for (t, p), pos in positions.items():
                self._committed[(group, t, p)] = pos

    def messages(self, topic: str) -> list[Message]:
        with self._lock:
            return [m for part in self._topic(topic) for m in part]

    def lag(self, group: str, topic: str) -> int:
        with self._lock:
            return sum(len(pl) - self._committed[(group, topic, p)] for p, pl in enumerate(self._topic(topic)))

    def producer(self) -> "InMemoryProducer":
        return InMemoryProducer(self)

    def consumer(self, *topics: str, group_id: str) -> "InMemoryConsumer":
        return InMemoryConsumer(self, list(topics), group_id)


class InMemoryProducer(EventProducer):
    def __init__(self, broker: InMemoryBroker) -> None:
        self.broker = broker

    def send(self, topic: str, value: str | dict, key: str | None = None) -> None:
        self.broker.append(topic, key, value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))


class InMemoryConsumer(EventConsumer):
    def __init__(self, broker: InMemoryBroker, topics: list[str], group_id: str) -> None:
        self.broker, self.topics, self.group_id = broker, topics, group_id
        self._positions: dict = {}

    def poll(self, max_records: int = 100, timeout_ms: int = 0) -> list[Message]:
        return self.broker.fetch(self.group_id, self.topics, self._positions, max_records)

    def commit(self) -> None:
        self.broker.commit(self.group_id, self._positions)


# --------------------------------------------------------------------------- real kafka
class KafkaProducerAdapter(EventProducer):
    def __init__(self, bootstrap: str) -> None:
        from kafka import KafkaProducer  # type: ignore

        self._p = KafkaProducer(bootstrap_servers=bootstrap, acks="all", retries=5, linger_ms=5, request_timeout_ms=10000,
                                key_serializer=lambda k: k.encode() if k else None,
                                value_serializer=lambda v: v.encode())

    def send(self, topic: str, value: str | dict, key: str | None = None) -> None:
        self._p.send(topic, key=key, value=value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))

    def flush(self) -> None:
        self._p.flush()


class KafkaConsumerAdapter(EventConsumer):
    def __init__(self, bootstrap: str, topics: list[str], group_id: str) -> None:
        from kafka import KafkaConsumer  # type: ignore

        self.group_id = group_id
        self._c = KafkaConsumer(*topics, bootstrap_servers=bootstrap, group_id=group_id,
                                enable_auto_commit=False, auto_offset_reset="earliest",
                                key_deserializer=lambda k: k.decode() if k else None,
                                value_deserializer=lambda v: v.decode())

    def poll(self, max_records: int = 100, timeout_ms: int = 500) -> list[Message]:
        batch = self._c.poll(timeout_ms=timeout_ms, max_records=max_records)
        return [Message(r.topic, r.partition, r.offset, r.key, r.value) for recs in batch.values() for r in recs]

    def commit(self) -> None:
        self._c.commit()


# --------------------------------------------------------------------------- factory helpers
_DEFAULT_BROKER = InMemoryBroker()


def default_broker() -> InMemoryBroker:
    return _DEFAULT_BROKER


def make_producer() -> EventProducer:
    bs = os.getenv("KAFKA_BOOTSTRAP")
    return KafkaProducerAdapter(bs) if bs else _DEFAULT_BROKER.producer()


def make_consumer(*topics: str, group_id: str) -> EventConsumer:
    bs = os.getenv("KAFKA_BOOTSTRAP")
    return KafkaConsumerAdapter(bs, list(topics), group_id) if bs else _DEFAULT_BROKER.consumer(*topics, group_id=group_id)


def run_consumer_loop(consumer: EventConsumer, handler: Callable[[Message], None], producer: EventProducer | None = None,
                      dlq_topic: str | None = None, max_retries: int = 3, stop: Callable[[], bool] = lambda: False,
                      once: bool = False) -> int:
    """At-least-once обработка: commit после успешной обработки батча.

    Сообщение, упавшее max_retries раз, отправляется в DLQ (если задан) и
    не блокирует партицию. Возвращает число обработанных сообщений.
    """
    processed = 0
    while not stop():
        batch = consumer.poll()
        if not batch:
            if once:  # режим «вычитать всё, что есть, и выйти» (тесты, batch-джобы)
                break
            continue
        group = getattr(consumer, "group_id", "") or ""
        if hasattr(handler, "handle_batch"):  # батчевая обработка (например, инференс модели на пачке сообщений)
            t0 = time.perf_counter()
            try:
                handler.handle_batch(batch)
                processed += len(batch)
                if MSG_PROCESSED is not None:
                    dt = (time.perf_counter() - t0) / len(batch)
                    for m in batch:
                        MSG_PROCESSED.labels(m.topic, group).inc()
                        MSG_LATENCY.labels(m.topic, group).observe(dt)
                consumer.commit()
                continue
            except Exception as e:  # noqa: BLE001 — откат на поштучную обработку, чтобы изолировать «ядовитое» сообщение
                log.warning("batch handler failed, falling back to per-message: %s", e)
        for msg in batch:
            for attempt in range(1, max_retries + 1):
                t0 = time.perf_counter()
                try:
                    handler(msg)
                    processed += 1
                    if MSG_PROCESSED is not None:
                        MSG_PROCESSED.labels(msg.topic, group).inc()
                        MSG_LATENCY.labels(msg.topic, group).observe(time.perf_counter() - t0)
                    break
                except Exception as e:  # noqa: BLE001
                    log.warning("handler failed (attempt %s/%s): %s", attempt, max_retries, e)
                    if MSG_FAILED is not None:
                        MSG_FAILED.labels(msg.topic, group).inc()
                    if attempt == max_retries and producer and dlq_topic:
                        producer.send(dlq_topic, key=msg.key,
                                      value=json.dumps({"error": str(e), "topic": msg.topic, "offset": msg.offset,
                                                        "original": msg.value}, ensure_ascii=False))
                        if MSG_DLQ is not None:
                            MSG_DLQ.labels(msg.topic, group).inc()
                    else:
                        time.sleep(min(0.05 * 2 ** attempt, 1.0))  # экспоненциальная пауза между ретраями
        consumer.commit()
    return processed
