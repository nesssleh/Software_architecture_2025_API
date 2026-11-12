"""Хранилища — паттерн Repository.

Бизнес-логика работает с абстракциями, конкретная СУБД выбирается конфигурацией:

* CustomerFeatureRepository — онлайн-состояние клиента: Redis (prod) / память (тесты);
* AnalyticsRepository       — сырой поток событий и скоринги для аналитики:
                              ClickHouse (prod) / SQLite (локально);
* CaseRepository            — кейсы расследований и решения аналитиков: PostgreSQL / SQLite.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, desc, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from ..features import CustomerState


# ----------------------------------------------------------------------------- online features
class CustomerFeatureRepository(ABC):
    @abstractmethod
    def get(self, customer_id: str) -> CustomerState: ...

    @abstractmethod
    def save(self, st: CustomerState) -> None: ...

    def seen_event(self, event_id: str) -> bool:
        """Дедупликация (идемпотентность consumer'а). True — событие уже обрабатывали."""
        return False


class InMemoryCustomerRepo(CustomerFeatureRepository):
    def __init__(self) -> None:
        self._d: dict[str, dict] = {}
        self._seen: set[str] = set()

    def get(self, customer_id: str) -> CustomerState:
        d = self._d.get(customer_id)
        return CustomerState.from_dict(json.loads(json.dumps(d))) if d else CustomerState(customer_id)

    def save(self, st: CustomerState) -> None:
        self._d[st.customer_id] = st.to_dict()

    def seen_event(self, event_id: str) -> bool:
        if event_id in self._seen:
            return True
        self._seen.add(event_id)
        return False


class RedisCustomerRepo(CustomerFeatureRepository):
    def __init__(self, url: str) -> None:
        import redis

        self.r = redis.Redis.from_url(url, decode_responses=True)

    def get(self, customer_id: str) -> CustomerState:
        raw = self.r.get(f"cust:{customer_id}")
        return CustomerState.from_dict(json.loads(raw)) if raw else CustomerState(customer_id)

    def save(self, st: CustomerState) -> None:
        self.r.set(f"cust:{st.customer_id}", json.dumps(st.to_dict()))

    def seen_event(self, event_id: str) -> bool:
        # SET NX с TTL 7 дней: атомарная проверка «видели ли событие»
        return not self.r.set(f"seen:{event_id}", 1, nx=True, ex=7 * 86400)


def make_customer_repo() -> CustomerFeatureRepository:
    url = os.getenv("REDIS_URL")
    return RedisCustomerRepo(url) if url else InMemoryCustomerRepo()


# ----------------------------------------------------------------------------- analytics
class AnalyticsRepository(ABC):
    @abstractmethod
    def insert_events(self, rows: list[dict]) -> None: ...

    @abstractmethod
    def insert_scores(self, rows: list[dict]) -> None: ...

    @abstractmethod
    def daily_summary(self) -> list[dict]: ...

    @abstractmethod
    def customer_timeline(self, customer_id: str, limit: int = 100) -> list[dict]: ...


class SQLiteAnalyticsRepo(AnalyticsRepository):
    """Локальная замена ClickHouse с той же схемой."""

    def __init__(self, path: str = ":memory:") -> None:
        self.c = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self.c.executescript("""
            PRAGMA journal_mode=WAL; PRAGMA synchronous=OFF;
            CREATE TABLE IF NOT EXISTS order_events(event_id TEXT PRIMARY KEY, ts TEXT, type TEXT, customer_id TEXT,
                order_id TEXT, amount REAL, payload TEXT);
            CREATE INDEX IF NOT EXISTS ix_ev_cust_ts ON order_events(customer_id, ts);
            CREATE TABLE IF NOT EXISTS return_scores(return_id TEXT PRIMARY KEY, ts TEXT, customer_id TEXT,
                score REAL, decision TEXT, decided_by TEXT, model_version TEXT, amount REAL);
        """)

    def insert_events(self, rows: list[dict]) -> None:
        with self._lock:
            self.c.executemany("INSERT OR IGNORE INTO order_events VALUES(?,?,?,?,?,?,?)",
                               [(r["event_id"], r["ts"], r["type"], r["customer_id"], r.get("order_id"),
                                 r.get("amount", 0.0), json.dumps(r, ensure_ascii=False)) for r in rows])
            self.c.commit()

    def insert_scores(self, rows: list[dict]) -> None:
        with self._lock:
            self.c.executemany("INSERT OR REPLACE INTO return_scores VALUES(?,?,?,?,?,?,?,?)",
                               [(r["return_id"], r["ts"], r["customer_id"], r["score"], r["decision"],
                                 r["decided_by"], r["model_version"], r["amount"]) for r in rows])
            self.c.commit()

    def daily_summary(self) -> list[dict]:
        cur = self.c.execute("""SELECT substr(ts,1,10) d, count(*), sum(decision='block'), sum(decision='review'),
                                avg(score), sum(CASE WHEN decision='block' THEN amount ELSE 0 END)
                                FROM return_scores GROUP BY d ORDER BY d""")
        return [{"day": d, "returns": n, "blocked": b, "review": r, "avg_score": round(s, 4), "blocked_amount": round(a, 2)}
                for d, n, b, r, s, a in cur.fetchall()]

    def customer_timeline(self, customer_id: str, limit: int = 100) -> list[dict]:
        cur = self.c.execute("SELECT payload FROM order_events WHERE customer_id=? ORDER BY ts DESC LIMIT ?",
                             (customer_id, limit))
        return [json.loads(p) for (p,) in cur.fetchall()]


class ClickHouseAnalyticsRepo(AnalyticsRepository):
    """ClickHouse через HTTP-клиент clickhouse-connect. DDL — в sql/clickhouse.sql."""

    def __init__(self, url: str) -> None:
        import clickhouse_connect  # type: ignore

        # async_insert: ClickHouse сам буферизует мелкие вставки потока событий (иначе — «too many parts»)
        self.client = clickhouse_connect.get_client(dsn=url, settings={"async_insert": 1, "wait_for_async_insert": 0})

    def insert_events(self, rows: list[dict]) -> None:
        self.client.insert("order_events", [[r["event_id"], datetime.fromisoformat(r["ts"]), r["type"], r["customer_id"],
                                             r.get("order_id", ""), float(r.get("amount", 0.0)),
                                             json.dumps(r, ensure_ascii=False)] for r in rows],
                           column_names=["event_id", "ts", "type", "customer_id", "order_id", "amount", "payload"])

    def insert_scores(self, rows: list[dict]) -> None:
        self.client.insert("return_scores", [[r["return_id"], datetime.fromisoformat(r["ts"]), r["customer_id"], r["score"],
                                              r["decision"], r["decided_by"], r["model_version"], r["amount"]] for r in rows],
                           column_names=["return_id", "ts", "customer_id", "score", "decision", "decided_by",
                                         "model_version", "amount"])

    def daily_summary(self) -> list[dict]:
        q = self.client.query("""SELECT toDate(ts) d, count(), countIf(decision='block'), countIf(decision='review'),
                                 avg(score), sumIf(amount, decision='block') FROM return_scores FINAL GROUP BY d ORDER BY d""")
        return [{"day": str(d), "returns": n, "blocked": b, "review": r, "avg_score": round(s, 4), "blocked_amount": round(a, 2)}
                for d, n, b, r, s, a in q.result_rows]

    def customer_timeline(self, customer_id: str, limit: int = 100) -> list[dict]:
        q = self.client.query("SELECT payload FROM order_events WHERE customer_id={c:String} ORDER BY ts DESC LIMIT {l:UInt32}",
                              parameters={"c": customer_id, "l": limit})
        return [json.loads(p) for (p,) in q.result_rows]


def make_analytics_repo() -> AnalyticsRepository:
    url = os.getenv("CLICKHOUSE_URL")
    return ClickHouseAnalyticsRepo(url) if url else SQLiteAnalyticsRepo(os.getenv("ANALYTICS_SQLITE", ":memory:"))


# ----------------------------------------------------------------------------- cases (PostgreSQL)
class Base(DeclarativeBase):
    pass


class CaseRow(Base):
    __tablename__ = "fraud_cases"
    case_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    return_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    customer_id: Mapped[str] = mapped_column(String(32), index=True)
    score: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(16), index=True)
    reasons: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    analyst_verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)
    analyst_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    amount: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CaseRepository:
    def __init__(self, url: str | None = None) -> None:
        self.engine = create_engine(url or os.getenv("DATABASE_URL", "sqlite:///cases.db"))
        _create_schema(self.engine, Base.metadata)

    def open_case(self, return_id: str, customer_id: str, score: float, decision: str, reasons: list[str], amount: float) -> int:
        with Session(self.engine) as s:
            existing = s.scalar(select(CaseRow).where(CaseRow.return_id == return_id))
            if existing:  # идемпотентно
                return existing.case_id
            row = CaseRow(return_id=return_id, customer_id=customer_id, score=score, decision=decision,
                          reasons=reasons, amount=amount)
            s.add(row)
            s.commit()
            return row.case_id

    def list(self, status: str | None = None, limit: int = 50, customer_id: str | None = None) -> list[dict]:
        with Session(self.engine) as s:
            q = select(CaseRow).order_by(desc(CaseRow.score), desc(CaseRow.case_id)).limit(limit)
            if status:
                q = q.where(CaseRow.status == status)
            if customer_id:
                q = q.where(CaseRow.customer_id == customer_id)
            return [self._d(r) for r in s.scalars(q)]

    def get(self, case_id: int) -> dict | None:
        with Session(self.engine) as s:
            r = s.get(CaseRow, case_id)
            return self._d(r) if r else None

    def resolve(self, case_id: int, verdict: str, comment: str | None) -> dict | None:
        with Session(self.engine) as s:
            r = s.get(CaseRow, case_id)
            if not r:
                return None
            r.status, r.analyst_verdict, r.analyst_comment = "closed", verdict, comment
            r.updated_at = datetime.now(timezone.utc)
            s.commit()
            return self._d(r)

    def counts(self) -> dict:
        with Session(self.engine) as s:
            return {st: n for st, n in s.execute(select(CaseRow.status, func.count()).group_by(CaseRow.status))}

    @staticmethod
    def _d(r: CaseRow) -> dict:
        return {"case_id": r.case_id, "return_id": r.return_id, "customer_id": r.customer_id, "score": round(r.score, 4),
                "decision": r.decision, "reasons": r.reasons, "status": r.status, "amount": r.amount,
                "analyst_verdict": r.analyst_verdict, "analyst_comment": r.analyst_comment,
                "created_at": r.created_at.isoformat() if r.created_at else None}


def _create_schema(engine, metadata, attempts: int = 5) -> None:
    """create_all, устойчивый к гонке: несколько сервисов стартуют одновременно и создают одни таблицы."""
    import time as _t

    from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError

    for i in range(attempts):
        try:
            metadata.create_all(engine)
            return
        except (IntegrityError, ProgrammingError, OperationalError):
            if i == attempts - 1:
                raise
            _t.sleep(0.5 * (i + 1))
