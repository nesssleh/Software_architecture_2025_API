"""Хранилища — паттерн Repository.

* TicketRepository — PostgreSQL (SQLAlchemy) / SQLite: тикеты, журнал команд (аудит).
* SearchRepository — Elasticsearch (полнотекстовый поиск + агрегации) / in-memory TF-IDF.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class TicketRow(Base):
    __tablename__ = "tickets"
    ticket_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    channel: Mapped[str] = mapped_column(String(16))
    customer_tier: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(32), index=True)
    category_conf: Mapped[float] = mapped_column(Float, default=0)
    priority: Mapped[str | None] = mapped_column(String(16), index=True)
    priority_conf: Mapped[float] = mapped_column(Float, default=0)
    team: Mapped[str | None] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    incident_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AuditRow(Base):
    __tablename__ = "ticket_audit"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(String(32), index=True)
    at: Mapped[str] = mapped_column(String(40))
    actor: Mapped[str] = mapped_column(String(32))
    command: Mapped[str] = mapped_column(String(32))
    details: Mapped[dict] = mapped_column(JSON)


class TicketRepository:
    def __init__(self, url: str | None = None) -> None:
        self.engine = create_engine(url or os.getenv("DATABASE_URL", "sqlite:///tickets.db"))
        _create_schema(self.engine, Base.metadata)

    def upsert(self, t: dict) -> None:
        fields = {c.name for c in TicketRow.__table__.columns}
        row = {k: v for k, v in t.items() if k in fields}
        if isinstance(row.get("created_at"), str):
            row["created_at"] = datetime.fromisoformat(row["created_at"])
        with Session(self.engine) as s:
            s.merge(TicketRow(**row))
            s.commit()

    def get(self, ticket_id: str) -> dict | None:
        with Session(self.engine) as s:
            r = s.get(TicketRow, ticket_id)
            if not r:
                return None
            d = {c.name: getattr(r, c.name) for c in TicketRow.__table__.columns}
            d["created_at"] = r.created_at.isoformat()
            return d

    def add_audit(self, entries: list[dict]) -> None:
        with Session(self.engine) as s:
            s.add_all([AuditRow(ticket_id=e["ticket_id"], at=e["at"], actor=e["actor"], command=e["command"],
                                details={k: v for k, v in e.items() if k not in ("ticket_id", "at", "actor", "command")})
                       for e in entries])
            s.commit()

    def audit(self, ticket_id: str) -> list[dict]:
        with Session(self.engine) as s:
            return [{"at": a.at, "actor": a.actor, "command": a.command, **a.details}
                    for a in s.scalars(select(AuditRow).where(AuditRow.ticket_id == ticket_id).order_by(AuditRow.id))]

    def stats(self) -> dict:
        with Session(self.engine) as s:
            by = lambda col: {k or "none": n for k, n in s.execute(select(col, func.count()).group_by(col))}  # noqa: E731
            return {"by_team": by(TicketRow.team), "by_priority": by(TicketRow.priority),
                    "by_category": by(TicketRow.category), "by_status": by(TicketRow.status)}


class SearchRepository(ABC):
    @abstractmethod
    def index(self, ticket: dict) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 10, team: str | None = None) -> list[dict]: ...


class InMemorySearch(SearchRepository):
    """Локальная замена Elasticsearch: TF-IDF по символьным n-граммам.

    Потокобезопасна (API обслуживает запросы в пуле потоков); словарь переобучается,
    только когда корпус вырос вдвое, новые документы векторизуются инкрементально.
    """

    def __init__(self) -> None:
        import threading

        self.docs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._vec = None
        self._ids: list[str] = []
        self._X = None
        self._fitted_on = 0

    def index(self, ticket: dict) -> None:
        with self._lock:
            self.docs[ticket["ticket_id"]] = ticket

    def _refresh(self) -> None:
        from scipy.sparse import vstack
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._vec is None or len(self.docs) >= 2 * self._fitted_on:
            self._ids = list(self.docs)
            self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(
                [self.docs[i]["text"] for i in self._ids])
            self._X = self._vec.transform([self.docs[i]["text"] for i in self._ids])
            self._fitted_on = len(self._ids)
        elif len(self.docs) > len(self._ids):
            new = [i for i in self.docs if i not in set(self._ids)]
            self._X = vstack([self._X, self._vec.transform([self.docs[i]["text"] for i in new])])
            self._ids += new

    def search(self, query, k=10, team=None):
        with self._lock:
            if not self.docs:
                return []
            self._refresh()
            sims = (self._X @ self._vec.transform([query]).T).toarray().ravel()
            ids, docs = list(self._ids), self.docs
        out = []
        for i in sims.argsort()[::-1]:
            d = docs[ids[i]]
            if sims[i] <= 0 or len(out) >= k:
                break
            if team is None or d.get("team") == team:
                out.append({"ticket_id": d["ticket_id"], "score": round(float(sims[i]), 4), "text": d["text"],
                            "category": d.get("category"), "team": d.get("team")})
        return out


class ElasticSearchRepository(SearchRepository):
    INDEX = "tickets"
    MAPPING = {"settings": {"analysis": {"analyzer": {"ru": {"type": "russian"}}}},
               "mappings": {"properties": {"text": {"type": "text", "analyzer": "russian"},
                                           "category": {"type": "keyword"}, "team": {"type": "keyword"},
                                           "priority": {"type": "keyword"}, "created_at": {"type": "date"},
                                           "tags": {"type": "keyword"}}}}

    def __init__(self, url: str) -> None:
        from elasticsearch import Elasticsearch  # type: ignore

        self.es = Elasticsearch(url)
        if not self.es.indices.exists(index=self.INDEX):
            self.es.indices.create(index=self.INDEX, **self.MAPPING)

    def index(self, ticket):
        self.es.index(index=self.INDEX, id=ticket["ticket_id"], document=ticket)

    def search(self, query, k=10, team=None):
        q = {"bool": {"must": [{"match": {"text": query}}]}}
        if team:
            q["bool"]["filter"] = [{"term": {"team": team}}]
        hits = self.es.search(index=self.INDEX, query=q, size=k)["hits"]["hits"]
        return [{"ticket_id": h["_id"], "score": h["_score"], **{f: h["_source"].get(f) for f in ("text", "category", "team")}}
                for h in hits]


def make_search() -> SearchRepository:
    url = os.getenv("ELASTICSEARCH_URL")
    return ElasticSearchRepository(url) if url else InMemorySearch()


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
