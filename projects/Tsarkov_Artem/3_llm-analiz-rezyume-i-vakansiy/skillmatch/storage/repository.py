"""Хранилище документов и векторный поиск — паттерн Repository.

* PgVectorRepository  — PostgreSQL + pgvector (HNSW-индекс по cosine), DDL в sql/schema.sql;
* InMemoryRepository  — numpy-реализация того же интерфейса (тесты, локальный запуск).
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Document:
    id: str
    type: str  # resume | vacancy
    title: str | None
    text: str
    skills: list[dict] = field(default_factory=list)
    must_have: list[str] = field(default_factory=list)
    nice_to_have: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    status: str = "processed"

    @property
    def skill_names(self) -> set[str]:
        return {s["name"] for s in self.skills}

    def public(self) -> dict:
        return {"id": self.id, "type": self.type, "title": self.title, "status": self.status, "skills": self.skills,
                "must_have": self.must_have, "nice_to_have": self.nice_to_have, "text": self.text[:2000]}


class DocumentRepository(ABC):
    @abstractmethod
    def save(self, doc: Document) -> None: ...

    @abstractmethod
    def get(self, doc_id: str) -> Document | None: ...

    def get_many(self, ids: list[str]) -> dict[str, Document]:
        out = {}
        for i in ids:
            d = self.get(i)
            if d:
                out[i] = d
        return out

    @abstractmethod
    def nearest(self, vec: np.ndarray, doc_type: str, k: int = 50) -> list[tuple[str, float]]: ...

    @abstractmethod
    def by_skills(self, skills: list[str], doc_type: str, limit: int = 200) -> list[str]:
        """Кандидаты по инвертированному индексу навыков (сколько навыков из списка есть у документа)."""

    @abstractmethod
    def count(self) -> dict: ...


class InMemoryRepository(DocumentRepository):
    def __init__(self) -> None:
        self.docs: dict[str, Document] = {}

    def save(self, doc: Document) -> None:
        self.docs[doc.id] = doc

    def get(self, doc_id: str) -> Document | None:
        return self.docs.get(doc_id)

    def nearest(self, vec, doc_type, k=50):
        cands = [d for d in self.docs.values() if d.type == doc_type and d.embedding is not None]
        if not cands:
            return []
        sims = np.stack([d.embedding for d in cands]) @ vec
        order = np.argsort(-sims)[:k]
        return [(cands[i].id, float(sims[i])) for i in order]

    def by_skills(self, skills, doc_type, limit=200):
        want = set(skills)
        scored = [(len(want & d.skill_names), d.id) for d in self.docs.values() if d.type == doc_type]
        return [i for n, i in sorted(scored, key=lambda x: -x[0]) if n > 0][:limit]

    def count(self) -> dict:
        out: dict[str, int] = {}
        for d in self.docs.values():
            out[d.type] = out.get(d.type, 0) + 1
        return out


class PgVectorRepository(DocumentRepository):
    """PostgreSQL + pgvector через пул соединений (API и воркер обслуживают запросы конкурентно)."""

    def __init__(self, dsn: str) -> None:
        from pgvector.psycopg import register_vector  # type: ignore
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(dsn, min_size=1, max_size=int(os.getenv("DB_POOL_SIZE", 8)), kwargs={"autocommit": True},
                                   configure=register_vector, open=True, timeout=30)

    def save(self, d: Document) -> None:
        with self.pool.connection() as c, c.transaction():
            c.execute(
                """INSERT INTO documents(id, type, title, text, skills, must_have, nice_to_have, embedding, status)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO UPDATE SET title=EXCLUDED.title, text=EXCLUDED.text, skills=EXCLUDED.skills,
                     must_have=EXCLUDED.must_have, nice_to_have=EXCLUDED.nice_to_have, embedding=EXCLUDED.embedding,
                     status=EXCLUDED.status, updated_at=now()""",
                (d.id, d.type, d.title, d.text, json.dumps(d.skills, ensure_ascii=False), d.must_have, d.nice_to_have,
                 d.embedding, d.status))
            c.execute("DELETE FROM document_skills WHERE doc_id=%s", (d.id,))
            with c.cursor() as cur:
                cur.executemany("INSERT INTO document_skills(doc_id, skill, method, confidence) VALUES (%s,%s,%s,%s)",
                                [(d.id, s["name"], s["method"], s["confidence"]) for s in d.skills])

    def get(self, doc_id):
        with self.pool.connection() as c:
            r = c.execute("SELECT id,type,title,text,skills,must_have,nice_to_have,embedding,status FROM documents "
                          "WHERE id=%s", (doc_id,)).fetchone()
        return None if not r else Document(r[0], r[1], r[2], r[3], r[4], list(r[5] or []), list(r[6] or []),
                                           _to_np(r[7]), r[8])

    def get_many(self, ids):
        """Один запрос вместо N (кандидатогенерация отдаёт до сотен id)."""
        if not ids:
            return {}
        with self.pool.connection() as c:
            rows = c.execute("SELECT id,type,title,text,skills,must_have,nice_to_have,embedding,status FROM documents "
                             "WHERE id = ANY(%s)", (list(ids),)).fetchall()
        return {r[0]: Document(r[0], r[1], r[2], r[3], r[4], list(r[5] or []), list(r[6] or []), _to_np(r[7]), r[8])
                for r in rows}

    def nearest(self, vec, doc_type, k=50):
        with self.pool.connection() as c:
            rows = c.execute("SELECT id, 1 - (embedding <=> %s) AS sim FROM documents WHERE type=%s "
                             "ORDER BY embedding <=> %s LIMIT %s", (vec, doc_type, vec, k)).fetchall()
        return [(i, float(s)) for i, s in rows]

    def by_skills(self, skills, doc_type, limit=200):
        with self.pool.connection() as c:
            rows = c.execute("""SELECT s.doc_id FROM document_skills s JOIN documents d ON d.id = s.doc_id
                                WHERE d.type = %s AND s.skill = ANY(%s)
                                GROUP BY s.doc_id ORDER BY count(*) DESC LIMIT %s""", (doc_type, list(skills), limit)).fetchall()
        return [r[0] for r in rows]

    def count(self):
        with self.pool.connection() as c:
            return dict(c.execute("SELECT type, count(*) FROM documents GROUP BY type").fetchall())


def _to_np(v) -> np.ndarray | None:
    if v is None:
        return None
    if hasattr(v, "to_numpy"):  # pgvector.Vector
        v = v.to_numpy()
    return np.asarray(v, dtype=np.float32)


def make_repository() -> DocumentRepository:
    dsn = os.getenv("DATABASE_URL")
    return PgVectorRepository(dsn) if dsn else InMemoryRepository()
