"""Хранилище итогов эпизодов (Repository). PostgreSQL в docker-compose, SQLite локально."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class EpisodeRow(Base):
    __tablename__ = "episodes"
    episode_id: Mapped[str] = mapped_column(String(32), primary_key=True)  # идемпотентность по episode_id
    policy: Mapped[str] = mapped_column(String(32), index=True)
    scenario: Mapped[str] = mapped_column(String(32), index=True)
    outcome: Mapped[str] = mapped_column(String(32))
    success: Mapped[bool] = mapped_column(Boolean)
    steps: Mapped[int] = mapped_column(Integer)
    ret: Mapped[float] = mapped_column(Float)
    hits: Mapped[int] = mapped_column(Integer)
    shots: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class EpisodeRepository:
    def __init__(self, url: str | None = None) -> None:
        self.engine = create_engine(url or os.getenv("DATABASE_URL", "sqlite:///drone.db"))
        _create_schema(self.engine, Base.metadata)

    def upsert(self, policy: str, episode_id: str, s: dict) -> None:
        with Session(self.engine) as ses:
            ses.merge(EpisodeRow(episode_id=episode_id, policy=policy, scenario=s["scenario"], outcome=s["outcome"],
                                 success=bool(s["success"]), steps=s["steps"], ret=s["return"], hits=s["hits"],
                                 shots=s["shots"]))
            ses.commit()

    def stats(self) -> list[dict]:
        with Session(self.engine) as ses:
            q = select(EpisodeRow.policy, EpisodeRow.scenario, func.count(), func.avg(EpisodeRow.ret),
                       func.avg(func.cast(EpisodeRow.success, Integer)), func.avg(EpisodeRow.steps)
                       ).group_by(EpisodeRow.policy, EpisodeRow.scenario)
            return [{"policy": p, "scenario": sc, "episodes": n, "mean_return": round(r, 3),
                     "success_rate": round(sr, 3), "mean_steps": round(st, 1)} for p, sc, n, r, sr, st in ses.execute(q)]

    def count(self) -> int:
        with Session(self.engine) as ses:
            return ses.scalar(select(func.count()).select_from(EpisodeRow))


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
