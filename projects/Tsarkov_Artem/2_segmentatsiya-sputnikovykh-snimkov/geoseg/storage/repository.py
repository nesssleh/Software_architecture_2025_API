"""Геопространственное хранилище результатов — паттерн Repository.

* PostGISLandCoverRepository — PostgreSQL + PostGIS (geometry, GIST-индекс, ST_Intersects);
* SQLiteLandCoverRepository  — локальная замена: GeoJSON + bbox-колонки с B-tree индексом.
Обе реализации дают одинаковый интерфейс сервисам и API.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone


class LandCoverRepository(ABC):
    @abstractmethod
    def upsert_raster(self, raster_id: str, meta: dict) -> None: ...

    @abstractmethod
    def set_status(self, raster_id: str, status: str, **extra) -> None: ...

    @abstractmethod
    def get_raster(self, raster_id: str) -> dict | None: ...

    @abstractmethod
    def save_polygons(self, raster_id: str, polygons: list[dict]) -> int: ...

    @abstractmethod
    def query(self, bbox: tuple[float, float, float, float] | None = None, cls: str | None = None,
              raster_id: str | None = None, limit: int = 1000) -> list[dict]: ...

    @abstractmethod
    def class_stats(self, raster_id: str) -> list[dict]: ...


class SQLiteLandCoverRepository(LandCoverRepository):
    def __init__(self, path: str = ":memory:") -> None:
        self.c = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.c.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS rasters(raster_id TEXT PRIMARY KEY, meta TEXT, status TEXT, extra TEXT,
                                               created_at TEXT, updated_at TEXT);
            CREATE TABLE IF NOT EXISTS landcover(id INTEGER PRIMARY KEY AUTOINCREMENT, raster_id TEXT, class TEXT,
                area_m2 REAL, minx REAL, miny REAL, maxx REAL, maxy REAL, geojson TEXT);
            CREATE INDEX IF NOT EXISTS ix_lc_raster_class ON landcover(raster_id, class);
            CREATE INDEX IF NOT EXISTS ix_lc_bbox ON landcover(minx, maxx, miny, maxy);
        """)

    def upsert_raster(self, raster_id: str, meta: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.lock:
            self.c.execute("INSERT OR REPLACE INTO rasters VALUES(?,?,?,?,?,?)",
                           (raster_id, json.dumps(meta), "ingested", "{}", now, now))
            self.c.commit()

    def set_status(self, raster_id: str, status: str, **extra) -> None:
        with self.lock:
            row = self.c.execute("SELECT extra FROM rasters WHERE raster_id=?", (raster_id,)).fetchone()
            ex = {**json.loads(row[0] if row else "{}"), **extra}
            self.c.execute("UPDATE rasters SET status=?, extra=?, updated_at=? WHERE raster_id=?",
                           (status, json.dumps(ex), datetime.now(timezone.utc).isoformat(), raster_id))
            self.c.commit()

    def get_raster(self, raster_id: str) -> dict | None:
        with self.lock:
            r = self.c.execute("SELECT raster_id, meta, status, extra, created_at, updated_at FROM rasters WHERE raster_id=?",
                               (raster_id,)).fetchone()
        if not r:
            return None
        return {"raster_id": r[0], "meta": json.loads(r[1]), "status": r[2], **json.loads(r[3]),
                "created_at": r[4], "updated_at": r[5]}

    def save_polygons(self, raster_id: str, polygons: list[dict]) -> int:
        with self.lock:
            self.c.execute("DELETE FROM landcover WHERE raster_id=?", (raster_id,))  # идемпотентная перезапись
            self.c.executemany("INSERT INTO landcover(raster_id,class,area_m2,minx,miny,maxx,maxy,geojson) VALUES(?,?,?,?,?,?,?,?)",
                               [(raster_id, p["class"], p["area_m2"], *p["bbox"], json.dumps(p["geometry"])) for p in polygons])
            self.c.commit()
        return len(polygons)

    def query(self, bbox=None, cls=None, raster_id=None, limit=1000) -> list[dict]:
        q, args = "SELECT id, raster_id, class, area_m2, geojson FROM landcover WHERE 1=1", []
        if bbox:
            q += " AND maxx >= ? AND minx <= ? AND maxy >= ? AND miny <= ?"
            args += [bbox[0], bbox[2], bbox[1], bbox[3]]
        if cls:
            q += " AND class = ?"
            args.append(cls)
        if raster_id:
            q += " AND raster_id = ?"
            args.append(raster_id)
        q += " ORDER BY area_m2 DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            rows = self.c.execute(q, args).fetchall()
        return [{"id": i, "raster_id": r, "class": c, "area_m2": a, "geometry": json.loads(g)} for i, r, c, a, g in rows]

    def class_stats(self, raster_id: str) -> list[dict]:
        with self.lock:
            rows = self.c.execute("SELECT class, count(*), sum(area_m2) FROM landcover WHERE raster_id=? GROUP BY class "
                                  "ORDER BY 3 DESC", (raster_id,)).fetchall()
        return [{"class": c, "polygons": n, "area_ha": round(a / 10_000, 3)} for c, n, a in rows]


class PostGISLandCoverRepository(LandCoverRepository):
    """DDL — sql/postgis.sql. Геометрии хранятся в исходной UTM-проекции (SRID из растра).

    Пул соединений: API обслуживает запросы в пуле потоков, одно соединение на всех привело бы
    к сериализации запросов и смешиванию транзакций.
    """

    def __init__(self, dsn: str) -> None:
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(dsn, min_size=1, max_size=int(os.getenv("DB_POOL_SIZE", 8)),
                                   kwargs={"autocommit": True}, open=True, timeout=30)

    def upsert_raster(self, raster_id: str, meta: dict) -> None:
        with self.pool.connection() as c:
            c.execute("""INSERT INTO rasters(raster_id, meta, status) VALUES (%s, %s, 'ingested')
                         ON CONFLICT (raster_id) DO UPDATE SET meta = EXCLUDED.meta, status = 'ingested', updated_at = now()""",
                      (raster_id, json.dumps(meta)))

    def set_status(self, raster_id: str, status: str, **extra) -> None:
        with self.pool.connection() as c:
            c.execute("UPDATE rasters SET status=%s, extra = extra || %s::jsonb, updated_at=now() WHERE raster_id=%s",
                      (status, json.dumps(extra), raster_id))

    def get_raster(self, raster_id: str) -> dict | None:
        with self.pool.connection() as c:
            r = c.execute("SELECT raster_id, meta, status, extra, created_at, updated_at FROM rasters WHERE raster_id=%s",
                          (raster_id,)).fetchone()
        return None if not r else {"raster_id": r[0], "meta": r[1], "status": r[2], **r[3],
                                   "created_at": r[4].isoformat(), "updated_at": r[5].isoformat()}

    def save_polygons(self, raster_id: str, polygons: list[dict]) -> int:
        srid = int(self.get_raster(raster_id)["meta"]["crs"].split(":")[1])
        with self.pool.connection() as c, c.transaction():
            c.execute("DELETE FROM landcover WHERE raster_id=%s", (raster_id,))
            with c.cursor() as cur:
                cur.executemany("""INSERT INTO landcover(raster_id, class, area_m2, geom)
                                   VALUES (%s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), %s))""",
                                [(raster_id, p["class"], p["area_m2"], json.dumps(p["geometry"]), srid) for p in polygons])
        return len(polygons)

    def query(self, bbox=None, cls=None, raster_id=None, limit=1000) -> list[dict]:
        q, args = "SELECT id, raster_id, class, area_m2, ST_AsGeoJSON(geom) FROM landcover WHERE true", []
        if bbox:
            q += " AND geom && ST_MakeEnvelope(%s, %s, %s, %s, ST_SRID(geom))"
            args += list(bbox)
        if cls:
            q += " AND class = %s"
            args.append(cls)
        if raster_id:
            q += " AND raster_id = %s"
            args.append(raster_id)
        q += " ORDER BY area_m2 DESC LIMIT %s"
        args.append(limit)
        with self.pool.connection() as c:
            rows = c.execute(q, args).fetchall()
        return [{"id": i, "raster_id": r, "class": c, "area_m2": a, "geometry": json.loads(g)} for i, r, c, a, g in rows]

    def class_stats(self, raster_id: str) -> list[dict]:
        with self.pool.connection() as c:
            rows = c.execute("SELECT class, count(*), sum(ST_Area(geom)) FROM landcover WHERE raster_id=%s "
                             "GROUP BY class ORDER BY 3 DESC", (raster_id,)).fetchall()
        return [{"class": c, "polygons": n, "area_ha": round(a / 10_000, 3)} for c, n, a in rows]


def make_repository() -> LandCoverRepository:
    dsn = os.getenv("POSTGIS_DSN")
    return PostGISLandCoverRepository(dsn) if dsn else SQLiteLandCoverRepository(os.getenv("GEO_SQLITE", ":memory:"))
