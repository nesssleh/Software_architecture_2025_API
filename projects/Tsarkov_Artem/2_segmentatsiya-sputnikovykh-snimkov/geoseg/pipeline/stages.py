"""Конвейер обработки растра — паттерн Pipeline, и единая точка входа — паттерн Facade.

Ingest → Tiling → Inference → Mosaic → Vectorize → Persist

Каждая стадия — объект с методом run(ctx) -> ctx; конвейер собирается из
стадий и логирует время каждой. В распределённом режиме те же стадии
исполняются разными сервисами, связанными топиками Kafka (см. services/).
"""
from __future__ import annotations

import hashlib
import io
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import rasterio
from rasterio import features as rfeatures
from rasterio.io import MemoryFile

from ..models.strategies import N_CLASSES, SegmentationStrategy
from ..storage.objects import ObjectStorage
from ..storage.repository import LandCoverRepository
from ..synth import CLASSES, PALETTE
from .tiling import Mosaic, TileWindow, make_windows

log = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram

    RASTERS_DONE = Counter("geoseg_rasters_completed_total", "Обработанные растры")
    AREA_HA = Counter("geoseg_area_hectares_total", "Площадь классов, га", ["cls"])
    STAGE_SEC = Histogram("geoseg_stage_seconds", "Длительность стадий конвейера", ["stage"],
                          buckets=(.01, .05, .1, .25, .5, 1, 2.5, 5, 10, 30))
except ImportError:  # pragma: no cover
    RASTERS_DONE = AREA_HA = STAGE_SEC = None


@dataclass
class RasterContext:
    raster_id: str
    raw: bytes | None = None
    image: np.ndarray | None = None      # [C,H,W] float32
    meta: dict = field(default_factory=dict)
    transform: Any = None
    windows: list[TileWindow] = field(default_factory=list)
    probas: list[np.ndarray] = field(default_factory=list)
    mask: np.ndarray | None = None
    polygons: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


class Stage(ABC):
    name = "stage"

    @abstractmethod
    def run(self, ctx: RasterContext) -> RasterContext: ...


def raster_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]  # контентный id => повторная загрузка идемпотентна


def read_raster(data: bytes) -> tuple[np.ndarray, dict, Any]:
    with MemoryFile(data) as mf, mf.open() as src:
        if src.count < 4:
            raise ValueError(f"expected 4 bands (R,G,B,NIR), got {src.count}")
        img = src.read([1, 2, 3, 4]).astype(np.float32)
        scale = 10000.0 if src.dtypes[0] in ("uint16", "int16") else 1.0
        meta = {"crs": src.crs.to_string() if src.crs else "EPSG:32637", "width": src.width, "height": src.height,
                "res": list(src.res), "bounds": list(src.bounds), "dtype": src.dtypes[0]}
        return img / scale, meta, src.transform


class IngestStage(Stage):
    name = "ingest"

    def __init__(self, storage: ObjectStorage, repo: LandCoverRepository) -> None:
        self.storage, self.repo = storage, repo

    def run(self, ctx: RasterContext) -> RasterContext:
        ctx.image, ctx.meta, ctx.transform = read_raster(ctx.raw)
        self.storage.put(f"raw/{ctx.raster_id}.tif", ctx.raw, "image/tiff")
        self.repo.upsert_raster(ctx.raster_id, ctx.meta)
        return ctx


class TilingStage(Stage):
    name = "tiling"

    def __init__(self, tile: int = 256, overlap: int = 32) -> None:
        self.tile, self.overlap = tile, overlap

    def run(self, ctx: RasterContext) -> RasterContext:
        ctx.windows = make_windows(ctx.image.shape[1], ctx.image.shape[2], self.tile, self.overlap)
        return ctx


class InferenceStage(Stage):
    name = "inference"

    def __init__(self, model: SegmentationStrategy, batch: int = 8) -> None:
        self.model, self.batch = model, batch

    def run(self, ctx: RasterContext) -> RasterContext:
        tiles = [ctx.image[:, w.row:w.row + w.height, w.col:w.col + w.width] for w in ctx.windows]
        if hasattr(self.model, "predict_proba_batch"):
            for i in range(0, len(tiles), self.batch):
                ctx.probas.extend(self.model.predict_proba_batch(np.stack(tiles[i:i + self.batch])))
        else:
            ctx.probas = [self.model.predict_proba(t) for t in tiles]
        return ctx


class MosaicStage(Stage):
    name = "mosaic"

    def __init__(self, storage: ObjectStorage, tile: int = 256, overlap: int = 32) -> None:
        self.storage, self.tile, self.overlap = storage, tile, overlap

    def run(self, ctx: RasterContext) -> RasterContext:
        h, w = ctx.image.shape[1:]
        mos = Mosaic(h, w, N_CLASSES, self.tile, self.overlap)
        for win, p in zip(ctx.windows, ctx.probas):
            mos.add(win, p)
        ctx.mask = mos.mask()
        self.storage.put(f"masks/{ctx.raster_id}.tif", mask_to_geotiff(ctx.mask, ctx.meta, ctx.transform), "image/tiff")
        self.storage.put(f"masks/{ctx.raster_id}.png", mask_to_png(ctx.mask), "image/png")
        return ctx


class VectorizeStage(Stage):
    """Маска → полигоны (rasterio.features.shapes) с упрощением и фильтром мелких объектов."""

    name = "vectorize"

    def __init__(self, min_area_m2: float = 50.0, simplify_m: float = 1.0) -> None:
        self.min_area, self.simplify = min_area_m2, simplify_m

    def run(self, ctx: RasterContext) -> RasterContext:
        from shapely.geometry import mapping, shape

        polys = []
        for geom, val in rfeatures.shapes(ctx.mask, transform=ctx.transform, connectivity=8):
            cls = CLASSES[int(val)]
            if cls == "other":
                continue
            g = shape(geom)
            if g.area < self.min_area:
                continue
            g = g.simplify(self.simplify, preserve_topology=True)
            polys.append({"class": cls, "area_m2": round(g.area, 2), "bbox": list(g.bounds), "geometry": mapping(g)})
        ctx.polygons = polys
        return ctx


class PersistStage(Stage):
    name = "persist"

    def __init__(self, repo: LandCoverRepository) -> None:
        self.repo = repo

    def run(self, ctx: RasterContext) -> RasterContext:
        self.repo.save_polygons(ctx.raster_id, ctx.polygons)
        if RASTERS_DONE is not None:
            RASTERS_DONE.inc()
        counts = np.bincount(ctx.mask.ravel(), minlength=N_CLASSES)
        px_area = abs(ctx.meta["res"][0] * ctx.meta["res"][1])
        if AREA_HA is not None:
            for c, n in zip(CLASSES, counts):
                AREA_HA.labels(c).inc(float(n * px_area / 1e4))
        self.repo.set_status(ctx.raster_id, "completed", polygons=len(ctx.polygons),
                             pixel_share={c: round(float(n / ctx.mask.size), 4) for c, n in zip(CLASSES, counts)},
                             area_ha={c: round(float(n * px_area / 1e4), 3) for c, n in zip(CLASSES, counts)},
                             timings=ctx.timings)
        return ctx


class Pipeline:
    def __init__(self, stages: list[Stage]) -> None:
        self.stages = stages

    def run(self, ctx: RasterContext) -> RasterContext:
        for st in self.stages:
            t = time.perf_counter()
            ctx = st.run(ctx)
            ctx.timings[st.name] = round((time.perf_counter() - t) * 1000, 1)
            if STAGE_SEC is not None:
                STAGE_SEC.labels(st.name).observe(ctx.timings[st.name] / 1000)
            log.info("raster %s: stage %s done in %.1f ms", ctx.raster_id, st.name, ctx.timings[st.name])
        return ctx


def mask_to_png(mask: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(PALETTE[mask]).save(buf, format="PNG")
    return buf.getvalue()


def mask_to_geotiff(mask: np.ndarray, meta: dict, transform) -> bytes:
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=mask.shape[0], width=mask.shape[1], count=1, dtype="uint8",
                     crs=meta["crs"], transform=transform, compress="deflate") as dst:
            dst.write(mask[None])
            dst.write_colormap(1, {i: tuple(c) + (255,) for i, c in enumerate(PALETTE)})
        return mf.read()


class LandCoverFacade:
    """Facade: одна простая операция process(bytes) скрывает хранилище, модель, тайлинг и БД."""

    def __init__(self, storage: ObjectStorage, repo: LandCoverRepository, model: SegmentationStrategy,
                 tile: int = 256, overlap: int = 32) -> None:
        self.storage, self.repo, self.model = storage, repo, model
        self.pipeline = Pipeline([IngestStage(storage, repo), TilingStage(tile, overlap), InferenceStage(model),
                                  MosaicStage(storage, tile, overlap), VectorizeStage(), PersistStage(repo)])

    def process(self, data: bytes, force: bool = False) -> dict:
        rid = raster_id_for(data)
        existing = self.repo.get_raster(rid)
        if existing and existing["status"] == "completed" and not force:  # force — пересчёт (новая модель, нагрузочный тест)
            return existing
        try:
            ctx = self.pipeline.run(RasterContext(rid, raw=data))
            self.repo.set_status(rid, "completed", timings=ctx.timings)  # с учётом времени стадии persist
        except Exception as e:
            if self.repo.get_raster(rid):
                self.repo.set_status(rid, "failed", error=str(e))
            raise
        return self.repo.get_raster(rid)

    def status(self, raster_id: str) -> dict | None:
        return self.repo.get_raster(raster_id)

    def mask_png(self, raster_id: str) -> bytes:
        return self.storage.get(f"masks/{raster_id}.png")

    def stats(self, raster_id: str) -> list[dict]:
        return self.repo.class_stats(raster_id)

    def geojson(self, bbox=None, cls=None, raster_id=None, limit=1000) -> dict:
        feats = [{"type": "Feature", "id": p["id"], "geometry": p["geometry"],
                  "properties": {"class": p["class"], "area_m2": p["area_m2"], "raster_id": p["raster_id"]}}
                 for p in self.repo.query(bbox, cls, raster_id, limit)]
        return {"type": "FeatureCollection", "features": feats}
