"""GeoSeg API.

POST /v1/rasters?mode=sync|async   — загрузка GeoTIFF (4 канала R,G,B,NIR)
GET  /v1/rasters/{id}              — статус и сводка
GET  /v1/rasters/{id}/mask.png     — цветная маска классов
GET  /v1/rasters/{id}/stats        — площади классов (га) по полигонам
GET  /v1/landcover?bbox=&cls=      — GeoJSON полигонов из PostGIS
GET  /v1/classes
"""
from __future__ import annotations

import os
from functools import lru_cache

from geoseg.obs import instrument
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from geoseg.models.strategies import make_strategy
from geoseg.pipeline.kafka_io import make_producer
from geoseg.pipeline.stages import IngestStage, LandCoverFacade, RasterContext, raster_id_for
from geoseg.storage.objects import make_storage
from geoseg.storage.repository import make_repository
from geoseg.synth import CLASSES, PALETTE

app = FastAPI(title="GeoSeg — Satellite Land-Cover Segmentation API", version="1.0.0")
instrument(app)
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", 512))


@lru_cache(maxsize=1)
def producer():
    """Один Kafka-producer на процесс (создание producer'а на каждый запрос — это сотни миллисекунд)."""
    return make_producer()


@lru_cache(maxsize=1)
def facade() -> LandCoverFacade:
    return LandCoverFacade(make_storage(), make_repository(), make_strategy(os.getenv("SEG_MODEL", "unet"),
                                                                             os.getenv("MODEL_PATH")))


@app.get("/health")
def health():
    return {"status": "ok", "model": facade().model.version}


@app.get("/v1/classes")
def classes():
    return [{"id": i, "name": c, "color": "#%02x%02x%02x" % tuple(PALETTE[i])} for i, c in enumerate(CLASSES)]


@app.post("/v1/rasters", status_code=201)
async def upload(file: UploadFile = File(...), mode: str = Query("sync", pattern="^(sync|async)$"),
                 force: bool = Query(False, description="пересчитать, даже если растр уже обработан")):
    data = await file.read()
    if len(data) > MAX_MB * 2 ** 20:
        raise HTTPException(413, "file too large")
    f = facade()
    try:
        if mode == "sync":
            return f.process(data, force=force)
        rid = raster_id_for(data)
        ctx = IngestStage(f.storage, f.repo).run(RasterContext(rid, raw=data))
        producer().send("raster.ingested", key=rid, value={"raster_id": rid, "key": f"raw/{rid}.tif", "meta": ctx.meta})
        return {"raster_id": rid, "status": "ingested"}
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # rasterio не смог прочитать файл
        raise HTTPException(422, f"cannot read raster: {e}") from e


@app.get("/v1/rasters/{raster_id}")
def status(raster_id: str):
    r = facade().status(raster_id)
    if not r:
        raise HTTPException(404, "raster not found")
    return r


@app.get("/v1/rasters/{raster_id}/mask.png")
def mask(raster_id: str):
    try:
        return Response(facade().mask_png(raster_id), media_type="image/png")
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(404, "mask not ready") from e


@app.get("/v1/rasters/{raster_id}/stats")
def stats(raster_id: str):
    if not facade().status(raster_id):
        raise HTTPException(404, "raster not found")
    return {"raster_id": raster_id, "classes": facade().stats(raster_id)}


@app.get("/v1/landcover")
def landcover(bbox: str | None = Query(None, description="minx,miny,maxx,maxy в CRS растра"),
              cls: str | None = None, raster_id: str | None = None, limit: int = Query(200, le=5000)):
    bb = None
    if bbox:
        try:
            bb = tuple(float(v) for v in bbox.split(","))
            assert len(bb) == 4
        except (ValueError, AssertionError) as e:
            raise HTTPException(422, "bbox must be 4 comma-separated numbers") from e
    if cls and cls not in CLASSES:
        raise HTTPException(422, f"cls must be one of {CLASSES}")
    return facade().geojson(bb, cls, raster_id, limit)
