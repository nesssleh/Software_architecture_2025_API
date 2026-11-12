"""Распределённый режим: стадии конвейера как отдельные Kafka-consumer'ы.

Топики (key = raster_id — все сообщения растра в одной партиции):
  raster.ingested   ingest-api  → tiler            {raster_id, key, meta}
  tiles.ready       tiler       → inference-worker {raster_id, window, total}   (key = raster_id:row:col)
  tiles.segmented   inference   → geostore         {raster_id, window, total, proba_key}
  raster.completed  geostore    → (подписчики)     {raster_id, polygons, area_ha}
  *.dlq             сообщения, не обработанные за 3 попытки
"""
from __future__ import annotations

import io
import json
import logging
from collections import defaultdict

import numpy as np

from geoseg.models.strategies import N_CLASSES, SegmentationStrategy
from geoseg.pipeline.kafka_io import EventProducer, Message
from geoseg.pipeline.stages import (MosaicStage, PersistStage, RasterContext, VectorizeStage, read_raster)
from geoseg.pipeline.tiling import TileWindow, make_windows
from geoseg.storage.objects import ObjectStorage
from geoseg.storage.repository import LandCoverRepository

log = logging.getLogger(__name__)
TILE, OVERLAP = 256, 32


class TilerWorker:
    group = "tiler"
    topic_in = "raster.ingested"

    def __init__(self, storage: ObjectStorage, repo: LandCoverRepository, producer: EventProducer) -> None:
        self.storage, self.repo, self.producer = storage, repo, producer

    def __call__(self, msg: Message) -> None:
        ev = msg.json()
        meta = ev["meta"]
        wins = make_windows(meta["height"], meta["width"], TILE, OVERLAP)
        self.repo.set_status(ev["raster_id"], "tiling", tiles_total=len(wins))
        for w in wins:
            # ключ raster_id:окно — тайлы одного растра расходятся по партициям и обрабатываются всеми
            # inference-воркерами параллельно; в tiles.segmented ключ снова raster_id (сборка в одном geostore)
            self.producer.send("tiles.ready", key=f"{ev['raster_id']}:{w.row}:{w.col}", value={
                "raster_id": ev["raster_id"], "raw_key": ev["key"], "total": len(wins),
                "window": [w.row, w.col, w.height, w.width]})


class InferenceWorker:
    group = "inference-worker"
    topic_in = "tiles.ready"

    def __init__(self, storage: ObjectStorage, model: SegmentationStrategy, producer: EventProducer) -> None:
        self.storage, self.model, self.producer = storage, model, producer
        self._cache: dict[str, np.ndarray] = {}

    def _image(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache = {key: read_raster(self.storage.get(key))[0]}  # держим в памяти один растр
        return self._cache[key]

    def __call__(self, msg: Message) -> None:
        ev = msg.json()
        r, c, h, w = ev["window"]
        proba = self.model.predict_proba(self._image(ev["raw_key"])[:, r:r + h, c:c + w])
        buf = io.BytesIO()
        np.savez_compressed(buf, p=(proba * 255).astype(np.uint8))  # квантование вероятностей: x4 меньше
        key = f"tiles/{ev['raster_id']}/r{r}_c{c}.npz"
        self.storage.put(key, buf.getvalue())
        self.producer.send("tiles.segmented", key=ev["raster_id"], value={**ev, "proba_key": key})


class GeoStoreWorker:
    """Собирает тайлы растра; когда пришли все — mosaic → vectorize → PostGIS → raster.completed."""

    group = "geostore"
    topic_in = "tiles.segmented"

    def __init__(self, storage: ObjectStorage, repo: LandCoverRepository, producer: EventProducer) -> None:
        self.storage, self.repo, self.producer = storage, repo, producer
        self.received: dict[str, dict[str, list]] = defaultdict(dict)

    def __call__(self, msg: Message) -> None:
        ev = msg.json()
        rid = ev["raster_id"]
        self.received[rid][ev["proba_key"]] = ev["window"]  # dict => дубликаты не считаются дважды
        if len(self.received[rid]) < ev["total"]:
            return
        img, meta, transform = read_raster(self.storage.get(ev["raw_key"]))
        ctx = RasterContext(rid, image=img, meta=meta, transform=transform)
        for key, (r, c, h, w) in self.received.pop(rid).items():
            ctx.windows.append(TileWindow(r, c, h, w))
            ctx.probas.append(np.load(io.BytesIO(self.storage.get(key)))["p"].astype(np.float32) / 255)
        for st in (MosaicStage(self.storage, TILE, OVERLAP), VectorizeStage(), PersistStage(self.repo)):
            ctx = st.run(ctx)
        info = self.repo.get_raster(rid)
        self.producer.send("raster.completed", key=rid, value=json.dumps(
            {"raster_id": rid, "polygons": info.get("polygons"), "area_ha": info.get("area_ha")}))
