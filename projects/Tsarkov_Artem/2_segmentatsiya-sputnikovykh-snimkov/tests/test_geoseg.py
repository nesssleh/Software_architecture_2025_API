import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from geoseg.models.strategies import SpectralIndexStrategy, make_strategy
from geoseg.pipeline.kafka_io import InMemoryBroker, run_consumer_loop
from geoseg.pipeline.stages import IngestStage, LandCoverFacade, RasterContext, raster_id_for
from geoseg.pipeline.tiling import Mosaic, blend_weights, make_windows
from geoseg.storage.objects import LocalStorage
from geoseg.storage.repository import SQLiteLandCoverRepository
from geoseg.synth import CLASSES, generate_scene, write_geotiff
from services.workers import GeoStoreWorker, InferenceWorker, TilerWorker

MODEL = Path("models/unet_landcover.pt")


@pytest.fixture(scope="module")
def tif_bytes(tmp_path_factory):
    img, m = generate_scene(np.random.default_rng(5), 300, 420)
    p = tmp_path_factory.mktemp("r") / "scene.tif"
    write_geotiff(p, img, m)
    return p.read_bytes()


def test_windows_cover_raster():
    wins = make_windows(300, 420, 256, 32)
    cov = np.zeros((300, 420), int)
    for w in wins:
        cov[w.row:w.row + w.height, w.col:w.col + w.width] += 1
    assert cov.min() >= 1


def test_mosaic_identity():
    """Если все тайлы предсказывают один и тот же класс, мозаика возвращает его везде."""
    h, w = 300, 420
    mos = Mosaic(h, w, 6, 256, 32)
    for win in make_windows(h, w, 256, 32):
        p = np.zeros((6, win.height, win.width), np.float32)
        p[2] = 1
        mos.add(win, p)
    assert (mos.mask() == 2).all()
    assert blend_weights(256, 32).min() > 0


def test_spectral_baseline_detects_water():
    img, m = generate_scene(np.random.default_rng(0), 256, 256)
    pred = SpectralIndexStrategy().predict(img)
    water = m == 2
    if water.sum() > 100:
        assert (pred[water] == 2).mean() > 0.7


def test_facade_end_to_end(tmp_path, tif_bytes):
    repo = SQLiteLandCoverRepository()
    f = LandCoverFacade(LocalStorage(tmp_path), repo, SpectralIndexStrategy())
    r = f.process(tif_bytes)
    assert r["status"] == "completed" and r["polygons"] > 0
    assert set(r["timings"]) == {"ingest", "tiling", "inference", "mosaic", "vectorize", "persist"}
    assert f.mask_png(r["raster_id"])[:4] == b"\x89PNG"
    stats = f.stats(r["raster_id"])
    assert all(s["class"] in CLASSES for s in stats)
    gj = f.geojson(cls="water")
    assert all(ft["properties"]["class"] == "water" for ft in gj["features"])
    # повторная загрузка того же файла не пересчитывается (идемпотентность по хэшу)
    assert f.process(tif_bytes)["updated_at"] == r["updated_at"]


def test_bbox_query(tmp_path, tif_bytes):
    repo = SQLiteLandCoverRepository()
    f = LandCoverFacade(LocalStorage(tmp_path), repo, SpectralIndexStrategy())
    r = f.process(tif_bytes)
    minx, miny, maxx, maxy = r["meta"]["bounds"]
    inside = repo.query(bbox=(minx, miny, minx + 50, miny + 50))
    allp = repo.query(limit=100000)
    assert 0 < len(inside) <= len(allp)
    assert repo.query(bbox=(0, 0, 1, 1)) == []


def test_distributed_workers_match_facade(tmp_path, tif_bytes):
    broker = InMemoryBroker()
    prod = broker.producer()
    storage, repo, model = LocalStorage(tmp_path), SQLiteLandCoverRepository(), SpectralIndexStrategy()
    rid = raster_id_for(tif_bytes)
    ctx = IngestStage(storage, repo).run(RasterContext(rid, raw=tif_bytes))
    prod.send("raster.ingested", key=rid, value={"raster_id": rid, "key": f"raw/{rid}.tif", "meta": ctx.meta})
    for w in (TilerWorker(storage, repo, prod), InferenceWorker(storage, model, prod), GeoStoreWorker(storage, repo, prod)):
        run_consumer_loop(broker.consumer(w.topic_in, group_id=w.group), w, prod, w.topic_in + ".dlq", once=True)
    assert len(broker.messages("raster.completed")) == 1
    assert repo.get_raster(rid)["status"] == "completed"
    assert len({m.partition for m in broker.messages("tiles.ready")}) > 1      # тайлы растра — в разных партициях
    assert len({m.partition for m in broker.messages("tiles.segmented")}) == 1  # сборка растра — в одной


def test_bad_raster_rejected(tmp_path):
    f = LandCoverFacade(LocalStorage(tmp_path), SQLiteLandCoverRepository(), SpectralIndexStrategy())
    with pytest.raises(Exception):
        f.process(b"not a tiff")


@pytest.mark.skipif(not MODEL.exists(), reason="trained model not present")
def test_unet_strategy_shapes():
    s = make_strategy("unet", str(MODEL))
    img, _ = generate_scene(np.random.default_rng(1), 128, 128)
    p = s.predict_proba(img)
    assert p.shape == (6, 128, 128) and np.allclose(p.sum(0), 1, atol=1e-4)


@pytest.mark.skipif(not MODEL.exists(), reason="trained model not present")
def test_api(tmp_path, monkeypatch, tif_bytes):
    monkeypatch.setenv("LOCAL_STORAGE", str(tmp_path))
    from services.ingest_api import app as m

    m.facade.cache_clear()
    c = TestClient(m.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/v1/rasters", files={"file": ("s.tif", io.BytesIO(tif_bytes), "image/tiff")})
    assert r.status_code == 201
    rid = r.json()["raster_id"]
    assert c.get(f"/v1/rasters/{rid}").json()["status"] == "completed"
    assert c.get(f"/v1/rasters/{rid}/mask.png").headers["content-type"] == "image/png"
    assert c.get(f"/v1/rasters/{rid}/stats").json()["classes"]
    assert c.get("/v1/landcover", params={"cls": "forest", "limit": 5}).json()["type"] == "FeatureCollection"
    assert c.get("/v1/landcover", params={"bbox": "1,2"}).status_code == 422
    assert c.post("/v1/rasters", files={"file": ("x.tif", io.BytesIO(b"junk"), "image/tiff")}).status_code == 422
    assert c.get("/v1/rasters/nope").status_code == 404
