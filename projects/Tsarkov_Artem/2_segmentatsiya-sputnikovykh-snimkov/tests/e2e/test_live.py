"""E2E против живого стека: API → MinIO → Kafka (tiler → inference ×2 → geostore) → PostGIS → API."""
import io
import os
import tempfile
import time

import httpx

httpx = httpx.Client(timeout=60)  # под нагрузкой стенда ответы бывают медленными
import numpy as np

from geoseg.synth import generate_scene, write_geotiff

from .helpers import TopicWatcher, kafka_admin_state, live, wait_for

pytestmark = live
API = os.getenv("API_URL", "http://api:8000")


def _tif(seed: int) -> bytes:
    img, _ = generate_scene(np.random.default_rng(seed), 512, 512)
    with tempfile.TemporaryDirectory() as d:
        write_geotiff(f"{d}/s.tif", img, None)
        return open(f"{d}/s.tif", "rb").read()


def test_api_healthy():
    r = httpx.get(f"{API}/health", timeout=10)
    print(r.json())
    assert r.status_code == 200


def test_kafka_topics_and_consumer_groups():
    topics, groups = kafka_admin_state()
    print("topics:", sorted(t for t in topics if not t.startswith("__")))
    print("groups:", sorted(groups))
    assert {"raster.ingested", "tiles.ready", "tiles.segmented", "raster.completed"} <= topics
    assert {"tiler", "inference-worker", "geostore"} <= groups


def test_async_pipeline_through_kafka():
    w = TopicWatcher("raster.completed")
    t0 = time.time()
    r = httpx.post(f"{API}/v1/rasters?mode=async", files={"file": ("a.tif", io.BytesIO(_tif(int(t0))), "image/tiff")}, timeout=60)
    assert r.status_code == 201, r.text
    rid = r.json()["raster_id"]
    part, off, msg = w.find(lambda v: v.get("raster_id") == rid, timeout=180)
    w.close()
    print(f"raster.completed partition={part} offset={off} за {time.time() - t0:.1f} с: {msg}")
    st = httpx.get(f"{API}/v1/rasters/{rid}").json()
    assert st["status"] == "completed" and st["polygons"] > 0
    assert httpx.get(f"{API}/v1/rasters/{rid}/mask.png").headers["content-type"] == "image/png"   # маска из MinIO
    stats = httpx.get(f"{API}/v1/rasters/{rid}/stats").json()["classes"]                          # площади из PostGIS
    print(stats)
    assert stats


def test_sync_and_geojson():
    r = httpx.post(f"{API}/v1/rasters", files={"file": ("b.tif", io.BytesIO(_tif(7)), "image/tiff")}, timeout=120)
    assert r.status_code == 201 and r.json()["status"] == "completed"
    b = r.json()["meta"]["bounds"]
    gj = wait_for(lambda: httpx.get(f"{API}/v1/landcover", params={"bbox": ",".join(map(str, b)), "limit": 5}).json()["features"],
                  30, 1, "полигоны в bbox (PostGIS)")
    print(gj[0]["properties"])
    assert httpx.post(f"{API}/v1/rasters", files={"file": ("x.tif", io.BytesIO(b"junk"), "image/tiff")}).status_code == 422
