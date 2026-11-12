"""Точка входа воркеров: python -m services.run_worker tiler|inference|geostore"""
from __future__ import annotations

import logging
import os
import sys

from geoseg.models.strategies import make_strategy
from geoseg.pipeline.kafka_io import make_consumer, make_producer, run_consumer_loop
from geoseg.storage.objects import make_storage
from geoseg.storage.repository import make_repository
from services.workers import GeoStoreWorker, InferenceWorker, TilerWorker


def build(kind: str):
    st, prod = make_storage(), make_producer()
    if kind == "tiler":
        return TilerWorker(st, make_repository(), prod)
    if kind == "inference":
        return InferenceWorker(st, make_strategy(os.getenv("SEG_MODEL", "unet"), os.getenv("MODEL_PATH")), prod)
    if kind == "geostore":
        return GeoStoreWorker(st, make_repository(), prod)
    raise SystemExit(f"unknown worker {kind}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from geoseg.obs import start_metrics_server

    start_metrics_server()
    w = build(sys.argv[1])
    run_consumer_loop(make_consumer(w.topic_in, group_id=w.group), w, w.producer, f"{w.topic_in}.dlq")
