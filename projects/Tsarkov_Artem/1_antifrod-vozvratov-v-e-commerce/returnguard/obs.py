"""Наблюдаемость: метрики Prometheus для FastAPI-сервисов и Kafka-воркеров.

* instrument(app)        — middleware: счётчик и гистограмма длительности запросов по шаблону пути,
                           эндпоинт /metrics для Prometheus;
* start_metrics_server() — HTTP-сервер метрик для воркеров (порт METRICS_PORT, по умолчанию 9100).
"""
from __future__ import annotations

import os
import time

from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest,
                               start_http_server)

# uvicorn --workers N: каждый процесс пишет метрики в общий каталог, /metrics агрегирует их (multiprocess mode)
MULTIPROC = bool(os.getenv("PROMETHEUS_MULTIPROC_DIR"))

HTTP_REQUESTS = Counter("http_requests_total", "HTTP-запросы", ["method", "path", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "Длительность HTTP-запроса", ["method", "path"],
                         buckets=(.0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10))
HTTP_INFLIGHT = Gauge("http_requests_in_progress", "Запросы в обработке", multiprocess_mode="livesum")


def instrument(app) -> None:
    from fastapi import Request, Response

    @app.middleware("http")
    async def _metrics(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        HTTP_INFLIGHT.inc()
        t0 = time.perf_counter()
        status = 500
        try:
            resp = await call_next(request)
            status = resp.status_code
            return resp
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            HTTP_LATENCY.labels(request.method, path).observe(time.perf_counter() - t0)
            HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            HTTP_INFLIGHT.dec()

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        if MULTIPROC:
            from prometheus_client import multiprocess

            reg = CollectorRegistry()
            multiprocess.MultiProcessCollector(reg)
            return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def start_metrics_server() -> None:
    start_http_server(int(os.getenv("METRICS_PORT", "9100")))
