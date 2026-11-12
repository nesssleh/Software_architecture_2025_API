"""Объектное хранилище растров и масок — паттерн Adapter.

Сервисы работают с интерфейсом ObjectStorage (put/get bytes). MinIOStorage
адаптирует S3-API (клиент minio-py; сервер — любой S3-совместимый: MinIO, SeaweedFS), LocalStorage — файловую систему (тесты, локальный запуск).
"""
from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from pathlib import Path


class ObjectStorage(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    def uri(self, key: str) -> str:
        return key


class LocalStorage(ObjectStorage):
    def __init__(self, root: str | Path = "storage") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return key

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def uri(self, key: str) -> str:
        return str((self.root / key).resolve())


class MinIOStorage(ObjectStorage):
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str = "rasters", secure: bool = False) -> None:
        from minio import Minio

        import time

        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket = bucket
        for attempt in range(60):      # хранилище может стартовать позже сервиса
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                break
            except Exception:
                if attempt == 59:
                    raise
                time.sleep(2)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(self.bucket, key, io.BytesIO(data), len(data), content_type=content_type)
        return key

    def get(self, key: str) -> bytes:
        r = self.client.get_object(self.bucket, key)
        try:
            return r.read()
        finally:
            r.close()
            r.release_conn()

    def exists(self, key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except Exception:
            return False

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"


def make_storage() -> ObjectStorage:
    ep = os.getenv("MINIO_ENDPOINT")
    if ep:
        return MinIOStorage(ep, os.getenv("MINIO_ACCESS_KEY", "minio"), os.getenv("MINIO_SECRET_KEY", "minio123"),
                            os.getenv("MINIO_BUCKET", "rasters"))
    return LocalStorage(os.getenv("LOCAL_STORAGE", "storage"))
