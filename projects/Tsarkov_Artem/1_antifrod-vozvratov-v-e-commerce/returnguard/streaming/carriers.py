"""Адаптеры статусов доставки разных служб — паттерн Adapter.

Каждая служба присылает вебхук в своём формате. Адаптер приводит его к
внутреннему событию order.delivered, которое понимает весь пайплайн.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone


class CarrierAdapter(ABC):
    carrier: str

    @abstractmethod
    def to_event(self, payload: dict) -> dict | None:
        """Вернуть order.delivered или None, если статус не финальный."""


class CdekAdapter(CarrierAdapter):
    carrier = "cdek"

    def to_event(self, p: dict) -> dict | None:
        if p["attributes"]["status_code"] != "DELIVERED":
            return None
        return {"type": "order.delivered", "ts": p["attributes"]["date_time"], "order_id": p["attributes"]["number"],
                "customer_id": p["attributes"]["customer_ref"], "carrier": self.carrier, "status": "DELIVERED",
                "proof": "code"}


class BoxberryAdapter(CarrierAdapter):
    carrier = "boxberry"

    def to_event(self, p: dict) -> dict | None:
        if p.get("Status") != "Выдано":
            return None
        ts = datetime.strptime(p["Date"], "%d.%m.%Y %H:%M").replace(tzinfo=timezone.utc).isoformat()
        return {"type": "order.delivered", "ts": ts, "order_id": p["OrderId"], "customer_id": p["ClientId"],
                "carrier": self.carrier, "status": "DELIVERED", "proof": "signature"}


class RussianPostAdapter(CarrierAdapter):
    carrier = "russian_post"

    def to_event(self, p: dict) -> dict | None:
        op = p["operation"]
        if (op["type"], op["attr"]) != (2, 1):  # 2/1 — «Вручение адресату»
            return None
        ts = datetime.fromtimestamp(op["epoch"], tz=timezone.utc).isoformat()
        return {"type": "order.delivered", "ts": ts, "order_id": p["barcode_ref"], "customer_id": p["recipient_ref"],
                "carrier": self.carrier, "status": "DELIVERED", "proof": "signature"}


ADAPTERS: dict[str, CarrierAdapter] = {a.carrier: a for a in (CdekAdapter(), BoxberryAdapter(), RussianPostAdapter())}


def normalize(carrier: str, payload: dict) -> dict | None:
    try:
        return ADAPTERS[carrier].to_event(payload)
    except KeyError as e:
        raise ValueError(f"unsupported carrier {carrier}") from e
