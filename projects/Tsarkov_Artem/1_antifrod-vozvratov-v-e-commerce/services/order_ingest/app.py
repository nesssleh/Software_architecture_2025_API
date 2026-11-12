"""Order ingest service — producer топика `orders.events`.

* POST /v1/events               — событие из OMS (order.created / return.requested)
* POST /v1/carriers/{carrier}   — вебхук службы доставки (через CarrierAdapter)
* CLI replay: python -m services.order_ingest.app replay data/events.jsonl
"""
from __future__ import annotations

import json
import sys
from typing import Literal

from returnguard.obs import instrument
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from returnguard.streaming.carriers import normalize
from returnguard.streaming.kafka_io import make_producer

TOPIC = "orders.events"
app = FastAPI(title="ReturnGuard — Order Ingest", version="1.0.0")
instrument(app)
producer = make_producer()


class Item(BaseModel):
    sku: str
    category: str
    price: float


class OrderEvent(BaseModel):
    type: Literal["order.created", "order.delivered", "return.requested"]
    ts: str
    customer_id: str
    order_id: str
    return_id: str | None = None
    items: list[Item] = []
    amount: float = 0.0
    payment_method: str | None = None
    address_id: str | None = None
    reason: str | None = None
    refund_method: str | None = None
    signup_ts: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "topic": TOPIC}


@app.post("/v1/events", status_code=202)
def ingest(ev: OrderEvent):
    if ev.type == "return.requested" and not (ev.return_id and ev.reason):
        raise HTTPException(422, "return.requested requires return_id and reason")
    producer.send(TOPIC, key=ev.customer_id, value=ev.model_dump_json(exclude_none=True))
    return {"accepted": True, "topic": TOPIC, "key": ev.customer_id}


@app.post("/v1/carriers/{carrier}", status_code=202)
def carrier_webhook(carrier: str, payload: dict):
    try:
        ev = normalize(carrier, payload)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    except (KeyError, TypeError) as e:
        raise HTTPException(422, f"bad payload: {e}") from e
    if ev is None:
        return {"accepted": False, "reason": "non-final status"}
    producer.send(TOPIC, key=ev["customer_id"], value=json.dumps(ev))
    return {"accepted": True, "event": ev}


def replay(path: str, limit: int | None = None) -> int:
    n = 0
    with open(path) as f:
        for line in f:
            ev = json.loads(line)
            producer.send(TOPIC, key=ev["customer_id"], value=line.strip())
            n += 1
            if limit and n >= limit:
                break
    producer.flush()
    return n


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "replay":
        print(f"sent {replay(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)} events")
