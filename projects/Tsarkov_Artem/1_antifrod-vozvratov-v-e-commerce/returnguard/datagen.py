"""Генератор синтетического потока событий e-commerce с разметкой злоупотреблений возвратами.

Реальные данные о мошенничестве с возвратами закрыты (персональные данные,
коммерческая тайна), поэтому используется симулятор с поведенческими
архетипами, описанными в отраслевых отчётах (NRF/Appriss «Consumer Returns»):

* normal        — обычный покупатель, редкие возвраты (размер, брак);
* heavy_legit   — «честный» частый возвратчик (fashion, примерка дома),
                  трудный негативный класс: высокий return rate, но медленные
                  возвраты дешёвых вещей с причиной «не подошёл размер»;
* wardrobing    — покупает дорогое, пользуется 1–4 дня, возвращает;
* damage_claim  — систематически заявляет «повреждено/пустая коробка»;
* inr_claim     — заявляет «не получил», хотя служба доставки подтвердила вручение;
* reseller      — много адресов и способов оплаты, массовые возвраты электроники.

Метка мошенничества ставится на конкретный возврат (return_id): возвраты
абьюзеров мошеннические с вероятностью p_fraud (абьюзеры иногда возвращают честно).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

CATEGORIES = {
    "apparel": (15, 120), "shoes": (30, 180), "electronics": (80, 1500),
    "home": (10, 200), "beauty": (5, 80), "luxury": (300, 3000),
}
REASONS = ["size", "changed_mind", "not_as_described", "damaged", "defect", "not_received"]
CARRIERS = ["cdek", "boxberry", "russian_post", "own_courier"]
START = datetime(2025, 1, 1, tzinfo=timezone.utc)


@dataclass
class Archetype:
    name: str
    share: float
    orders_per_month: float
    cat_p: dict
    return_p: float
    days_to_return: tuple[float, float]
    reason_p: dict
    p_fraud: float
    n_addresses: tuple[int, int] = (1, 2)
    n_payments: tuple[int, int] = (1, 2)
    value_mult: float = 1.0


ARCHETYPES = [
    Archetype("normal", 0.82, 1.0, {"apparel": .3, "shoes": .1, "electronics": .15, "home": .25, "beauty": .2},
              0.12, (1, 14), {"size": .4, "defect": .2, "changed_mind": .2, "damaged": .1, "not_as_described": .07,
                              "not_received": .03}, 0.0),
    Archetype("heavy_legit", 0.10, 2.5, {"apparel": .55, "shoes": .3, "beauty": .1, "luxury": .05},
              0.45, (2, 12), {"size": .6, "changed_mind": .25, "not_as_described": .1, "damaged": .05}, 0.0),
    Archetype("wardrobing", 0.02, 1.5, {"apparel": .35, "luxury": .25, "electronics": .2, "shoes": .2},
              0.55, (1, 6), {"changed_mind": .35, "not_as_described": .3, "size": .35}, 0.7, value_mult=1.2),
    Archetype("damage_claim", 0.02, 1.2, {"electronics": .45, "home": .35, "beauty": .2},
              0.40, (1, 10), {"damaged": .5, "defect": .3, "not_as_described": .2}, 0.7),
    Archetype("inr_claim", 0.015, 1.2, {"electronics": .4, "apparel": .3, "beauty": .3},
              0.35, (1, 5), {"not_received": .6, "damaged": .2, "changed_mind": .2}, 0.75),
    Archetype("reseller", 0.025, 2.0, {"electronics": .6, "luxury": .2, "apparel": .2},
              0.35, (2, 10), {"changed_mind": .45, "defect": .3, "not_as_described": .25}, 0.6, (1, 5), (1, 4)),
]


def _pick(rng: np.random.Generator, p: dict) -> str:
    keys = list(p)
    w = np.array([p[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=w / w.sum())]


def _iso(t: datetime) -> str:
    return t.isoformat()


def generate(n_customers: int = 6000, months: int = 12, seed: int = 7) -> tuple[list[dict], dict[str, int], dict[str, str]]:
    """Возвращает (events, labels{return_id: 0/1}, customer_archetype{customer_id: name})."""
    rng = np.random.default_rng(seed)
    shares = np.array([a.share for a in ARCHETYPES])
    events: list[dict] = []
    labels: dict[str, int] = {}
    arche: dict[str, str] = {}
    horizon = months * 30
    oid = rid = 0
    for c in range(n_customers):
        a = ARCHETYPES[rng.choice(len(ARCHETYPES), p=shares / shares.sum())]
        cid = f"C{c:06d}"
        arche[cid] = a.name
        signup = START - timedelta(days=float(rng.uniform(0, 900 if a.name in ("normal", "heavy_legit") else 500)))
        addresses = [f"A{c}-{i}" for i in range(int(rng.integers(a.n_addresses[0], a.n_addresses[1] + 1)))]
        payments = list(rng.choice(["card", "sbp", "cash_on_delivery", "wallet", "credit"],
                                   size=int(rng.integers(a.n_payments[0], a.n_payments[1] + 1)), replace=False))
        n_orders = rng.poisson(a.orders_per_month * months) + 1
        for _ in range(n_orders):
            oid += 1
            t0 = START + timedelta(days=float(rng.uniform(0, horizon)), hours=float(rng.uniform(8, 23)))
            items = []
            for _ in range(int(rng.integers(1, 4))):
                cat = _pick(rng, a.cat_p)
                lo, hi = CATEGORIES[cat]
                price = round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) * a.value_mult, 2)
                items.append({"sku": f"{cat[:3].upper()}{int(rng.integers(1, 5000)):05d}", "category": cat, "price": price})
            order_id = f"O{oid:08d}"
            events.append({"type": "order.created", "ts": _iso(t0), "order_id": order_id, "customer_id": cid,
                           "items": items, "amount": round(sum(i["price"] for i in items), 2),
                           "payment_method": str(rng.choice(payments)), "address_id": str(rng.choice(addresses)),
                           "signup_ts": _iso(signup)})
            delivered = t0 + timedelta(days=float(rng.uniform(1, 6)))
            lost = a.name == "normal" and rng.random() < 0.03  # реально потерянная посылка
            if not lost:
                events.append({"type": "order.delivered", "ts": _iso(delivered), "order_id": order_id, "customer_id": cid,
                               "carrier": str(rng.choice(CARRIERS)), "status": "DELIVERED",
                               "proof": str(rng.choice(["signature", "photo", "code"]))})
            fraud_mode = rng.random() < a.p_fraud
            beh = a if (fraud_mode or a.p_fraud == 0) else ARCHETYPES[0]
            if lost or rng.random() < beh.return_p:
                rid += 1
                lo, hi = beh.days_to_return
                rt = delivered + timedelta(days=float(rng.uniform(lo, hi)))
                n_ret = int(rng.integers(1, len(items) + 1))
                ret_items = items[:n_ret]
                reason = "not_received" if lost else _pick(rng, beh.reason_p)
                return_id = f"R{rid:08d}"
                events.append({"type": "return.requested", "ts": _iso(rt), "return_id": return_id, "order_id": order_id,
                               "customer_id": cid, "items": ret_items,
                               "amount": round(sum(i["price"] for i in ret_items), 2), "reason": reason,
                               "refund_method": str(rng.choice(payments + (["wallet"] if fraud_mode else [])))})
                labels[return_id] = int(fraud_mode and a.p_fraud > 0)
    events.sort(key=lambda e: e["ts"])
    return events, labels, arche


def save(path: str | Path, **kw) -> None:
    events, labels, arche = generate(**kw)
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (p / "labels.json").write_text(json.dumps(labels))
    (p / "customers.json").write_text(json.dumps(arche))


if __name__ == "__main__":
    import os

    save("data", n_customers=int(os.getenv("DATAGEN_CUSTOMERS", 6000)))
