"""Онлайн-расчёт признаков клиента по потоку событий.

Один и тот же код используется (1) при офлайн-построении обучающей выборки и
(2) в feature-service, читающем Kafka, — это исключает train/serve skew.
Состояние клиента хранится в CustomerFeatureRepository (Redis / память).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

HIGH_VALUE = 300.0
FEATURE_NAMES = [
    "account_age_days", "n_orders", "n_returns_prev", "return_rate", "return_value_share", "returns_30d",
    "avg_days_to_return", "share_fast_returns", "share_damaged", "share_not_described", "n_not_received_claims",
    "n_addresses", "n_payment_methods", "share_high_value_returned", "refund_method_mismatch_rate",
    # признаки текущего возврата
    "ret_amount", "ret_n_items", "ret_days_since_delivery", "ret_is_high_value", "ret_refund_mismatch",
    "ret_reason", "ret_category",
]
CAT_FEATURES = ["ret_reason", "ret_category"]


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


@dataclass
class CustomerState:
    customer_id: str
    signup_ts: str | None = None
    n_orders: int = 0
    ordered_value: float = 0.0
    n_returns: int = 0
    returned_value: float = 0.0
    sum_days_to_return: float = 0.0
    n_fast_returns: int = 0
    n_damaged: int = 0
    n_not_described: int = 0
    n_not_received: int = 0
    n_high_value_returned: int = 0
    n_refund_mismatch: int = 0
    addresses: list[str] = field(default_factory=list)
    payments: list[str] = field(default_factory=list)
    return_ts: list[str] = field(default_factory=list)
    orders: dict = field(default_factory=dict)  # order_id -> {ts, delivered_ts, payment, amount}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CustomerState":
        return cls(**d)


class FeatureBuilder:
    """Применяет события к состоянию клиента и строит вектор признаков на момент возврата."""

    def apply(self, st: CustomerState, ev: dict) -> None:
        t = ev["type"]
        if t == "order.created":
            st.signup_ts = st.signup_ts or ev.get("signup_ts")
            st.n_orders += 1
            st.ordered_value += ev["amount"]
            if ev["address_id"] not in st.addresses:
                st.addresses.append(ev["address_id"])
            if ev["payment_method"] not in st.payments:
                st.payments.append(ev["payment_method"])
            st.orders[ev["order_id"]] = {"ts": ev["ts"], "delivered_ts": None, "payment": ev["payment_method"],
                                         "amount": ev["amount"]}
        elif t == "order.delivered":
            if ev["order_id"] in st.orders:
                st.orders[ev["order_id"]]["delivered_ts"] = ev["ts"]
        elif t == "return.requested":
            o = st.orders.get(ev["order_id"], {})
            days = self._days_since_delivery(o, ev)
            st.n_returns += 1
            st.returned_value += ev["amount"]
            st.sum_days_to_return += days
            st.n_fast_returns += days < 3
            st.n_damaged += ev["reason"] in ("damaged", "defect")
            st.n_not_described += ev["reason"] == "not_as_described"
            st.n_not_received += ev["reason"] == "not_received" and o.get("delivered_ts") is not None
            st.n_high_value_returned += ev["amount"] >= HIGH_VALUE
            st.n_refund_mismatch += ev.get("refund_method") != o.get("payment")
            st.return_ts = (st.return_ts + [ev["ts"]])[-50:]

    @staticmethod
    def _days_since_delivery(o: dict, ev: dict) -> float:
        base = o.get("delivered_ts") or o.get("ts") or ev["ts"]
        return max((_ts(ev["ts"]) - _ts(base)).total_seconds() / 86400, 0.0)

    def features(self, st: CustomerState, ret: dict) -> dict:
        """Признаки на момент запроса возврата ret (ДО применения ret к состоянию)."""
        now = _ts(ret["ts"])
        o = st.orders.get(ret["order_id"], {})
        n_ret = st.n_returns
        cats = [i["category"] for i in ret.get("items", [])]
        return {
            "account_age_days": (now - _ts(st.signup_ts)).days if st.signup_ts else 0,
            "n_orders": st.n_orders,
            "n_returns_prev": n_ret,
            "return_rate": n_ret / max(st.n_orders, 1),
            "return_value_share": st.returned_value / max(st.ordered_value, 1.0),
            "returns_30d": sum((now - _ts(t)).days <= 30 for t in st.return_ts),
            "avg_days_to_return": st.sum_days_to_return / n_ret if n_ret else -1.0,
            "share_fast_returns": st.n_fast_returns / n_ret if n_ret else 0.0,
            "share_damaged": st.n_damaged / n_ret if n_ret else 0.0,
            "share_not_described": st.n_not_described / n_ret if n_ret else 0.0,
            "n_not_received_claims": st.n_not_received,
            "n_addresses": len(st.addresses),
            "n_payment_methods": len(st.payments),
            "share_high_value_returned": st.n_high_value_returned / n_ret if n_ret else 0.0,
            "refund_method_mismatch_rate": st.n_refund_mismatch / n_ret if n_ret else 0.0,
            "ret_amount": ret["amount"],
            "ret_n_items": len(ret.get("items", [])),
            "ret_days_since_delivery": self._days_since_delivery(o, ret),
            "ret_is_high_value": int(ret["amount"] >= HIGH_VALUE),
            "ret_refund_mismatch": int(ret.get("refund_method") != o.get("payment")),
            "ret_reason": ret["reason"],
            "ret_category": max(set(cats), key=cats.count) if cats else "unknown",
        }
