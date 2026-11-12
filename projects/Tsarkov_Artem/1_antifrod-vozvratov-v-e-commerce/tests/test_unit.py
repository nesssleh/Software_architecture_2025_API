import pytest

from returnguard.features import CustomerState, FeatureBuilder
from returnguard.scoring.chain import ReturnContext, build_chain
from returnguard.scoring.strategies import BlendStrategy, RuleBasedStrategy, ScoringStrategy, make_strategy
from returnguard.streaming.carriers import normalize

ORDER = {"type": "order.created", "ts": "2025-03-01T10:00:00+00:00", "order_id": "O1", "customer_id": "C1",
         "items": [{"sku": "APP1", "category": "apparel", "price": 500.0}], "amount": 500.0, "payment_method": "card",
         "address_id": "A1", "signup_ts": "2025-01-01T00:00:00+00:00"}
DELIV = {"type": "order.delivered", "ts": "2025-03-03T10:00:00+00:00", "order_id": "O1", "customer_id": "C1"}
RET = {"type": "return.requested", "ts": "2025-03-04T10:00:00+00:00", "return_id": "R1", "order_id": "O1",
       "customer_id": "C1", "items": ORDER["items"], "amount": 500.0, "reason": "not_received", "refund_method": "wallet"}


class Const(ScoringStrategy):
    name = version = "const"

    def __init__(self, v):
        self.v = v

    def score(self, f):
        return self.v


def _state():
    fb, st = FeatureBuilder(), CustomerState("C1")
    fb.apply(st, ORDER)
    fb.apply(st, DELIV)
    return fb, st


def test_features_point_in_time():
    fb, st = _state()
    f = fb.features(st, RET)
    assert f["n_returns_prev"] == 0  # текущий возврат не учитывается в истории (нет утечки)
    assert f["ret_days_since_delivery"] == pytest.approx(1.0)
    assert f["ret_is_high_value"] == 1 and f["ret_refund_mismatch"] == 1
    fb.apply(st, RET)
    assert st.n_not_received == 1 and st.n_fast_returns == 1


def test_state_roundtrip():
    _, st = _state()
    assert CustomerState.from_dict(st.to_dict()) == st


def test_chain_allowlist_short_circuits():
    fb, st = _state()
    ctx = build_chain(Const(0.99), allowlist={"C1"}).handle(ReturnContext("R1", "C1", fb.features(st, RET), 500))
    assert ctx.decision == "approve" and ctx.decided_by == "allowlist"


def test_chain_hard_rule_blocks_repeat_inr():
    fb, st = _state()
    f = fb.features(st, RET) | {"n_not_received_claims": 2}
    ctx = build_chain(Const(0.0)).handle(ReturnContext("R1", "C1", f, 500))
    assert ctx.decision == "block" and ctx.decided_by == "hard_rule"


@pytest.mark.parametrize("score,expected", [(0.95, "block"), (0.7, "review"), (0.1, "approve")])
def test_chain_thresholds(score, expected):
    fb, st = _state()
    ctx = build_chain(Const(score), block=0.9, review=0.6).handle(ReturnContext("R1", "C1", fb.features(st, RET), 500))
    assert ctx.decision == expected and ctx.decided_by == "model"


def test_velocity_forces_review():
    fb, st = _state()
    f = fb.features(st, RET) | {"returns_30d": 9}
    ctx = build_chain(Const(0.1)).handle(ReturnContext("R1", "C1", f, 500))
    assert ctx.decision == "review"


def test_strategies():
    fb, st = _state()
    f = fb.features(st, RET)
    assert 0 <= RuleBasedStrategy().score(f) <= 1
    cb = make_strategy("catboost")
    assert 0 <= cb.score(f) <= 1
    assert len(cb.explain(f, top=3)) == 3
    assert BlendStrategy([(Const(1.0), 1), (Const(0.0), 1)]).score(f) == 0.5
    with pytest.raises(ValueError):
        make_strategy("xgboost")


def test_carrier_adapters():
    e = normalize("cdek", {"attributes": {"status_code": "DELIVERED", "date_time": "2025-03-03T10:00:00+00:00",
                                          "number": "O1", "customer_ref": "C1"}})
    assert e["type"] == "order.delivered" and e["carrier"] == "cdek"
    e = normalize("boxberry", {"Status": "Выдано", "Date": "03.03.2025 10:00", "OrderId": "O1", "ClientId": "C1"})
    assert e["order_id"] == "O1"
    assert normalize("russian_post", {"operation": {"type": 8, "attr": 2, "epoch": 0}, "barcode_ref": "O", "recipient_ref": "C"}) is None
    with pytest.raises(ValueError):
        normalize("dhl", {})
