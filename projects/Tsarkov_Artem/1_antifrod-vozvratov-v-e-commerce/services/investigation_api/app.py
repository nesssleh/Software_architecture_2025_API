"""Investigation API — рабочее место аналитика антифрода.

GET  /health
POST /v1/score                       — синхронный скоринг возврата (для OMS, SLA < 50 мс)
GET  /v1/customers/{id}/risk         — профиль риска клиента + признаки + объяснение (SHAP)
GET  /v1/customers/{id}/timeline     — история событий клиента (ClickHouse)
GET  /v1/cases?status=open           — очередь кейсов
GET  /v1/cases/{case_id}
POST /v1/cases/{case_id}/decision    — решение аналитика (fraud / legit)
GET  /v1/analytics/daily             — дневная сводка решений
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from returnguard.obs import instrument
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from returnguard.features import FeatureBuilder
from returnguard.scoring.chain import ReturnContext, build_chain
from returnguard.scoring.strategies import CatBoostStrategy, make_strategy
from returnguard.storage.repositories import CaseRepository, make_analytics_repo, make_customer_repo

app = FastAPI(title="ReturnGuard — Investigation API", version="1.0.0")
instrument(app)
API_TOKEN = os.getenv("API_TOKEN", "dev-token")


def auth(x_api_token: str = Header(default="")) -> None:
    if x_api_token != API_TOKEN:
        raise HTTPException(401, "invalid token")


@lru_cache(maxsize=1)
def deps():
    strategy = make_strategy(os.getenv("SCORING_STRATEGY", "catboost"), os.getenv("MODEL_PATH"))
    return {"strategy": strategy, "chain": build_chain(strategy, block=float(os.getenv("BLOCK_THRESHOLD", 0.9)),
                                                       review=float(os.getenv("REVIEW_THRESHOLD", 0.6))),
            "customers": make_customer_repo(), "analytics": make_analytics_repo(), "cases": CaseRepository()}


class ReturnItem(BaseModel):
    sku: str
    category: str
    price: float


class ScoreRequest(BaseModel):
    return_id: str
    customer_id: str
    order_id: str
    ts: str
    amount: float = Field(ge=0)
    reason: Literal["size", "changed_mind", "not_as_described", "damaged", "defect", "not_received"]
    refund_method: str | None = None
    items: list[ReturnItem]


class Decision(BaseModel):
    verdict: Literal["fraud", "legit"]
    comment: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "strategy": deps()["strategy"].version}


@app.post("/v1/score", dependencies=[Depends(auth)])
def score(req: ScoreRequest):
    d = deps()
    st = d["customers"].get(req.customer_id)
    feats = FeatureBuilder().features(st, req.model_dump())
    ctx = d["chain"].handle(ReturnContext(req.return_id, req.customer_id, feats, req.amount))
    return {"return_id": req.return_id, "score": round(ctx.score or 0.0, 4), "decision": ctx.decision,
            "decided_by": ctx.decided_by, "reasons": ctx.reasons, "model_version": d["strategy"].version}


@app.get("/v1/customers/{customer_id}/risk", dependencies=[Depends(auth)])
def customer_risk(customer_id: str):
    d = deps()
    st = d["customers"].get(customer_id)
    if st.n_orders == 0:
        raise HTTPException(404, "customer not found")
    last = {"ts": st.return_ts[-1] if st.return_ts else st.orders[next(iter(st.orders))]["ts"], "order_id": "",
            "amount": 0.0, "reason": "size", "items": []}
    feats = FeatureBuilder().features(st, last)
    s = d["strategy"]
    out = {"customer_id": customer_id, "risk_score": round(s.score(feats), 4),
           "profile": {k: feats[k] for k in ("n_orders", "n_returns_prev", "return_rate", "share_fast_returns",
                                              "n_not_received_claims", "n_addresses", "account_age_days")}}
    if isinstance(s, CatBoostStrategy):
        out["explanation"] = s.explain(feats)
    return out


@app.get("/v1/customers/{customer_id}/timeline", dependencies=[Depends(auth)])
def timeline(customer_id: str, limit: int = 50):
    return {"customer_id": customer_id, "events": deps()["analytics"].customer_timeline(customer_id, limit)}


@app.get("/v1/cases", dependencies=[Depends(auth)])
def cases(status: str | None = "open", limit: int = 20, customer_id: str | None = None):
    return {"counts": deps()["cases"].counts(), "items": deps()["cases"].list(status, limit, customer_id)}


@app.get("/v1/cases/{case_id}", dependencies=[Depends(auth)])
def case(case_id: int):
    c = deps()["cases"].get(case_id)
    if not c:
        raise HTTPException(404, "case not found")
    return c


@app.post("/v1/cases/{case_id}/decision", dependencies=[Depends(auth)])
def decide(case_id: int, body: Decision):
    c = deps()["cases"].resolve(case_id, body.verdict, body.comment)
    if not c:
        raise HTTPException(404, "case not found")
    return c


@app.get("/v1/analytics/daily", dependencies=[Depends(auth)])
def daily():
    return {"days": deps()["analytics"].daily_summary()}
