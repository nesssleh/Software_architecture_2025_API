"""Сквозной прогон пайплайна: ingest → orders.events → feature-service → returns.to_score →
scoring-service → returns.scored / fraud.alerts / кейсы.

Kafka — InMemoryBroker (та же семантика партиций и consumer groups) либо реальный
Kafka при KAFKA_BOOTSTRAP. Redis/PostgreSQL/ClickHouse берутся из переменных окружения
(REDIS_URL, DATABASE_URL, CLICKHOUSE_URL / ANALYTICS_SQLITE).

python scripts/run_pipeline_demo.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score  # noqa: E402

from returnguard.datagen import generate  # noqa: E402
from returnguard.scoring.strategies import make_strategy  # noqa: E402
from returnguard.storage.repositories import CaseRepository, make_analytics_repo, make_customer_repo  # noqa: E402
from returnguard.streaming.kafka_io import InMemoryBroker  # noqa: E402
from services.feature_service.service import FeatureService  # noqa: E402
from services.scoring_service.service import ScoringService  # noqa: E402
from returnguard.streaming.kafka_io import run_consumer_loop  # noqa: E402


def main() -> None:
    broker = InMemoryBroker(partitions=6)
    prod = broker.producer()
    events, labels, _ = generate(n_customers=int(os.getenv('DEMO_CUSTOMERS', 2000)), seed=2026)  # новая популяция, не видна модели
    customers, analytics, cases = make_customer_repo(), make_analytics_repo(), CaseRepository()
    fs = FeatureService(customers, analytics, prod)
    ss = ScoringService(make_strategy("catboost"), cases, analytics, prod, block=0.9, review=0.6)

    t0 = time.time()
    for ev in events:
        prod.send("orders.events", key=ev["customer_id"], value=json.dumps(ev, ensure_ascii=False))
    t_ingest = time.time() - t0

    t1 = time.time()
    n_feat = run_consumer_loop(broker.consumer("orders.events", group_id="feature-builder"), fs.handle, prod,
                               "orders.events.dlq", once=True)
    t_feat = time.time() - t1
    t2 = time.time()
    lat = []
    cons = broker.consumer("returns.to_score", group_id="return-scorer")

    def timed(msg):
        s = time.perf_counter()
        ss.handle(msg)
        lat.append((time.perf_counter() - s) * 1000)

    n_score = run_consumer_loop(cons, timed, prod, "returns.to_score.dlq", once=True)
    t_score = time.time() - t2

    scored = [m.json() for m in broker.messages("returns.scored")]
    test = scored  # все возвраты новой популяции — out-of-sample для модели
    y = np.array([labels[s["return_id"]] for s in test])
    p = np.array([s["score"] for s in test])
    flagged = np.array([s["decision"] in ("review", "block") for s in test])
    blocked = np.array([s["decision"] == "block" for s in test])
    dec = Counter(s["decision"] for s in scored)
    by = Counter(s["decided_by"] for s in scored)
    fraud_amount = sum(s["amount"] for s in test if labels[s["return_id"]])
    caught_amount = sum(s["amount"] for s, f in zip(test, flagged) if f and labels[s["return_id"]])

    summary = {
        "events": len(events), "returns_scored": n_score, "feature_msgs": n_feat,
        "throughput_feature_eps": round(n_feat / t_feat), "throughput_scoring_rps": round(n_score / t_score),
        "ingest_seconds": round(t_ingest, 2),
        "scoring_latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2),
                               "p99": round(float(np.percentile(lat, 99)), 2)},
        "decisions": dict(dec), "decided_by": dict(by),
        "topics": {t: len(broker.messages(t)) for t in ["orders.events", "returns.to_score", "returns.scored",
                                                         "fraud.alerts", "orders.events.dlq", "returns.to_score.dlq"]},
        "consumer_lag": {"feature-builder": broker.lag("feature-builder", "orders.events"),
                         "return-scorer": broker.lag("return-scorer", "returns.to_score")},
        "online_eval_new_population": {"returns": len(test), "roc_auc": round(roc_auc_score(y, p), 4),
                               "pr_auc": round(average_precision_score(y, p), 4),
                               "flag_precision": round(precision_score(y, flagged), 4),
                               "flag_recall": round(recall_score(y, flagged), 4),
                               "block_precision": round(precision_score(y, blocked), 4),
                               "fraud_amount_caught_share": round(caught_amount / fraud_amount, 4)},
        "cases": cases.counts(),
    }
    Path("models").mkdir(exist_ok=True)
    Path("models/pipeline_run.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))

    # «аналитический экран»: дневная сводка из хранилища
    days = analytics.daily_summary()
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    x = np.arange(len(days))
    ax[0].plot(x, [d["returns"] for d in days], label="возвратов", color="#7a7a7a")
    ax[0].bar(x, [d["review"] for d in days], label="на проверку", color="#f2a93b")
    ax[0].bar(x, [d["blocked"] for d in days], bottom=[d["review"] for d in days], label="заблокировано", color="#c8423b")
    ax[0].legend(loc="upper left")
    ax[0].set_title("ReturnGuard: решения по возвратам по дням")
    ax[1].plot(x, np.cumsum([d["blocked_amount"] for d in days]), color="#3b6fb6")
    ax[1].set_title("Накопленная сумма заблокированных возвратов, у.е.")
    step = max(len(days) // 12, 1)
    ax[1].set_xticks(x[::step], [d["day"][5:] for d in days][::step], rotation=45)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("docs/img/dashboard_daily.png", dpi=120)


if __name__ == "__main__":
    main()
