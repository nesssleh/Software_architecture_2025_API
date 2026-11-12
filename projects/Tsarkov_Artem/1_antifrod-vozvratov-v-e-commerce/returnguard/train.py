"""Офлайн-обучение: replay событий → признаки → CatBoost + baseline'ы → метрики и графики.

python -m returnguard.train
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostClassifier, Pool  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,  # noqa: E402
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from .datagen import generate  # noqa: E402
from .features import CAT_FEATURES, FEATURE_NAMES, CustomerState, FeatureBuilder  # noqa: E402
from .scoring.strategies import RuleBasedStrategy  # noqa: E402

OUT_MODELS = Path("models")
OUT_IMG = Path("docs/img")


def build_dataset(events: list[dict], labels: dict[str, int]) -> pd.DataFrame:
    fb = FeatureBuilder()
    states: dict[str, CustomerState] = {}
    rows = []
    for ev in events:
        st = states.setdefault(ev["customer_id"], CustomerState(ev["customer_id"]))
        if ev["type"] == "return.requested":
            f = fb.features(st, ev)
            f.update(return_id=ev["return_id"], customer_id=ev["customer_id"], ts=ev["ts"], label=labels[ev["return_id"]])
            rows.append(f)
        fb.apply(st, ev)
    return pd.DataFrame(rows)


def metrics(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    k = max(int(0.05 * len(p)), 1)
    top = np.argsort(-p)[:k]
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
            "f1": f1_score(y, yhat), "recall_at_fpr1": float(np.interp(0.01, fpr, tpr)),
            "precision_at_top5pct": float(np.mean(np.asarray(y)[top]))}


def main() -> None:
    OUT_MODELS.mkdir(exist_ok=True)
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    events, labels, arche = generate()
    df = build_dataset(events, labels)
    df = df.sort_values("ts").reset_index(drop=True)
    cut = df["ts"].iloc[int(len(df) * 0.75)]  # out-of-time split: последние 25% возвратов — тест
    train, test = df[df.ts < cut], df[df.ts >= cut]
    val_cut = train["ts"].iloc[int(len(train) * 0.85)]
    tr, va = train[train.ts < val_cut], train[train.ts >= val_cut]
    X = lambda d: d[FEATURE_NAMES].astype({c: str for c in CAT_FEATURES})  # noqa: E731

    dataset_info = {"events": len(events), "customers": len(arche), "returns": len(df), "fraud_share": float(df.label.mean()),
                    "train": len(tr), "val": len(va), "test": len(test), "split_ts": cut,
                    "archetypes": pd.Series(arche).value_counts().to_dict()}

    model = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05, loss_function="Logloss", eval_metric="PRAUC",
                               auto_class_weights="Balanced", random_seed=42, verbose=0, od_type="Iter", od_wait=80,
                               thread_count=2)
    model.fit(Pool(X(tr), tr.label, cat_features=CAT_FEATURES), eval_set=Pool(X(va), va.label, cat_features=CAT_FEATURES),
              use_best_model=True)
    p_cb = model.predict_proba(X(test))[:, 1]

    rules = RuleBasedStrategy()
    p_rules = rules.score_batch(test[FEATURE_NAMES].to_dict("records"))

    num = [f for f in FEATURE_NAMES if f not in CAT_FEATURES]
    lr = make_pipeline(ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)]),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(train[FEATURE_NAMES], train.label)
    p_lr = lr.predict_proba(test[FEATURE_NAMES])[:, 1]

    y = test.label.values
    # порог для CatBoost подбираем на валидации по максимуму F1
    pv = model.predict_proba(X(va))[:, 1]
    pr, rc, th = precision_recall_curve(va.label, pv)
    f1s = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    best_thr = float(th[np.argmax(f1s[:-1])])
    res = {"rules (baseline)": metrics(y, p_rules, 0.5), "logreg (baseline)": metrics(y, p_lr, 0.5),
           "catboost": metrics(y, p_cb, best_thr)}

    version = time.strftime("cb-%Y%m%d")
    model.save_model(str(OUT_MODELS / "catboost_returns.cbm"))
    (OUT_MODELS / "catboost_returns.json").write_text(json.dumps({"version": version, "threshold_f1": best_thr,
                                                                  "features": FEATURE_NAMES}, indent=1))
    imp = sorted(zip(FEATURE_NAMES, model.get_feature_importance()), key=lambda x: -x[1])

    # ---------- графики
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, p in [("Rules", p_rules), ("LogReg", p_lr), ("CatBoost", p_cb)]:
        f, t, _ = roc_curve(y, p)
        ax[0].plot(f, t, label=f"{name} AUC={roc_auc_score(y, p):.3f}")
        pr_, rc_, _ = precision_recall_curve(y, p)
        ax[1].plot(rc_, pr_, label=f"{name} AP={average_precision_score(y, p):.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC (test, out-of-time)", xlabel="FPR", ylabel="TPR")
    ax[1].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    for a in ax:
        a.legend(loc="lower right" if a is ax[0] else "upper right", fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_IMG / "roc_pr.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    names, vals = zip(*imp[:12])
    ax.barh(names[::-1], vals[::-1], color="#3b6fb6")
    ax.set(title="CatBoost feature importance (top-12)", xlabel="importance")
    fig.tight_layout()
    fig.savefig(OUT_IMG / "feature_importance.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(p_cb[y == 0], bins=50, alpha=0.6, label="legit", density=True)
    ax.hist(p_cb[y == 1], bins=50, alpha=0.6, label="fraud", density=True)
    ax.axvline(best_thr, color="k", ls="--", label=f"threshold={best_thr:.2f}")
    ax.set(title="Распределение скоров CatBoost на тесте", xlabel="score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_IMG / "score_hist.png", dpi=130)
    plt.close(fig)

    report = {"dataset": dataset_info, "metrics": res, "threshold": best_thr, "best_iteration": model.get_best_iteration(),
              "feature_importance": [[n, round(v, 2)] for n, v in imp], "seconds": round(time.time() - t0, 1)}
    (OUT_MODELS / "metrics.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(report["metrics"], indent=1, default=float))


if __name__ == "__main__":
    main()
