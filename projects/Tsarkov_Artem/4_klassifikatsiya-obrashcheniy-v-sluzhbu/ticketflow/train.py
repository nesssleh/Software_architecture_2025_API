"""Обучение и сравнение классификаторов обращений.

Разбиение: train / val / test (случайное, без точных дубликатов текста между частями)
+ отдельный срез test_unseen — обращения по шаблонам формулировок, которых не было в train
(проверка обобщения на новые формулировки).

python -m ticketflow.train
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers  # noqa: E402
from transformers import BertConfig, PreTrainedTokenizerFast  # noqa: E402

from .classify.hf_model import TicketModel  # noqa: E402
from .classify.strategies import (EnsembleStrategy, HFTransformerStrategy, KeywordStrategy, TfidfStrategy,
                                  _with_tier)  # noqa: E402
from .datagen import CATEGORIES, PRIORITIES, T, generate  # noqa: E402

M, IMG = Path("models"), Path("docs/img")


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def split(data, seed=0):
    rng = np.random.default_rng(seed)
    held = {f"{c}:{len(T[c]) - 1}" for c in CATEGORIES}  # последний шаблон каждой категории — только в тесте
    seen = [d for d in data if d["tpl"] not in held]
    unseen = [d for d in data if d["tpl"] in held]
    idx = rng.permutation(len(seen))
    n = len(seen)
    train = [seen[i] for i in idx[: int(0.7 * n)]]
    val = [seen[i] for i in idx[int(0.7 * n): int(0.8 * n)]]
    tr_txt = {norm(d["text"]) for d in train}
    test = [seen[i] for i in idx[int(0.8 * n):] if norm(seen[i]["text"]) not in tr_txt]
    return train, val, test, unseen


def evaluate(strategy, data) -> dict:
    t = time.perf_counter()
    pc, pp = strategy.predict_proba([d["text"] for d in data], [d["customer_tier"] for d in data])
    ms = (time.perf_counter() - t) / len(data) * 1000
    yc = [CATEGORIES.index(d["category"]) for d in data]
    yp = [PRIORITIES.index(d["priority"]) for d in data]
    hc, hp = pc.argmax(1), pp.argmax(1)
    hi = np.array([p >= 2 for p in yp])
    return {"category_acc": accuracy_score(yc, hc), "category_macro_f1": f1_score(yc, hc, average="macro"),
            "priority_acc": accuracy_score(yp, hp), "priority_macro_f1": f1_score(yp, hp, average="macro"),
            "high_critical_recall": float((hp[hi] >= 2).mean()), "routing_acc": accuracy_score(yc, hc),
            "ms_per_ticket": round(ms, 3), "_pred_cat": hc.tolist()}


def train_hf(train, val, epochs=6, bs=64, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    texts = _with_tier([d["text"] for d in train], [d["customer_tier"] for d in train])
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=6000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]")
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=128, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=256, max_position_embeddings=128, pad_token_id=tok.pad_token_id)
    model = TicketModel(cfg, len(CATEGORIES), len(PRIORITIES))
    enc = tok(texts, padding=True, truncation=True, max_length=96, return_tensors="pt")
    yc = torch.tensor([CATEGORIES.index(d["category"]) for d in train])
    yp = torch.tensor([PRIORITIES.index(d["priority"]) for d in train])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    ce = torch.nn.CrossEntropyLoss()
    hist = []
    out_dir = M / "hf_ticket_model"
    best = -1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(train))
        losses = []
        for i in range(0, len(train), bs):
            b = perm[i:i + bs]
            lc, lp = model(enc["input_ids"][b], enc["attention_mask"][b])
            loss = ce(lc, yc[b]) + ce(lp, yp[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        model.save(out_dir)
        tok.save_pretrained(out_dir)
        (out_dir / "meta.json").write_text(json.dumps({"version": f"bert-tiny-scratch-ep{ep + 1}",
                                                      "categories": CATEGORIES, "priorities": PRIORITIES}))
        v = evaluate(HFTransformerStrategy(out_dir), val)
        hist.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_cat_f1": v["category_macro_f1"],
                     "val_pri_f1": v["priority_macro_f1"]})
        print(json.dumps(hist[-1]), flush=True)
        score = v["category_macro_f1"] + v["priority_macro_f1"]
        if score > best:
            best = score
            model.save(M / "hf_ticket_model_best")
    # лучшая эпоха -> основной путь
    import shutil

    shutil.copy(M / "hf_ticket_model_best" / "model.pt", out_dir / "model.pt")
    shutil.rmtree(M / "hf_ticket_model_best")
    n_params = sum(p.numel() for p in model.parameters())
    return hist, n_params


def main():
    import sys

    M.mkdir(exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    data = generate()
    train, val, test, unseen = split(data)
    if "--eval-only" in sys.argv:  # переиспользовать обученные модели и историю
        prev = json.loads((M / "metrics.json").read_text())
        tfidf = TfidfStrategy(M / "tfidf_logreg.joblib")
        hist, n_params = prev["hf_history"], prev["hf_params"]
        t_tfidf, t_hf = prev["train_seconds"]["tfidf_logreg"], prev["train_seconds"]["hf_transformer"]
        return report(data, train, val, test, unseen, tfidf, hist, n_params, t_tfidf, t_hf)
    t0 = time.time()
    tfidf = TfidfStrategy(models=TfidfStrategy.fit([d["text"] for d in train], [d["customer_tier"] for d in train],
                                                  [d["category"] for d in train], [d["priority"] for d in train]))
    joblib.dump({"vec": tfidf.vec, "cat": tfidf.cat, "pri": tfidf.pri}, M / "tfidf_logreg.joblib", compress=3)
    t_tfidf = time.time() - t0
    t0 = time.time()
    hist, n_params = train_hf(train, val)
    t_hf = time.time() - t0
    report(data, train, val, test, unseen, tfidf, hist, n_params, t_tfidf, t_hf)


def report(data, train, val, test, unseen, tfidf, hist, n_params, t_tfidf, t_hf):
    hf = HFTransformerStrategy(M / "hf_ticket_model")
    ens = EnsembleStrategy([(hf, 0.5, 1.0), (KeywordStrategy(), 0.5, 0.0)])
    res = {}
    for s in (KeywordStrategy(), tfidf, hf, ens):
        res[s.name] = {"test": evaluate(s, test), "test_unseen_templates": evaluate(s, unseen)}
    best = "hf_transformer"

    yc = [CATEGORIES.index(d["category"]) for d in test]
    cm = confusion_matrix(yc, res[best]["test"]["_pred_cat"], labels=range(len(CATEGORIES)))
    report_txt = classification_report(yc, res[best]["test"]["_pred_cat"], target_names=CATEGORIES, digits=3)
    for v in res.values():
        for part in v.values():
            part.pop("_pred_cat")
    out = {"dataset": {"total": len(data), "train": len(train), "val": len(val), "test": len(test), "test_unseen": len(unseen),
                       "category_dist": {c: sum(d["category"] == c for d in data) for c in CATEGORIES},
                       "priority_dist": {p: sum(d["priority"] == p for d in data) for p in PRIORITIES}},
           "results": res, "best_model": best, "production_model": "ensemble", "hf_history": hist, "hf_params": n_params,
           "train_seconds": {"tfidf_logreg": round(t_tfidf, 1), "hf_transformer": round(t_hf, 1)},
           "classification_report_best": report_txt}
    (M / "metrics.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(report_txt)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues")
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45, ha="right")
    ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if cmn[i, j] > .5 else "k")
    ax.set(title=f"Категории: матрица ошибок ({best}, test)")
    fig.tight_layout()
    fig.savefig(IMG / "confusion_category.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = list(res)
    for k, part in enumerate(["test", "test_unseen_templates"]):
        x = np.arange(len(names))
        ax[k].bar(x - 0.2, [res[n][part]["category_macro_f1"] for n in names], 0.4, label="category macro-F1")
        ax[k].bar(x + 0.2, [res[n][part]["priority_macro_f1"] for n in names], 0.4, label="priority macro-F1")
        ax[k].set_xticks(x, names, fontsize=8)
        ax[k].set(title=part, ylim=(0, 1))
        ax[k].grid(alpha=.3, axis="y")
        ax[k].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "model_comparison.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], "o-", label="train loss")
    ax2 = ax.twinx()
    ax2.plot([h["epoch"] for h in hist], [h["val_cat_f1"] for h in hist], "s-", color="g", label="val cat F1")
    ax2.plot([h["epoch"] for h in hist], [h["val_pri_f1"] for h in hist], "^-", color="r", label="val pri F1")
    ax.set(title="Обучение HF-модели", xlabel="epoch")
    fig.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "hf_training.png", dpi=130)
    print(json.dumps({k: v["test"] for k, v in res.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
