"""Оценка качества: извлечение навыков, разбиение требований, ранжирование кандидатов, поиск пробелов.

python scripts/evaluate.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ndcg_score, roc_auc_score  # noqa: E402

from skillmatch.datagen import fit_label, generate  # noqa: E402
from skillmatch.embedding.embedders import SpacyVectorEmbedder, TfidfEmbedder  # noqa: E402
from skillmatch.explain.explainers import TemplateExplainer  # noqa: E402
from skillmatch.extraction.extractors import GazetteerExtractor, HybridExtractor, SemanticExtractor  # noqa: E402
from skillmatch.facade import SkillMatchFacade  # noqa: E402
from skillmatch.matching import SkillMatcher  # noqa: E402
from skillmatch.parsing.parsers import split_title  # noqa: E402
from skillmatch.storage.repository import InMemoryRepository  # noqa: E402
from skillmatch.taxonomy import SKILLS  # noqa: E402

OUT = Path("models")
IMG = Path("docs/img")


def prf(pred: list[set], gold: list[set]) -> dict:
    tp = sum(len(p & g) for p, g in zip(pred, gold))
    fp = sum(len(p - g) for p, g in zip(pred, gold))
    fn = sum(len(g - p) for p, g in zip(pred, gold))
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(P, 4), "recall": round(R, 4), "f1": round(2 * P * R / max(P + R, 1e-9), 4)}


def rank_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """scores/labels: [n_vac, n_res]."""
    nd = [ndcg_score(labels[i:i + 1], scores[i:i + 1], k=10) for i in range(len(scores)) if labels[i].any()]
    p10 = [labels[i][np.argsort(-scores[i])[:10]].mean() for i in range(len(scores)) if labels[i].any()]
    return {"roc_auc": round(roc_auc_score(labels.ravel(), scores.ravel()), 4), "ndcg@10": round(float(np.mean(nd)), 4),
            "p@10": round(float(np.mean(p10)), 4)}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    resumes, vacancies = generate(600, 60, seed=1)
    docs = resumes + vacancies
    emb = SpacyVectorEmbedder()
    gaz = GazetteerExtractor()
    hybrid = HybridExtractor(gaz, SemanticExtractor(emb))
    report: dict = {"dataset": {"resumes": len(resumes), "vacancies": len(vacancies), "skills_in_taxonomy": len(SKILLS),
                                "avg_skills_resume": round(float(np.mean([len(r["skills"]) for r in resumes])), 2)}}

    # 1. извлечение навыков
    gold = [set(d["skills"]) for d in docs]
    ext = {}
    for name, ex in (("gazetteer", gaz), ("hybrid (gazetteer+semantic)", hybrid)):
        t = time.time()
        pred = [{e.name for e in ex.extract(split_title(d["text"])[1])} for d in docs]
        ext[name] = prf(pred, gold) | {"ms_per_doc": round((time.time() - t) / len(docs) * 1000, 2)}
    report["extraction"] = ext

    # 2. полный пайплайн через фасад
    repo = InMemoryRepository()
    f = SkillMatchFacade(repo, hybrid, emb, TemplateExplainer())
    t = time.time()
    for d in docs:
        f.ingest(d["type"], d["text"], d["id"])
    report["ingest_ms_per_doc"] = round((time.time() - t) / len(docs) * 1000, 2)
    must_pred = [set(repo.get(v["id"]).must_have) for v in vacancies]
    report["requirements_split"] = {"must_have": prf(must_pred, [set(v["must_have"]) for v in vacancies]),
                                    "nice_to_have": prf([set(repo.get(v["id"]).nice_to_have) for v in vacancies],
                                                        [set(v["nice_to_have"]) for v in vacancies])}

    # 3. ранжирование кандидатов
    labels = np.array([[fit_label(r, v) for r in resumes] for v in vacancies])
    tfidf = TfidfEmbedder([d["text"] for d in docs])
    R_t, V_t = tfidf.embed([r["text"] for r in resumes]), tfidf.embed([v["text"] for v in vacancies])
    R_e = np.stack([repo.get(r["id"]).embedding for r in resumes])
    V_e = np.stack([repo.get(v["id"]).embedding for v in vacancies])
    matcher = SkillMatcher(emb)
    t = time.time()
    s_ours = np.array([[matcher.match(repo.get(r["id"]).skill_names, repo.get(v["id"]).must_have,
                                      repo.get(v["id"]).nice_to_have, float(R_e[j] @ V_e[i])).score
                        for j, r in enumerate(resumes)] for i, v in enumerate(vacancies)])
    match_ms = (time.time() - t) / labels.size * 1000
    s_oracle = np.array([[matcher.match(set(r["skills"]), v["must_have"], v["nice_to_have"]).score for r in resumes]
                         for v in vacancies])
    s_exact = np.array([[len(set(v["must_have"]) & repo.get(r["id"]).skill_names) / len(v["must_have"]) for r in resumes]
                        for v in vacancies])
    report["ranking"] = {
        "tfidf_cosine (baseline)": rank_metrics(V_t @ R_t.T, labels),
        "embedding_cosine (spaCy)": rank_metrics(V_e @ R_e.T, labels),
        "skill_matcher: exact only (ablation)": rank_metrics(s_exact, labels),
        "skill_matcher: exact+related+semantic (default)": rank_metrics(s_ours, labels),
        "skill_matcher on gold skills (upper bound)": rank_metrics(s_oracle, labels),
    }
    report["ranking_positive_share"] = round(float(labels.mean()), 4)
    report["match_ms_per_pair"] = round(match_ms, 3)
    # полнота кандидатогенерации гибридного поиска (сколько подходящих попадает в выдачу top-50)
    rec = []
    for i, v in enumerate(vacancies):
        pos = {resumes[j]["id"] for j in np.where(labels[i])[0]}
        if pos:
            got = {c["resume_id"] for c in f.candidates(v["id"], k=50)}
            rec.append(len(pos & got) / min(len(pos), 50))
    report["candidates_recall@50"] = round(float(np.mean(rec)), 4)

    # 4. качество объяснения пробелов: missing_must vs эталонные пробелы
    gp, gg = [], []
    for v in vacancies[:20]:
        for r in resumes[:100]:
            res = f.match(r["id"], v["id"], explain=True)
            gp.append(set(res["explanation"]["facts"]["missing_must"]))
            gg.append(set(v["must_have"]) - set(r["skills"]) - {m["requirement"] for m in res["matches"] if m["kind"] in ("related", "semantic")})
    report["gap_detection"] = prf(gp, gg)
    v0, r0 = vacancies[0], max(resumes, key=lambda r: s_ours[0][resumes.index(r)] * (1 - fit_label(r, vacancies[0])))
    report["example"] = {"vacancy": v0["text"], "resume": r0["text"], "result": f.match(r0["id"], v0["id"])}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    # графики
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    names = list(report["ranking"])
    for k, m in enumerate(["roc_auc", "ndcg@10", "p@10"]):
        ax[0].bar(np.arange(len(names)) + (k - 1) * 0.27, [report["ranking"][n][m] for n in names], 0.27, label=m)
    ax[0].set_xticks(range(len(names)), [n.split(" (")[0] for n in names], rotation=20, ha="right", fontsize=8)
    ax[0].set(title="Ранжирование кандидатов", ylim=(0, 1.05))
    ax[0].legend(fontsize=8)
    en = list(ext)
    for k, m in enumerate(["precision", "recall", "f1"]):
        ax[1].bar(np.arange(len(en)) + (k - 1) * 0.27, [ext[n][m] for n in en], 0.27, label=m)
    ax[1].set_xticks(range(len(en)), en, fontsize=8)
    ax[1].set(title="Извлечение навыков", ylim=(0.8, 1.0))
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(IMG / "eval_summary.png", dpi=130)
    print(json.dumps({k: report[k] for k in ("extraction", "requirements_split", "ranking", "gap_detection",
                                             "ingest_ms_per_doc", "match_ms_per_pair", "candidates_recall@50")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
