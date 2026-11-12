"""Синтетический корпус резюме и вакансий с эталонной разметкой навыков.

Каждый документ генерируется из набора навыков роли; навык упоминается
либо каноническим именем, либо синонимом, либо «неявно» — описанием опыта
(например, «настраивал пайплайны выгрузки в оркестраторе Airflow»).
Это позволяет честно измерить precision/recall извлечения и качество
сопоставления. Для пар (резюме, вакансия) известна метка «подходит»:
кандидат покрывает >= 80% обязательных навыков вакансии.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .taxonomy import SKILLS

ROLES = {
    "Python Backend Developer": (["Python", "FastAPI", "PostgreSQL", "Docker", "REST API", "Git"],
                                 ["Redis", "Apache Kafka", "Kubernetes", "Django", "Unit Testing", "CI/CD", "Microservices", "Linux"]),
    "Data Scientist": (["Python", "Machine Learning", "pandas", "scikit-learn", "SQL", "Statistics"],
                       ["CatBoost", "PyTorch", "A/B Testing", "NumPy", "Deep Learning", "Git", "Apache Spark"]),
    "ML Engineer (NLP)": (["Python", "PyTorch", "NLP", "Hugging Face", "Docker"],
                          ["LLM", "MLOps", "Kubernetes", "FastAPI", "Deep Learning", "Linux", "Git"]),
    "Data Engineer": (["Python", "SQL", "Apache Airflow", "Apache Spark", "ClickHouse"],
                      ["Apache Kafka", "Hadoop", "dbt", "Docker", "PostgreSQL", "Scala", "Kubernetes"]),
    "Data Analyst": (["SQL", "Excel", "Data Analysis", "Statistics"],
                     ["Python", "pandas", "Tableau", "Power BI", "A/B Testing", "ClickHouse"]),
    "Java Developer": (["Java", "Spring", "PostgreSQL", "REST API", "Git"],
                       ["Kotlin", "Apache Kafka", "Docker", "Microservices", "Unit Testing", "Redis", "Kubernetes"]),
    "Frontend Developer": (["JavaScript", "TypeScript", "React", "HTML/CSS", "Git"],
                           ["Vue.js", "REST API", "GraphQL", "Unit Testing", "CI/CD"]),
    "DevOps Engineer": (["Linux", "Docker", "Kubernetes", "CI/CD", "Terraform"],
                        ["Ansible", "Prometheus", "Grafana", "AWS", "Yandex Cloud", "Python", "Go", "Git"]),
    "Go Developer": (["Go", "PostgreSQL", "Docker", "gRPC", "Microservices"],
                     ["Redis", "Apache Kafka", "Kubernetes", "Prometheus", "Linux", "Git"]),
    "Team Lead": (["Team Leadership", "System Design", "Agile", "Communication"],
                  ["Project Management", "Python", "Java", "Microservices", "English"]),
}

IMPLICIT = {  # «неявные» формулировки, которые словарь синонимов не ловит
    "Apache Airflow": ["настраивал DAG-и для ежедневной загрузки данных в оркестраторе",
                       "строил регламентные ETL-процессы с расписанием и ретраями"],
    "Docker": ["упаковывал сервисы в образы и описывал окружение через compose-файлы"],
    "Kubernetes": ["деплоил сервисы в кластер, писал Helm-чарты и манифесты деплойментов"],
    "Machine Learning": ["обучал модели классификации и регрессии, подбирал гиперпараметры"],
    "Statistics": ["проверял статистические гипотезы, считал доверительные интервалы"],
    "A/B Testing": ["проводил онлайн-эксперименты и оценивал прирост конверсии между группами"],
    "Unit Testing": ["покрывал код автотестами с фикстурами и моками"],
    "CI/CD": ["настраивал автоматическую сборку и выкладку при каждом мёрже в main"],
    "Team Leadership": ["руководил группой из 6 разработчиков, проводил one-to-one и ревью"],
    "System Design": ["проектировал архитектуру высоконагруженного сервиса на 10k RPS"],
    "Data Analysis": ["готовил аналитические отчёты и исследовал метрики продукта"],
    "Communication": ["выступал на митапах, взаимодействовал с заказчиками и смежными командами"],
    "NLP": ["решал задачи классификации текстов и извлечения сущностей из документов"],
    "Microservices": ["распиливал монолит на независимые сервисы с отдельными базами"],
    "REST API": ["проектировал HTTP-эндпоинты с OpenAPI-спецификацией"],
    "LLM": ["строил чат-бота на большой языковой модели с поиском по базе знаний"],
}

RES_PHRASES = ["Опыт работы с {s}.", "Использовал {s} в продакшене.", "Уверенно владею {s}.", "Писал сервисы с применением {s}.",
               "Работал с {s} более {y} лет.", "Знаком с {s}.", "Стек: {s}.", "Есть коммерческий опыт с {s}."]
VAC_PHRASES = ["Требуется опыт с {s}.", "Обязательно знание {s}.", "Опыт работы с {s} от {y} лет.", "Ожидаем уверенное владение {s}."]
VAC_NICE = ["Будет плюсом: {s}.", "Желательно знакомство с {s}.", "Плюсом будет опыт с {s}."]
FILLER = ["Ответственный, быстро обучаюсь.", "Участвовал в код-ревью.", "Работал в продуктовой компании.",
          "Есть опыт менторства стажёров.", "Интересуюсь новыми технологиями.", "Готов к переезду.",
          "Компания развивает финтех-продукты.", "Удалённая работа, гибкий график.", "ДМС и обучение за счёт компании."]


def _mention(rng, skill: str, implicit_p: float) -> str:
    s = SKILLS[skill]
    if skill in IMPLICIT and rng.random() < implicit_p:
        return str(rng.choice(IMPLICIT[skill])) + "."
    variants = [s.name] + list(s.aliases)
    w = np.array([3.0] + [1.0] * len(s.aliases))
    return variants[rng.choice(len(variants), p=w / w.sum())]


def make_resume(rng, role: str, quality: float) -> dict:
    must, nice = ROLES[role]
    skills = [s for s in must if rng.random() < quality] + [s for s in nice if rng.random() < 0.45]
    other_role = ROLES[list(ROLES)[rng.integers(len(ROLES))]]
    skills += [s for s in other_role[1] if rng.random() < 0.12 and s not in skills]
    skills = list(dict.fromkeys(skills))
    lines = [f"Резюме. Желаемая должность: {role}."]
    for s in rng.permutation(skills):
        m = _mention(rng, s, 0.25)
        lines.append(m if m.endswith(".") else str(rng.choice(RES_PHRASES)).format(s=m, y=int(rng.integers(1, 7))))
    lines += list(rng.choice(FILLER, 2, replace=False))
    return {"type": "resume", "role": role, "text": " ".join(lines), "skills": sorted(skills)}


def make_vacancy(rng, role: str) -> dict:
    must, nice = ROLES[role]
    nice_sel = [s for s in nice if rng.random() < 0.4]
    lines = [f"Вакансия: {role}."]
    for s in must:
        m = _mention(rng, s, 0.15)
        lines.append(m if m.endswith(".") else str(rng.choice(VAC_PHRASES)).format(s=m, y=int(rng.integers(1, 5))))
    for s in nice_sel:
        lines.append(str(rng.choice(VAC_NICE)).format(s=SKILLS[s].name))
    lines += list(rng.choice(FILLER[6:], 1))
    return {"type": "vacancy", "role": role, "text": " ".join(lines), "skills": sorted(must + nice_sel),
            "must_have": sorted(must), "nice_to_have": sorted(nice_sel)}


def generate(n_resumes=600, n_vacancies=60, seed=1) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    roles = list(ROLES)
    resumes = [make_resume(rng, roles[rng.integers(len(roles))], float(rng.uniform(0.4, 1.0))) | {"id": f"res-{i:04d}"}
               for i in range(n_resumes)]
    vacancies = [make_vacancy(rng, roles[i % len(roles)]) | {"id": f"vac-{i:03d}"} for i in range(n_vacancies)]
    return resumes, vacancies


def fit_label(resume: dict, vacancy: dict, thr: float = 0.8) -> int:
    must = set(vacancy["must_have"])
    return int(len(must & set(resume["skills"])) / len(must) >= thr)


if __name__ == "__main__":
    r, v = generate()
    Path("data").mkdir(exist_ok=True)
    Path("data/resumes.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in r))
    Path("data/vacancies.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in v))
