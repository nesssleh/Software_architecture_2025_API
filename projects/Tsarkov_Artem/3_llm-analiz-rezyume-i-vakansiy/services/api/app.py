"""SkillMatch API.

POST /v1/documents               — загрузка резюме/вакансии (JSON с текстом), mode=sync|async
POST /v1/documents/upload        — загрузка файла (txt/pdf/docx)
GET  /v1/documents/{id}          — документ и извлечённые навыки
POST /v1/match                   — сопоставление резюме и вакансии + объяснение пробелов
GET  /v1/vacancies/{id}/candidates?k=10 — лучшие кандидаты (pgvector + rerank)
GET  /v1/skills                  — таксономия
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from skillmatch.obs import instrument
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from skillmatch.factory import build_facade
from skillmatch.kafka_io import make_producer
from skillmatch.parsing.parsers import parse
from skillmatch.taxonomy import SKILLS

app = FastAPI(title="SkillMatch — LLM-анализ резюме и вакансий", version="1.0.0")
instrument(app)


@lru_cache(maxsize=1)
def producer():
    """Один Kafka-producer на процесс (создание producer'а на каждый запрос — это сотни миллисекунд)."""
    return make_producer()


@lru_cache(maxsize=1)
def facade():
    return build_facade()


class DocIn(BaseModel):
    type: Literal["resume", "vacancy"]
    text: str = Field(min_length=20, max_length=50_000)
    id: str | None = None


class MatchIn(BaseModel):
    resume_id: str
    vacancy_id: str
    explain: bool = True


@app.get("/health")
def health():
    f = facade()
    return {"status": "ok", "embedder": f.embedder.name, "explainer": f.explainer.name, "documents": f.repo.count()}


@app.post("/v1/documents", status_code=201)
def create(doc: DocIn, mode: str = Query("sync", pattern="^(sync|async)$")):
    f = facade()
    if mode == "async":
        did = doc.id or f.doc_id(doc.type, doc.text)
        producer().send("documents.ingested", key=did, value={"id": did, "type": doc.type, "text": doc.text})
        return {"id": did, "status": "queued"}
    try:
        return f.ingest(doc.type, doc.text, doc.id).public()
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/v1/documents/upload", status_code=201)
async def upload(type: Literal["resume", "vacancy"] = Form(...), file: UploadFile = File(...)):
    try:
        text = parse(await file.read(), file.filename or "doc.txt")
        return facade().ingest(type, text).public()
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/v1/documents/{doc_id}")
def get_doc(doc_id: str):
    d = facade().repo.get(doc_id)
    if not d:
        raise HTTPException(404, "document not found")
    return d.public()


@app.post("/v1/match")
def match(body: MatchIn):
    try:
        return facade().match(body.resume_id, body.vacancy_id, body.explain)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/v1/vacancies/{vacancy_id}/candidates")
def candidates(vacancy_id: str, k: int = Query(10, ge=1, le=100)):
    try:
        return {"vacancy_id": vacancy_id, "candidates": facade().candidates(vacancy_id, k)}
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/v1/skills")
def skills():
    return [{"name": s.name, "category": s.category, "aliases": list(s.aliases), "related": list(s.related)}
            for s in SKILLS.values()]
