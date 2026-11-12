CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           VARCHAR(64) PRIMARY KEY,
    type         VARCHAR(16) NOT NULL CHECK (type IN ('resume', 'vacancy')),
    title        TEXT,
    text         TEXT NOT NULL,
    skills       JSONB NOT NULL DEFAULT '[]',     -- [{name, category, evidence, method, confidence}]
    must_have    TEXT[] NOT NULL DEFAULT '{}',
    nice_to_have TEXT[] NOT NULL DEFAULT '{}',
    embedding    vector(300),                      -- ru_core_news_md; 384 для multilingual-e5-small
    status       VARCHAR(16) NOT NULL DEFAULT 'processed',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- ANN-поиск по косинусной близости
CREATE INDEX IF NOT EXISTS ix_documents_embedding ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS ix_documents_type ON documents (type);

-- Инвертированный индекс навыков: кандидатогенерация «у кого есть навыки вакансии»
CREATE TABLE IF NOT EXISTS document_skills (
    doc_id     VARCHAR(64) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    skill      VARCHAR(64) NOT NULL,
    method     VARCHAR(16) NOT NULL,              -- gazetteer | semantic | llm
    confidence REAL NOT NULL,
    PRIMARY KEY (doc_id, skill)
);
CREATE INDEX IF NOT EXISTS ix_document_skills_skill ON document_skills (skill);
