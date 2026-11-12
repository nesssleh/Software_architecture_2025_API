-- Кейсы расследований (PostgreSQL). Таблица также создаётся автоматически через SQLAlchemy.
CREATE TABLE IF NOT EXISTS fraud_cases (
    case_id          SERIAL PRIMARY KEY,
    return_id        VARCHAR(32) NOT NULL UNIQUE,      -- идемпотентное открытие кейса
    customer_id      VARCHAR(32) NOT NULL,
    score            DOUBLE PRECISION NOT NULL,
    decision         VARCHAR(16) NOT NULL,             -- review | block
    reasons          JSONB NOT NULL,
    status           VARCHAR(16) NOT NULL DEFAULT 'open',
    analyst_verdict  VARCHAR(16),
    analyst_comment  TEXT,
    amount           DOUBLE PRECISION NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_cases_status_score ON fraud_cases (status, score DESC);  -- очередь аналитика
CREATE INDEX IF NOT EXISTS ix_cases_customer ON fraud_cases (customer_id);
