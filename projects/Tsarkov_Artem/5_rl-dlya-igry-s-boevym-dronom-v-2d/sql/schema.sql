-- Итоги эпизодов (PostgreSQL). Таблица также создаётся SQLAlchemy при старте сервисов.
CREATE TABLE IF NOT EXISTS episodes (
    episode_id VARCHAR(32) PRIMARY KEY,           -- идемпотентная запись при повторной доставке
    policy     VARCHAR(32) NOT NULL,
    scenario   VARCHAR(32) NOT NULL,
    outcome    VARCHAR(32) NOT NULL,
    success    BOOLEAN NOT NULL,
    steps      INTEGER NOT NULL,
    ret        DOUBLE PRECISION NOT NULL,
    hits       INTEGER NOT NULL,
    shots      INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_episodes_policy ON episodes (policy);
CREATE INDEX IF NOT EXISTS ix_episodes_scenario ON episodes (scenario);
CREATE INDEX IF NOT EXISTS ix_episodes_created_at ON episodes (created_at);
