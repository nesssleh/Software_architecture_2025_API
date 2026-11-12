-- Аналитическое хранилище ReturnGuard (ClickHouse)
CREATE TABLE IF NOT EXISTS order_events
(
    event_id    String,
    ts          DateTime64(3, 'UTC'),
    type        LowCardinality(String),
    customer_id String,
    order_id    String,
    amount      Float64,
    payload     String CODEC(ZSTD(3))
)
ENGINE = ReplacingMergeTree            -- дедупликация повторных доставок по event_id
PARTITION BY toYYYYMM(ts)
ORDER BY (customer_id, ts, event_id)   -- быстрый таймлайн клиента
TTL toDateTime(ts) + INTERVAL 3 YEAR;  -- retention

CREATE TABLE IF NOT EXISTS return_scores
(
    return_id     String,
    ts            DateTime64(3, 'UTC'),
    customer_id   String,
    score         Float32,
    decision      LowCardinality(String),
    decided_by    LowCardinality(String),
    model_version LowCardinality(String),
    amount        Float64,
    INDEX ix_decision decision TYPE set(3) GRANULARITY 4
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (ts, return_id)
TTL toDateTime(ts) + INTERVAL 3 YEAR;

-- Витрина для дашборда: дневные агрегаты считаются при вставке
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_decisions
ENGINE = SummingMergeTree ORDER BY (day, decision)
AS SELECT toDate(ts) AS day, decision, count() AS n, sum(amount) AS amount
FROM return_scores GROUP BY day, decision;
