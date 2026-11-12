-- TicketFlow: PostgreSQL (таблицы также создаются SQLAlchemy при старте)
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id     VARCHAR(32) PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL,
    channel       VARCHAR(16) NOT NULL,
    customer_tier VARCHAR(16) NOT NULL,
    text          TEXT NOT NULL,
    category      VARCHAR(32),
    category_conf DOUBLE PRECISION DEFAULT 0,
    priority      VARCHAR(16),
    priority_conf DOUBLE PRECISION DEFAULT 0,
    team          VARCHAR(32),
    status        VARCHAR(16) NOT NULL,
    tags          JSONB NOT NULL DEFAULT '[]',
    incident_id   VARCHAR(32),
    model         VARCHAR(64)
);
CREATE INDEX IF NOT EXISTS ix_tickets_team_status ON tickets (team, status);    -- очередь команды
CREATE INDEX IF NOT EXISTS ix_tickets_priority_created ON tickets (priority, created_at);
CREATE INDEX IF NOT EXISTS ix_tickets_incident ON tickets (incident_id) WHERE incident_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS ticket_audit (                -- журнал команд (Command pattern)
    id        BIGSERIAL PRIMARY KEY,
    ticket_id VARCHAR(32) NOT NULL,
    at        VARCHAR(40) NOT NULL,
    actor     VARCHAR(32) NOT NULL,
    command   VARCHAR(32) NOT NULL,
    details   JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_audit_ticket ON ticket_audit (ticket_id, id);
