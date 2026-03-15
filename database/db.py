import sqlite3
import os
import secrets

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "skill2.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS orgs (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            api_key     TEXT NOT NULL UNIQUE,
            join_code   TEXT,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS workers (
            id          TEXT PRIMARY KEY,
            org_id      TEXT NOT NULL REFERENCES orgs(id),
            name        TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            worker_id   TEXT NOT NULL REFERENCES workers(id),
            skill_id    TEXT NOT NULL,
            started_at  TEXT NOT NULL,
            ended_at    TEXT,
            avg_score   REAL,
            rep_count   INTEGER DEFAULT 0,
            debrief     TEXT
        );

        CREATE TABLE IF NOT EXISTS reps (
            id           TEXT PRIMARY KEY,
            session_id   TEXT NOT NULL REFERENCES sessions(id),
            score        INTEGER NOT NULL,
            is_good_form INTEGER NOT NULL,
            angles_json  TEXT NOT NULL,
            coaching_tip TEXT NOT NULL,
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS skills (
            id         TEXT PRIMARY KEY,
            org_id     TEXT,
            definition TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS certifications (
            id              TEXT PRIMARY KEY,
            worker_id       TEXT NOT NULL REFERENCES workers(id),
            skill_id        TEXT NOT NULL,
            org_id          TEXT NOT NULL REFERENCES orgs(id),
            avg_score       REAL NOT NULL,
            good_form_rate  REAL NOT NULL,
            sessions_used   TEXT NOT NULL,
            issued_at       TEXT NOT NULL,
            expires_at      TEXT NOT NULL,
            token           TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS programs (
            id          TEXT PRIMARY KEY,
            org_id      TEXT,
            name        TEXT NOT NULL,
            description TEXT,
            skill_ids   TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );
    """)

    conn.commit()
    _migrate(conn)
    conn.close()


def _migrate(conn):
    """Applies additive schema migrations idempotently.

    Each migration is attempted individually; SQLite raises an error if the
    column already exists, which is silently swallowed. This makes the
    migration list append-only and safe to re-run on every startup.

    Args:
        conn: An open SQLite connection (must have WAL mode already set).
    """
    migrations = [
        # Historic migrations
        "ALTER TABLE orgs     ADD COLUMN join_code      TEXT",
        "ALTER TABLE sessions ADD COLUMN debrief        TEXT",
        # FSM state tracking (v2)
        "ALTER TABLE sessions ADD COLUMN state          TEXT DEFAULT 'ACTIVE'",
        "ALTER TABLE sessions ADD COLUMN state_history  TEXT",
        # Compliance audit blob (v2)
        "ALTER TABLE reps     ADD COLUMN compliance_json TEXT",
        "ALTER TABLE reps     ADD COLUMN confidence      REAL",
    ]
    for sql in migrations:
        try:
            conn.execute(sql)
            conn.commit()
        except Exception:
            pass  # column already exists — safe to ignore

    # Backfill join codes for orgs that pre-date the join_code column.
    rows = conn.execute("SELECT id FROM orgs WHERE join_code IS NULL").fetchall()
    for row in rows:
        conn.execute(
            "UPDATE orgs SET join_code = ? WHERE id = ?",
            (secrets.token_hex(3).upper(), row["id"]),
        )
    if rows:
        conn.commit()
