import uuid
import json
import jwt
import os
import secrets
from datetime import datetime, timezone, timedelta
from .db import get_connection

CERT_SECRET = os.getenv("CERT_SECRET", "changeme-set-in-env")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ── Orgs ─────────────────────────────────────────────────────

def create_org(name: str) -> dict:
    org = {
        "id": str(uuid.uuid4()),
        "name": name,
        "api_key": "sk-" + str(uuid.uuid4()).replace("-", ""),
        "join_code": secrets.token_hex(3).upper(),
        "created_at": now_iso(),
    }
    conn = get_connection()
    conn.execute(
        "INSERT INTO orgs (id, name, api_key, join_code, created_at) VALUES (?, ?, ?, ?, ?)",
        (org["id"], org["name"], org["api_key"], org["join_code"], org["created_at"])
    )
    conn.commit()
    conn.close()
    return org


def get_org(org_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM orgs WHERE id = ?", (org_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_org_by_api_key(api_key: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM orgs WHERE api_key = ?", (api_key,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_org_by_join_code(join_code: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM orgs WHERE join_code = ?", (join_code.upper(),)).fetchone()
    conn.close()
    return dict(row) if row else None


def save_debrief(session_id: str, text: str):
    conn = get_connection()
    conn.execute("UPDATE sessions SET debrief = ? WHERE id = ?", (text, session_id))
    conn.commit()
    conn.close()


# ── Workers ───────────────────────────────────────────────────

def create_worker(org_id: str, name: str) -> dict:
    worker = {
        "id": str(uuid.uuid4()),
        "org_id": org_id,
        "name": name,
        "created_at": now_iso(),
    }
    conn = get_connection()
    conn.execute(
        "INSERT INTO workers (id, org_id, name, created_at) VALUES (?, ?, ?, ?)",
        (worker["id"], worker["org_id"], worker["name"], worker["created_at"])
    )
    conn.commit()
    conn.close()
    return worker


def get_worker(worker_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM workers WHERE id = ?", (worker_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_workers_by_org(org_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM workers WHERE org_id = ?", (org_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Sessions ──────────────────────────────────────────────────

def create_session(worker_id: str, skill_id: str) -> dict:
    """Creates a new training session in PENDING state.

    Args:
        worker_id: UUID of the worker starting the session.
        skill_id: Identifier of the skill to be trained.

    Returns:
        Session dict with ``state: PENDING`` and all audit fields.
    """
    session = {
        "id":         str(uuid.uuid4()),
        "worker_id":  worker_id,
        "skill_id":   skill_id,
        "state":      "PENDING",
        "started_at": now_iso(),
        "ended_at":   None,
        "avg_score":  None,
        "rep_count":  0,
    }
    conn = get_connection()
    conn.execute(
        "INSERT INTO sessions (id, worker_id, skill_id, state, started_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (session["id"], session["worker_id"], session["skill_id"],
         session["state"], session["started_at"]),
    )
    conn.commit()
    conn.close()
    return session


def update_session_state(session_id: str, new_state: str) -> None:
    """Persists a validated FSM state transition to the sessions table.

    Also appends a timestamped entry to the ``state_history`` JSON array
    for the audit trail.

    Args:
        session_id: UUID of the session to update.
        new_state: The target state (already validated by SessionFSM).
    """
    conn = get_connection()
    row  = conn.execute(
        "SELECT state_history FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()

    history: list = []
    if row and row["state_history"]:
        try:
            history = json.loads(row["state_history"])
        except (json.JSONDecodeError, TypeError):
            history = []

    history.append({"state": new_state, "at": now_iso()})

    conn.execute(
        "UPDATE sessions SET state = ?, state_history = ? WHERE id = ?",
        (new_state, json.dumps(history), session_id),
    )
    conn.commit()
    conn.close()


def end_session(session_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT AVG(score) as avg, COUNT(*) as cnt FROM reps WHERE session_id = ?",
        (session_id,)
    ).fetchone()

    avg_score = round(row["avg"], 1) if row["avg"] is not None else 0
    rep_count = row["cnt"]
    ended_at = now_iso()

    conn.execute(
        "UPDATE sessions SET ended_at = ?, avg_score = ?, rep_count = ? WHERE id = ?",
        (ended_at, avg_score, rep_count, session_id)
    )
    conn.commit()

    session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(session) if session else None


def get_session(session_id: str) -> dict | None:
    conn = get_connection()
    session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not session:
        conn.close()
        return None
    reps = conn.execute(
        "SELECT * FROM reps WHERE session_id = ? ORDER BY created_at", (session_id,)
    ).fetchall()
    conn.close()
    result = dict(session)
    result["reps"] = [dict(r) for r in reps]
    return result


def get_sessions_by_worker(worker_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sessions WHERE worker_id = ? ORDER BY started_at DESC",
        (worker_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_worker_sessions(worker_id: str, skill_id: str) -> list[dict]:
    """Returns all completed sessions for a worker on a specific skill.

    Used by the coaching engine to determine how experienced the worker is
    with this skill, so coaching tips can be calibrated accordingly.

    Args:
        worker_id: UUID of the worker.
        skill_id: Skill identifier to filter by.

    Returns:
        List of completed session dicts, oldest first.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sessions WHERE worker_id = ? AND skill_id = ? AND state = 'COMPLETED' ORDER BY started_at ASC",
        (worker_id, skill_id)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Reps ──────────────────────────────────────────────────────

def record_rep(
    session_id:      str,
    score:           int,
    is_good_form:    bool,
    angles:          dict,
    coaching_tip:    str,
    compliance_json: str | None = None,
    confidence:      float | None = None,
) -> dict:
    """Records a single evaluated rep to the audit log.

    Args:
        session_id: UUID of the parent training session.
        score: Integer 0–100 from the skill score_formula.
        is_good_form: True when no form_rule violations were detected.
        angles: Dict of joint name → angle in degrees.
        coaching_tip: Claude-generated coaching sentence for this rep.
        compliance_json: Optional serialized ComplianceMetadata blob.
            When provided, enables full post-hoc audit reconstruction.
        confidence: Mean landmark visibility score [0.0–1.0].
            Stored as a separate indexed column for fast certification queries.

    Returns:
        The created rep dict.
    """
    rep = {
        "id":              str(uuid.uuid4()),
        "session_id":      session_id,
        "score":           score,
        "is_good_form":    int(is_good_form),
        "angles_json":     json.dumps(angles),
        "coaching_tip":    coaching_tip,
        "compliance_json": compliance_json,
        "confidence":      confidence,
        "created_at":      now_iso(),
    }
    conn = get_connection()
    conn.execute(
        "INSERT INTO reps "
        "(id, session_id, score, is_good_form, angles_json, coaching_tip, "
        " compliance_json, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (rep["id"], rep["session_id"], rep["score"], rep["is_good_form"],
         rep["angles_json"], rep["coaching_tip"],
         rep["compliance_json"], rep["confidence"], rep["created_at"]),
    )
    conn.commit()
    conn.close()
    return rep


# ── Skills ───────────────────────────────────────────────────

def upsert_skill(definition: dict, org_id: str | None = None):
    conn = get_connection()
    existing = conn.execute("SELECT id FROM skills WHERE id = ?", (definition["skill_id"],)).fetchone()
    if existing:
        conn.execute(
            "UPDATE skills SET definition = ? WHERE id = ?",
            (json.dumps(definition), definition["skill_id"])
        )
    else:
        conn.execute(
            "INSERT INTO skills (id, org_id, definition, created_at) VALUES (?, ?, ?, ?)",
            (definition["skill_id"], org_id, json.dumps(definition), now_iso())
        )
    conn.commit()
    conn.close()


def get_skill(skill_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    result["definition"] = json.loads(result["definition"])
    return result


def list_skills(org_id: str | None = None) -> list[dict]:
    conn = get_connection()
    # Built-in skills (org_id IS NULL) + org's own skills
    if org_id:
        rows = conn.execute(
            "SELECT * FROM skills WHERE org_id IS NULL OR org_id = ?", (org_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM skills WHERE org_id IS NULL").fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d["definition"] = json.loads(d["definition"])
        results.append(d)
    return results


# ── Certifications ───────────────────────────────────────────

def get_worker_skill_history(worker_id: str, skill_id: str) -> dict:
    """Returns all completed sessions + reps for a worker/skill combo."""
    conn = get_connection()
    sessions = conn.execute("""
        SELECT * FROM sessions
        WHERE worker_id = ? AND skill_id = ? AND ended_at IS NOT NULL
        ORDER BY started_at
    """, (worker_id, skill_id)).fetchall()

    result = []
    total_reps = 0
    good_reps = 0

    for s in sessions:
        reps = conn.execute(
            "SELECT * FROM reps WHERE session_id = ?", (s["id"],)
        ).fetchall()
        total_reps += len(reps)
        good_reps += sum(1 for r in reps if r["is_good_form"])
        result.append({"session": dict(s), "reps": [dict(r) for r in reps]})

    conn.close()
    return {
        "sessions": result,
        "total_reps": total_reps,
        "good_reps": good_reps,
    }


def issue_certification(worker_id: str, skill_id: str, org_id: str,
                        avg_score: float, good_form_rate: float,
                        session_ids: list[str], valid_days: int) -> dict:
    cert_id = str(uuid.uuid4())
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(days=valid_days)

    payload = {
        "cert_id": cert_id,
        "worker_id": worker_id,
        "skill_id": skill_id,
        "org_id": org_id,
        "avg_score": avg_score,
        "good_form_rate": good_form_rate,
        "issued_at": issued_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "exp": int(expires_at.timestamp()),
    }
    token = jwt.encode(payload, CERT_SECRET, algorithm="HS256")

    cert = {
        "id": cert_id,
        "worker_id": worker_id,
        "skill_id": skill_id,
        "org_id": org_id,
        "avg_score": avg_score,
        "good_form_rate": good_form_rate,
        "sessions_used": json.dumps(session_ids),
        "issued_at": issued_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "token": token,
    }

    conn = get_connection()
    conn.execute("""
        INSERT INTO certifications
        (id, worker_id, skill_id, org_id, avg_score, good_form_rate, sessions_used, issued_at, expires_at, token)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (cert["id"], cert["worker_id"], cert["skill_id"], cert["org_id"],
          cert["avg_score"], cert["good_form_rate"], cert["sessions_used"],
          cert["issued_at"], cert["expires_at"], cert["token"]))
    conn.commit()
    conn.close()
    return cert


def verify_certification(cert_id: str) -> dict:
    conn = get_connection()
    row = conn.execute("SELECT * FROM certifications WHERE id = ?", (cert_id,)).fetchone()
    conn.close()

    if not row:
        return {"valid": False, "reason": "Certificate not found"}

    cert = dict(row)
    try:
        jwt.decode(cert["token"], CERT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return {"valid": False, "reason": "Certificate has expired", "cert_id": cert_id,
                "expired_at": cert["expires_at"]}
    except jwt.InvalidTokenError:
        return {"valid": False, "reason": "Certificate signature is invalid", "cert_id": cert_id}

    worker = get_worker(cert["worker_id"])
    return {
        "valid": True,
        "cert_id": cert_id,
        "worker_name": worker["name"] if worker else "Unknown",
        "skill_id": cert["skill_id"],
        "avg_score": cert["avg_score"],
        "good_form_rate": cert["good_form_rate"],
        "issued_at": cert["issued_at"],
        "expires_at": cert["expires_at"],
        "sessions_evaluated": len(json.loads(cert["sessions_used"])),
    }


def get_latest_certification(worker_id: str, skill_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM certifications WHERE worker_id = ? AND skill_id = ? ORDER BY issued_at DESC LIMIT 1",
        (worker_id, skill_id)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_certifications_by_worker(worker_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM certifications WHERE worker_id = ? ORDER BY issued_at DESC",
        (worker_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Analytics ─────────────────────────────────────────────────

def get_org_analytics(org_id: str) -> dict:
    conn = get_connection()

    workers = conn.execute(
        "SELECT COUNT(*) as cnt FROM workers WHERE org_id = ?", (org_id,)
    ).fetchone()["cnt"]

    sessions = conn.execute("""
        SELECT COUNT(*) as cnt, AVG(s.avg_score) as avg
        FROM sessions s
        JOIN workers w ON w.id = s.worker_id
        WHERE w.org_id = ? AND s.ended_at IS NOT NULL
    """, (org_id,)).fetchone()

    top_workers = conn.execute("""
        SELECT w.id, w.name, AVG(s.avg_score) as avg_score, COUNT(s.id) as sessions
        FROM workers w
        JOIN sessions s ON s.worker_id = w.id
        WHERE w.org_id = ? AND s.ended_at IS NOT NULL
        GROUP BY w.id
        ORDER BY avg_score DESC
        LIMIT 5
    """, (org_id,)).fetchall()

    conn.close()
    return {
        "org_id": org_id,
        "total_workers": workers,
        "total_sessions": sessions["cnt"] or 0,
        "avg_score_across_sessions": round(sessions["avg"], 1) if sessions["avg"] else None,
        "top_workers": [dict(r) for r in top_workers],
    }


# ── Programs ───────────────────────────────────────────────────

def create_program(name: str, description: str, skill_ids: list, org_id: str | None = None) -> dict:
    program = {
        "id": str(uuid.uuid4()),
        "org_id": org_id,
        "name": name,
        "description": description,
        "skill_ids": skill_ids,
        "created_at": now_iso(),
    }
    conn = get_connection()
    conn.execute(
        "INSERT INTO programs (id, org_id, name, description, skill_ids, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (program["id"], org_id, name, description, json.dumps(skill_ids), program["created_at"])
    )
    conn.commit()
    conn.close()
    return program


def get_program(program_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM programs WHERE id = ?", (program_id,)).fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    result["skill_ids"] = json.loads(result["skill_ids"])
    return result


def list_programs(org_id: str | None = None) -> list[dict]:
    conn = get_connection()
    if org_id:
        rows = conn.execute(
            "SELECT * FROM programs WHERE org_id IS NULL OR org_id = ? ORDER BY created_at DESC",
            (org_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM programs WHERE org_id IS NULL ORDER BY created_at DESC").fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d["skill_ids"] = json.loads(d["skill_ids"])
        results.append(d)
    return results
