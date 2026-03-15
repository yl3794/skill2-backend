"""
Training session lifecycle endpoints with Finite State Machine enforcement.

Every write to the ``sessions`` table goes through ``SessionFSM.transition()``,
which validates the state edge before the database is updated. Illegal
transitions return HTTP 409 Conflict with a precise error message.

State flow:
    POST /sessions/start         → creates session in PENDING state
    POST /sessions/{id}/activate → PENDING → ACTIVE (called by coaching loop)
    POST /sessions/{id}/pause    → ACTIVE  → PAUSED
    POST /sessions/{id}/resume   → PAUSED  → ACTIVE
    POST /sessions/{id}/end      → ACTIVE  → COMPLETING → COMPLETED
    POST /sessions/{id}/debrief  → reads COMPLETED session; calls Claude
    GET  /sessions/{id}          → returns current state + full rep list
"""

from fastapi import APIRouter, HTTPException, Depends

from database import operations as db
from auth import require_org, assert_org_owns_session
from schemas.session import SessionState, StartSessionRequest, fsm
from services.claude import async_client

router = APIRouter(prefix="/sessions", tags=["Sessions"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_session_or_404(session_id: str) -> dict:
    """Fetches a session or raises HTTP 404.

    Args:
        session_id: UUID of the session to fetch.

    Returns:
        Session dict including ``reps`` list.

    Raises:
        HTTPException(404): If no session exists with this ID.
    """
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _fsm_transition(session: dict, target: SessionState) -> None:
    """Validates and applies an FSM transition, updating the DB.

    Args:
        session: Current session dict (must have ``id`` and ``state`` keys).
        target: Desired next state.

    Raises:
        HTTPException(409): If the transition is not permitted by the FSM.
    """
    try:
        new_state = fsm.transition(session.get("state", SessionState.ACTIVE), target)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    db.update_session_state(session["id"], new_state)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start", summary="Start a new training session")
async def start_session(
    body:        StartSessionRequest,
    current_org: dict = Depends(require_org),
):
    """Creates a new training session in PENDING state.

    A PENDING session does not record reps until the coaching loop confirms
    a pose is detected (POST /sessions/{id}/activate). This prevents
    empty sessions from inflating session counts.

    Args:
        body: Payload with ``worker_id`` and ``skill_id``.
        current_org: Injected by ``require_org``.

    Returns:
        Created session dict including the initial ``state: PENDING``.

    Raises:
        HTTPException(404): If the worker does not exist.
        HTTPException(403): If the worker belongs to a different org.
    """
    from auth import assert_org_owns_worker
    worker = db.get_worker(body.worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    assert_org_owns_worker(worker, current_org)
    return db.create_session(body.worker_id, body.skill_id)


@router.post("/{session_id}/activate", summary="Transition PENDING → ACTIVE")
async def activate_session(
    session_id:  str,
    current_org: dict = Depends(require_org),
):
    """Transitions a session from PENDING to ACTIVE.

    Called by the coaching loop as soon as a pose is detected in-frame.
    Only ACTIVE sessions accept rep recordings.

    Args:
        session_id: UUID of the session.
        current_org: Injected by ``require_org``.

    Returns:
        Dict with ``state: ACTIVE``.

    Raises:
        HTTPException(404): If session not found.
        HTTPException(403): If session belongs to a different org.
        HTTPException(409): If the transition is not permitted by the FSM.
    """
    session = _get_session_or_404(session_id)
    assert_org_owns_session(session, current_org, db.get_worker)
    _fsm_transition(session, SessionState.ACTIVE)
    return {"session_id": session_id, "state": SessionState.ACTIVE}


@router.post("/{session_id}/pause", summary="Pause an active session")
async def pause_session(
    session_id:  str,
    current_org: dict = Depends(require_org),
):
    """Transitions a session from ACTIVE to PAUSED.

    No reps are recorded while paused. Use POST /resume to continue.

    Args:
        session_id: UUID of the session.
        current_org: Injected by ``require_org``.

    Returns:
        Dict with ``state: PAUSED``.

    Raises:
        HTTPException(409): If the session is not currently ACTIVE.
    """
    session = _get_session_or_404(session_id)
    assert_org_owns_session(session, current_org, db.get_worker)
    _fsm_transition(session, SessionState.PAUSED)
    return {"session_id": session_id, "state": SessionState.PAUSED}


@router.post("/{session_id}/resume", summary="Resume a paused session")
async def resume_session(
    session_id:  str,
    current_org: dict = Depends(require_org),
):
    """Transitions a session from PAUSED back to ACTIVE.

    Args:
        session_id: UUID of the session.
        current_org: Injected by ``require_org``.

    Returns:
        Dict with ``state: ACTIVE``.

    Raises:
        HTTPException(409): If the session is not currently PAUSED.
    """
    session = _get_session_or_404(session_id)
    assert_org_owns_session(session, current_org, db.get_worker)
    _fsm_transition(session, SessionState.ACTIVE)
    return {"session_id": session_id, "state": SessionState.ACTIVE}


@router.post("/{session_id}/end", summary="End a session and compute final stats")
async def end_session(
    session_id:  str,
    current_org: dict = Depends(require_org),
):
    """Closes a session and writes final aggregate statistics.

    Transitions: ACTIVE → COMPLETING → COMPLETED (both transitions happen
    within this request so the COMPLETING state is transient).

    Args:
        session_id: UUID of the session.
        current_org: Injected by ``require_org``.

    Returns:
        Completed session dict with ``avg_score``, ``rep_count``, and
        ``state: COMPLETED``.

    Raises:
        HTTPException(404): If session not found.
        HTTPException(403): If session belongs to a different org.
        HTTPException(409): If the session is not in a state that permits ending.
    """
    session = _get_session_or_404(session_id)
    assert_org_owns_session(session, current_org, db.get_worker)

    # Lock the session against concurrent writes while stats are computed.
    _fsm_transition(session, SessionState.COMPLETING)
    result = db.end_session(session_id)

    # Mark as fully closed once the DB write completes.
    db.update_session_state(session_id, SessionState.COMPLETED)
    if result:
        result["state"] = SessionState.COMPLETED
    return result


@router.get("/{session_id}", summary="Get session with rep history")
async def get_session(
    session_id:  str,
    current_org: dict = Depends(require_org),
):
    """Returns a session record including the full rep history.

    Args:
        session_id: UUID of the session.
        current_org: Injected by ``require_org``.

    Returns:
        Session dict with nested ``reps`` list.

    Raises:
        HTTPException(404): If session not found.
        HTTPException(403): If session belongs to a different org.
    """
    session = _get_session_or_404(session_id)
    assert_org_owns_session(session, current_org, db.get_worker)
    return session


@router.post("/{session_id}/debrief", summary="Generate AI coaching debrief")
async def generate_debrief(
    session_id:  str,
    current_org: dict = Depends(require_org),
):
    """Generates a personalized coaching debrief using Claude Haiku.

    Analyses the session's rep history and produces a 2–3 sentence debrief:
    performance summary, positive reinforcement, and one concrete improvement tip.
    The debrief is persisted to the session record and returned.

    Args:
        session_id: UUID of the session. Must be in COMPLETED state.
        current_org: Injected by ``require_org``.

    Returns:
        Dict with ``debrief`` key containing the coaching text.

    Raises:
        HTTPException(404): If session not found.
        HTTPException(403): If session belongs to a different org.
        HTTPException(409): If the session is not yet COMPLETED.
    """
    session = _get_session_or_404(session_id)
    assert_org_owns_session(session, current_org, db.get_worker)

    state = session.get("state", "")
    if state not in (SessionState.COMPLETED, ""):
        # Allow empty string for backward-compat with sessions that predate FSM.
        if state:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Debrief requires a COMPLETED session. "
                    f"Current state: {state}"
                ),
            )

    reps = session.get("reps", [])
    if not reps:
        return {"debrief": "No reps were recorded in this session."}

    worker     = db.get_worker(session["worker_id"])
    skill_row  = db.get_skill(session["skill_id"])
    skill_name = skill_row["definition"].get("display_name", session["skill_id"]) if skill_row else session["skill_id"]

    scores      = [r["score"] for r in reps]
    good_count  = sum(1 for r in reps if r["is_good_form"])
    avg         = round(sum(scores) / len(scores), 1)
    trend       = (
        "improving"  if len(scores) > 2 and scores[-1] > scores[0] + 5 else
        "declining"  if len(scores) > 2 and scores[-1] < scores[0] - 5 else
        "consistent"
    )
    violations  = [r["coaching_tip"] for r in reps if not r["is_good_form"]]
    recent_issues = "; ".join(set(violations[-3:])) if violations else ""

    prompt = (
        f"You are a professional physical skills coach reviewing a training session.\n\n"
        f"Worker: {worker['name'] if worker else 'Unknown'}\n"
        f"Skill: {skill_name}\n"
        f"Reps analyzed: {len(reps)}\n"
        f"Good form: {good_count}/{len(reps)} ({round(good_count / len(reps) * 100)}%)\n"
        f"Average score: {avg}/100\n"
        f"Performance trend: {trend}\n"
        f"{'Recent issues: ' + recent_issues if recent_issues else 'No form issues detected.'}\n\n"
        "Write a personalized 2–3 sentence coaching debrief. Be specific about their "
        "performance, mention what they did well, and give one concrete improvement tip. "
        "Be encouraging but honest. No markdown, no asterisks."
    )

    message = await async_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text
    db.save_debrief(session_id, text)
    return {"debrief": text}
