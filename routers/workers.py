"""
Worker management and progress tracking endpoints.

A Worker belongs to exactly one Org. Workers authenticate implicitly via
their org's API key — the org key is required to read or modify any worker
record. Worker self-registration (POST /workers/register) is the only
unauthenticated write path, and it requires a valid org join code.
"""

from typing import Optional

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends

from database import operations as db
from auth import require_org, assert_org_owns_worker

router = APIRouter(prefix="/workers", tags=["Workers"])


# ── Request schemas ───────────────────────────────────────────────────────────

class CreateWorkerRequest(BaseModel):
    """Request body for POST /workers (org-managed creation).

    Attributes:
        org_id: UUID of the org this worker belongs to.
        name: Worker's full name.
    """
    org_id: str
    name:   str


class WorkerRegisterRequest(BaseModel):
    """Request body for POST /workers/register (worker self-registration).

    Attributes:
        name: Worker's full name.
        join_code: Optional 6-character org join code. If omitted, a personal
            workspace org is created automatically.
    """
    name:      str
    join_code: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", summary="Worker self-registration")
async def register_worker(body: WorkerRegisterRequest):
    """Registers a new worker, optionally joining an existing org.

    No authentication required — this is the entry point for workers on their
    first day. If a valid ``join_code`` is provided, the worker is attached to
    that org. Otherwise, a personal workspace org is created.

    Args:
        body: Registration payload with name and optional join code.

    Returns:
        Dict with ``worker`` record, ``api_key`` (org key), and ``org_name``.

    Raises:
        HTTPException(400): If the join code does not match any org.
    """
    if body.join_code:
        org = db.get_org_by_join_code(body.join_code)
        if not org:
            raise HTTPException(status_code=400, detail="Invalid join code")
    else:
        org = db.create_org(f"{body.name.strip()}'s Workspace")

    worker = db.create_worker(org["id"], body.name.strip())
    return {"worker": worker, "api_key": org["api_key"], "org_name": org["name"]}


@router.post("", summary="Create a worker (org-managed)")
async def create_worker(
    body:        CreateWorkerRequest,
    current_org: dict = Depends(require_org),
):
    """Creates a new worker record under the authenticated org.

    Args:
        body: Worker creation payload.
        current_org: Injected by the ``require_org`` dependency.

    Returns:
        Created worker dict.

    Raises:
        HTTPException(403): If ``body.org_id`` does not match the authenticated org.
    """
    if body.org_id != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return db.create_worker(body.org_id, body.name)


@router.get("/{worker_id}", summary="Get worker with session stats")
async def get_worker(worker_id: str, current_org: dict = Depends(require_org)):
    """Returns a worker record enriched with aggregate session statistics.

    Args:
        worker_id: UUID of the worker.
        current_org: Injected by the ``require_org`` dependency.

    Returns:
        Worker dict extended with ``total_sessions`` and ``avg_score``.

    Raises:
        HTTPException(404): If the worker does not exist.
        HTTPException(403): If the worker belongs to a different org.
    """
    worker = db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    assert_org_owns_worker(worker, current_org)

    sessions = db.get_sessions_by_worker(worker_id)
    scores   = [s["avg_score"] for s in sessions if s["avg_score"] is not None]
    return {
        **worker,
        "total_sessions": len(sessions),
        "avg_score":      round(sum(scores) / len(scores), 1) if scores else None,
    }


@router.get("/{worker_id}/sessions", summary="List worker sessions")
async def worker_sessions(worker_id: str, current_org: dict = Depends(require_org)):
    """Returns all training sessions for a worker, newest first.

    Args:
        worker_id: UUID of the worker.
        current_org: Injected by the ``require_org`` dependency.

    Returns:
        List of session dicts.

    Raises:
        HTTPException(404): If the worker does not exist.
        HTTPException(403): If the worker belongs to a different org.
    """
    worker = db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    assert_org_owns_worker(worker, current_org)
    return db.get_sessions_by_worker(worker_id)


@router.get("/{worker_id}/progress", summary="Worker progress across all programs")
async def worker_progress(worker_id: str, current_org: dict = Depends(require_org)):
    """Returns per-program, per-skill training progress for a worker.

    For each program assigned to the org, returns skill-level stats:
    sessions completed vs. required, average score, and certification status.

    Args:
        worker_id: UUID of the worker.
        current_org: Injected by the ``require_org`` dependency.

    Returns:
        List of program progress dicts, each containing ``skill_progress``,
        ``completion_pct``, and ``completed_skills``.

    Raises:
        HTTPException(404): If the worker does not exist.
        HTTPException(403): If the worker belongs to a different org.
    """
    worker = db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    assert_org_owns_worker(worker, current_org)

    programs = db.list_programs(org_id=current_org["id"])
    result   = []
    for program in programs:
        skill_progress = []
        for skill_id in program["skill_ids"]:
            skill_row    = db.get_skill(skill_id)
            history      = db.get_worker_skill_history(worker_id, skill_id)
            cert         = db.get_latest_certification(worker_id, skill_id)
            sessions_done = len(history["sessions"])
            scores        = [
                s["session"]["avg_score"] for s in history["sessions"]
                if s["session"]["avg_score"] is not None
            ]
            skill_progress.append({
                "skill_id":     skill_id,
                "display_name": skill_row["definition"].get("display_name", skill_id) if skill_row else skill_id,
                "sessions_done": sessions_done,
                "passed":       cert is not None,
                "certified":    cert is not None,
                "avg_score":    round(sum(scores) / len(scores), 1) if scores else None,
            })

        completed = sum(1 for s in skill_progress if s["certified"])
        total     = len(program["skill_ids"])
        result.append({
            "program_id":       program["id"],
            "program_name":     program["name"],
            "description":      program.get("description", ""),
            "skill_progress":   skill_progress,
            "completed_skills": completed,
            "total_skills":     total,
            "completion_pct":   round(completed / total * 100) if total else 0,
        })
    return result


@router.post("/{worker_id}/certify/{skill_id}", summary="Evaluate certification eligibility")
async def certify_worker(
    worker_id:   str,
    skill_id:    str,
    current_org: dict = Depends(require_org),
):
    """Evaluates whether a worker has met all certification criteria for a skill.

    Aggregates all completed sessions for the worker/skill pair and checks
    them against the skill's ``certification`` thresholds. Issues a JWT-backed
    certificate if all criteria pass.

    Args:
        worker_id: UUID of the worker.
        skill_id: Identifier of the skill to certify.
        current_org: Injected by the ``require_org`` dependency.

    Returns:
        Dict with ``certified: True`` and certificate details on success, or
        ``certified: False`` with a human-readable ``reason`` on failure.

    Raises:
        HTTPException(404): If the worker or skill does not exist.
        HTTPException(400): If the skill has no certification criteria defined.
        HTTPException(403): If the worker belongs to a different org.
    """
    worker = db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    assert_org_owns_worker(worker, current_org)

    skill_row = db.get_skill(skill_id)
    if not skill_row:
        raise HTTPException(status_code=404, detail=f"Unknown skill: {skill_id}")
    criteria = skill_row["definition"].get("certification")
    if not criteria:
        raise HTTPException(
            status_code=400,
            detail=f"Skill '{skill_id}' has no certification criteria defined",
        )

    history    = db.get_worker_skill_history(worker_id, skill_id)
    sessions   = history["sessions"]

    if not sessions:
        return {
            "certified": False,
            "reason": "Complete at least one training session to earn certification.",
        }

    # Evaluate the most recent session only — one passing session is enough.
    latest     = sessions[-1]
    lat_score  = latest["session"]["avg_score"] or 0
    lat_reps   = latest["reps"]
    total_reps = len(lat_reps)
    good_reps  = sum(1 for r in lat_reps if r["is_good_form"])

    avg_score      = round(lat_score, 1)
    good_form_rate = round(good_reps / total_reps, 2) if total_reps > 0 else 0

    failures = []
    if avg_score < criteria["min_avg_score"]:
        failures.append(
            f"Average score {avg_score} below required {criteria['min_avg_score']}"
        )
    if good_form_rate < criteria["min_good_form_rate"]:
        failures.append(
            f"Good form rate {good_form_rate:.0%} below required "
            f"{criteria['min_good_form_rate']:.0%}"
        )

    if failures:
        return {
            "certified":      False,
            "reason":         " | ".join(failures),
            "avg_score":      avg_score,
            "good_form_rate": good_form_rate,
        }

    session_ids = [latest["session"]["id"]]
    cert = db.issue_certification(
        worker_id=worker_id,
        skill_id=skill_id,
        org_id=worker["org_id"],
        avg_score=avg_score,
        good_form_rate=good_form_rate,
        session_ids=session_ids,
        valid_days=criteria["cert_valid_days"],
    )
    return {
        "certified":      True,
        "cert_id":        cert["id"],
        "worker_name":    worker["name"],
        "skill_id":       skill_id,
        "avg_score":      avg_score,
        "good_form_rate": good_form_rate,
        "issued_at":      cert["issued_at"],
        "expires_at":     cert["expires_at"],
        "token":          cert["token"],
    }


@router.get("/{worker_id}/certifications", summary="List worker certifications")
async def worker_certifications(worker_id: str, current_org: dict = Depends(require_org)):
    """Returns all certifications issued to a worker, newest first.

    Args:
        worker_id: UUID of the worker.
        current_org: Injected by the ``require_org`` dependency.

    Returns:
        List of certification dicts including JWT tokens and expiry dates.

    Raises:
        HTTPException(404): If the worker does not exist.
        HTTPException(403): If the worker belongs to a different org.
    """
    worker = db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    assert_org_owns_worker(worker, current_org)
    return db.get_certifications_by_worker(worker_id)
