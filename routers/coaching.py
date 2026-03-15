"""
Real-time coaching endpoints.

POST /coach reads the latest pose from the CV pipeline, evaluates it against
the requested skill definition, calls Claude Haiku for a coaching tip, records
the rep, and returns the result. Target latency: < 50ms for evaluation;
the Claude call adds ~200ms but runs async so the event loop is not blocked.

POST /evaluate accepts an explicit pose payload for offline testing or
integration without a live camera.
"""

from typing import Optional

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, Body

import anthropic

from cv.pipeline import pipeline
from database import operations as db
from skills.engine import evaluate_skill, build_coaching_prompt
from services.claude import async_client
from schemas.compliance import build_compliance_metadata

router = APIRouter(tags=["Coaching"])


# ── Request / Response schemas ────────────────────────────────────────────────

class PoseData(BaseModel):
    """Explicit joint angle payload for POST /evaluate.

    All angles are in degrees [0°–180°]. Use this endpoint for testing
    skill definitions without a live camera feed.

    Attributes:
        left_elbow: Interior angle at the left elbow.
        right_elbow: Interior angle at the right elbow.
        left_shoulder: Angle at the left shoulder.
        right_shoulder: Angle at the right shoulder.
        left_hip: Angle at the left hip.
        right_hip: Angle at the right hip.
        left_knee: Angle at the left knee.
        right_knee: Angle at the right knee.
        left_ankle: Angle at the left ankle.
        right_ankle: Angle at the right ankle.
        spine: Sagittal spine angle.
    """
    left_elbow:     float
    right_elbow:    float
    left_shoulder:  float
    right_shoulder: float
    left_hip:       float
    right_hip:      float
    left_knee:      float
    right_knee:     float
    left_ankle:     float
    right_ankle:    float
    spine:          float


class CoachRequest(BaseModel):
    """Optional request body for POST /coach.

    When ``angles`` is provided (sent by the client-side CV pipeline), those
    values are used directly. When omitted, angles are read from the server-side
    camera pipeline as a fallback.

    Attributes:
        angles: Dict of joint name → angle in degrees from the client.
        phase: Detected movement phase string (e.g. 'standing', 'mid_lift').
        skill_id: Skill to evaluate (overrides query param if provided).
    """
    angles:   Optional[dict] = None
    phase:    Optional[str]  = None
    skill_id: Optional[str]  = None


class EvaluationResponse(BaseModel):
    """Response returned by both /coach and /evaluate.

    Attributes:
        coaching_tip: Single-sentence coaching cue from Claude Haiku.
        is_good_form: True when zero form_rule violations were detected.
        score: Integer 0–100 from the skill score_formula.
    """
    coaching_tip: str
    is_good_form: bool
    score:        int


# ── Shared coaching logic ─────────────────────────────────────────────────────

async def _evaluate_and_coach(
    angles:     dict,
    skill_id:   str,
    session_id: Optional[str],
    worker_id:  Optional[str],
    org_id:     Optional[str],
    frame_index: int = 0,
) -> EvaluationResponse:
    """Core evaluation + coaching pipeline shared by /coach and /evaluate.

    Evaluates the pose against the skill definition, calls Claude Haiku for
    a coaching tip, writes a rep record (if session_id provided), and returns
    the evaluation response.

    Args:
        angles: Dict of joint name → angle in degrees.
        skill_id: Skill definition identifier.
        session_id: Optional session UUID. If provided, the rep is persisted.
        worker_id: Optional worker UUID (required for compliance blob).
        org_id: Optional org UUID (required for compliance blob).
        frame_index: CV pipeline frame counter at time of capture.

    Returns:
        EvaluationResponse with coaching_tip, is_good_form, and score.

    Raises:
        HTTPException(404): If the skill definition does not exist.
        HTTPException(503): If the Claude API is unavailable.
    """
    skill_row = db.get_skill(skill_id)
    if not skill_row:
        raise HTTPException(status_code=404, detail=f"Unknown skill: {skill_id}")
    definition = skill_row["definition"]

    is_good, score, violations = evaluate_skill(definition, angles)

    # ── Gather context for intelligent coaching ───────────────────────────────
    worker_name    = None
    session_number = 1
    rep_history    = []

    if session_id:
        session = db.get_session(session_id)
        if session:
            rep_history = session.get("reps", [])

    if worker_id:
        worker = db.get_worker(worker_id)
        if worker:
            worker_name = worker.get("name")
            # Count how many completed sessions this worker has had for this skill
            all_sessions = db.list_worker_sessions(worker_id, skill_id)
            session_number = len(all_sessions) + 1  # +1 for current in-progress

    prompt = build_coaching_prompt(
        definition=definition,
        pose=angles,
        violations=violations,
        rep_history=rep_history,
        worker_name=worker_name,
        session_number=session_number,
    )

    try:
        message = await async_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        coaching_tip = message.content[0].text.strip()
    except anthropic.APIError as exc:
        raise HTTPException(status_code=503, detail=f"Claude API error: {exc}") from exc

    if session_id:
        # Build compliance blob and persist the rep.
        compliance = None
        if worker_id and org_id:
            compliance = build_compliance_metadata(
                session_id=session_id,
                worker_id=worker_id,
                skill_id=skill_id,
                org_id=org_id,
                angles=angles,
                score=score,
                is_good_form=is_good,
                violations=violations,
                coaching_tip=coaching_tip,
                frame_index=frame_index,
            )
        db.record_rep(
            session_id=session_id,
            score=score,
            is_good_form=is_good,
            angles=angles,
            coaching_tip=coaching_tip,
            compliance_json=compliance.model_dump_json() if compliance else None,
            confidence=compliance.overall_confidence if compliance else None,
        )

    return EvaluationResponse(
        coaching_tip=coaching_tip,
        is_good_form=is_good,
        score=score,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/coach", response_model=EvaluationResponse, summary="Real-time coaching tick")
async def coach(
    body:       CoachRequest  = Body(default=CoachRequest()),
    session_id: Optional[str] = None,
    skill_id:   str           = "lifting",
):
    """Evaluates a completed rep and returns a coaching cue.

    Called by the frontend once per detected rep (movement cycle completion).
    Uses angles from the request body when provided by the client-side CV
    pipeline; falls back to the server-side camera pipeline otherwise.

    Args:
        body: Optional angles/phase from the client-side MediaPipe pipeline.
        session_id: Optional session UUID. If provided, the rep is recorded.
        skill_id: Skill to evaluate against. Defaults to ``"lifting"``.

    Returns:
        EvaluationResponse with coaching_tip, is_good_form, and score.

    Raises:
        HTTPException(503): If no pose data is available from any source.
    """
    # Prefer client-provided angles (more accurate, privacy-first).
    # Fall back to server-side camera pipeline if body is empty.
    angles      = body.angles or pipeline.get_angles()
    frame_index = pipeline.frame_index
    skill_id    = body.skill_id or skill_id

    if not angles:
        raise HTTPException(
            status_code=503,
            detail="No pose detected. Verify the worker is visible in the camera frame.",
        )

    # Resolve worker/org for compliance blob when session is active.
    worker_id = org_id = None
    if session_id:
        session = db.get_session(session_id)
        if session:
            worker   = db.get_worker(session["worker_id"])
            worker_id = session["worker_id"]
            org_id    = worker["org_id"] if worker else None

    return await _evaluate_and_coach(
        angles=angles,
        skill_id=skill_id,
        session_id=session_id,
        worker_id=worker_id,
        org_id=org_id,
        frame_index=frame_index,
    )


@router.post("/evaluate", response_model=EvaluationResponse, summary="Evaluate an explicit pose payload")
async def evaluate_pose(
    pose:       PoseData,
    session_id: Optional[str] = None,
    skill_id:   str           = "lifting",
):
    """Evaluates an explicitly provided pose against a skill definition.

    Use this endpoint for integration testing, replay of recorded sessions,
    or evaluation without a live camera feed.

    Args:
        pose: Explicit joint angle values.
        session_id: Optional session UUID for rep recording.
        skill_id: Skill to evaluate against.

    Returns:
        EvaluationResponse with coaching_tip, is_good_form, and score.
    """
    return await _evaluate_and_coach(
        angles=pose.model_dump(),
        skill_id=skill_id,
        session_id=session_id,
        worker_id=None,
        org_id=None,
    )
