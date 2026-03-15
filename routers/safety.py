"""
Real-time safety monitoring endpoints.

Combines stateless skill-engine form checks (lifting, overhead reach,
awkward posture) with time-based and pattern-based tracker state
(spinal twisting, repetitive motion RSI). All hazard data is read from
the CVPipeline singleton — no DB queries.
"""

import time

from fastapi import APIRouter

from cv.pipeline import pipeline
from database import operations as db
from skills.engine import evaluate_skill

router = APIRouter(prefix="/safety", tags=["Safety"])


# ── Internal helpers ──────────────────────────────────────────────────────────

def _overall_status(alerts: list) -> str:
    """Derives an aggregate status label from a list of alert dicts.

    Args:
        alerts: List of alert dicts, each with a ``severity`` key.

    Returns:
        ``"danger"`` if any alert is high-severity, ``"caution"`` if any is
        medium-severity, otherwise ``"safe"``.
    """
    severities = {a.get("severity") for a in alerts}
    if "high"   in severities:
        return "danger"
    if "medium" in severities:
        return "caution"
    return "safe"


def _max_severity(alerts: list) -> str | None:
    """Returns the highest severity level present in a list of alerts.

    Args:
        alerts: List of alert dicts.

    Returns:
        ``"high"``, ``"medium"``, ``"low"``, or ``None`` if list is empty.
    """
    if not alerts:
        return None
    if any(a.get("severity") == "high"   for a in alerts):
        return "high"
    if any(a.get("severity") == "medium" for a in alerts):
        return "medium"
    return "low"


def _lifting_alerts(angles: dict) -> list:
    """Runs a stateless lifting form check against the current joint angles.

    Args:
        angles: Dict of joint name → angle in degrees from the CV pipeline.

    Returns:
        List of alert dicts (type, severity, message). Empty list if no
        violations or if the lifting skill definition is unavailable.
    """
    if not angles:
        return []
    try:
        lifting_row = db.get_skill("lifting")
        if not lifting_row:
            return []
        _, _, violations = evaluate_skill(lifting_row["definition"], angles)
        return [
            {
                "type":     "lifting",
                "severity": "high",
                "message":  v.get("violation_tip", "Lifting form issue"),
            }
            for v in violations
        ]
    except Exception:
        return []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status", summary="Real-time safety status")
async def safety_status():
    """Returns the latest combined safety status from the CV pipeline.

    Merges stateless skill-engine checks (lifting, overhead reach, awkward
    posture) with stateful SafetyTracker results (spinal twist, RSI).

    Returns:
        Dict with ``overall_status``, ``alerts``, ``overhead_duration``,
        ``cycle_counts``, and ``current_angles``.
    """
    sr     = pipeline.get_safety()
    angles = pipeline.get_angles()

    tracker_alerts = sr.get("alerts", [])
    lift_alerts    = _lifting_alerts(angles)
    all_alerts     = lift_alerts + tracker_alerts

    return {
        "timestamp":        time.time(),
        "overall_status":   _overall_status(all_alerts),
        "alerts":           all_alerts,
        "overhead_duration": sr.get("overhead_duration", {"left": 0.0, "right": 0.0}),
        "cycle_counts":     sr.get("cycle_counts", {}),
        "current_angles":   angles,
    }


@router.get("/dashboard", summary="Per-hazard safety dashboard")
async def safety_dashboard():
    """Returns structured data for a five-card safety hazard dashboard.

    Each hazard type is represented as a card dict with ``id``, ``name``,
    ``description``, ``active``, ``severity``, and ``alerts``. Designed
    for direct consumption by a frontend card-per-hazard layout.

    Returns:
        Dict with ``overall_status``, ``active_hazard_count``, ``hazards``
        list, and ``current_angles``.
    """
    sr     = pipeline.get_safety()
    angles = pipeline.get_angles()

    tracker_alerts = sr.get("alerts", [])
    overhead       = sr.get("overhead_duration", {"left": 0.0, "right": 0.0})
    cycles         = sr.get("cycle_counts", {})

    by_type: dict[str, list] = {}
    for a in tracker_alerts:
        by_type.setdefault(a["type"], []).append(a)

    lift_alerts = _lifting_alerts(angles)

    hazards = [
        {
            "id":          "lifting",
            "name":        "Improper Lifting",
            "description": "Bending at waist instead of knees — spine injury risk",
            "icon":        "weight",
            "alerts":      lift_alerts,
            "active":      bool(lift_alerts),
            "severity":    _max_severity(lift_alerts),
        },
        {
            "id":                         "overhead_reach",
            "name":                       "Sustained Overhead Work",
            "description":                "Arms above shoulder level — rotator cuff injury risk",
            "icon":                       "arms-up",
            "alerts":                     by_type.get("overhead_reach", []),
            "active":                     bool(by_type.get("overhead_reach")),
            "severity":                   _max_severity(by_type.get("overhead_reach", [])),
            "overhead_duration_left_s":   overhead.get("left",  0.0),
            "overhead_duration_right_s":  overhead.get("right", 0.0),
        },
        {
            "id":          "awkward_posture",
            "name":        "Awkward Kneeling / Squatting",
            "description": "Extreme knee bend + hunched back — knee & spine injury risk",
            "icon":        "squat",
            "alerts":      by_type.get("awkward_posture", []),
            "active":      bool(by_type.get("awkward_posture")),
            "severity":    _max_severity(by_type.get("awkward_posture", [])),
        },
        {
            "id":          "spinal_twist",
            "name":        "Twisting While Carrying",
            "description": "Rotating spine under load — #1 cause of back injuries",
            "icon":        "twist",
            "alerts":      by_type.get("spinal_twist", []),
            "active":      bool(by_type.get("spinal_twist")),
            "severity":    _max_severity(by_type.get("spinal_twist", [])),
        },
        {
            "id":           "repetitive_motion",
            "name":         "Repetitive Motion",
            "description":  "Same movement pattern cycling — RSI risk",
            "icon":         "repeat",
            "alerts":       by_type.get("repetitive_motion", []),
            "active":       bool(by_type.get("repetitive_motion")),
            "severity":     _max_severity(by_type.get("repetitive_motion", [])),
            "cycle_counts": {j: c for j, c in cycles.items() if c > 0},
        },
    ]

    all_alerts = lift_alerts + tracker_alerts
    return {
        "timestamp":           time.time(),
        "overall_status":      _overall_status(all_alerts),
        "active_hazard_count": sum(1 for h in hazards if h["active"]),
        "hazards":             hazards,
        "current_angles":      angles,
    }


@router.post("/reset-twist-baseline", summary="Re-calibrate spinal twist detector")
async def reset_twist_baseline():
    """Resets the spinal-twist neutral baseline so the detector re-calibrates.

    Call this when a worker repositions to a new neutral stance. The twist
    detector will collect the next 30 frames as the new baseline before
    alerting again.

    Returns:
        ``{"reset": True}``
    """
    pipeline.reset_twist_baseline()
    return {"reset": True}
