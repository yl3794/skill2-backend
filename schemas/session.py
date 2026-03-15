"""
Session schemas and Finite State Machine for training session lifecycle.

A session transitions through defined states. Every state change is validated
against the allowed transition table before the database is written. Illegal
transitions raise an HTTP 409 Conflict so the client always knows the reason.

State diagram:
    PENDING ──► ACTIVE ──► COMPLETING ──► COMPLETED
                  │ ▲          │
                  ▼ │       (terminal)
                PAUSED
                  │
              ABANDONED (terminal — idle timeout or explicit abort)

    PENDING is the initial state set by POST /sessions/start.
    ACTIVE is set immediately when the first coaching frame is processed
    (i.e., when the CV pipeline confirms a pose is detected in-frame).
    COMPLETING is a short-lived write-lock state set by POST /sessions/{id}/end
    before avg_score is computed. The router sets COMPLETED when the DB write
    succeeds.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


# ── State machine ─────────────────────────────────────────────────────────────

class SessionState(StrEnum):
    """Enumeration of valid training session lifecycle states.

    Attributes:
        PENDING: Session created; worker not yet confirmed in frame.
        ACTIVE: Worker detected; reps are being recorded.
        PAUSED: Operator paused; no reps recorded until resumed.
        COMPLETING: End request received; final stats being written.
        COMPLETED: Session fully closed; eligible for debrief and certification.
        ABANDONED: Session expired without completion (idle timeout or abort).
    """

    PENDING    = "PENDING"
    ACTIVE     = "ACTIVE"
    PAUSED     = "PAUSED"
    COMPLETING = "COMPLETING"
    COMPLETED  = "COMPLETED"
    ABANDONED  = "ABANDONED"


# Allowed (from_state, to_state) transitions. Any pair not in this set is illegal.
_ALLOWED: frozenset[tuple[SessionState, SessionState]] = frozenset({
    (SessionState.PENDING,    SessionState.ACTIVE),
    (SessionState.PENDING,    SessionState.COMPLETING),  # end before first rep
    (SessionState.PENDING,    SessionState.ABANDONED),
    (SessionState.ACTIVE,     SessionState.PAUSED),
    (SessionState.ACTIVE,     SessionState.COMPLETING),
    (SessionState.ACTIVE,     SessionState.ABANDONED),
    (SessionState.PAUSED,     SessionState.ACTIVE),
    (SessionState.PAUSED,     SessionState.ABANDONED),
    (SessionState.COMPLETING, SessionState.COMPLETED),
    (SessionState.COMPLETING, SessionState.ABANDONED),
})

# Terminal states — no outbound transitions permitted.
_TERMINAL: frozenset[SessionState] = frozenset({
    SessionState.COMPLETED,
    SessionState.ABANDONED,
})


class SessionFSM:
    """Validates and executes training session state transitions.

    This class is stateless — it operates on the ``state`` value read from the
    database. Call ``transition()`` before every ``UPDATE sessions SET state``
    to ensure only legal transitions are written.

    Example:
        >>> fsm = SessionFSM()
        >>> fsm.transition("ACTIVE", "COMPLETING")  # returns SessionState.COMPLETING
        >>> fsm.transition("COMPLETED", "ACTIVE")   # raises ValueError
    """

    def transition(
        self,
        current: str | SessionState,
        target: str | SessionState,
    ) -> SessionState:
        """Validates a requested state transition and returns the target state.

        Args:
            current: The session's present state (from the database).
            target: The desired next state.

        Returns:
            The validated ``target`` as a ``SessionState`` enum member.

        Raises:
            ValueError: If the transition is not in the allowed table, if
                ``current`` is a terminal state, or if either value is not a
                recognized ``SessionState``.
        """
        try:
            from_state = SessionState(current)
            to_state   = SessionState(target)
        except ValueError as exc:
            raise ValueError(f"Unknown session state: {exc}") from exc

        if from_state in _TERMINAL:
            raise ValueError(
                f"Session is in terminal state '{from_state}' — "
                "no further transitions are permitted."
            )

        if (from_state, to_state) not in _ALLOWED:
            raise ValueError(
                f"Illegal transition: {from_state} → {to_state}. "
                f"Allowed targets from {from_state}: "
                + str([t for (f, t) in _ALLOWED if f == from_state])
            )

        return to_state


# Module-level singleton — import this directly in routers.
fsm = SessionFSM()


# ── Request / Response schemas ────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    """Request body for POST /sessions/start.

    Attributes:
        worker_id: UUID of the worker starting the session.
        skill_id: Identifier of the skill to be trained.
    """

    worker_id: str = Field(min_length=1, description="Worker UUID")
    skill_id:  str = Field(min_length=1, description="Skill identifier")


class SessionResponse(BaseModel):
    """Response body for session read/mutate endpoints.

    Attributes:
        id: Session UUID.
        worker_id: Worker UUID.
        skill_id: Skill identifier.
        state: Current FSM state.
        started_at: ISO-8601 UTC start timestamp.
        ended_at: ISO-8601 UTC end timestamp, or None if still active.
        avg_score: Mean score across all recorded reps, or None.
        rep_count: Number of coaching frames recorded in this session.
        debrief: Claude-generated coaching debrief, or None.
    """

    id:         str
    worker_id:  str
    skill_id:   str
    state:      str
    started_at: str
    ended_at:   Optional[str]  = None
    avg_score:  Optional[float] = None
    rep_count:  int = 0
    debrief:    Optional[str]  = None
