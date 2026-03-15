"""
Compliance metadata schema — the immutable audit blob attached to every recorded rep.

Every Pass/Fail evaluation writes one ComplianceMetadata blob. This blob is the
ground truth for post-hoc compliance review, liability protection, and HR
certificate generation. It is frozen at write time and must never be mutated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Pose primitives ───────────────────────────────────────────────────────────

class JointVector3D(BaseModel):
    """Normalized 3D world coordinates for a single MediaPipe landmark.

    MediaPipe Pose outputs world-space coordinates centered at the mid-hip.
    All values are metric (meters) but normalized so the mid-hip is origin.

    Attributes:
        x: World-space x coordinate (positive = right).
        y: World-space y coordinate (positive = up).
        z: World-space z coordinate (negative = toward camera).
        visibility: Landmark visibility confidence in [0.0, 1.0].
            Values below 0.5 indicate the joint is likely occluded.
        presence: Landmark presence confidence in [0.0, 1.0].
            Distinguishes "visible but low-confidence" from "not in frame".
    """

    model_config = {"frozen": True}

    x: Annotated[float, Field(ge=-3.0, le=3.0, description="World x (meters, right-positive)")]
    y: Annotated[float, Field(ge=-3.0, le=3.0, description="World y (meters, up-positive)")]
    z: Annotated[float, Field(ge=-3.0, le=3.0, description="World z (meters, negative = toward cam)")]
    visibility: Annotated[float, Field(ge=0.0, le=1.0, description="Landmark visibility score")]
    presence: Annotated[float, Field(ge=0.0, le=1.0, description="Landmark presence score")]


class JointAngles(BaseModel):
    """Computed 2D joint angles derived from pixel-space landmark triplets.

    Each angle is the interior angle at the named joint, computed via the
    dot-product formula on the two connecting bone vectors. Units: degrees.

    Attributes:
        left_elbow: Interior angle at the left elbow [0°–180°].
        right_elbow: Interior angle at the right elbow [0°–180°].
        left_shoulder: Angle at left shoulder (hip → shoulder → elbow).
        right_shoulder: Angle at right shoulder (hip → shoulder → elbow).
        left_hip: Angle at left hip (shoulder → hip → knee).
        right_hip: Angle at right hip (shoulder → hip → knee).
        left_knee: Angle at left knee (hip → knee → ankle).
        right_knee: Angle at right knee (hip → knee → ankle).
        left_ankle: Angle at left ankle (knee → ankle → foot_index).
        right_ankle: Angle at right ankle (knee → ankle → foot_index).
        spine: Sagittal spine angle (nose → mid_shoulder → mid_hip).
            >150° = upright; <120° = significantly hunched.
    """

    model_config = {"frozen": True}

    left_elbow:     Annotated[float, Field(ge=0.0, le=180.0)]
    right_elbow:    Annotated[float, Field(ge=0.0, le=180.0)]
    left_shoulder:  Annotated[float, Field(ge=0.0, le=180.0)]
    right_shoulder: Annotated[float, Field(ge=0.0, le=180.0)]
    left_hip:       Annotated[float, Field(ge=0.0, le=180.0)]
    right_hip:      Annotated[float, Field(ge=0.0, le=180.0)]
    left_knee:      Annotated[float, Field(ge=0.0, le=180.0)]
    right_knee:     Annotated[float, Field(ge=0.0, le=180.0)]
    left_ankle:     Annotated[float, Field(ge=0.0, le=180.0)]
    right_ankle:    Annotated[float, Field(ge=0.0, le=180.0)]
    spine:          Annotated[float, Field(ge=0.0, le=180.0)]


# Canonical landmark names that map to JointVector3D entries.
LandmarkName = Literal[
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
    "left_foot_index", "right_foot_index",
]

# Minimum per-landmark visibility required for this rep to be audit-eligible.
CONFIDENCE_GATE: float = 0.6


# ── Compliance audit blob ─────────────────────────────────────────────────────

class ComplianceMetadata(BaseModel):
    """Full audit blob attached to every recorded training rep.

    This is the immutable record stored in the ``reps`` table as
    ``compliance_json``. It satisfies the audit-trail requirement: every
    Pass/Fail decision is fully reproducible from this blob alone.

    The blob is frozen (``model_config = {"frozen": True}``) so no code path
    can mutate it after construction. Serialize with ``model.model_dump_json()``
    before writing to the database.

    Attributes:
        schema_version: Schema revision for forward-migration. Always ``"1"``.
        captured_at: UTC wall-clock time the source camera frame was processed.
        session_id: UUID of the parent training session (FK → sessions.id).
        worker_id: UUID of the worker being evaluated (FK → workers.id).
        skill_id: Identifier of the skill definition used (FK → skills.id).
        org_id: UUID of the industrial site that owns this record.
        joint_vectors: Dict mapping each tracked landmark to its 3D world
            coordinates and MediaPipe confidence scores.
        joint_angles: Computed 2D angle at each joint in degrees.
        overall_confidence: Arithmetic mean of all landmark visibility scores.
        is_high_confidence: True when ``overall_confidence >= CONFIDENCE_GATE``.
            Only high-confidence reps count toward certification thresholds.
        score: Integer 0–100 produced by the skill ``score_formula``.
        is_good_form: True when zero ``form_rule`` violations were detected.
        violations: List of ``form_rule`` dicts that fired during this rep.
        coaching_tip: Single-sentence coaching cue generated by Claude Haiku.
        model_complexity: MediaPipe ``model_complexity`` setting (0 = lite,
            1 = full, 2 = heavy). Recorded for reproducibility.
        frame_index: Monotonic camera frame counter at capture time.
    """

    model_config = {"frozen": True}

    schema_version: Literal["1"] = "1"
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the source camera frame.",
    )

    # ── Identity ──────────────────────────────────────────────────────────────
    session_id: str = Field(min_length=36, max_length=36)
    worker_id:  str = Field(min_length=36, max_length=36)
    skill_id:   str = Field(min_length=1,  max_length=256)
    org_id:     str = Field(min_length=36, max_length=36)

    # ── Raw pose data ─────────────────────────────────────────────────────────
    joint_vectors: dict[LandmarkName, JointVector3D] = Field(
        description="3D world-space coordinates + confidence for each landmark.",
    )
    joint_angles: JointAngles = Field(
        description="Computed interior angles at each joint in degrees.",
    )

    # ── Confidence gate ───────────────────────────────────────────────────────
    overall_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        description=(
            f"Mean visibility across all joint_vectors. "
            f"Reps below {CONFIDENCE_GATE} are excluded from certification."
        ),
    )
    is_high_confidence: bool = Field(
        description=f"True when overall_confidence >= {CONFIDENCE_GATE}.",
    )

    # ── Evaluation results ────────────────────────────────────────────────────
    score:        Annotated[int, Field(ge=0, le=100)]
    is_good_form: bool
    violations:   list[dict] = Field(
        description="form_rule dicts that fired; empty list when is_good_form=True.",
    )
    coaching_tip: str = Field(min_length=1)

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    model_complexity: Literal[0, 1, 2] = Field(
        default=1,
        description="MediaPipe model_complexity used when this frame was captured.",
    )
    frame_index: Annotated[int, Field(ge=0)] = Field(
        description="Monotonic camera frame counter at time of capture.",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("captured_at")
    @classmethod
    def must_be_utc(cls, v: datetime) -> datetime:
        """Rejects naive datetimes — all timestamps must carry UTC tzinfo.

        Args:
            v: The datetime value to validate.

        Returns:
            The original datetime if it carries timezone info.

        Raises:
            ValueError: If ``v`` is timezone-naive.
        """
        if v.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware (UTC required)")
        return v

    @model_validator(mode="after")
    def validate_confidence_consistency(self) -> "ComplianceMetadata":
        """Ensures ``is_high_confidence`` is consistent with ``overall_confidence``.

        Returns:
            Self, if consistent.

        Raises:
            ValueError: If ``is_high_confidence`` contradicts ``overall_confidence``.
        """
        expected = self.overall_confidence >= CONFIDENCE_GATE
        if self.is_high_confidence != expected:
            raise ValueError(
                f"is_high_confidence={self.is_high_confidence} is inconsistent "
                f"with overall_confidence={self.overall_confidence:.3f} "
                f"(gate={CONFIDENCE_GATE})"
            )
        return self


# ── Builder helper ────────────────────────────────────────────────────────────

def build_compliance_metadata(
    *,
    session_id: str,
    worker_id: str,
    skill_id: str,
    org_id: str,
    angles: dict[str, float],
    score: int,
    is_good_form: bool,
    violations: list[dict],
    coaching_tip: str,
    frame_index: int,
    world_landmarks: list | None = None,
) -> ComplianceMetadata:
    """Constructs a ComplianceMetadata blob from pipeline outputs.

    Called by the coaching router immediately after ``evaluate_skill`` returns.
    When ``world_landmarks`` are unavailable (e.g., unit tests), the
    ``joint_vectors`` dict is populated with zero-vectors at visibility=0.

    Args:
        session_id: UUID of the parent training session.
        worker_id: UUID of the worker being evaluated.
        skill_id: Skill definition identifier.
        org_id: Industrial site UUID.
        angles: Computed joint angles from ``angle.calculate_angle``.
        score: Integer 0–100 from ``evaluate_skill``.
        is_good_form: True when no form_rule violations detected.
        violations: Form-rule dicts that fired.
        coaching_tip: Claude-generated coaching sentence.
        frame_index: Monotonic camera frame counter.
        world_landmarks: Optional list of MediaPipe ``NormalizedLandmark``
            objects from ``pose_world_landmarks``. If None, vectors default
            to zero with visibility=0.

    Returns:
        A frozen, validated ComplianceMetadata instance ready for serialization.
    """
    LANDMARK_MAP: list[tuple[int, LandmarkName]] = [
        (0,  "nose"),
        (11, "left_shoulder"),  (12, "right_shoulder"),
        (13, "left_elbow"),     (14, "right_elbow"),
        (15, "left_wrist"),     (16, "right_wrist"),
        (23, "left_hip"),       (24, "right_hip"),
        (25, "left_knee"),      (26, "right_knee"),
        (27, "left_ankle"),     (28, "right_ankle"),
        (31, "left_foot_index"),(32, "right_foot_index"),
    ]

    _zero = JointVector3D(x=0.0, y=0.0, z=0.0, visibility=0.0, presence=0.0)

    if world_landmarks:
        vectors: dict[LandmarkName, JointVector3D] = {}
        for idx, name in LANDMARK_MAP:
            try:
                lm = world_landmarks[idx]
                vectors[name] = JointVector3D(
                    x=round(lm.x, 4),
                    y=round(lm.y, 4),
                    z=round(lm.z, 4),
                    visibility=round(lm.visibility, 4),
                    presence=round(lm.presence if hasattr(lm, "presence") else lm.visibility, 4),
                )
            except (IndexError, AttributeError):
                vectors[name] = _zero
    else:
        vectors = {name: _zero for _, name in LANDMARK_MAP}

    visibilities = [v.visibility for v in vectors.values()]
    overall_conf = round(sum(visibilities) / len(visibilities), 4) if visibilities else 0.0

    # Build JointAngles — default missing joints to 0.0 (no pose detected).
    joint_angles = JointAngles(
        left_elbow=     angles.get("left_elbow",     0.0),
        right_elbow=    angles.get("right_elbow",    0.0),
        left_shoulder=  angles.get("left_shoulder",  0.0),
        right_shoulder= angles.get("right_shoulder", 0.0),
        left_hip=       angles.get("left_hip",       0.0),
        right_hip=      angles.get("right_hip",      0.0),
        left_knee=      angles.get("left_knee",      0.0),
        right_knee=     angles.get("right_knee",     0.0),
        left_ankle=     angles.get("left_ankle",     0.0),
        right_ankle=    angles.get("right_ankle",    0.0),
        spine=          angles.get("spine",          0.0),
    )

    return ComplianceMetadata(
        session_id=session_id,
        worker_id=worker_id,
        skill_id=skill_id,
        org_id=org_id,
        joint_vectors=vectors,
        joint_angles=joint_angles,
        overall_confidence=overall_conf,
        is_high_confidence=overall_conf >= CONFIDENCE_GATE,
        score=score,
        is_good_form=is_good_form,
        violations=violations,
        coaching_tip=coaching_tip,
        frame_index=frame_index,
    )
