"""
Computer-vision pipeline: MediaPipe Pose estimation, angle computation,
safety tracking, and MJPEG frame generation.

All shared mutable state (angle_data, safety state) that was previously
scattered as globals in main.py is encapsulated in ``CVPipeline``. Routers
that need live pose data import the ``pipeline`` singleton from this module.

Thread safety:
    ``get_angles()`` and ``get_safety()`` acquire short-lived locks and return
    copies — callers never hold a reference into the internal dicts.
"""

from __future__ import annotations

import threading
from typing import Any

import cv2
import mediapipe as mp

from angle import calculate_angle
from safety.tracker import SafetyTracker


class CVPipeline:
    """Encapsulates the MediaPipe pose pipeline and shared CV state.

    Attributes:
        angle_data: Latest computed joint angles (degrees), keyed by joint name.
        latest_safety: Latest SafetyTracker result dict.
        frame_index: Monotonic counter incremented on every processed frame.

    Example:
        >>> from cv.pipeline import pipeline
        >>> angles = pipeline.get_angles()
        >>> safety = pipeline.get_safety()
    """

    def __init__(self, camera_index: int = 1) -> None:
        self._angle_data:    dict[str, float] = {}
        self._latest_safety: dict[str, Any]   = {}
        self._lock         = threading.Lock()
        self._safety_lock  = threading.Lock()
        self.frame_index   = 0
        self._camera_index = camera_index

        self._safety_tracker = SafetyTracker()

        # MediaPipe solution handles
        self._mp_pose    = mp.solutions.pose
        self._mp_hands   = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils

        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap.set(cv2.CAP_PROP_FPS,          30)

    # ── Public accessors (thread-safe copies) ─────────────────────────────────

    def get_angles(self) -> dict[str, float]:
        """Returns a copy of the latest joint angle dict.

        Returns:
            Dict mapping joint name → angle in degrees. Empty dict if no
            pose has been detected since startup.
        """
        with self._lock:
            return dict(self._angle_data)

    def get_safety(self) -> dict[str, Any]:
        """Returns a copy of the latest SafetyTracker result.

        Returns:
            Dict with keys ``alerts``, ``overhead_duration``, ``cycle_counts``.
            Empty dict if no frames have been processed yet.
        """
        with self._safety_lock:
            return dict(self._latest_safety)

    def reset_twist_baseline(self) -> None:
        """Resets the spinal-twist detector so it re-calibrates from the next frame.

        Call this when a worker repositions to a new neutral stance.
        """
        self._safety_tracker.reset_twist_baseline()

    # ── MJPEG frame generator ─────────────────────────────────────────────────

    def generate_frames(self):
        """Yields MJPEG boundary-delimited JPEG bytes for StreamingResponse.

        Runs the full MediaPipe Pose + Hands pipeline on every other frame to
        balance latency against CPU load. Angles and safety state are updated
        inside the same loop so all consumers read consistent data.

        Yields:
            Bytes chunks in ``multipart/x-mixed-replace`` format.
        """
        local_frame_count = 0
        last_pose_results = None
        last_hand_results = None
        cached_angles:  dict[str, float]    = {}
        cached_coords:  dict[str, list[int]] = {}

        with self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose, self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:

            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break

                local_frame_count += 1
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                if local_frame_count % 2 == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_pose_results = pose.process(rgb)
                    last_hand_results = hands.process(rgb)

                    if last_pose_results.pose_landmarks:
                        lms = last_pose_results.pose_landmarks.landmark
                        MP  = self._mp_pose.PoseLandmark

                        def _px(idx):
                            lm = lms[idx]
                            return [int(lm.x * w), int(lm.y * h)]

                        nose       = _px(MP.NOSE.value)
                        l_shoulder = _px(MP.LEFT_SHOULDER.value)
                        r_shoulder = _px(MP.RIGHT_SHOULDER.value)
                        l_elbow    = _px(MP.LEFT_ELBOW.value)
                        r_elbow    = _px(MP.RIGHT_ELBOW.value)
                        l_wrist    = _px(MP.LEFT_WRIST.value)
                        r_wrist    = _px(MP.RIGHT_WRIST.value)
                        l_hip      = _px(MP.LEFT_HIP.value)
                        r_hip      = _px(MP.RIGHT_HIP.value)
                        l_knee     = _px(MP.LEFT_KNEE.value)
                        r_knee     = _px(MP.RIGHT_KNEE.value)
                        l_ankle    = _px(MP.LEFT_ANKLE.value)
                        r_ankle    = _px(MP.RIGHT_ANKLE.value)
                        l_foot     = _px(MP.LEFT_FOOT_INDEX.value)
                        r_foot     = _px(MP.RIGHT_FOOT_INDEX.value)

                        mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) // 2,
                                        (l_shoulder[1] + r_shoulder[1]) // 2]
                        mid_hip      = [(l_hip[0] + r_hip[0]) // 2,
                                        (l_hip[1] + r_hip[1]) // 2]

                        cached_coords = {
                            "nose": nose,
                            "l_shoulder": l_shoulder, "r_shoulder": r_shoulder,
                            "l_elbow": l_elbow,       "r_elbow": r_elbow,
                            "l_wrist": l_wrist,       "r_wrist": r_wrist,
                            "l_hip": l_hip,           "r_hip": r_hip,
                            "l_knee": l_knee,         "r_knee": r_knee,
                            "l_ankle": l_ankle,       "r_ankle": r_ankle,
                            "l_foot": l_foot,         "r_foot": r_foot,
                            "mid_shoulder": mid_shoulder,
                            "mid_hip":      mid_hip,
                        }

                        cached_angles = {
                            "left_elbow":     calculate_angle(l_shoulder, l_elbow, l_wrist),
                            "right_elbow":    calculate_angle(r_shoulder, r_elbow, r_wrist),
                            "left_shoulder":  calculate_angle(l_hip,      l_shoulder, l_elbow),
                            "right_shoulder": calculate_angle(r_hip,      r_shoulder, r_elbow),
                            "left_hip":       calculate_angle(l_shoulder, l_hip,    l_knee),
                            "right_hip":      calculate_angle(r_shoulder, r_hip,    r_knee),
                            "left_knee":      calculate_angle(l_hip,      l_knee,   l_ankle),
                            "right_knee":     calculate_angle(r_hip,      r_knee,   r_ankle),
                            "left_ankle":     calculate_angle(l_knee,     l_ankle,  l_foot),
                            "right_ankle":    calculate_angle(r_knee,     r_ankle,  r_foot),
                            "spine":          calculate_angle(nose, mid_shoulder, mid_hip),
                        }

                        with self._lock:
                            self._angle_data = cached_angles
                            self.frame_index = local_frame_count

                        safety_result = self._safety_tracker.update(
                            cached_angles, cached_coords
                        )
                        with self._safety_lock:
                            self._latest_safety = safety_result

                # ── Overlay: skeleton ─────────────────────────────────────────
                if last_pose_results and last_pose_results.pose_landmarks:
                    self._mp_drawing.draw_landmarks(
                        frame,
                        last_pose_results.pose_landmarks,
                        self._mp_pose.POSE_CONNECTIONS,
                        self._mp_drawing.DrawingSpec(
                            color=(0, 255, 255), thickness=2, circle_radius=4
                        ),
                        self._mp_drawing.DrawingSpec(
                            color=(255, 255, 255), thickness=2
                        ),
                    )

                if cached_coords and cached_angles:
                    cv2.line(
                        frame,
                        tuple(cached_coords["mid_shoulder"]),
                        tuple(cached_coords["mid_hip"]),
                        (0, 165, 255), 2,
                    )
                    cv2.putText(
                        frame, f"SP: {cached_angles['spine']:.0f}°",
                        tuple(cached_coords["mid_shoulder"]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                    )
                    cv2.putText(
                        frame, f"LK: {cached_angles['left_knee']:.0f}°",
                        tuple(cached_coords["l_knee"]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                    )
                    cv2.putText(
                        frame, f"RK: {cached_angles['right_knee']:.0f}°",
                        tuple(cached_coords["r_knee"]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                    )

                # ── Overlay: hands ────────────────────────────────────────────
                if last_hand_results and last_hand_results.multi_hand_landmarks:
                    for hand_lms in last_hand_results.multi_hand_landmarks:
                        self._mp_drawing.draw_landmarks(
                            frame, hand_lms, self._mp_hands.HAND_CONNECTIONS
                        )

                # ── Overlay: safety panel ─────────────────────────────────────
                with self._safety_lock:
                    _sr = dict(self._latest_safety)
                if _sr:
                    _draw_safety_overlay(frame, _sr)

                # ── Encode and yield ──────────────────────────────────────────
                _, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50]
                )
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )


# ── Safety overlay renderer ───────────────────────────────────────────────────

def _draw_safety_overlay(frame, safety_result: dict) -> None:
    """Draws status badge, alert list, overhead bars, and cycle counts onto frame.

    Args:
        frame: BGR OpenCV frame to draw on (mutated in place).
        safety_result: Dict returned by ``SafetyTracker.update()``.
    """
    alerts = safety_result.get("alerts", [])
    h, w   = frame.shape[:2]

    severities = {a.get("severity") for a in alerts}
    if "high" in severities:
        status_color, status_text = (0, 0, 200), "DANGER"
    elif "medium" in severities:
        status_color, status_text = (0, 120, 255), "CAUTION"
    else:
        status_color, status_text = (0, 180, 50), "SAFE"

    bx = w - 190
    cv2.rectangle(frame, (bx, 8), (w - 8, 40), status_color, -1)
    cv2.rectangle(frame, (bx, 8), (w - 8, 40), (255, 255, 255), 1)
    cv2.putText(
        frame, f"SAFETY: {status_text}", (bx + 6, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )

    ay = 50
    for alert in alerts[:6]:
        col = (60, 60, 255) if alert.get("severity") == "high" else (40, 140, 255)
        msg = alert.get("message", "")[:54]
        cv2.rectangle(frame, (w - 330, ay), (w - 8, ay + 20), (25, 25, 25), -1)
        cv2.putText(
            frame, f"! {msg}", (w - 325, ay + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1,
        )
        ay += 24

    overhead = safety_result.get("overhead_duration", {})
    for i, (side, dur) in enumerate(overhead.items()):
        if dur <= 0:
            continue
        bx2   = 10 + i * 210
        label = f"{side.title()} overhead: {dur:.0f}s"
        cv2.putText(
            frame, label, (bx2, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1,
        )
        cv2.rectangle(frame, (bx2, h - 20), (bx2 + 190, h - 8), (60, 60, 60), -1)
        fill    = min(int(dur / 60.0 * 190), 190)
        bar_col = (0, 0, 200) if dur > 30 else (0, 120, 255) if dur >= 5 else (0, 180, 50)
        if fill > 0:
            cv2.rectangle(frame, (bx2, h - 20), (bx2 + fill, h - 8), bar_col, -1)

    cycles        = safety_result.get("cycle_counts", {})
    active_cycles = {j: c for j, c in cycles.items() if c > 0}
    if active_cycles:
        cy = h - 8 - len(active_cycles) * 18
        for joint, count in active_cycles.items():
            col   = (0, 120, 255) if count >= 12 else (0, 180, 50)
            label = f"{joint.replace('_', ' ').title()}: {count}/min"
            cv2.putText(
                frame, label, (w - 220, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1,
            )
            cy += 18


# ── Singleton ─────────────────────────────────────────────────────────────────

# All routers import this object. It initialises the camera at import time,
# matching the existing behaviour of the monolithic main.py.
pipeline = CVPipeline(camera_index=1)
