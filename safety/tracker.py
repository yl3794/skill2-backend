"""
Real-time safety hazard tracker for time-based and pattern-based risks
that can't be captured by the stateless skill engine.

Handles:
  - Overhead reaching: sustained shoulder elevation timing
  - Spinal twisting:   shoulder/hip width ratio deviation from baseline
  - Repetitive motion: joint direction-reversal cycle counting (RSI)

Awkward posture is handled by the skill engine (stateless threshold).
"""

import time
from collections import deque
from typing import Optional


class SafetyTracker:
    # Overhead reaching
    OVERHEAD_SHOULDER_THRESHOLD = 90   # degrees — arm at/above shoulder level
    OVERHEAD_ALERT_SECONDS = 5.0       # alert after 5s sustained elevation

    # Spinal twisting (shoulder/hip width ratio method)
    TWIST_CALIBRATION_FRAMES = 30      # frames to establish neutral baseline
    TWIST_DEVIATION_THRESHOLD = 0.15   # 15% deviation from baseline triggers alert

    # Repetitive motion (RSI)
    REPETITION_WINDOW_SECONDS = 60.0   # count cycles per minute
    REPETITION_ALERT_COUNT = 12        # alert at ≥12 cycles/min
    REPETITION_MIN_DELTA = 4.0         # minimum angle change (°) to count as direction reversal

    _TRACKED_JOINTS = ("left_elbow", "right_elbow", "left_shoulder", "right_shoulder")

    def __init__(self) -> None:
        # Overhead state
        self._overhead_start: dict[str, Optional[float]] = {"left": None, "right": None}
        self.overhead_duration: dict[str, float] = {"left": 0.0, "right": 0.0}

        # Twist state
        self._twist_ratios: deque = deque(maxlen=60)
        self._twist_baseline: Optional[float] = None

        # Repetitive motion state
        self._prev_angle: dict[str, Optional[float]] = {j: None for j in self._TRACKED_JOINTS}
        self._direction: dict[str, Optional[str]] = {j: None for j in self._TRACKED_JOINTS}
        self.cycle_counts: dict[str, int] = {j: 0 for j in self._TRACKED_JOINTS}
        self._rep_window_start: float = time.time()

    # ── Public API ────────────────────────────────────────────

    def update(self, angles: dict, coords: dict) -> dict:
        """
        Call once per detected pose frame.
        Returns a status dict with 'alerts', 'overhead_duration', 'cycle_counts'.
        """
        now = time.time()
        alerts: list[dict] = []
        alerts.extend(self._check_overhead(angles, now))
        alerts.extend(self._check_twisting(coords))
        alerts.extend(self._check_repetitive(angles, now))
        return {
            "alerts": alerts,
            "overhead_duration": dict(self.overhead_duration),
            "cycle_counts": dict(self.cycle_counts),
        }

    def reset_twist_baseline(self) -> None:
        """Call when worker repositions so a new neutral baseline is captured."""
        self._twist_ratios.clear()
        self._twist_baseline = None

    # ── Hazard checks ─────────────────────────────────────────

    def _check_overhead(self, angles: dict, now: float) -> list:
        alerts = []
        for side in ("left", "right"):
            angle = angles.get(f"{side}_shoulder", 0.0)
            if angle > self.OVERHEAD_SHOULDER_THRESHOLD:
                if self._overhead_start[side] is None:
                    self._overhead_start[side] = now
                elapsed = now - self._overhead_start[side]
                self.overhead_duration[side] = elapsed
                if elapsed >= self.OVERHEAD_ALERT_SECONDS:
                    alerts.append({
                        "type": "overhead_reach",
                        "severity": "high" if elapsed > 30 else "medium",
                        "side": side,
                        "duration_s": round(elapsed, 1),
                        "message": (
                            f"{side.title()} arm overhead {elapsed:.0f}s"
                            " — rotator cuff injury risk"
                        ),
                    })
            else:
                self._overhead_start[side] = None
                self.overhead_duration[side] = 0.0
        return alerts

    def _check_twisting(self, coords: dict) -> list:
        """
        Detects torso rotation by comparing shoulder-width / hip-width ratio
        against a calibrated neutral baseline.

        When the person twists, one shoulder comes forward while the other
        goes back, narrowing the apparent shoulder span in the 2D camera view.
        """
        if not coords:
            return []

        ls = coords.get("l_shoulder")
        rs = coords.get("r_shoulder")
        lh = coords.get("l_hip")
        rh = coords.get("r_hip")
        if not all([ls, rs, lh, rh]):
            return []

        hip_w = abs(rh[0] - lh[0])
        if hip_w < 10:
            return []

        ratio = abs(rs[0] - ls[0]) / hip_w
        self._twist_ratios.append(ratio)

        # Calibrate baseline from first N frames
        if self._twist_baseline is None:
            if len(self._twist_ratios) >= self.TWIST_CALIBRATION_FRAMES:
                self._twist_baseline = sum(self._twist_ratios) / len(self._twist_ratios)
            return []

        if len(self._twist_ratios) < 5:
            return []

        recent = sum(list(self._twist_ratios)[-5:]) / 5
        deviation = abs(recent - self._twist_baseline) / max(self._twist_baseline, 0.01)

        if deviation > self.TWIST_DEVIATION_THRESHOLD:
            direction = "right" if recent < self._twist_baseline else "left"
            return [{
                "type": "spinal_twist",
                "severity": "high",
                "message": (
                    f"Torso twisting {direction} ({deviation * 100:.0f}%)"
                    " — #1 cause of back injuries"
                ),
                "deviation_pct": round(deviation * 100, 1),
                "direction": direction,
            }]
        return []

    def _check_repetitive(self, angles: dict, now: float) -> list:
        """
        Counts direction reversals in joint angle trajectories.
        Each up→down transition = 1 completed cycle.
        Alerts when cycles-per-minute exceed threshold.
        """
        # Reset counts at the start of each new minute window
        if now - self._rep_window_start >= self.REPETITION_WINDOW_SECONDS:
            self.cycle_counts = {j: 0 for j in self._TRACKED_JOINTS}
            self._rep_window_start = now

        alerts = []
        for joint in self._TRACKED_JOINTS:
            val = angles.get(joint)
            if val is None:
                continue

            prev = self._prev_angle[joint]
            if prev is not None:
                delta = val - prev
                if abs(delta) >= self.REPETITION_MIN_DELTA:
                    curr_dir = "up" if delta > 0 else "down"
                    if self._direction[joint] == "up" and curr_dir == "down":
                        self.cycle_counts[joint] += 1
                    self._direction[joint] = curr_dir
            else:
                self._direction[joint] = None

            self._prev_angle[joint] = val

            count = self.cycle_counts[joint]
            if count >= self.REPETITION_ALERT_COUNT:
                alerts.append({
                    "type": "repetitive_motion",
                    "severity": "medium",
                    "message": (
                        f"{joint.replace('_', ' ').title()}: {count} cycles/min"
                        " — RSI risk"
                    ),
                    "joint": joint,
                    "cycles_per_minute": count,
                })
        return alerts
