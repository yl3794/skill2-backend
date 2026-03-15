AWKWARD_POSTURE_SKILL = {
    "skill_id": "awkward_posture",
    "display_name": "Kneeling & Squatting Safety",
    "coaching_context": (
        "coaching plumbers and electricians working in cramped spaces. "
        "Extreme knee flexion (knee angle < 90°) combined with a hunched spine (spine < 140°) "
        "is a primary cause of serious workplace knee and back injuries. "
        "Good form avoids acute knee angles and maintains spine above 140°."
    ),
    "phases": [
        {
            "name": "standing",
            "condition": {"joint": "left_knee", "op": "gt", "value": 150},
            "override_good_form": True,
            "override_score": 100,
            "coaching_instruction": "Worker is standing. Safe position — maintain this posture when possible.",
        }
    ],
    "form_rules": [
        {
            # Violation fires when knee <= 90 (acute squat)
            "joint": "left_knee",
            "op": "gt",
            "value": 90,
            "violation_tip": "Extreme knee bend — use knee pads and open up your stance!",
        },
        {
            "joint": "right_knee",
            "op": "gt",
            "value": 90,
            "violation_tip": "Extreme knee bend — shift to a less acute angle!",
        },
        {
            # Violation fires when spine <= 140 (hunched)
            "joint": "spine",
            "op": "gt",
            "value": 140,
            "violation_tip": "Hunched back in cramped position — reposition to protect your spine!",
        },
    ],
    "score_formula": [
        {
            # 60° (deep bend) → 0pts, 130° (mild bend) → 100pts
            "joint": "left_knee",
            "scale": {"from": [60, 130], "to": [0, 100]},
            "weight": 0.35,
        },
        {
            "joint": "right_knee",
            "scale": {"from": [60, 130], "to": [0, 100]},
            "weight": 0.35,
        },
        {
            # 110° (very hunched) → 0pts, 165° (straight) → 100pts
            "joint": "spine",
            "scale": {"from": [110, 165], "to": [0, 100]},
            "weight": 0.30,
        },
    ],
    "certification": {
        "min_sessions": 2,
        "min_avg_score": 70,
        "min_good_form_rate": 0.75,
        "cert_valid_days": 365,
    },
}
