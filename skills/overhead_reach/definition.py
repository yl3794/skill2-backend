OVERHEAD_REACH_SKILL = {
    "skill_id": "overhead_reach",
    "display_name": "Overhead Work Safety",
    "coaching_context": (
        "coaching an automotive or assembly worker doing overhead installation work. "
        "Sustained arm elevation above shoulder level (shoulder angle > 90°) causes "
        "rotator cuff injuries. Good form keeps both shoulder angles below 90°."
    ),
    # No standing phase override — overhead is always actively monitored
    "form_rules": [
        {
            "joint": "left_shoulder",
            "op": "lt",
            "value": 90,
            "violation_tip": "Lower your left arm — sustained overhead work causes rotator cuff injuries!",
        },
        {
            "joint": "right_shoulder",
            "op": "lt",
            "value": 90,
            "violation_tip": "Lower your right arm — take a break from overhead work!",
        },
    ],
    "score_formula": [
        {
            # 0° = arm at side (100pts), 120° = arm well overhead (0pts)
            "joint": "left_shoulder",
            "scale": {"from": [0, 120], "to": [100, 0]},
            "weight": 0.5,
        },
        {
            "joint": "right_shoulder",
            "scale": {"from": [0, 120], "to": [100, 0]},
            "weight": 0.5,
        },
    ],
    "certification": {
        "min_sessions": 2,
        "min_avg_score": 70,
        "min_good_form_rate": 0.75,
        "cert_valid_days": 365,
    },
}
