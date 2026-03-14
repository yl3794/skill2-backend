LIFTING_SKILL = {
    "skill_id": "lifting",
    "display_name": "Safe Lifting Technique",
    "coaching_context": "coaching someone on how to safely pick up an object from the floor. Good form means a straight back (spine > 150°) and bent knees during the lift",
    "phases": [
        {
            "name": "standing",
            "condition": {"joint": "left_knee", "op": "gt", "value": 155},
            "override_good_form": True,
            "override_score": 100,
            "coaching_instruction": "The person is standing upright in ready position. Say: Ready position. Now squat down to pick up the object.",
        }
    ],
    "form_rules": [
        {
            "joint": "spine",
            "op": "gt",
            "value": 150,
            "violation_tip": "Keep your back straight!",
        },
        {
            "joint": "left_knee",
            "op": "lt",
            "value": 140,
            "violation_tip": "Bend your knees, not your back!",
        },
    ],
    "score_formula": [
        {
            "joint": "spine",
            "scale": {"from": [100, 180], "to": [0, 100]},
            "weight": 0.7,
        },
        {
            "joint": "left_knee",
            "bonus_if": {"op": "lt", "value": 140},
            "bonus": 10,
            "else": -20,
            "weight": 0.3,
        },
    ],
    "certification": {
        "min_sessions": 3,
        "min_avg_score": 75,
        "min_good_form_rate": 0.70,
        "cert_valid_days": 365,
    },
}
