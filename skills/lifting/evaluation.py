def evaluate(pose) -> tuple[bool, int]:
    """
    Evaluate lifting form based on joint angles.
    Returns (is_good_form, score 0-100)

    Good lifting form:
    - Spine above 160 degrees (straight back)
    - At least one knee below 150 degrees (bending knees)
    - Hips above 160 degrees (stable base)
    """
    spine_good = pose.spine > 160
    knees_good = pose.left_knee < 150 or pose.right_knee < 150
    hips_good = pose.left_hip > 160 or pose.right_hip > 160

    is_good = spine_good and knees_good and hips_good

    # Score based on spine angle as primary metric
    spine_score = min(100, max(0, int(pose.spine - 60)))

    # Deduct points for bad knees or hips
    knee_penalty = 0 if knees_good else 20
    hip_penalty = 0 if hips_good else 10

    score = min(100, max(0, spine_score - knee_penalty - hip_penalty))

    return is_good, score