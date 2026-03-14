def evaluate(pose) -> tuple[bool, int]:
    """
    Phases:
    - Standing (both knees > 155): neutral, score 100
    - Mid-lift (at least one knee < 155): evaluate back and knees
    """
    is_standing = pose.left_knee > 155 and pose.right_knee > 155

    if is_standing:
        # Person is standing upright — good ready position
        return True, 100

    # Person is in a squat/lift — evaluate form
    spine_good = pose.spine > 150
    knees_bending = pose.left_knee < 140 or pose.right_knee < 140


    is_good = spine_good and knees_bending

    spine_score = min(100, max(0, int((pose.spine - 100) * 1.5)))
    knee_bonus = 10 if knees_bending else -20
    score = min(100, max(0, spine_score + knee_bonus))

    return is_good, score