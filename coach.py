#  coach.py — Posture analysis + LLM feedback
#  Later: replace analyze_posture() with a real LLM call

def analyze_posture(angles: dict) -> str:
    """
    TODO: Replace this with a real LLM call later.

    Example prompt to send to Claude/GPT:
    f"These are the joint angles: {angles}.
      Is the person's posture good? What should they fix?"
    """
    if not angles:
        return "No pose detected yet."

    feedback = []

    if "spine" in angles:
        if angles["spine"] < 160:
            feedback.append("⚠️ Spine: You may be leaning forward.")
        else:
            feedback.append("✅ Spine looks good.")

    if "left_knee" in angles and "right_knee" in angles:
        if angles["left_knee"] < 90 or angles["right_knee"] < 90:
            feedback.append("⚠️ Knees: Deep bend detected.")

    return "\n".join(feedback) if feedback else "✅ Posture looks okay."