"""
Generic skill evaluation engine.
Takes a skill definition (dict) + pose data (dict of joint: angle)
and returns is_good_form, score, violations, and a Claude prompt.
"""


def _check_op(value: float, op: str, threshold: float) -> bool:
    return {
        "gt":  value > threshold,
        "gte": value >= threshold,
        "lt":  value < threshold,
        "lte": value <= threshold,
        "eq":  value == threshold,
    }.get(op, True)


def _detect_phase(definition: dict, pose: dict) -> dict | None:
    for phase in definition.get("phases", []):
        cond = phase["condition"]
        joint_val = pose.get(cond["joint"])
        if joint_val is not None and _check_op(joint_val, cond["op"], cond["value"]):
            return phase
    return None


def _evaluate_rules(rules: list, pose: dict) -> list[dict]:
    violations = []
    for rule in rules:
        joint_val = pose.get(rule["joint"])
        if joint_val is None:
            continue
        if not _check_op(joint_val, rule["op"], rule["value"]):
            violations.append(rule)
    return violations


def _compute_score(formula: list, pose: dict) -> int:
    if not formula:
        return 100

    total_weight = sum(f.get("weight", 1) for f in formula)
    weighted = 0.0

    for f in formula:
        joint_val = pose.get(f["joint"])
        if joint_val is None:
            continue
        weight = f.get("weight", 1)

        if "scale" in f:
            lo, hi = f["scale"]["from"]
            out_lo, out_hi = f["scale"]["to"]
            t = max(0.0, min(1.0, (joint_val - lo) / (hi - lo)))
            joint_score = out_lo + t * (out_hi - out_lo)
        elif "bonus_if" in f:
            bi = f["bonus_if"]
            passes = _check_op(joint_val, bi["op"], bi["value"])
            joint_score = 50 + (f.get("bonus", 10) if passes else f.get("else", -20))
        else:
            joint_score = 50

        weighted += joint_score * weight

    return min(100, max(0, int(weighted / total_weight)))


def evaluate_skill(definition: dict, pose: dict) -> tuple[bool, int, list[dict]]:
    """
    Returns (is_good_form, score, violations).
    Violations are the form_rules that failed.
    """
    phase = _detect_phase(definition, pose)

    if phase:
        if "override_good_form" in phase:
            return phase["override_good_form"], phase.get("override_score", 100), []

    violations = _evaluate_rules(definition.get("form_rules", []), pose)
    is_good = len(violations) == 0
    score = _compute_score(definition.get("score_formula", []), pose)
    return is_good, score, violations


def build_coaching_prompt(
    definition:     dict,
    pose:           dict,
    violations:     list[dict],
    rep_history:    list[dict] | None = None,
    worker_name:    str | None = None,
    session_number: int = 1,
) -> str:
    """Builds a context-rich coaching prompt for Claude.

    Instead of passing raw angle numbers, this prompt gives Claude a physical
    interpretation of what went wrong, a summary of the worker's recent rep
    history, and any persistent patterns — so it can reason like a real coach
    rather than a threshold-checker.

    Args:
        definition: Skill definition dict with form_rules, coaching_context, etc.
        pose: Current rep joint angles (degrees).
        violations: Form rules that failed this rep.
        rep_history: Last N reps from the session (score, is_good_form, coaching_tip).
        worker_name: Worker's name for personalisation.
        session_number: How many sessions this worker has done for this skill.

    Returns:
        Prompt string ready to send to Claude.
    """
    display_name  = definition.get("display_name", "this skill")
    context       = definition.get("coaching_context", "")
    form_rules    = definition.get("form_rules", [])
    rep_history   = rep_history or []
    name          = worker_name or "the worker"
    rep_num       = len(rep_history) + 1  # this rep will be rep N

    # ── Physical violation descriptions ──────────────────────────────────────
    # Translate angle + threshold into a human-readable physical description.
    _joint_labels = {
        "spine":          "back/spine",
        "left_knee":      "left knee",
        "right_knee":     "right knee",
        "left_hip":       "left hip",
        "right_hip":      "right hip",
        "left_shoulder":  "left shoulder",
        "right_shoulder": "right shoulder",
        "left_elbow":     "left elbow",
        "right_elbow":    "right elbow",
    }
    _op_phrases = {
        "gt":  ("should be above", "was below"),
        "gte": ("should be at least", "was below"),
        "lt":  ("should be below", "was above"),
        "lte": ("should be at most", "was above"),
    }

    # Severity: how far off is the joint from the threshold?
    def _severity(v):
        actual    = pose.get(v.get("joint"), 0)
        threshold = v.get("value", 0)
        diff      = abs(actual - threshold)
        if diff >= 20: return "significantly"
        if diff >= 10: return "noticeably"
        return "slightly"

    if violations:
        violation_lines = []
        for v in violations:
            tip      = v.get("violation_tip", "")
            severity = _severity(v)
            violation_lines.append(f"- {tip} ({severity} off)")
        this_rep_block = "FORM ISSUES THIS REP:\n" + "\n".join(violation_lines)
        this_rep_form  = "needs work"
    else:
        this_rep_block = "FORM THIS REP: Good — no violations detected."
        this_rep_form  = "good"

    # ── Session history summary ───────────────────────────────────────────────
    if rep_history:
        scores       = [r["score"] for r in rep_history]
        good_count   = sum(1 for r in rep_history if r.get("is_good_form"))
        avg          = sum(scores) / len(scores)
        recent5      = scores[-5:]

        # Score trend over last 5 reps
        if len(recent5) >= 3:
            if recent5[-1] > recent5[0] + 8:
                trend = "improving"
            elif recent5[-1] < recent5[0] - 8:
                trend = "declining"
            else:
                trend = "consistent"
        else:
            trend = "not enough data"

        scores_str = ", ".join(str(s) for s in scores[-5:])
        history_block = (
            f"SESSION HISTORY (last {len(rep_history)} reps):\n"
            f"- Scores: {scores_str} → current rep is #{rep_num}\n"
            f"- Good form rate: {good_count}/{len(rep_history)} reps\n"
            f"- Trend: {trend}"
        )

        # Detect persistent issues across last 5 reps
        recent_tips = [r.get("coaching_tip", "") for r in rep_history[-5:] if not r.get("is_good_form")]
        persistent_block = ""
        if len(recent_tips) >= 3:
            persistent_block = (
                f"\nPERSISTENT ISSUE: {name} has had form problems in "
                f"{len(recent_tips)} of the last {min(5, len(rep_history))} reps. "
                "This is a recurring pattern, not a one-off mistake."
            )
    else:
        history_block    = f"SESSION HISTORY: This is {name}'s first rep of this session."
        persistent_block = ""

    # ── Session experience level ──────────────────────────────────────────────
    if session_number <= 1:
        experience = f"This is {name}'s first session on this skill."
    elif session_number <= 3:
        experience = f"This is session {session_number} for {name} on this skill — still learning."
    else:
        experience = f"{name} has completed {session_number - 1} previous sessions on this skill."

    # ── Form requirements (what good looks like) ─────────────────────────────
    rules_str = "\n".join(
        f"- {r.get('violation_tip', '')}" for r in form_rules if r.get("violation_tip")
    ) or "Follow the technique described above."

    return f"""You are a physical skills coach. You are training {name} on: {display_name}.

ABOUT THIS SKILL:
{context or f'Proper technique for {display_name}.'}

WHAT GOOD FORM REQUIRES:
{rules_str}

{experience}

{history_block}{persistent_block}

THIS REP (form: {this_rep_form}):
{this_rep_block}

Give a 1–2 sentence coaching response. Use directional, physical language ("round your back less", "bend your knees deeper", "keep the load closer") — never mention degrees, numbers, or measurements. If there is a persistent pattern, address it directly. If they are improving, acknowledge it. Sound like a real coach talking to someone mid-workout.

No markdown, no asterisks, no degree symbols, no numbers, no filler phrases like "Great job!" unless they truly earned it."""
