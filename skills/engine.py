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


def build_coaching_prompt(definition: dict, pose: dict, violations: list[dict]) -> str:
    display_name = definition.get("display_name", "this skill")
    context = definition.get("coaching_context", f"coaching someone on {display_name}")

    phase = _detect_phase(definition, pose)
    if phase and "coaching_instruction" in phase:
        phase_block = f"Situation: {phase['coaching_instruction']}"
    elif violations:
        tips = "\n".join(f"- {v['violation_tip']}" for v in violations if "violation_tip" in v)
        phase_block = f"Issues detected:\n{tips}\nGive one short, direct coaching cue for the most critical issue."
    else:
        phase_block = "Form looks good. Give brief positive reinforcement in one sentence."

    angles_str = "\n".join(f"- {k}: {v}°" for k, v in pose.items())

    return f"""You are a coach {context}.

Current joint angles:
{angles_str}

{phase_block}

Respond in one sentence. No markdown, no asterisks."""
