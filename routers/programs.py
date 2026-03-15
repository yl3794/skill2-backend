"""
Training program management and SOP ingestion endpoints.

A Program is a named collection of Skills generated from a Standard Operating
Procedure (SOP) document. The SOP is processed by Claude Sonnet, which
identifies posture-critical tasks and generates a complete skill definition
(form_rules, score_formula, certification thresholds) for each one.

Supported SOP formats: PDF, Word (.docx), plain text (.txt, .md).
"""

import io
import json
from typing import Optional

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File

from database import operations as db
from auth import require_org
from services.claude import async_client

router = APIRouter(prefix="/programs", tags=["Programs"])


class AssessmentRequest(BaseModel):
    worker_id: str


class AssessmentSubmitRequest(BaseModel):
    worker_id: str
    questions: list[str]
    answers: list[str]


class CreateProgramFromManualRequest(BaseModel):
    """Request body for POST /programs/from-manual.

    Attributes:
        manual_text: Raw SOP text content (max ~5000 chars are sent to Claude).
        program_name: Optional override for the generated program name.
    """
    manual_text:  str
    program_name: Optional[str] = None


# Joints the skill engine can evaluate from MediaPipe pose landmarks.
_EVALUABLE_JOINTS = [
    "spine", "left_knee", "right_knee",
    "left_elbow", "right_elbow",
    "left_hip", "right_hip",
    "left_shoulder", "right_shoulder",
]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/extract-text", summary="Extract text from an uploaded SOP file")
async def extract_text(
    file:        UploadFile = File(...),
    current_org: dict       = Depends(require_org),
):
    """Extracts plain text from a PDF, DOCX, or plain-text SOP file.

    This is a pre-processing step before POST /programs/from-manual. The
    extracted text can be reviewed by the operator before submission.

    Args:
        file: Uploaded SOP file (.pdf, .docx, .txt, .md).
        current_org: Injected by ``require_org``.

    Returns:
        Dict with ``text``, ``filename``, and ``chars`` (character count).

    Raises:
        HTTPException(400): If the file cannot be parsed or yields no text.
    """
    content  = await file.read()
    filename = (file.filename or "").lower()

    if filename.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            text   = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}")

    elif filename.endswith(".docx"):
        try:
            import docx
            doc  = docx.Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read DOCX: {exc}")

    else:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in file")

    return {"text": text, "filename": file.filename, "chars": len(text)}


@router.post("/from-manual", summary="Generate a training program from SOP text")
async def create_program_from_manual(
    body:        CreateProgramFromManualRequest,
    current_org: dict = Depends(require_org),
):
    """Parses an SOP with Claude Sonnet and creates a training program.

    Claude identifies every posture-critical task in the SOP and generates
    a complete skill definition for each: form_rules (joint thresholds),
    score_formula (weighted scoring), and certification criteria. The
    generated skills and program are persisted immediately.

    Args:
        body: SOP text content and optional program name override.
        current_org: Injected by ``require_org``.

    Returns:
        Dict with the created ``program``, ``skills_created`` count, and the
        raw ``program_def`` returned by Claude (for client-side preview).

    Raises:
        HTTPException(400): If no posture skills could be extracted.
        HTTPException(500): If Claude returns malformed JSON.
    """
    prompt = (
        "You are a physical skills training analyst. Analyze this Standard Operating "
        "Procedure and extract all physical/posture-related tasks that can be monitored "
        "using body joint angles from a camera.\n\n"
        f"Available joints for monitoring: {', '.join(_EVALUABLE_JOINTS)}\n"
        "Joint values are angles in degrees (0–180).\n\n"
        f"SOP content:\n{body.manual_text[:5000]}\n\n"
        "Return ONLY valid JSON — no markdown fences, no explanation:\n"
        "{\n"
        '  "program_name": "descriptive name",\n'
        '  "description": "1-2 sentence overview",\n'
        '  "skills": [\n'
        "    {\n"
        '      "skill_id": "snake_case_id",\n'
        '      "display_name": "Human Readable Name",\n'
        '      "coaching_context": "describe the physical activity and what good form means",\n'
        '      "form_rules": [\n'
        '        {"joint": "spine", "op": "gt", "value": 150, "violation_tip": "Keep your back straight!"}\n'
        "      ],\n"
        '      "score_formula": [\n'
        '        {"joint": "spine", "scale": {"from": [100, 180], "to": [0, 100]}, "weight": 1.0}\n'
        "      ],\n"
        '      "certification": {\n'
        '        "min_sessions": 3, "min_avg_score": 70,\n'
        '        "min_good_form_rate": 0.7, "cert_valid_days": 365\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Only include skills with measurable body posture (lifting, bending, reaching, squatting, etc.)\n"
        "- Each skill needs at least 1 form_rule\n"
        "- score_formula weights must sum to 1.0\n"
        "- op must be 'gt' (greater than) or 'lt' (less than)\n"
        "- Spine > 150 = straight back; < 120 = hunched\n"
        "- Knee > 155 = standing straight; < 140 = properly bent\n"
        "- Generate 2–5 skills based on what the SOP covers\n"
        f"- Program name: {body.program_name or 'auto-generate a descriptive name'}"
    )

    message = await async_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    # Strip markdown fences if the model wraps the output anyway.
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        program_def = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Claude returned malformed JSON: {exc}",
        )

    skill_ids: list[str] = []
    for skill_def in program_def.get("skills", []):
        skill_def["skill_id"] = f"{current_org['id']}:{skill_def['skill_id']}"
        db.upsert_skill(skill_def, org_id=current_org["id"])
        skill_ids.append(skill_def["skill_id"])

    if not skill_ids:
        raise HTTPException(
            status_code=400,
            detail="No posture-critical skills could be extracted from the SOP",
        )

    program = db.create_program(
        name=program_def.get("program_name", "Training Program"),
        description=program_def.get("description", ""),
        skill_ids=skill_ids,
        org_id=current_org["id"],
    )
    return {
        "program":        program,
        "skills_created": len(skill_ids),
        "program_def":    program_def,
    }


@router.get("", summary="List all programs for the authenticated org")
async def list_programs(current_org: dict = Depends(require_org)):
    """Returns all programs owned by the authenticated org, enriched with skill names.

    Args:
        current_org: Injected by ``require_org``.

    Returns:
        List of program dicts, each extended with a ``skills_info`` list
        containing ``skill_id`` and ``display_name`` for each skill.
    """
    programs = db.list_programs(org_id=current_org["id"])
    result   = []
    for p in programs:
        skills_info = []
        for sid in p["skill_ids"]:
            sk = db.get_skill(sid)
            skills_info.append({
                "skill_id":     sid,
                "display_name": sk["definition"].get("display_name", sid) if sk else sid,
            })
        result.append({**p, "skills_info": skills_info})
    return result


@router.get("/{program_id}", summary="Get a program by ID")
async def get_program(program_id: str, current_org: dict = Depends(require_org)):
    """Returns a single program record.

    Args:
        program_id: UUID of the program.
        current_org: Injected by ``require_org``.

    Returns:
        Program dict with ``skill_ids`` list.

    Raises:
        HTTPException(404): If the program does not exist.
        HTTPException(403): If the program belongs to a different org.
    """
    program = db.get_program(program_id)
    if not program:
        raise HTTPException(status_code=404, detail="Program not found")
    if program["org_id"] and program["org_id"] != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return program


@router.post("/{program_id}/assessment", summary="Generate knowledge assessment quiz")
async def generate_assessment(
    program_id: str,
    body: AssessmentRequest,
    current_org: dict = Depends(require_org),
):
    """Generates 5 knowledge assessment questions from the program's skill definitions.

    Called after a worker has completed all skills in a program. Uses Claude Haiku
    to produce safety-knowledge questions tailored to the specific physical tasks
    in this program.

    Args:
        program_id: UUID of the program.
        body: Contains ``worker_id`` of the worker being assessed.
        current_org: Injected by ``require_org``.

    Returns:
        Dict with ``questions`` list and ``program_name``.

    Raises:
        HTTPException(404): If the program does not exist.
        HTTPException(403): If the program belongs to a different org.
    """
    program = db.get_program(program_id)
    if not program:
        raise HTTPException(status_code=404, detail="Program not found")
    if program["org_id"] and program["org_id"] != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    skill_summaries = []
    for sid in program["skill_ids"]:
        sk = db.get_skill(sid)
        if not sk:
            continue
        defn = sk["definition"]
        rules = "; ".join(r.get("violation_tip", "") for r in defn.get("form_rules", []))
        skill_summaries.append(
            f"- {defn.get('display_name', sid)}: {defn.get('coaching_context', '')} Key rules: {rules}"
        )

    prompt = (
        "You are a workplace safety training examiner. A worker has completed hands-on "
        "training in the following skills:\n\n"
        + "\n".join(skill_summaries)
        + "\n\nGenerate exactly 5 knowledge assessment questions to verify their understanding. "
        "Questions should be clear and answerable with 1–3 sentences. Test: "
        "(1) why proper form prevents injury, (2) what correct technique looks like, "
        "(3) warning signs of incorrect form, (4) what to do when feeling discomfort, "
        "(5) a specific safety rule from the training.\n\n"
        'Return ONLY a JSON array of 5 question strings — no markdown, no explanation:\n'
        '["question1", "question2", "question3", "question4", "question5"]'
    )

    message = await async_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1][4:] if len(parts) > 1 and parts[1].startswith("json") else parts[1]
    try:
        questions = json.loads(raw.strip())
    except json.JSONDecodeError:
        questions = [raw]  # fallback

    return {"questions": questions, "program_name": program["name"]}


@router.post("/{program_id}/assessment/submit", summary="Grade a completed assessment")
async def submit_assessment(
    program_id: str,
    body: AssessmentSubmitRequest,
    current_org: dict = Depends(require_org),
):
    """Grades a worker's answers to the assessment questions using Claude Haiku.

    Args:
        program_id: UUID of the program.
        body: Contains ``worker_id``, ``questions`` list, and ``answers`` list.
        current_org: Injected by ``require_org``.

    Returns:
        Dict with ``score`` (0–100), ``passed`` (bool), ``feedback``, and
        ``correct`` list of per-question booleans.

    Raises:
        HTTPException(404): If the program does not exist.
    """
    program = db.get_program(program_id)
    if not program:
        raise HTTPException(status_code=404, detail="Program not found")

    if not body.answers:
        raise HTTPException(status_code=400, detail="No answers provided")

    qa_pairs = "\n".join(
        f"Q{i+1}: {q}\nA{i+1}: {a}"
        for i, (q, a) in enumerate(zip(body.questions, body.answers))
    )

    prompt = (
        "You are a workplace safety training examiner. Grade these answers from a worker "
        "who just completed hands-on safety training.\n\n"
        f"{qa_pairs}\n\n"
        "An answer PASSES if it shows basic safety awareness — even if imperfectly worded. "
        "An answer FAILS only if it is blank, completely wrong, or shows a dangerous misunderstanding.\n\n"
        "Return ONLY valid JSON — no markdown, no explanation:\n"
        '{"score": 0-100, "passed": true/false, '
        '"feedback": "1-2 sentences of constructive overall feedback", '
        '"correct": [true/false, ...]}'
    )

    message = await async_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1][4:] if len(parts) > 1 and parts[1].startswith("json") else parts[1]
    try:
        result = json.loads(raw.strip())
    except json.JSONDecodeError:
        result = {"score": 0, "passed": False, "feedback": "Grading error. Please retry.", "correct": []}

    return result


@router.get("/{program_id}/certifications", summary="Get certification route")
async def get_certifications(program_id: str, current_org: dict = Depends(require_org)):
    """Returns the certification URI for each skill in a program.

    Args:
        program_id: UUID of the program.
        current_org: Injected by ``require_org``.

    Returns:
        Dict with ``verify`` endpoint template for external HR/compliance systems.
    """
    program = db.get_program(program_id)
    if not program:
        raise HTTPException(status_code=404, detail="Program not found")
    if program["org_id"] and program["org_id"] != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return {
        "program_id":  program_id,
        "verify_endpoint": "/certifications/{cert_id}/verify",
        "skill_ids":   program["skill_ids"],
    }
