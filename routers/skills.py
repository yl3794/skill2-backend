"""
Skill definition management endpoints.

Skills are the atomic units of a training program. Each skill definition
contains ``form_rules`` (what constitutes good form), a ``score_formula``
(how to compute a 0–100 score), and ``certification`` thresholds.

Built-in skills (lifting, overhead_reach, awkward_posture) are registered at
startup and are accessible to all orgs. Org-scoped custom skills are created
via POST /skills and are prefixed with the org UUID to prevent collisions.
"""

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends

from database import operations as db
from auth import require_org

router = APIRouter(prefix="/skills", tags=["Skills"])


class RegisterSkillRequest(BaseModel):
    """Request body for POST /skills.

    Attributes:
        definition: Full skill definition dict. Must contain ``skill_id``
            and ``form_rules``. See the ``SkillDefinition`` format in the
            skill engine docs.
    """
    definition: dict


@router.post("", summary="Register a custom skill")
async def register_skill(
    body:        RegisterSkillRequest,
    current_org: dict = Depends(require_org),
):
    """Registers a new org-scoped skill definition.

    The ``skill_id`` is namespaced with the org UUID
    (``{org_id}:{skill_id}``) to prevent cross-org collisions. Pass the
    bare snake_case ``skill_id`` in the request — namespacing is applied
    automatically.

    Args:
        body: Skill registration payload containing the ``definition`` dict.
        current_org: Injected by ``require_org``.

    Returns:
        Dict with the namespaced ``skill_id`` and ``registered: True``.

    Raises:
        HTTPException(400): If ``skill_id`` or ``form_rules`` are missing.
    """
    definition = body.definition
    if "skill_id" not in definition:
        raise HTTPException(status_code=400, detail="definition must include 'skill_id'")
    if "form_rules" not in definition:
        raise HTTPException(status_code=400, detail="definition must include 'form_rules'")
    definition["skill_id"] = f"{current_org['id']}:{definition['skill_id']}"
    db.upsert_skill(definition, org_id=current_org["id"])
    return {"skill_id": definition["skill_id"], "registered": True}


@router.get("", summary="List all accessible skills")
async def list_skills(current_org: dict = Depends(require_org)):
    """Returns all skills accessible to the authenticated org.

    Includes built-in skills (org_id IS NULL) and skills created by this org.

    Args:
        current_org: Injected by ``require_org``.

    Returns:
        List of skill dicts, each with a parsed ``definition`` object.
    """
    return db.list_skills(org_id=current_org["id"])


@router.get("/{skill_id:path}", summary="Get a skill by ID")
async def get_skill(skill_id: str, current_org: dict = Depends(require_org)):
    """Returns a single skill definition.

    Args:
        skill_id: The skill identifier (built-in: e.g. ``lifting``;
            org-scoped: e.g. ``{org_id}:forklift_operation``).
        current_org: Injected by ``require_org``.

    Returns:
        Skill dict with parsed ``definition``.

    Raises:
        HTTPException(404): If the skill does not exist.
        HTTPException(403): If the skill is org-scoped to a different org.
    """
    skill = db.get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    if skill["org_id"] and skill["org_id"] != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return skill
