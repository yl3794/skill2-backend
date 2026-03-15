"""
Organization management endpoints.

An Org represents a single industrial site. It owns workers, programs, and
skills. Authentication is via API key (``X-API-Key`` header); the key is
issued at org creation and never regenerated without explicit request.
"""

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends

from database import operations as db
from auth import require_org

router = APIRouter(prefix="/orgs", tags=["Organizations"])


class CreateOrgRequest(BaseModel):
    """Request body for POST /orgs.

    Attributes:
        name: Human-readable organization name (e.g. "Acme Automotive — Plant 3").
    """
    name: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", summary="Create a new organization")
async def create_org(body: CreateOrgRequest):
    """Creates a new industrial-site organization.

    Returns the full org record including the API key and worker join code.
    The API key is shown only once — store it immediately.

    Args:
        body: Organization creation payload.

    Returns:
        Full org dict including ``api_key`` and ``join_code``.
    """
    return db.create_org(body.name)


@router.get("/me", summary="Get the authenticated organization")
async def get_my_org(current_org: dict = Depends(require_org)):
    """Returns the organization record associated with the current API key.

    Args:
        current_org: Injected by the ``require_org`` dependency.

    Returns:
        Org dict (id, name, join_code, created_at).
    """
    return current_org


@router.get("/join/{join_code}", summary="Look up org by worker join code")
async def lookup_join_code(join_code: str):
    """Returns the org name for a given worker join code.

    Used by the worker registration UI to preview which org they are joining
    before submitting their name.

    Args:
        join_code: Six-character alphanumeric code printed on the worker's
            onboarding sheet.

    Returns:
        Dict with ``id`` and ``name`` of the matching org.

    Raises:
        HTTPException(404): If the join code does not match any org.
    """
    org = db.get_org_by_join_code(join_code)
    if not org:
        raise HTTPException(status_code=404, detail="Invalid join code")
    return {"id": org["id"], "name": org["name"]}


@router.get("/{org_id}", summary="Get organization by ID")
async def get_org(org_id: str, current_org: dict = Depends(require_org)):
    """Returns an org record. Callers may only access their own org.

    Args:
        org_id: UUID of the org to retrieve.
        current_org: Injected by the ``require_org`` dependency.

    Returns:
        Org dict.

    Raises:
        HTTPException(403): If ``org_id`` does not match the authenticated org.
    """
    if org_id != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return current_org


@router.get("/{org_id}/workers", summary="List all workers in an org")
async def list_workers(org_id: str, current_org: dict = Depends(require_org)):
    """Returns all worker records belonging to this org.

    Args:
        org_id: UUID of the org.
        current_org: Injected by the ``require_org`` dependency.

    Returns:
        List of worker dicts.

    Raises:
        HTTPException(403): If ``org_id`` does not match the authenticated org.
    """
    if org_id != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return db.get_workers_by_org(org_id)


@router.get("/{org_id}/workforce", summary="Worker roster with session stats")
async def org_workforce(org_id: str, current_org: dict = Depends(require_org)):
    """Returns workers enriched with session statistics and certification status.

    Args:
        org_id: UUID of the org.
        current_org: Injected by the ``require_org`` dependency.

    Returns:
        List of worker dicts extended with ``total_sessions``, ``avg_score``,
        ``certified``, and ``cert_expires``.

    Raises:
        HTTPException(403): If ``org_id`` does not match the authenticated org.
    """
    if org_id != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    workers = db.get_workers_by_org(org_id)
    result  = []
    for w in workers:
        sessions = db.get_sessions_by_worker(w["id"])
        scores   = [s["avg_score"] for s in sessions if s["avg_score"] is not None]
        cert     = db.get_latest_certification(w["id"], "lifting")
        result.append({
            **w,
            "total_sessions": len(sessions),
            "avg_score":      round(sum(scores) / len(scores), 1) if scores else None,
            "certified":      cert is not None,
            "cert_expires":   cert["expires_at"] if cert else None,
        })
    return result


@router.get("/{org_id}/analytics", summary="Aggregate analytics for an org")
async def org_analytics(org_id: str, current_org: dict = Depends(require_org)):
    """Returns aggregate training analytics for the org.

    Args:
        org_id: UUID of the org.
        current_org: Injected by the ``require_org`` dependency.

    Returns:
        Dict with ``total_workers``, ``total_sessions``,
        ``avg_score_across_sessions``, and ``top_workers`` list.

    Raises:
        HTTPException(403): If ``org_id`` does not match the authenticated org.
    """
    if org_id != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return db.get_org_analytics(org_id)
