from fastapi import Header, HTTPException
from database import operations as db


async def require_org(x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    org = db.get_org_by_api_key(x_api_key)
    if not org:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return org


def assert_org_owns_worker(worker: dict, org: dict):
    if worker["org_id"] != org["id"]:
        raise HTTPException(status_code=403, detail="Worker does not belong to your org")


def assert_org_owns_session(session: dict, org: dict, worker_lookup_fn):
    worker = worker_lookup_fn(session["worker_id"])
    if not worker or worker["org_id"] != org["id"]:
        raise HTTPException(status_code=403, detail="Session does not belong to your org")
