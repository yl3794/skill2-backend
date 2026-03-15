"""
VisionTrain API — entry point.

Registers all routers and runs startup initialization (DB schema, built-in
skill upserts). All application logic lives in ``routers/``, ``schemas/``,
``services/``, and ``cv/``. This file should remain configuration-only.

Run with:
    ./venv/bin/uvicorn main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from database.db import init_db
from database import operations as db
from skills.lifting.definition import LIFTING_SKILL
from skills.overhead_reach.definition import OVERHEAD_REACH_SKILL
from skills.awkward_posture.definition import AWKWARD_POSTURE_SKILL

from routers import (
    orgs,
    workers,
    sessions,
    skills,
    certifications,
    programs,
    coaching,
    safety,
    video,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs startup and teardown logic around the application lifespan.

    Startup: initialises the SQLite schema (idempotent migrations) and
    upserts the three built-in skill definitions so they are always present.
    """
    init_db()
    db.upsert_skill(LIFTING_SKILL)
    db.upsert_skill(OVERHEAD_REACH_SKILL)
    db.upsert_skill(AWKWARD_POSTURE_SKILL)
    yield
    # Teardown: nothing required for SQLite.


app = FastAPI(
    title="VisionTrain API",
    description=(
        "AI-powered workplace safety training platform. "
        "Upload SOPs → Claude generates training programs → "
        "workers train with real-time computer-vision coaching."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Router registration ───────────────────────────────────────────────────────
# Order matters for OpenAPI tag grouping; functional order does not.

for _router in [
    video.router,
    coaching.router,
    safety.router,
    orgs.router,
    workers.router,
    sessions.router,
    skills.router,
    certifications.router,
    programs.router,
]:
    app.include_router(_router)
