# Protocol.ai — AI-Powered Workplace Safety Training

Protocol.ai turns any Standard Operating Procedure into a structured, AI-coached training program. Workers train against real-time computer vision feedback. Managers issue verifiable safety certifications. Every rep is audited.

## Who It's For

**Safety managers and workers at industrial worksites** — warehouses, manufacturing floors, construction sites, logistics hubs — where musculoskeletal injuries from improper lifting, reaching, and posture account for 33% of all workplace injuries (BLS, 2023).

| Role               | What they do                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| **Safety Manager** | Uploads SOPs → AI generates training programs → monitors workforce → exports compliance records |
| **Worker**         | Joins org with a code → trains with live AI coaching → earns verifiable certificates            |
| **Auditor / HR**   | Verifies certificates via a public endpoint — no login required                                 |

---

## Quickstart

### Local development

```bash
cp .env.example .env        # set ANTHROPIC_API_KEY
pip install -r requirements.txt
uvicorn main:app --reload
```

### Docker (one command)

```bash
docker compose up --build
```

API at `http://localhost:8000` · Docs at `http://localhost:8000/docs`

### Environment variables

| Variable            | Required | Description                                                  |
| ------------------- | -------- | ------------------------------------------------------------ |
| `ANTHROPIC_API_KEY` | Yes      | Used for real-time coaching (Haiku) and SOP parsing (Sonnet) |
| `CERT_SECRET`       | No       | JWT signing key for certificates (auto-generated if unset)   |

---

## How It Works

```
Safety Manager uploads SOP (PDF/DOCX/TXT)
    │
    ▼  Claude Sonnet
Training program generated (form rules, scoring, certification thresholds)
    │
    ▼
Worker selects skill → sees instructions → starts session
    │
    ▼  MediaPipe (on-device, client-side — no video sent to server)
Joint angles captured at peak of each rep
    │
    ▼  Claude Haiku
Personalized coaching tip — knows the worker's name, session history,
recurring issues, and current trend across the session
    │
    ▼
Session ends → debrief generated → certification issued if thresholds met
    │
    ▼  JWT-signed certificate
Verifiable by any HR/EHS system at GET /certifications/{id}/verify
```

**Privacy-first:** Only joint angle JSON is sent to the backend. No video ever leaves the worker's device.

---

## Performance

| Metric                   | Value                                       |
| ------------------------ | ------------------------------------------- |
| Coaching tip latency     | ~250ms (Claude Haiku)                       |
| Pose detection           | 30 FPS client-side (MediaPipe GPU delegate) |
| Skill rule evaluation    | <1ms (pure arithmetic)                      |
| SOP → program generation | ~4s (Claude Sonnet, one-time per upload)    |

---

## Scalability

### Current (demo / pilot deployment)

SQLite, single process — runs on any laptop or $5 VPS with no configuration.

### Path to production scale

**1. Database swap (one file change)**
All queries are in `database/operations.py`. Replacing SQLite with PostgreSQL requires changing that one file — zero business logic changes.

**2. Horizontal API scaling**
The API is fully stateless. All session and rep state lives in the database. Run behind any load balancer with `uvicorn main:app --workers N`.

**3. Skill coverage**
The skill engine is JSON-driven. Any physical task measurable by body joint angles works:

- Upload an SOP → Claude auto-generates the skill definition
- Or define rules manually via `POST /skills`
- 11 joints tracked: spine, both knees, hips, shoulders, elbows, ankles

**4. Enterprise integrations**

- Certificates are JWT-signed and publicly verifiable — no API key needed for auditors
- `GET /orgs/{id}/analytics` exposes aggregate data for LMS/EHS integration
- Per-rep compliance blobs stored for OSHA / ISO audit exports

---

## API Overview

```bash
# Org setup
POST /orgs                              → { api_key, join_code }

# Upload SOP → generate training program
POST /programs/extract-text             → { text }
POST /programs/from-manual              → { program, skills_created }

# Worker trains
POST /workers/register                  → { worker }
POST /sessions/start                    → { session_id }
POST /coach                             → { coaching_tip, score, is_good_form }
POST /sessions/{id}/end                 → { avg_score, rep_count }
POST /sessions/{id}/debrief             → { debrief }

# Certification
POST /workers/{id}/certify/{skill_id}   → { certified, certificate }
GET  /certifications/{id}/verify        → { valid, worker, skill, issued_at }
```

Full interactive docs: `http://localhost:8000/docs`

---

## Built-in Skills

Three skills are pre-loaded at startup and available to all orgs:

| Skill                 | Use case                             | Key joints   |
| --------------------- | ------------------------------------ | ------------ |
| Safe Lifting          | Warehouse, logistics                 | Spine, knees |
| Overhead Reach Safety | Assembly, automotive                 | Shoulders    |
| Awkward Posture       | Plumbing, electrical, cramped spaces | Knees, spine |

Any industry-specific skill can be added by uploading an SOP.
