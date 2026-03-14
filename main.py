from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import anthropic
import os
import threading
import cv2
import mediapipe as mp
from dotenv import load_dotenv

from angle import calculate_angle
from skills.engine import evaluate_skill, build_coaching_prompt
from skills.lifting.definition import LIFTING_SKILL
from database.db import init_db
from database import operations as db
from auth import require_org, assert_org_owns_worker, assert_org_owns_session

load_dotenv()

app = FastAPI(title="Skill2 API", description="AI-powered physical skills coaching backend")

@app.on_event("startup")
def startup():
    init_db()
    db.upsert_skill(LIFTING_SKILL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── MediaPipe setup ──────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

angle_data = {}
lock = threading.Lock()

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)


def get_coords(landmarks, index, w, h):
    lm = landmarks[index]
    return [int(lm.x * w), int(lm.y * h)]


def put_angle(frame, label, angle, position):
    cv2.putText(frame, f"{label}: {angle}°", position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


def generate_frames():
    global angle_data
    frame_count = 0
    last_pose_results = None
    last_hand_results = None
    cached_angles = {}
    cached_coords = {}

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose, \
        mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if frame_count % 2 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_pose_results = pose.process(rgb)
                last_hand_results = hands.process(rgb)

                if last_pose_results.pose_landmarks:
                    landmarks = last_pose_results.pose_landmarks.landmark

                    nose       = get_coords(landmarks, mp_pose.PoseLandmark.NOSE.value, w, h)
                    l_shoulder = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
                    r_shoulder = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
                    l_elbow    = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value, w, h)
                    r_elbow    = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value, w, h)
                    l_wrist    = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value, w, h)
                    r_wrist    = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value, w, h)
                    l_hip      = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, w, h)
                    r_hip      = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)
                    l_knee     = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value, w, h)
                    r_knee     = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value, w, h)
                    l_ankle    = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h)
                    r_ankle    = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, w, h)
                    l_foot     = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value, w, h)
                    r_foot     = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value, w, h)

                    mid_shoulder = [(l_shoulder[0]+r_shoulder[0])//2, (l_shoulder[1]+r_shoulder[1])//2]
                    mid_hip      = [(l_hip[0]+r_hip[0])//2, (l_hip[1]+r_hip[1])//2]

                    cached_coords = {
                        "nose": nose, "l_shoulder": l_shoulder, "r_shoulder": r_shoulder,
                        "l_elbow": l_elbow, "r_elbow": r_elbow,
                        "l_wrist": l_wrist, "r_wrist": r_wrist,
                        "l_hip": l_hip, "r_hip": r_hip,
                        "l_knee": l_knee, "r_knee": r_knee,
                        "l_ankle": l_ankle, "r_ankle": r_ankle,
                        "l_foot": l_foot, "r_foot": r_foot,
                        "mid_shoulder": mid_shoulder, "mid_hip": mid_hip
                    }

                    cached_angles = {
                        "left_elbow":     calculate_angle(l_shoulder, l_elbow, l_wrist),
                        "right_elbow":    calculate_angle(r_shoulder, r_elbow, r_wrist),
                        "left_shoulder":  calculate_angle(l_hip, l_shoulder, l_elbow),
                        "right_shoulder": calculate_angle(r_hip, r_shoulder, r_elbow),
                        "left_hip":       calculate_angle(l_shoulder, l_hip, l_knee),
                        "right_hip":      calculate_angle(r_shoulder, r_hip, r_knee),
                        "left_knee":      calculate_angle(l_hip, l_knee, l_ankle),
                        "right_knee":     calculate_angle(r_hip, r_knee, r_ankle),
                        "left_ankle":     calculate_angle(l_knee, l_ankle, l_foot),
                        "right_ankle":    calculate_angle(r_knee, r_ankle, r_foot),
                        "spine":          calculate_angle(nose, mid_shoulder, mid_hip)
                    }

                    with lock:
                        angle_data = cached_angles

            if last_pose_results and last_pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    last_pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

            if cached_coords and cached_angles:
                cv2.line(frame, tuple(cached_coords["mid_shoulder"]), tuple(cached_coords["mid_hip"]), (0, 165, 255), 2)
                put_angle(frame, "SP", cached_angles["spine"], tuple(cached_coords["mid_shoulder"]))
                put_angle(frame, "LK", cached_angles["left_knee"], tuple(cached_coords["l_knee"]))
                put_angle(frame, "RK", cached_angles["right_knee"], tuple(cached_coords["r_knee"]))

            if last_hand_results and last_hand_results.multi_hand_landmarks:
                for hand_landmarks in last_hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 50]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ── Pydantic models ──────────────────────────────────────────
class PoseData(BaseModel):
    left_ankle: float
    left_elbow: float
    left_hip: float
    left_knee: float
    left_shoulder: float
    right_ankle: float
    right_elbow: float
    right_hip: float
    right_knee: float
    right_shoulder: float
    spine: float


class EvaluationResponse(BaseModel):
    coaching_tip: str
    is_good_form: bool
    score: int


# ── Platform models ───────────────────────────────────────────
class CreateOrgRequest(BaseModel):
    name: str

class CreateWorkerRequest(BaseModel):
    org_id: str
    name: str

class StartSessionRequest(BaseModel):
    worker_id: str
    skill_id: str

class WorkerRegisterRequest(BaseModel):
    name: str
    join_code: Optional[str] = None


# ── Shared coaching logic ────────────────────────────────────
async def get_coaching_response(
    pose: PoseData,
    session_id: Optional[str] = None,
    skill_id: str = "lifting",
) -> EvaluationResponse:
    skill_row = db.get_skill(skill_id)
    if not skill_row:
        raise HTTPException(status_code=404, detail=f"Unknown skill: {skill_id}")
    definition = skill_row["definition"]

    pose_dict = pose.model_dump()
    is_good, score, violations = evaluate_skill(definition, pose_dict)
    prompt = build_coaching_prompt(definition, pose_dict, violations)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    coaching_tip = message.content[0].text

    if session_id:
        db.record_rep(session_id, score, is_good, pose_dict, coaching_tip)

    return EvaluationResponse(coaching_tip=coaching_tip, is_good_form=is_good, score=score)


# ── Routes ───────────────────────────────────────────────────
@app.get("/video")
async def video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/angles")
async def angles():
    with lock:
        return angle_data


@app.post("/coach", response_model=EvaluationResponse)
async def coach(session_id: Optional[str] = None, skill_id: str = "lifting"):
    try:
        with lock:
            angles = angle_data.copy()
        if not angles:
            raise HTTPException(status_code=503, detail="No pose detected yet. Make sure you are in frame.")
        pose = PoseData(**angles)
        return await get_coaching_response(pose, session_id=session_id, skill_id=skill_id)
    except HTTPException:
        raise
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_pose(pose: PoseData, session_id: Optional[str] = None, skill_id: str = "lifting"):
    try:
        return await get_coaching_response(pose, session_id=session_id, skill_id=skill_id)
    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/orgs/join/{join_code}")
async def lookup_join_code(join_code: str):
    org = db.get_org_by_join_code(join_code)
    if not org:
        raise HTTPException(status_code=404, detail="Invalid join code")
    return {"id": org["id"], "name": org["name"]}


@app.post("/workers/register")
async def register_worker(body: WorkerRegisterRequest):
    if body.join_code:
        org = db.get_org_by_join_code(body.join_code)
        if not org:
            raise HTTPException(status_code=400, detail="Invalid join code")
    else:
        org = db.create_org(f"{body.name.strip()}'s Workspace")
    worker = db.create_worker(org["id"], body.name.strip())
    return {"worker": worker, "api_key": org["api_key"], "org_name": org["name"]}


@app.post("/sessions/{session_id}/debrief")
async def generate_debrief(session_id: str, current_org: dict = Depends(require_org)):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    assert_org_owns_session(session, current_org, db.get_worker)

    reps = session.get("reps", [])
    if not reps:
        return {"debrief": "No reps were recorded in this session."}

    worker = db.get_worker(session["worker_id"])
    scores = [r["score"] for r in reps]
    good_count = sum(1 for r in reps if r["is_good_form"])
    avg = round(sum(scores) / len(scores), 1)

    if len(scores) > 2:
        trend = "improving" if scores[-1] > scores[0] + 5 else "declining" if scores[-1] < scores[0] - 5 else "consistent"
    else:
        trend = "consistent"

    violations = [r["coaching_tip"] for r in reps if not r["is_good_form"]]

    prompt = f"""You are a professional physical skills coach reviewing a training session.

Worker: {worker['name'] if worker else 'Unknown'}
Skill: Safe Lifting Technique
Reps analyzed: {len(reps)}
Good form: {good_count}/{len(reps)} ({round(good_count / len(reps) * 100)}%)
Average score: {avg}/100
Performance trend: {trend}
{"Recent issues flagged: " + "; ".join(set(violations[-3:])) if violations else "No form issues detected."}

Write a personalized 2-3 sentence coaching debrief. Be specific about their performance, mention what they did well, and give one concrete improvement tip. Be encouraging but honest. No markdown, no asterisks."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    text = message.content[0].text
    db.save_debrief(session_id, text)
    return {"debrief": text}


# ── Org routes ────────────────────────────────────────────────

@app.post("/orgs")
async def create_org(body: CreateOrgRequest):
    return db.create_org(body.name)


@app.get("/orgs/me")
async def get_my_org(current_org: dict = Depends(require_org)):
    return current_org


@app.get("/orgs/{org_id}")
async def get_org(org_id: str, current_org: dict = Depends(require_org)):
    if org_id != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return current_org


@app.get("/orgs/{org_id}/workers")
async def list_workers(org_id: str, current_org: dict = Depends(require_org)):
    if org_id != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return db.get_workers_by_org(org_id)


@app.get("/orgs/{org_id}/workforce")
async def org_workforce(org_id: str, current_org: dict = Depends(require_org)):
    if org_id != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    workers = db.get_workers_by_org(org_id)
    result = []
    for w in workers:
        sessions = db.get_sessions_by_worker(w["id"])
        scores = [s["avg_score"] for s in sessions if s["avg_score"] is not None]
        cert = db.get_latest_certification(w["id"], "lifting")
        result.append({
            **w,
            "total_sessions": len(sessions),
            "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
            "certified": cert is not None,
            "cert_expires": cert["expires_at"] if cert else None,
        })
    return result


@app.get("/orgs/{org_id}/analytics")
async def org_analytics(org_id: str, current_org: dict = Depends(require_org)):
    if org_id != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return db.get_org_analytics(org_id)


# ── Worker routes ─────────────────────────────────────────────

@app.post("/workers")
async def create_worker(body: CreateWorkerRequest, current_org: dict = Depends(require_org)):
    if body.org_id != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return db.create_worker(body.org_id, body.name)


@app.get("/workers/{worker_id}")
async def get_worker(worker_id: str, current_org: dict = Depends(require_org)):
    worker = db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    assert_org_owns_worker(worker, current_org)
    sessions = db.get_sessions_by_worker(worker_id)
    scores = [s["avg_score"] for s in sessions if s["avg_score"] is not None]
    return {
        **worker,
        "total_sessions": len(sessions),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
    }


@app.get("/workers/{worker_id}/sessions")
async def worker_sessions(worker_id: str, current_org: dict = Depends(require_org)):
    worker = db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    assert_org_owns_worker(worker, current_org)
    return db.get_sessions_by_worker(worker_id)


# ── Session routes ────────────────────────────────────────────

@app.post("/sessions/start")
async def start_session(body: StartSessionRequest, current_org: dict = Depends(require_org)):
    worker = db.get_worker(body.worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    assert_org_owns_worker(worker, current_org)
    return db.create_session(body.worker_id, body.skill_id)


@app.post("/sessions/{session_id}/end")
async def end_session(session_id: str, current_org: dict = Depends(require_org)):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    assert_org_owns_session(session, current_org, db.get_worker)
    return db.end_session(session_id)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, current_org: dict = Depends(require_org)):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    assert_org_owns_session(session, current_org, db.get_worker)
    return session


# ── Skill routes ──────────────────────────────────────────────

class RegisterSkillRequest(BaseModel):
    definition: dict


@app.post("/skills")
async def register_skill(body: RegisterSkillRequest, current_org: dict = Depends(require_org)):
    definition = body.definition
    if "skill_id" not in definition:
        raise HTTPException(status_code=400, detail="definition must include skill_id")
    if "form_rules" not in definition:
        raise HTTPException(status_code=400, detail="definition must include form_rules")
    definition["skill_id"] = f"{current_org['id']}:{definition['skill_id']}"
    db.upsert_skill(definition, org_id=current_org["id"])
    return {"skill_id": definition["skill_id"], "registered": True}


@app.get("/skills")
async def list_skills(current_org: dict = Depends(require_org)):
    return db.list_skills(org_id=current_org["id"])


@app.get("/skills/{skill_id}")
async def get_skill(skill_id: str, current_org: dict = Depends(require_org)):
    skill = db.get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    if skill["org_id"] and skill["org_id"] != current_org["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return skill


# ── Certification routes ───────────────────────────────────────

@app.post("/workers/{worker_id}/certify/{skill_id}")
async def certify_worker(worker_id: str, skill_id: str, current_org: dict = Depends(require_org)):
    worker = db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    assert_org_owns_worker(worker, current_org)

    skill_row = db.get_skill(skill_id)
    if not skill_row:
        raise HTTPException(status_code=404, detail=f"Unknown skill: {skill_id}")
    criteria = skill_row["definition"].get("certification")
    if not criteria:
        raise HTTPException(status_code=400, detail=f"Skill '{skill_id}' has no certification criteria defined")

    history = db.get_worker_skill_history(worker_id, skill_id)
    sessions = history["sessions"]
    total_reps = history["total_reps"]
    good_reps = history["good_reps"]

    # Check minimum sessions
    if len(sessions) < criteria["min_sessions"]:
        return {
            "certified": False,
            "reason": f"Not enough sessions. Completed {len(sessions)}/{criteria['min_sessions']} required.",
        }

    # Compute aggregate metrics
    scores = [s["session"]["avg_score"] for s in sessions if s["session"]["avg_score"] is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    good_form_rate = round(good_reps / total_reps, 2) if total_reps > 0 else 0

    failures = []
    if avg_score < criteria["min_avg_score"]:
        failures.append(f"Average score {avg_score} is below required {criteria['min_avg_score']}")
    if good_form_rate < criteria["min_good_form_rate"]:
        failures.append(f"Good form rate {good_form_rate:.0%} is below required {criteria['min_good_form_rate']:.0%}")

    if failures:
        return {"certified": False, "reason": " | ".join(failures), "avg_score": avg_score,
                "good_form_rate": good_form_rate}

    session_ids = [s["session"]["id"] for s in sessions]
    cert = db.issue_certification(
        worker_id=worker_id,
        skill_id=skill_id,
        org_id=worker["org_id"],
        avg_score=avg_score,
        good_form_rate=good_form_rate,
        session_ids=session_ids,
        valid_days=criteria["cert_valid_days"],
    )

    return {
        "certified": True,
        "cert_id": cert["id"],
        "worker_name": worker["name"],
        "skill_id": skill_id,
        "avg_score": avg_score,
        "good_form_rate": good_form_rate,
        "issued_at": cert["issued_at"],
        "expires_at": cert["expires_at"],
        "token": cert["token"],
    }


@app.get("/workers/{worker_id}/certifications")
async def worker_certifications(worker_id: str, current_org: dict = Depends(require_org)):
    worker = db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    assert_org_owns_worker(worker, current_org)
    return db.get_certifications_by_worker(worker_id)


@app.get("/certifications/{cert_id}/verify")
async def verify_certification(cert_id: str):
    return db.verify_certification(cert_id)