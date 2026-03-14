from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic
import os
import threading
import cv2
import mediapipe as mp
from dotenv import load_dotenv

from angle import calculate_angle
from skills.lifting.prompts import get_coaching_prompt
from skills.lifting.evaluation import evaluate

load_dotenv()

app = FastAPI(title="Skill2 API", description="AI-powered physical skills coaching backend")

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


# ── Shared coaching logic ────────────────────────────────────
async def get_coaching_response(pose: PoseData) -> EvaluationResponse:
    prompt = get_coaching_prompt(pose)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    coaching_tip = message.content[0].text
    is_good, score = evaluate(pose)
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
async def coach():
    try:
        with lock:
            angles = angle_data.copy()
        if not angles:
            raise HTTPException(status_code=503, detail="No pose detected yet. Make sure you are in frame.")
        pose = PoseData(**angles)
        return await get_coaching_response(pose)
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_pose(pose: PoseData):
    try:
        return await get_coaching_response(pose)
    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}