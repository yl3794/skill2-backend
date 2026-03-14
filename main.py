from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

# This is the main endpoint that the frontend will call with pose data
# Can change later to return more detailed feedback if needed and depending on what the model can provide
@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_pose(pose: PoseData):
    prompt = f"""You are a physical trainer coaching someone on safe lifting form.
    Current joint angles:
    - Spine: {pose.spine} degrees (good form = above 160, means straight back)
    - Left knee: {pose.left_knee} degrees, Right knee: {pose.right_knee} degrees (good form = below 150 when lifting)
    - Left hip: {pose.left_hip} degrees, Right hip: {pose.right_hip} degrees

    Give exactly one coaching instruction under 10 words if form needs correction.
    If form is good, say 'Good form, keep it up.'
    Do not use any markdown formatting or asterisks in your response."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )

    tip = message.content[0].text
    is_good = pose.spine > 160 and (pose.left_knee < 150 or pose.right_knee < 150)
    score = 100 if is_good else min(100, max(0, int(pose.spine - 60)))

    return EvaluationResponse(
        coaching_tip=tip,
        is_good_form=is_good,
        score=score
    )

@app.get("/health")
async def health():
    return {"status": "ok"}