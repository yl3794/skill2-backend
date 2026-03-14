from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import os
import httpx
from dotenv import load_dotenv

from skills.lifting.prompts import get_coaching_prompt
from skills.lifting.evaluation import evaluate

load_dotenv()

app = FastAPI(title="Skill2 API", description="AI-powered physical skills coaching backend")
POSE_FLASK_URL = os.getenv("POSE_FLASK_URL", "http://192.168.2.54:5000/angles")

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

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_pose(pose: PoseData):
    try:
        return await get_coaching_response(pose)
    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)}")

@app.post("/coach", response_model=EvaluationResponse)
async def coach():
    try:
        async with httpx.AsyncClient() as client_http:
            pose_response = await client_http.get(POSE_FLASK_URL)
            angles = pose_response.json()
        print("Angles received:", angles)  # add this
        pose = PoseData(**angles)
        return await get_coaching_response(pose)
    except Exception as e:
        print("ERROR:", e)  # add this
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}