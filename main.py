from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import os
from dotenv import load_dotenv

from skills.lifting.prompts import get_coaching_prompt
from skills.lifting.evaluation import evaluate

load_dotenv()

app = FastAPI(title="Skill2 API", description="AI-powered physical skills coaching backend")

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


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_pose(pose: PoseData):
    try:
        prompt = get_coaching_prompt(pose)

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )

        coaching_tip = message.content[0].text
        is_good, score = evaluate(pose)

        return EvaluationResponse(
            coaching_tip=coaching_tip,
            is_good_form=is_good,
            score=score
        )

    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}