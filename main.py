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

# This is where to plug in Ayush's format
class PoseData(BaseModel):
    back_angle: float
    knee_angle: float
    # add more fields here once we know the output format

class EvaluationResponse(BaseModel):
    coaching_tip: str
    is_good_form: bool
    score: int

# This is the main endpoint that the frontend will call with pose data
# Can change later to return more detailed feedback if needed and depending on what the model can provide
@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_pose(pose: PoseData):
    prompt = f"""You are a physical trainer coaching someone on safe lifting form.
Current joint angles: back angle {pose.back_angle} degrees, knee angle {pose.knee_angle} degrees.
Good form means: back angle under 30 degrees from vertical, knee angle around 90 degrees when lifting.
Give exactly one coaching instruction under 10 words if form needs correction.
If form is good, say 'Good form, keep it up.'
Also return whether form is good (true/false) and a score 0-100."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )

    tip = message.content[0].text
    is_good = pose.back_angle < 30 and 80 < pose.knee_angle < 110
    score = 100 if is_good else max(0, 100 - int(abs(pose.back_angle - 20)))

    return EvaluationResponse(
        coaching_tip=tip,
        is_good_form=is_good,
        score=score
    )

@app.get("/health")
async def health():
    return {"status": "ok"}