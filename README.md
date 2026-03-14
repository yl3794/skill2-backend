# Skill2 — Backend

> AI-powered physical skills coaching backend for [Skill2](https://github.com/yl3794/skill2-backend)

Skill2 is a real-time coaching platform that uses AI to evaluate physical form and guide workers through trade skills training. This backend receives pose/angle data from the frontend, calls Claude AI to generate coaching tips, and returns feedback to be spoken aloud to the user.

---

## How It Works

1. Frontend captures camera feed and extracts joint angles using pose detection
2. Angles are sent to the `/evaluate` endpoint every 2 seconds
3. Backend calls Claude AI with the angles and a coaching prompt
4. A short coaching tip is returned and spoken aloud to the user
5. After a session, a competency score is calculated and returned

---

## Tech Stack

- **FastAPI** — Python web framework
- **Anthropic Claude API** — AI coaching language generation
- **Uvicorn** — ASGI server
- **Python-dotenv** — environment variable management

---

## Project Structure

```
skill2-backend/
├── main.py                  # FastAPI app and /evaluate endpoint
├── requirements.txt
├── .env                     # API keys (never commit this)
├── .gitignore
└── skills/
    ├── lifting/
    │   ├── prompts.py       # Claude prompts for lifting coaching
    │   └── evaluation.py    # Angle thresholds and scoring logic
    └── plumbing/            # Placeholder for future skill modules
        ├── prompts.py       # potential
        └── evaluation.py    # potential
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yl3794/skill2-backend
cd skill2-backend
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your `.env` file

Create a `.env` file in the root directory:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get your API key from [console.anthropic.com](https://console.anthropic.com).

### 5. Run the server

```bash
./venv/bin/uvicorn main:app --reload
```

Server runs at `http://localhost:8000`

---

## API Endpoints

### `POST /evaluate`

Receives joint angles and returns a coaching tip and score.

**Request body:**

```json
{
  "back_angle": 45.0,
  "knee_angle": 60.0
}
```

**Response:**

```json
{
  "coaching_tip": "Bend your knees more to protect your back.",
  "is_good_form": false,
  "score": 72
}
```

### `GET /health`

Returns server status.

```json
{ "status": "ok" }
```

---

## Interactive API Docs

Once the server is running, visit:

```
http://localhost:8000/docs
```

You can test all endpoints directly from the browser.

---

## Adding a New Skill Module

To extend Skill2 to a new trade (e.g. plumbing, electrical):

1. Create a new folder under `skills/your_skill/`
2. Add `prompts.py` with your Claude coaching prompt
3. Add `evaluation.py` with your angle thresholds and scoring logic
4. Register the new skill in `main.py`

---

## Team

Built for the AI Hackathon.
