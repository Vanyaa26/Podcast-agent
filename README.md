# Agentic Podcast RAG Demo

Standalone prototype showing how SARAL's PDF-to-podcast flow can become an **agentic RAG pipeline**.

This repo is intentionally separate from [SARAL](../SARAL). It mirrors SARAL's backend shape (FastAPI routes, services, staged pipeline, local artifacts) without coupling to the main codebase.

## Pipeline

```text
Upload PDF
  → Parse
  → Index (chunk + embed)
  → Summarize (paper brief)
  → Plan (podcast outline)
  → STOP — user reviews / revises / approves the plan
  → Generate script (RAG-grounded dialogue, honoring the user's directive)
  → Voice (OpenAI TTS, two voices + cue track)
```

## Quick start

### Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install fastapi "uvicorn[standard]" python-multipart pypdf openai numpy pydantic python-dotenv httpx pydub
cp .env.example .env   # add OPENAI_API_KEY for best results
uvicorn app.main:app --reload --port 8100
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The UI uploads a PDF, builds an episode plan, and pauses so you can revise or approve it before any audio is generated. After approval it generates the grounded podcast and lets you raise your hand to ask questions mid-episode.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/podcast/upload-pdf` | Upload PDF, build the episode plan (stops for approval) |
| GET | `/api/podcast/{paper_id}/status` | Pipeline stage status |
| GET | `/api/podcast/{paper_id}/brief` | Paper summary |
| GET | `/api/podcast/{paper_id}/outline` | Proposed podcast outline |
| POST | `/api/podcast/{paper_id}/outline/revise` | Revise the plan with a natural-language instruction |
| POST | `/api/podcast/{paper_id}/generate` | Approve the plan and generate script + audio |
| GET | `/api/podcast/{paper_id}/script` | Grounded dialogue |

## Docs

- **[Presentation doc](docs/PRESENTATION.md)** — slide-ready narrative for demos: user flow, agentic vs deterministic, SARAL comparison, Realtime API proposal
- [Architecture & pipeline details](docs/AGENTIC_RAG_PIPELINE.md)
