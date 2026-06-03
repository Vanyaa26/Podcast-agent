from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.podcast import router as podcast_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Agentic Podcast RAG Demo",
    description="SARAL-like staged pipeline for grounded paper-to-podcast generation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(podcast_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "pipeline": "approval-gated-v2"}
