from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data")).resolve()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
OPENAI_REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")

# Optional: Tavily gives higher-quality web results for the interrupt agent.
# Without it, DuckDuckGo is used (no extra key).
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))

PAPERS_DIR = DATA_DIR / "papers"
INDEXES_DIR = DATA_DIR / "indexes"
BRIEFS_DIR = DATA_DIR / "briefs"
OUTLINES_DIR = DATA_DIR / "outlines"
SCRIPTS_DIR = DATA_DIR / "scripts"
STATUS_DIR = DATA_DIR / "status"
AUDIO_DIR = DATA_DIR / "audio"
CUES_DIR = DATA_DIR / "cues"

for path in (
    PAPERS_DIR,
    INDEXES_DIR,
    BRIEFS_DIR,
    OUTLINES_DIR,
    SCRIPTS_DIR,
    STATUS_DIR,
    AUDIO_DIR,
    CUES_DIR,
):
    path.mkdir(parents=True, exist_ok=True)
