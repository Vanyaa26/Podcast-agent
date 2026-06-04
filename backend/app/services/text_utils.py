from __future__ import annotations

import re
from typing import Any


_CHUNK_ID_RE = re.compile(r"\b(?:chunk_id|chunk ids?|source_chunk_ids?)\s*[:=]\s*[\w, \-\[\]\"']+", re.IGNORECASE)
_BARE_CHUNK_RE = re.compile(r"\b[a-f0-9]{8,32}_c\d+\b", re.IGNORECASE)


def normalize_pdf_text(text: str) -> str:
    """Clean deterministic PDF extraction artifacts before chunking."""
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


def clean_for_speech(text: str) -> str:
    """Remove metadata and whitespace that should never be read aloud."""
    text = text or ""
    text = _CHUNK_ID_RE.sub("", text)
    text = _BARE_CHUNK_RE.sub("", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" -:\n\t")


def sanitize_spoken_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a model turn with only spoken text sanitized."""
    cleaned = dict(turn)
    cleaned["text"] = clean_for_speech(str(cleaned.get("text") or ""))
    return cleaned
