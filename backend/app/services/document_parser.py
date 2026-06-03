from __future__ import annotations

import hashlib
import re
import uuid
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.config import PAPERS_DIR
from app.models import ChunkRecord, ParsedDocument
from app.services.text_utils import normalize_pdf_text


def new_paper_id() -> str:
    return uuid.uuid4().hex[:12]


def _guess_title(text: str, filename: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:20]:
        if 10 <= len(line) <= 180:
            return line
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, paper_id: str, chunk_size: int = 900, overlap: int = 120) -> list[ChunkRecord]:
    sentences = _split_sentences(text)
    chunks: list[ChunkRecord] = []
    current: list[str] = []
    current_len = 0
    idx = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            body = " ".join(current).strip()
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{paper_id}_c{idx}",
                    paper_id=paper_id,
                    text=body,
                    section="body",
                )
            )
            idx += 1
            tail = body[-overlap:] if overlap else ""
            current = [tail, sentence] if tail else [sentence]
            current_len = sum(len(s) for s in current)
        else:
            current.append(sentence)
            current_len += len(sentence)

    if current:
        chunks.append(
            ChunkRecord(
                chunk_id=f"{paper_id}_c{idx}",
                paper_id=paper_id,
                text=" ".join(current).strip(),
                section="body",
            )
        )
    return chunks


class DocumentParser:
    def __init__(self, papers_dir: Path = PAPERS_DIR) -> None:
        self.papers_dir = papers_dir

    def save_upload(self, paper_id: str, filename: str, content: bytes) -> Path:
        dest = self.papers_dir / f"{paper_id}_{Path(filename).name}"
        dest.write_bytes(content)
        return dest

    def parse_pdf_bytes(self, paper_id: str, filename: str, content: bytes) -> ParsedDocument:
        reader = PdfReader(BytesIO(content))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n\n".join(pages).strip()
        text = normalize_pdf_text(text)
        title = _guess_title(text, filename)
        chunks = chunk_text(text, paper_id)
        return ParsedDocument(
            paper_id=paper_id,
            filename=filename,
            title=title,
            text=text,
            page_count=len(reader.pages),
            chunks=chunks,
        )

    @staticmethod
    def content_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()[:16]
