from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    INDEXING = "indexing"
    SUMMARIZING = "summarizing"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    SCRIPTING = "scripting"
    VOICE = "voice"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageInfo(BaseModel):
    status: StageStatus = StageStatus.PENDING
    message: str = ""
    updated_at: datetime | None = None


class PipelineStatus(BaseModel):
    paper_id: str
    filename: str
    current_stage: PipelineStage = PipelineStage.UPLOADED
    stages: dict[str, StageInfo] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class PaperBrief(BaseModel):
    paper_id: str
    title: str
    one_liner: str
    problem: str
    method: str
    results: str
    limitations: str
    key_terms: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)


class OutlineSegment(BaseModel):
    segment_id: str
    title: str
    goal: str
    retrieval_queries: list[str] = Field(default_factory=list)
    estimated_seconds: int = 60


class PodcastOutline(BaseModel):
    paper_id: str
    episode_title: str
    hook: str
    segments: list[OutlineSegment] = Field(default_factory=list)
    closing: str = ""


class ScriptTurn(BaseModel):
    speaker: Literal["host", "guest"]
    text: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class ScriptSegment(BaseModel):
    segment_id: str
    title: str
    turns: list[ScriptTurn] = Field(default_factory=list)
    recap: str = ""


class PodcastScript(BaseModel):
    paper_id: str
    episode_title: str
    segments: list[ScriptSegment] = Field(default_factory=list)


class TurnCue(BaseModel):
    index: int
    segment_id: str
    segment_title: str
    speaker: Literal["host", "guest"]
    text: str
    start_ms: int
    end_ms: int
    source_chunk_ids: list[str] = Field(default_factory=list)


class CueTrack(BaseModel):
    paper_id: str
    total_ms: int
    cues: list[TurnCue] = Field(default_factory=list)


class AnswerTurn(BaseModel):
    speaker: Literal["host", "guest"]
    text: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class ToolCallTrace(BaseModel):
    """One tool invocation the agent chose to make, for inspectability."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    paper_id: str
    question: str
    grounded: bool
    answer_turns: list[AnswerTurn] = Field(default_factory=list)
    audio_url: str | None = None
    resume_ms: int | None = None
    trace: list[ToolCallTrace] = Field(default_factory=list)


class UploadResponse(BaseModel):
    paper_id: str
    message: str
    status_url: str


class ReviseOutlineRequest(BaseModel):
    """A natural-language command to revise the proposed episode plan."""

    instruction: str


class GenerateRequest(BaseModel):
    """Approve the plan and start script + audio generation.

    ``instruction`` is an optional global directive the user wants the writer to
    honor (e.g. tone, focus, what to skip). It is sent to the agent as a user
    command and threaded into every beat.
    """

    instruction: str = ""


class GenerateResponse(BaseModel):
    paper_id: str
    message: str


class VoiceStatus(BaseModel):
    paper_id: str
    status: Literal["pending", "running", "completed", "failed", "not_implemented"]
    message: str
    audio_url: str | None = None


class ChunkRecord(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    section: str = "body"
    page: int | None = None
    embedding: list[float] | None = None


class ParsedDocument(BaseModel):
    paper_id: str
    filename: str
    title: str
    text: str
    page_count: int
    chunks: list[ChunkRecord] = Field(default_factory=list)


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    section: str = "body"
    page: int | None = None


class PipelineArtifacts(BaseModel):
    parsed: ParsedDocument | None = None
    brief: PaperBrief | None = None
    outline: PodcastOutline | None = None
    script: PodcastScript | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
