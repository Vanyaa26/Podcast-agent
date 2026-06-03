from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.models import (
    AskResponse,
    CueTrack,
    GenerateRequest,
    GenerateResponse,
    PaperBrief,
    PipelineStage,
    PipelineStatus,
    PodcastOutline,
    PodcastScript,
    ReviseOutlineRequest,
    StageStatus,
    TurnCue,
    UploadResponse,
    VoiceStatus,
)
from app.services.document_parser import new_paper_id
from app.services.interaction_service import InterruptAgent
from app.services.pipeline import ArtifactStore, PipelineOrchestrator
from app.services.status_store import StatusStore
from app.services.voice_service import VoiceService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/podcast", tags=["podcast"])

_executor = ThreadPoolExecutor(max_workers=2)
_orchestrator = PipelineOrchestrator()
_status = StatusStore()
_artifacts = ArtifactStore()
_voice = VoiceService()
_interrupt = InterruptAgent(voice=_voice)


def _run_planning(paper_id: str, filename: str, content: bytes) -> None:
    _orchestrator.run_planning(paper_id, filename, content)


def _run_generation(paper_id: str, directive: str) -> None:
    _orchestrator.run_generation(paper_id, directive=directive)


def _plan_review_open(status: PipelineStatus) -> bool:
    """True when planning finished and the user must approve before audio."""
    approval = status.stages.get(PipelineStage.AWAITING_APPROVAL.value)
    scripting = status.stages.get(PipelineStage.SCRIPTING.value)
    if approval is None or approval.status != StageStatus.RUNNING:
        return False
    if scripting is not None and scripting.status != StageStatus.PENDING:
        return False
    return True


@router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    paper_id = new_paper_id()
    _status.create(paper_id, file.filename)
    background_tasks.add_task(_run_planning, paper_id, file.filename, content)

    return UploadResponse(
        paper_id=paper_id,
        message="Upload accepted. Building the episode plan.",
        status_url=f"/api/podcast/{paper_id}/status",
    )


@router.get("/{paper_id}/status", response_model=PipelineStatus)
async def get_status(paper_id: str) -> PipelineStatus:
    status = _status.load(paper_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return status


@router.get("/{paper_id}/brief", response_model=PaperBrief)
async def get_brief(paper_id: str) -> PaperBrief:
    brief = _artifacts.load_brief(paper_id)
    if brief is None:
        raise HTTPException(status_code=404, detail="Brief not ready yet.")
    return brief


@router.get("/{paper_id}/outline", response_model=PodcastOutline)
async def get_outline(paper_id: str) -> PodcastOutline:
    outline = _artifacts.load_outline(paper_id)
    if outline is None:
        raise HTTPException(status_code=404, detail="Outline not ready yet.")
    return outline


@router.post("/{paper_id}/outline/revise", response_model=PodcastOutline)
async def revise_outline(paper_id: str, body: ReviseOutlineRequest) -> PodcastOutline:
    status = _status.load(paper_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if not _plan_review_open(status):
        raise HTTPException(
            status_code=409,
            detail="The plan can only be revised before generation starts.",
        )

    brief = _artifacts.load_brief(paper_id)
    current = _artifacts.load_outline(paper_id)
    if brief is None or current is None:
        raise HTTPException(status_code=404, detail="Plan not ready yet.")

    instruction = (body.instruction or "").strip()
    if not instruction:
        raise HTTPException(status_code=422, detail="Tell the agent what to change.")

    revised = _orchestrator.planner.revise(brief, current, instruction)
    _artifacts.save_outline(revised)
    return revised


@router.post("/{paper_id}/generate", response_model=GenerateResponse)
async def generate_podcast(
    paper_id: str,
    background_tasks: BackgroundTasks,
    body: GenerateRequest | None = None,
) -> GenerateResponse:
    status = _status.load(paper_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if not _plan_review_open(status):
        raise HTTPException(
            status_code=409,
            detail="Generation has already started or the plan is not ready yet.",
        )
    if _artifacts.load_outline(paper_id) is None:
        raise HTTPException(status_code=404, detail="Plan not ready yet.")

    directive = (body.instruction if body else "") or ""
    # Lock approval synchronously so a stale/old worker cannot double-start and
    # so duplicate clicks get a clear 409 on the next request.
    _status.set_stage(
        paper_id,
        PipelineStage.AWAITING_APPROVAL,
        StageStatus.COMPLETED,
        "Plan approved — generating podcast.",
    )
    background_tasks.add_task(_run_generation, paper_id, directive)

    return GenerateResponse(
        paper_id=paper_id,
        message="Plan approved. Generating the podcast.",
    )


@router.get("/{paper_id}/script", response_model=PodcastScript)
async def get_script(paper_id: str) -> PodcastScript:
    script = _artifacts.load_script(paper_id)
    if script is None:
        raise HTTPException(status_code=404, detail="Script not ready yet.")
    return script


@router.get("/{paper_id}/voice", response_model=VoiceStatus)
async def get_voice_status(paper_id: str) -> VoiceStatus:
    status = _status.load(paper_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Paper not found.")

    voice_stage = status.stages.get("voice")
    audio_path = _voice.get_audio_path(paper_id)

    if audio_path:
        return VoiceStatus(
            paper_id=paper_id,
            status="completed",
            message="Podcast audio is ready.",
            audio_url=f"/api/podcast/{paper_id}/audio/stream",
        )

    if voice_stage and voice_stage.status.value == "running":
        return VoiceStatus(
            paper_id=paper_id,
            status="running",
            message=voice_stage.message or "Synthesizing podcast audio…",
            audio_url=None,
        )

    if voice_stage and voice_stage.status.value == "failed":
        return VoiceStatus(
            paper_id=paper_id,
            status="failed",
            message=voice_stage.message or "Voice synthesis failed.",
            audio_url=None,
        )

    return VoiceStatus(
        paper_id=paper_id,
        status="pending",
        message="Waiting for script generation.",
        audio_url=None,
    )


@router.get("/{paper_id}/audio/stream")
async def stream_audio(paper_id: str) -> FileResponse:
    audio_path = _voice.get_audio_path(paper_id)
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Audio not ready yet.")
    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/{paper_id}/audio/download")
async def download_audio(paper_id: str) -> FileResponse:
    audio_path = _voice.get_audio_path(paper_id)
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Audio not ready yet.")
    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename=f"podcast_{paper_id}.wav",
    )


@router.get("/{paper_id}/cues", response_model=CueTrack)
async def get_cues(paper_id: str) -> CueTrack:
    cue_track = _voice.get_cue_track(paper_id)
    if cue_track is None:
        raise HTTPException(status_code=404, detail="Cue track not ready yet.")
    return cue_track


@router.post("/{paper_id}/ask-voice", response_model=AskResponse)
async def ask_voice(
    paper_id: str,
    file: UploadFile = File(...),
    current_index: int = Form(-1),
    resume_ms: int = Form(-1),
) -> AskResponse:
    if _status.load(paper_id) is None:
        raise HTTPException(status_code=404, detail="Paper not found.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio.")

    try:
        question = _interrupt.transcribe(audio_bytes, file.filename or "question.webm")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    if not question:
        raise HTTPException(status_code=422, detail="Could not understand the question.")

    current_cue: TurnCue | None = None
    recent_turns: list[dict] = []
    cue_track = _voice.get_cue_track(paper_id)
    if cue_track and 0 <= current_index < len(cue_track.cues):
        current_cue = cue_track.cues[current_index]
        window_start = max(0, current_index - 3)
        recent_turns = [
            {"speaker": c.speaker, "text": c.text}
            for c in cue_track.cues[window_start : current_index + 1]
        ]

    # Re-anchor resume to the START of the interrupted turn so the episode picks
    # up at a clean sentence boundary (not mid-word) for a smooth re-entry.
    resume_target = resume_ms if resume_ms >= 0 else None
    if current_cue is not None:
        resume_target = current_cue.start_ms

    return _interrupt.answer(
        paper_id=paper_id,
        question=question,
        current_cue=current_cue,
        recent_turns=recent_turns,
        resume_ms=resume_target,
    )


@router.get("/{paper_id}/interjection/{interjection_id}")
async def stream_interjection(paper_id: str, interjection_id: str) -> FileResponse:
    path = _voice.get_interjection_path(paper_id, interjection_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Interjection not found.")
    return FileResponse(path=path, media_type="audio/wav")
