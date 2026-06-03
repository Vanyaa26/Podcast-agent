from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import BRIEFS_DIR, OUTLINES_DIR, SCRIPTS_DIR
from app.models import (
    OutlineSegment,
    PaperBrief,
    ParsedDocument,
    PodcastOutline,
    PodcastScript,
)
from app.services.document_parser import DocumentParser
from app.services.summary_service import OpenAIClient, SummaryService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class PlannerService:
    def __init__(self, client: OpenAIClient | None = None) -> None:
        self.client = client or OpenAIClient()

    def plan(self, doc: ParsedDocument, brief: PaperBrief) -> PodcastOutline:
        if self.client.enabled:
            payload = self.client.generate_json(
                system=(
                    "Create a podcast outline grounded in a research paper. "
                    "Return JSON: episode_title, hook, closing, segments (array of "
                    "{segment_id, title, goal, retrieval_queries, estimated_seconds}). "
                    "Use 4-5 segments. retrieval_queries should help fetch paper evidence."
                ),
                user=json.dumps(
                    {
                        "title": brief.title,
                        "one_liner": brief.one_liner,
                        "themes": brief.themes,
                        "key_terms": brief.key_terms,
                    },
                    indent=2,
                ),
            )
            outline = self._outline_from_payload(payload, doc.paper_id, brief)
            if outline is not None:
                return outline

        return self._fallback_outline(doc.paper_id, brief)

    def revise(
        self,
        brief: PaperBrief,
        current: PodcastOutline,
        instruction: str,
    ) -> PodcastOutline:
        """Rewrite the proposed plan according to a user's natural-language command.

        The user is in control here: they can ask to remove a section, add a
        comparison, reorder, go deeper on one part, change tone, or drop hype.
        The agent returns a new structured outline; we never silently ignore the
        instruction.
        """
        instruction = (instruction or "").strip()
        if not instruction or not self.client.enabled:
            return current

        payload = self.client.generate_json(
            system=(
                "You revise a podcast episode plan for a research paper based on the "
                "user's instruction. The user decides what the episode should cover. "
                "Apply the instruction faithfully: remove, add, reorder, retitle, refocus, "
                "merge, or re-scope segments as asked. Keep segments grounded in the paper "
                "and regenerate retrieval_queries so each segment can fetch the right "
                "evidence. Do NOT invent findings not implied by the brief. "
                "Return JSON: episode_title, hook, closing, segments (array of "
                "{segment_id, title, goal, retrieval_queries, estimated_seconds})."
            ),
            user=json.dumps(
                {
                    "paper_title": brief.title,
                    "one_liner": brief.one_liner,
                    "themes": brief.themes,
                    "key_terms": brief.key_terms,
                    "current_plan": current.model_dump(),
                    "user_instruction": instruction,
                },
                indent=2,
            ),
            temperature=0.4,
        )
        outline = self._outline_from_payload(payload, current.paper_id, brief)
        return outline if outline is not None else current

    def _outline_from_payload(
        self, payload: dict, paper_id: str, brief: PaperBrief
    ) -> PodcastOutline | None:
        if not payload or not payload.get("segments"):
            return None
        segments = []
        for idx, seg in enumerate(payload["segments"]):
            if not isinstance(seg, dict):
                continue
            segments.append(
                {
                    "segment_id": str(seg.get("segment_id") or f"seg_{idx+1}"),
                    "title": str(seg.get("title") or f"Segment {idx+1}"),
                    "goal": str(seg.get("goal") or ""),
                    "retrieval_queries": [
                        str(q) for q in seg.get("retrieval_queries", [])[:3]
                    ]
                    or [brief.title],
                    "estimated_seconds": int(seg.get("estimated_seconds") or 60),
                }
            )
        if not segments:
            return None
        return PodcastOutline(
            paper_id=paper_id,
            episode_title=str(payload.get("episode_title") or brief.title),
            hook=str(payload.get("hook") or brief.one_liner),
            segments=segments,  # type: ignore[arg-type]
            closing=str(payload.get("closing") or "Thanks for listening."),
        )

    def _fallback_outline(self, paper_id: str, brief: PaperBrief) -> PodcastOutline:
        return PodcastOutline(
            paper_id=paper_id,
            episode_title=f"Understanding {brief.title}",
            hook=brief.one_liner,
            segments=[
                {
                    "segment_id": "seg_1",
                    "title": "Why this paper matters",
                    "goal": "Introduce the problem and stakes",
                    "retrieval_queries": [brief.problem, brief.title],
                    "estimated_seconds": 60,
                },
                {
                    "segment_id": "seg_2",
                    "title": "How they approached it",
                    "goal": "Explain the method in plain language",
                    "retrieval_queries": [brief.method, "methodology"],
                    "estimated_seconds": 75,
                },
                {
                    "segment_id": "seg_3",
                    "title": "What they found",
                    "goal": "Cover results with evidence",
                    "retrieval_queries": [brief.results, "results evaluation"],
                    "estimated_seconds": 75,
                },
                {
                    "segment_id": "seg_4",
                    "title": "Limits and what's next",
                    "goal": "Discuss limitations and future work",
                    "retrieval_queries": [brief.limitations, "future work"],
                    "estimated_seconds": 60,
                },
            ],  # type: ignore[arg-type]
            closing="That wraps our grounded walkthrough of the paper.",
        )


class PaperKB:
    """The search_paper_kb tool: retrieves grounded evidence from one paper."""

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.vector_store = vector_store or VectorStore()

    def search(self, paper_id: str, queries: list[str], top_k: int = 3) -> list[dict]:
        hits = self.vector_store.search_many(paper_id, queries, top_k=top_k)
        return [
            {
                "chunk_id": hit.chunk_id,
                "score": round(hit.score, 3),
                "text": hit.text[:500],
            }
            for hit in hits
        ]


class DialogueService:
    """Agentic chunked writer.

    Writes the episode beat-by-beat. For each beat it (1) calls the
    search_paper_kb tool for fresh evidence, (2) reads running memory
    (covered points + last verbatim turns), (3) writes the beat in a
    role-aware way (opening greets once, middle transitions, closing wraps),
    and (4) emits a one-line recap that feeds forward. No giant single prompt,
    no repeated greetings.
    """

    MAX_TURNS_PER_BEAT = 8

    def __init__(
        self,
        client: OpenAIClient | None = None,
        vector_store: VectorStore | None = None,
        kb: PaperKB | None = None,
    ) -> None:
        self.client = client or OpenAIClient()
        self.kb = kb or PaperKB(vector_store)

    def generate(
        self,
        brief: PaperBrief,
        outline: PodcastOutline,
        directive: str = "",
    ) -> PodcastScript:
        script_segments: list[dict] = []
        covered_points: list[str] = []
        last_turns: list[dict] = []
        total = len(outline.segments)
        directive = (directive or "").strip()

        for idx, seg in enumerate(outline.segments):
            role = "opening" if idx == 0 else ("closing" if idx == total - 1 else "middle")
            evidence = self.kb.search(outline.paper_id, seg.retrieval_queries, top_k=3)

            turns, recap = self._write_beat(
                role=role,
                position=idx + 1,
                total=total,
                seg=seg,
                brief=brief,
                outline=outline,
                evidence=evidence,
                covered_points=covered_points,
                last_turns=last_turns,
                directive=directive,
            )

            script_segments.append(
                {
                    "segment_id": seg.segment_id,
                    "title": seg.title,
                    "turns": turns,
                    "recap": recap,
                }
            )
            if recap:
                covered_points.append(recap)
            last_turns = turns[-2:]

        return PodcastScript(
            paper_id=outline.paper_id,
            episode_title=outline.episode_title,
            segments=script_segments,  # type: ignore[arg-type]
        )

    def _write_beat(
        self,
        role: str,
        position: int,
        total: int,
        seg: OutlineSegment,
        brief: PaperBrief,
        outline: PodcastOutline,
        evidence: list[dict],
        covered_points: list[str],
        last_turns: list[dict],
        directive: str = "",
    ) -> tuple[list[dict], str]:
        if self.client.enabled and evidence:
            role_rules = {
                "opening": (
                    "This is the OPENING beat of the episode. Greet the listener ONCE, "
                    f"use this hook to start: \"{outline.hook}\". Introduce the topic naturally."
                ),
                "middle": (
                    "This is a MIDDLE beat. Do NOT greet or welcome the listener again. "
                    "Open by transitioning naturally from the previous point (use the "
                    "'last_turns' provided), then move into this beat."
                ),
                "closing": (
                    "This is the CLOSING beat. Do NOT greet. Wrap up the discussion, "
                    f"land the takeaway, and close with: \"{outline.closing}\". Introduce no new topics."
                ),
            }[role]

            directive_rule = (
                "The user gave a directive for this episode; follow it for tone, depth, "
                "focus, and what to emphasize or avoid. "
                if directive
                else ""
            )

            system = (
                "You are writing ONE beat of a single, continuous two-person podcast "
                "about a research paper. Speakers are 'host' (curious, asks questions) and "
                "'guest' (the expert, explains with evidence). "
                "Ground every paper-specific claim in the provided evidence chunks; cite the "
                "chunk_ids you used in source_chunk_ids. Keep turns short and spoken (under ~45 words). "
                "It must read as a continuation of one ongoing conversation, never a fresh episode. "
                f"{role_rules} {directive_rule}"
                "Return JSON: {turns: [{speaker, text, source_chunk_ids}], recap: one short sentence "
                "summarizing what THIS beat covered (for continuity memory)}. "
                f"Use {3 if role != 'middle' else 4}-6 turns."
            )

            user = json.dumps(
                {
                    "paper_title": brief.title,
                    "user_directive": directive,
                    "beat_position": f"{position} of {total}",
                    "beat_title": seg.title,
                    "beat_goal": seg.goal,
                    "points_already_covered": covered_points,
                    "last_turns": last_turns,
                    "evidence": evidence,
                },
                indent=2,
            )

            payload = self.client.generate_json(system, user, temperature=0.6)
            turns = self._normalize_turns(payload.get("turns"))
            recap = str(payload.get("recap") or "").strip()
            if turns:
                if not recap:
                    recap = f"Discussed {seg.title.lower()}."
                return turns, recap

        return self._fallback_beat(role, seg, brief, evidence)

    def _normalize_turns(self, turns) -> list[dict]:
        if not isinstance(turns, list):
            return []
        normalized = []
        for turn in turns[: self.MAX_TURNS_PER_BEAT]:
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            normalized.append(
                {
                    "speaker": "guest" if turn.get("speaker") == "guest" else "host",
                    "text": text,
                    "source_chunk_ids": [str(x) for x in turn.get("source_chunk_ids", [])],
                }
            )
        return normalized

    def _fallback_beat(
        self,
        role: str,
        seg: OutlineSegment,
        brief: PaperBrief,
        evidence: list[dict],
    ) -> tuple[list[dict], str]:
        chunk_ids = [e["chunk_id"] for e in evidence[:2]]
        snippet = evidence[0]["text"][:220] if evidence else brief.one_liner

        if role == "opening":
            opener = {
                "speaker": "host",
                "text": f"Welcome! Today we're unpacking {brief.title}. {brief.one_liner}",
                "source_chunk_ids": chunk_ids[:1],
            }
        elif role == "closing":
            opener = {
                "speaker": "host",
                "text": f"So to wrap up on {seg.title.lower()} - what's the takeaway?",
                "source_chunk_ids": [],
            }
        else:
            opener = {
                "speaker": "host",
                "text": f"Building on that, let's turn to {seg.title.lower()}.",
                "source_chunk_ids": [],
            }

        turns = [
            opener,
            {
                "speaker": "guest",
                "text": f"{seg.goal}. From the paper: {snippet}",
                "source_chunk_ids": chunk_ids,
            },
            {
                "speaker": "host",
                "text": "What should listeners take from this part?",
                "source_chunk_ids": [],
            },
            {
                "speaker": "guest",
                "text": f"In short: {brief.one_liner}",
                "source_chunk_ids": chunk_ids[:1],
            },
        ]
        return turns, f"Discussed {seg.title.lower()}."


class ArtifactStore:
    def __init__(self) -> None:
        self.briefs_dir = BRIEFS_DIR
        self.outlines_dir = OUTLINES_DIR
        self.scripts_dir = SCRIPTS_DIR

    def save_brief(self, brief: PaperBrief) -> None:
        self._write(self.briefs_dir / f"{brief.paper_id}.json", brief)

    def save_outline(self, outline: PodcastOutline) -> None:
        self._write(self.outlines_dir / f"{outline.paper_id}.json", outline)

    def save_script(self, script: PodcastScript) -> None:
        self._write(self.scripts_dir / f"{script.paper_id}.json", script)

    def load_brief(self, paper_id: str) -> PaperBrief | None:
        return self._load(self.briefs_dir / f"{paper_id}.json", PaperBrief)

    def load_outline(self, paper_id: str) -> PodcastOutline | None:
        return self._load(self.outlines_dir / f"{paper_id}.json", PodcastOutline)

    def load_script(self, paper_id: str) -> PodcastScript | None:
        return self._load(self.scripts_dir / f"{paper_id}.json", PodcastScript)

    @staticmethod
    def _write(path: Path, model) -> None:
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _load(path: Path, model_cls):
        if not path.exists():
            return None
        return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


class PipelineOrchestrator:
    def __init__(self) -> None:
        from app.services.status_store import StatusStore

        self.parser = DocumentParser()
        self.vector_store = VectorStore()
        self.kb = PaperKB(self.vector_store)
        self.summary = SummaryService()
        self.planner = PlannerService()
        self.dialogue = DialogueService(kb=self.kb)
        self.artifacts = ArtifactStore()
        self.status = StatusStore()

    def run_planning(self, paper_id: str, filename: str, content: bytes) -> None:
        """Stage 1: parse -> index -> brief -> outline, then STOP for user approval.

        We deliberately do not generate script or audio yet. The user reviews the
        proposed episode plan (and can steer it) before any expensive synthesis.
        """
        from app.models import PipelineStage, StageStatus

        try:
            self.status.set_stage(
                paper_id,
                PipelineStage.PARSING,
                StageStatus.RUNNING,
                "Extracting text from PDF",
            )
            self.parser.save_upload(paper_id, filename, content)
            doc = self.parser.parse_pdf_bytes(paper_id, filename, content)
            self.status.set_stage(
                paper_id,
                PipelineStage.PARSING,
                StageStatus.COMPLETED,
                f"Parsed {doc.page_count} pages, {len(doc.chunks)} chunks",
            )

            self.status.set_stage(
                paper_id,
                PipelineStage.INDEXING,
                StageStatus.RUNNING,
                "Building paper knowledge base",
            )
            indexed = self.vector_store.index_chunks(doc.chunks)
            self.status.set_stage(
                paper_id,
                PipelineStage.INDEXING,
                StageStatus.COMPLETED,
                f"Indexed {len(indexed)} chunks",
            )

            self.status.set_stage(
                paper_id,
                PipelineStage.SUMMARIZING,
                StageStatus.RUNNING,
                "Generating paper brief",
            )
            brief = self.summary.summarize(doc)
            self.artifacts.save_brief(brief)
            self.status.set_stage(
                paper_id,
                PipelineStage.SUMMARIZING,
                StageStatus.COMPLETED,
                "Paper brief ready",
            )

            self.status.set_stage(
                paper_id,
                PipelineStage.PLANNING,
                StageStatus.RUNNING,
                "Planning podcast outline",
            )
            outline = self.planner.plan(doc, brief)
            self.artifacts.save_outline(outline)
            self.status.set_stage(
                paper_id,
                PipelineStage.PLANNING,
                StageStatus.COMPLETED,
                f"{len(outline.segments)} segments planned",
            )

            self.status.set_stage(
                paper_id,
                PipelineStage.AWAITING_APPROVAL,
                StageStatus.RUNNING,
                "Episode plan ready — review it and generate when you're happy.",
            )
        except Exception:
            logger.exception("Planning failed for %s", paper_id)
            self.status.mark_failed(paper_id, "Planning failed.")

    def run_generation(self, paper_id: str, directive: str = "") -> None:
        """Stage 2: script -> audio, using the approved (possibly edited) outline."""
        from app.models import PipelineStage, StageStatus

        try:
            status = self.status.load(paper_id)
            if status is None:
                raise RuntimeError("Paper status missing.")

            approval = status.stages.get(PipelineStage.AWAITING_APPROVAL.value)
            if approval is None or approval.status != StageStatus.COMPLETED:
                raise RuntimeError(
                    "Generation refused: plan was not approved. "
                    "Restart the backend if you still see auto-generation after upload."
                )

            brief = self.artifacts.load_brief(paper_id)
            outline = self.artifacts.load_outline(paper_id)
            if brief is None or outline is None:
                raise RuntimeError("Brief or outline missing; cannot generate.")

            self.status.set_stage(
                paper_id,
                PipelineStage.SCRIPTING,
                StageStatus.RUNNING,
                "Generating RAG-grounded dialogue",
            )
            script = self.dialogue.generate(brief, outline, directive=directive)
            self.artifacts.save_script(script)
            self.status.set_stage(
                paper_id,
                PipelineStage.SCRIPTING,
                StageStatus.COMPLETED,
                f"{sum(len(s.turns) for s in script.segments)} dialogue turns",
            )

            from app.services.voice_service import VoiceService

            voice_service = VoiceService()
            turn_count = sum(len(s.turns) for s in script.segments)
            self.status.set_stage(
                paper_id,
                PipelineStage.VOICE,
                StageStatus.RUNNING,
                f"Synthesizing {turn_count} dialogue turns",
            )
            audio_path = voice_service.synthesize_podcast(paper_id, script)
            self.status.set_stage(
                paper_id,
                PipelineStage.VOICE,
                StageStatus.COMPLETED,
                f"Podcast audio ready ({audio_path.name})",
            )
            self.status.set_stage(
                paper_id,
                PipelineStage.COMPLETED,
                StageStatus.COMPLETED,
                "Pipeline complete — podcast ready to play",
            )
        except Exception as exc:
            logger.exception("Generation failed for %s", paper_id)
            self.status.mark_failed(paper_id, str(exc))
