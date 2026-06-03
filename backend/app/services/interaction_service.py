from __future__ import annotations

import json
import logging
import tempfile
import uuid
from pathlib import Path

from app.config import OPENAI_API_KEY
from app.models import AnswerTurn, AskResponse, ToolCallTrace, TurnCue
from app.services.agent_runner import AgentRunner, extract_json
from app.services.pipeline import ArtifactStore, PaperKB
from app.services.summary_service import OpenAIClient
from app.services.text_utils import clean_for_speech, sanitize_spoken_turn
from app.services.tools import ToolRegistry, make_search_paper_kb, make_web_search
from app.services.voice_service import VoiceService

logger = logging.getLogger(__name__)


class InterruptAgent:
    """Handles a listener jumping into the conversation.

    Truly agentic: the model gets search_paper_kb (paper evidence) and web_search
    (general background) and decides when to call each. Paper KB is always first;
    web is for undefined terms the paper assumes. Every tool call is traced.
    """

    def __init__(
        self,
        client: OpenAIClient | None = None,
        kb: PaperKB | None = None,
        voice: VoiceService | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> None:
        self.client = client or OpenAIClient()
        self.kb = kb or PaperKB()
        self.voice = voice or VoiceService()
        self.artifacts = artifacts or ArtifactStore()

    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY required for transcription.")

        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        suffix = Path(filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
            handle.write(audio_bytes)
            handle.flush()
            with open(handle.name, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
        return (result.text or "").strip()

    def answer(
        self,
        paper_id: str,
        question: str,
        current_cue: TurnCue | None,
        resume_ms: int | None,
        recent_turns: list[dict] | None = None,
    ) -> AskResponse:
        brief = self.artifacts.load_brief(paper_id)
        paper_title = brief.title if brief else "the paper"
        recent_turns = recent_turns or []

        grounded, answer_turns, trace = self._run_agent(
            paper_id, paper_title, question, current_cue, recent_turns
        )

        audio_url = None
        interjection_id = uuid.uuid4().hex[:10]
        try:
            turns_payload = [t.model_dump() for t in answer_turns]
            self.voice.synthesize_interjection(paper_id, interjection_id, turns_payload)
            audio_url = f"/api/podcast/{paper_id}/interjection/{interjection_id}"
        except Exception as exc:
            logger.warning("Interjection synthesis failed: %s", exc)

        return AskResponse(
            paper_id=paper_id,
            question=question,
            grounded=grounded,
            answer_turns=answer_turns,
            audio_url=audio_url,
            resume_ms=resume_ms,
            trace=trace,
        )

    def _run_agent(
        self,
        paper_id: str,
        paper_title: str,
        question: str,
        current_cue: TurnCue | None,
        recent_turns: list[dict],
    ) -> tuple[bool, list[AnswerTurn], list[ToolCallTrace]]:
        context_text = current_cue.text if current_cue else ""
        context_topic = current_cue.segment_title if current_cue else ""

        if not self.client.enabled:
            return self._fallback(paper_id, question, current_cue)

        registry = (
            ToolRegistry()
            .add(make_search_paper_kb(self.kb, paper_id))
            .add(make_web_search())
        )
        runner = AgentRunner(
            self.client,
            registry,
            max_iters=4,
            temperature=0.5,
            required_first_tool="search_paper_kb",
        )

        system = (
            "You are the two podcast hosts ('host' and 'guest') in a LIVE episode about a "
            "research paper. A listener just interrupted with a question. "
            "You have two tools:\n"
            "1) search_paper_kb — passages from THIS paper (always try this first).\n"
            "2) web_search — public web background for concepts the paper uses but does "
            "not define (e.g. 'what is RAG', 'what are iPSCs').\n"
            "Tool policy:\n"
            "- The FIRST tool call MUST be search_paper_kb. Include the listener's question, "
            "the exact line just said, and the current topic when available.\n"
            "- After KB results: if the paper clearly answers, use that. If the question is "
            "about a term/concept the paper assumes but never defines, call web_search for "
            "brief general background.\n"
            "- Never use web_search for paper-specific claims, numbers, or results — only "
            "for missing definitions/background.\n"
            "Answer policy:\n"
            "(a) CLARIFICATION of something already discussed: explain using KB + transcript; "
            "add web background only if needed; never say 'paper doesn't cover that' for a "
            "clarification of something just said.\n"
            "(b) Paper-specific fact not in KB: grounded=false, say the paper doesn't go there.\n"
            "(c) General definition filled by web_search: grounded=false, but explain clearly "
            "and say it's general background, not a claim from this paper.\n"
            "(d) OUT OF SCOPE: briefly decline.\n"
            "Keep 2-3 short spoken turns. End with a natural hand-back in your own words. "
            "Cite chunk_ids from the paper in source_chunk_ids; leave source_chunk_ids empty "
            "for web-only background.\n"
            "When finished, reply with ONLY JSON: "
            '{"grounded": bool, "turns": [{"speaker": "host"|"guest", "text": str, '
            '"source_chunk_ids": [str]}]}.'
        )
        user = json.dumps(
            {
                "paper_title": paper_title,
                "currently_discussing": context_topic,
                "recent_transcript": recent_turns,
                "last_said_on_air": context_text,
                "listener_question": question,
            },
            indent=2,
        )

        result = runner.run(system, user)
        payload = extract_json(result.content)
        answer_turns = self._parse_turns(payload.get("turns"))

        if answer_turns:
            used_paper = any(t.chunk_ids for t in result.trace if t.tool == "search_paper_kb")
            grounded = bool(payload.get("grounded", used_paper))
            return grounded, answer_turns, result.trace

        # Model produced nothing usable: fall back, but keep whatever trace exists.
        grounded, turns, fb_trace = self._fallback(paper_id, question, current_cue)
        return grounded, turns, result.trace or fb_trace

    @staticmethod
    def _parse_turns(turns) -> list[AnswerTurn]:
        if not isinstance(turns, list):
            return []
        parsed: list[AnswerTurn] = []
        for turn in turns[:4]:
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            parsed.append(
                AnswerTurn(
                    speaker="guest" if turn.get("speaker") == "guest" else "host",
                    text=text,
                    source_chunk_ids=[str(x) for x in turn.get("source_chunk_ids", [])],
                )
            )
        return parsed

    def _fallback(
        self,
        paper_id: str,
        question: str,
        current_cue: TurnCue | None,
    ) -> tuple[bool, list[AnswerTurn], list[ToolCallTrace]]:
        """Deterministic safety net when the LLM is unavailable.

        We still perform a real retrieval (so the trace is honest) and surface
        the evidence verbatim or decline — never invent claims.
        """
        queries = [question]
        if current_cue is not None:
            queries.append(current_cue.text)
            queries.append(current_cue.segment_title)
        evidence = self.kb.search(paper_id, queries, top_k=4)

        chunk_ids = [e["chunk_id"] for e in evidence]
        trace = [
            ToolCallTrace(
                tool="search_paper_kb",
                arguments={"queries": [q for q in queries if q]},
                result_summary=(
                    f"{len(evidence)} passage(s): {', '.join(chunk_ids)}"
                    if evidence
                    else "no matching passages found"
                ),
                chunk_ids=chunk_ids,
            )
        ]

        if evidence:
            return (
                True,
                [
                    AnswerTurn(
                        speaker="guest",
                        text=f"From the paper: {evidence[0]['text'][:240]}",
                        source_chunk_ids=[evidence[0]["chunk_id"]],
                    )
                ],
                trace,
            )
        return (
            False,
            [
                AnswerTurn(
                    speaker="host",
                    text="The paper doesn't cover that, so we'd only be guessing.",
                    source_chunk_ids=[],
                )
            ],
            trace,
        )
