from __future__ import annotations

import logging
from pathlib import Path

from pydub import AudioSegment

from app.config import AUDIO_DIR, CUES_DIR, OPENAI_API_KEY, OPENAI_TTS_MODEL
from app.models import CueTrack, PodcastScript, TurnCue

logger = logging.getLogger(__name__)

HOST_VOICE = "onyx"
GUEST_VOICE = "nova"
MAX_TURN_CHARS = 600
PAUSE_MS = 350


class VoiceService:
    def __init__(self) -> None:
        self.enabled = bool(OPENAI_API_KEY)

    def synthesize_podcast(self, paper_id: str, script: PodcastScript) -> Path:
        if not self.enabled:
            raise RuntimeError("OPENAI_API_KEY is required for voice synthesis.")

        output_dir = AUDIO_DIR / paper_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "podcast.wav"

        combined = AudioSegment.empty()
        pause = AudioSegment.silent(duration=PAUSE_MS)
        cues: list[TurnCue] = []
        index = 0
        cursor_ms = 0
        clip_paths: list[Path] = []

        for segment in script.segments:
            for turn in segment.turns:
                text = turn.text.strip()
                if not text:
                    continue
                voice = HOST_VOICE if turn.speaker == "host" else GUEST_VOICE
                clip_path = output_dir / f"turn_{index:03d}.wav"
                self._synthesize_clip(text[:MAX_TURN_CHARS], voice, clip_path)
                clip_paths.append(clip_path)

                clip = AudioSegment.from_wav(str(clip_path))
                start_ms = cursor_ms
                end_ms = start_ms + len(clip)
                cues.append(
                    TurnCue(
                        index=index,
                        segment_id=segment.segment_id,
                        segment_title=segment.title,
                        speaker=turn.speaker,
                        text=text,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        source_chunk_ids=turn.source_chunk_ids,
                    )
                )

                combined += clip + pause
                cursor_ms = end_ms + PAUSE_MS
                index += 1

        if index == 0:
            raise RuntimeError("Script has no dialogue turns to synthesize.")

        combined.export(str(output_path), format="wav")

        cue_track = CueTrack(paper_id=paper_id, total_ms=len(combined), cues=cues)
        (CUES_DIR / f"{paper_id}.json").write_text(
            cue_track.model_dump_json(indent=2), encoding="utf-8"
        )

        for clip_path in clip_paths:
            clip_path.unlink(missing_ok=True)
        return output_path

    def synthesize_interjection(self, paper_id: str, interjection_id: str, turns: list[dict]) -> Path:
        if not self.enabled:
            raise RuntimeError("OPENAI_API_KEY is required for voice synthesis.")

        output_dir = AUDIO_DIR / paper_id / "interjections"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{interjection_id}.wav"

        combined = AudioSegment.empty()
        pause = AudioSegment.silent(duration=PAUSE_MS)
        clip_paths: list[Path] = []

        for i, turn in enumerate(turns):
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            speaker = "guest" if turn.get("speaker") == "guest" else "host"
            voice = HOST_VOICE if speaker == "host" else GUEST_VOICE
            clip_path = output_dir / f"{interjection_id}_{i:02d}.wav"
            self._synthesize_clip(text[:MAX_TURN_CHARS], voice, clip_path)
            clip_paths.append(clip_path)
            combined += AudioSegment.from_wav(str(clip_path))
            if i < len(turns) - 1:
                combined += pause

        if len(combined) == 0:
            raise RuntimeError("No interjection turns to synthesize.")

        combined.export(str(output_path), format="wav")
        for clip_path in clip_paths:
            clip_path.unlink(missing_ok=True)
        return output_path

    def get_audio_path(self, paper_id: str) -> Path | None:
        path = AUDIO_DIR / paper_id / "podcast.wav"
        return path if path.exists() and path.stat().st_size > 1000 else None

    def get_cue_track(self, paper_id: str) -> CueTrack | None:
        path = CUES_DIR / f"{paper_id}.json"
        if not path.exists():
            return None
        return CueTrack.model_validate_json(path.read_text(encoding="utf-8"))

    def get_interjection_path(self, paper_id: str, interjection_id: str) -> Path | None:
        path = AUDIO_DIR / paper_id / "interjections" / f"{interjection_id}.wav"
        return path if path.exists() else None

    def _synthesize_clip(self, text: str, voice: str, output_path: Path) -> None:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=voice,
            input=text,
            response_format="wav",
        )
        output_path.write_bytes(response.content)
