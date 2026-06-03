# Agentic RAG Podcast Plan

This was the initial lightweight planning note. The full presentation-ready architecture document is now:

[Agentic RAG Podcast Pipeline](AGENTIC_RAG_PIPELINE.md)

Standalone demo repo for validating a SARAL-style refactor of paper-to-podcast.

## Why this exists

SARAL's current podcast path is upload → extract text → single LLM prompt → TTS. This demo replaces that with a staged, inspectable, RAG-grounded pipeline.

## Target pipeline

1. Upload PDF
2. Parse + chunk
3. Index (embeddings + local vector store)
4. Summarize (paper brief)
5. Plan (podcast outline with retrieval queries)
6. Script (host/guest dialogue grounded on retrieved chunks)
7. Voice (OpenAI Realtime adapter — stubbed in MVP)

## SARAL alignment

| SARAL concept | Demo equivalent |
|---------------|-----------------|
| `routes/podcast.py` | `backend/app/routes/podcast.py` |
| `services/podcast_service.py` | `backend/app/services/pipeline.py` |
| temp status JSON | `data/status/` |
| frontend upload + poll | `frontend/src/main.js` |

## MVP scope

- Working upload → status → brief → outline → script
- OpenAI when `OPENAI_API_KEY` is set; deterministic fallback otherwise
- Voice endpoint returns stub message until Realtime integration

## Next steps

- Add `search_paper_kb` tool interface for agent loop
- Optional critic/grounding pass on script turns
- OpenAI Realtime session for two-voice synthesis
- Port proven pieces back into SARAL on a feature branch
