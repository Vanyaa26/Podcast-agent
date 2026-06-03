"""Tool layer for the agentic loop.

A Tool bundles together (1) the JSON schema the model sees, (2) a Python
handler that actually runs, and (3) an optional trace extractor that turns a
result into a short, human-readable summary for the agent trace shown in the UI.

The model — not our code — decides when to call these tools. That is what makes
the pipeline agentic rather than a fixed retrieve-then-generate chain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from app.services.pipeline import PaperKB
from app.services.web_search_service import WebSearchService


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    # Optional: (result) -> (summary_text, chunk_ids). Falls back to a generic
    # summary when not provided.
    trace_extractor: Callable[[Any], tuple[str, list[str]]] | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def summarize(self, result: Any) -> tuple[str, list[str]]:
        if self.trace_extractor is not None:
            return self.trace_extractor(result)
        text = json.dumps(result, default=str)
        return (text[:160] + ("…" if len(text) > 160 else ""), [])


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> "ToolRegistry":
        self.tools[tool.name] = tool
        return self

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)


def make_search_paper_kb(kb: PaperKB, paper_id: str, default_top_k: int = 3) -> Tool:
    """Build the search_paper_kb tool bound to a single paper.

    The model passes one or more natural-language queries; we embed them once
    and return the most relevant chunks (id, score, trimmed text).
    """

    def handler(args: dict[str, Any]) -> list[dict[str, Any]]:
        raw = args.get("queries")
        if isinstance(raw, str):
            raw = [raw]
        queries = [str(q).strip() for q in (raw or []) if str(q).strip()][:4]
        if not queries:
            return []
        top_k = args.get("top_k", default_top_k)
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = default_top_k
        top_k = max(1, min(top_k, 5))
        return kb.search(paper_id, queries, top_k=top_k)

    def extract_trace(result: Any) -> tuple[str, list[str]]:
        if not isinstance(result, list) or not result:
            return ("no matching passages found", [])
        ids = [str(r.get("chunk_id")) for r in result if isinstance(r, dict)]
        top = result[0].get("score") if isinstance(result[0], dict) else None
        summary = f"{len(result)} passage(s) (top score {top}): {', '.join(ids)}"
        return (summary, ids)

    return Tool(
        name="search_paper_kb",
        description=(
            "Search the indexed knowledge base of THIS research paper for passages "
            "relevant to one or more queries. Returns the most relevant chunks with "
            "their chunk_id, similarity score, and text. Call this whenever you need "
            "evidence from the paper to answer accurately or to re-find the exact "
            "passage behind something the hosts just said. You may call it more than "
            "once with different queries if the first results are not enough."
        ),
        parameters={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "1-4 search phrases. For a clarification, include BOTH the "
                        "listener's question and the exact line that was just said."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many chunks to return (1-5). Default 3.",
                },
            },
            "required": ["queries"],
        },
        handler=handler,
        trace_extractor=extract_trace,
    )


def make_web_search(service: WebSearchService | None = None, default_max: int = 3) -> Tool:
    """Build the web_search tool for general background the paper may not define."""

    web = service or WebSearchService()

    def handler(args: dict[str, Any]) -> list[dict[str, Any]]:
        query = str(args.get("query") or "").strip()
        if not query:
            return []
        max_results = args.get("max_results", default_max)
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = default_max
        max_results = max(1, min(max_results, 5))
        return web.search(query, max_results=max_results)

    def extract_trace(result: Any) -> tuple[str, list[str]]:
        if not isinstance(result, list) or not result:
            return ("no web results found", [])
        titles = [str(r.get("title", ""))[:40] for r in result if isinstance(r, dict)]
        return (f"{len(result)} web hit(s): {'; '.join(t for t in titles if t)}", [])

    return Tool(
        name="web_search",
        description=(
            "Search the public web for general background on a concept, acronym, or "
            "method that the paper uses but does not define (e.g. 'what is RAG', "
            "'what are iPSCs'). Use AFTER search_paper_kb when the paper KB is weak or "
            "does not contain a definition. Do NOT use for paper-specific claims, results, "
            "or numbers — those must come from search_paper_kb. Results are external "
            "background; say so briefly when you use them."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise web search query for the missing background.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return (1-5). Default 3.",
                },
            },
            "required": ["query"],
        },
        handler=handler,
        trace_extractor=extract_trace,
    )
