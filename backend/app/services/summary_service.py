from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import OPENAI_API_KEY, OPENAI_TEXT_MODEL
from app.models import PaperBrief, ParsedDocument

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(self) -> None:
        self.enabled = bool(OPENAI_API_KEY)

    def generate_json(self, system: str, user: str, temperature: float = 0.3) -> dict[str, Any]:
        if self.enabled:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=OPENAI_TEXT_MODEL,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:
                logger.warning("OpenAI JSON generation failed: %s", exc)
        return {}

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.4,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """One turn of a tool-calling chat.

        Returns ``(assistant_message, tool_calls)`` where ``assistant_message`` is
        an OpenAI-wire-format dict ready to append to the running conversation and
        ``tool_calls`` is a normalized list of ``{id, name, arguments}``. When the
        model returns a final answer instead of calling tools, ``tool_calls`` is
        empty and the answer is in ``assistant_message["content"]``.
        """
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        kwargs: dict[str, Any] = {
            "model": OPENAI_TEXT_MODEL,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        tool_calls: list[dict[str, Any]] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(
                {"id": call.id, "name": call.function.name, "arguments": arguments}
            )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                    },
                }
                for call in message.tool_calls
            ]

        return assistant_message, tool_calls


def _first_paragraph(text: str, max_len: int = 500) -> str:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return text[:max_len]
    return paras[0][:max_len]


def _extract_keywords(text: str, limit: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z]{5,}", text.lower())
    freq: dict[str, int] = {}
    stop = {"paper", "study", "results", "method", "using", "based", "model", "approach"}
    for word in words:
        if word in stop:
            continue
        freq[word] = freq.get(word, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w.title() for w, _ in ranked[:limit]]


class SummaryService:
    def __init__(self, client: OpenAIClient | None = None) -> None:
        self.client = client or OpenAIClient()

    def summarize(self, doc: ParsedDocument) -> PaperBrief:
        excerpt = doc.text[:12000]
        if self.client.enabled:
            payload = self.client.generate_json(
                system=(
                    "You summarize research papers for podcast planning. "
                    "Return JSON with keys: title, one_liner, problem, method, results, "
                    "limitations, key_terms (array), themes (array)."
                ),
                user=f"Paper title guess: {doc.title}\n\nPaper text:\n{excerpt}",
            )
            if payload:
                return PaperBrief(
                    paper_id=doc.paper_id,
                    title=str(payload.get("title") or doc.title),
                    one_liner=str(payload.get("one_liner") or _first_paragraph(doc.text, 200)),
                    problem=str(payload.get("problem") or _first_paragraph(doc.text, 300)),
                    method=str(payload.get("method") or "See paper methodology section."),
                    results=str(payload.get("results") or "See paper results section."),
                    limitations=str(payload.get("limitations") or "Limitations not explicitly extracted."),
                    key_terms=[str(x) for x in payload.get("key_terms", [])][:8],
                    themes=[str(x) for x in payload.get("themes", [])][:6],
                )

        return PaperBrief(
            paper_id=doc.paper_id,
            title=doc.title,
            one_liner=_first_paragraph(doc.text, 200),
            problem=_first_paragraph(doc.text, 350),
            method="The paper proposes an approach described in the uploaded document.",
            results="Key findings are summarized from the extracted PDF text.",
            limitations="Automated extraction may miss figures, tables, and nuanced caveats.",
            key_terms=_extract_keywords(doc.text),
            themes=["Research overview", "Method", "Findings"],
        )
