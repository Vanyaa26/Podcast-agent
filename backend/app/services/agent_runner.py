"""A small, budgeted tool-calling loop.

This is the agentic core: we hand the model a set of tools and let IT decide
whether and how to call them. We execute whatever it asks for, feed the results
back, and repeat — up to a hard budget so "agentic" never means "unbounded
spend". Every tool call is recorded in a trace so the decision process is
inspectable.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.models import ToolCallTrace
from app.services.summary_service import OpenAIClient
from app.services.tools import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    content: str | None
    trace: list[ToolCallTrace] = field(default_factory=list)
    iterations: int = 0


def extract_json(text: str | None) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences/prose."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def _urls_from_tool_result(tool_name: str, result: Any) -> list[str]:
    if tool_name != "web_search" or not isinstance(result, list):
        return []
    urls: list[str] = []
    for item in result:
        if isinstance(item, dict):
            url = str(item.get("url") or "").strip()
            if url:
                urls.append(url)
    return urls


class AgentRunner:
    """Runs an OpenAI chat with tools until the model returns a final answer.

    The model may issue several rounds of tool calls. After ``max_iters`` rounds
    we make one final call with tools disabled so it is forced to answer.
    """

    def __init__(
        self,
        client: OpenAIClient,
        registry: ToolRegistry,
        max_iters: int = 3,
        temperature: float = 0.4,
        required_first_tool: str | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.max_iters = max_iters
        self.temperature = temperature
        self.required_first_tool = required_first_tool

    def run(self, system: str, user: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        trace: list[ToolCallTrace] = []
        schemas = self.registry.schemas()

        for iteration in range(1, self.max_iters + 1):
            # On the last allowed iteration, disable tools to force a final answer.
            allow_tools = iteration < self.max_iters
            tool_choice: str | dict[str, Any] = "auto"
            if allow_tools and iteration == 1 and self.required_first_tool:
                tool_choice = {
                    "type": "function",
                    "function": {"name": self.required_first_tool},
                }
            assistant_message, tool_calls = self.client.chat_with_tools(
                messages,
                tools=schemas if allow_tools else None,
                temperature=self.temperature,
                tool_choice=tool_choice,
            )
            messages.append(assistant_message)

            if not tool_calls:
                return AgentResult(
                    content=assistant_message.get("content"),
                    trace=trace,
                    iterations=iteration,
                )

            for call in tool_calls:
                name = call["name"]
                args = call.get("arguments") or {}
                tool = self.registry.get(name)
                if tool is None:
                    result: Any = {"error": f"unknown tool '{name}'"}
                    summary, chunk_ids = (f"unknown tool '{name}'", [])
                    source_urls: list[str] = []
                else:
                    try:
                        result = tool.handler(args)
                        summary, chunk_ids = tool.summarize(result)
                    except Exception as exc:  # keep the loop alive on tool failure
                        logger.warning("Tool %s failed: %s", name, exc)
                        result = {"error": str(exc)}
                        summary, chunk_ids = (f"error: {exc}", [])
                    source_urls = _urls_from_tool_result(name, result)

                trace.append(
                    ToolCallTrace(
                        tool=name,
                        arguments=args if isinstance(args, dict) else {},
                        result_summary=summary,
                        chunk_ids=chunk_ids,
                        source_urls=source_urls,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, default=str),
                    }
                )

        # Budget exhausted without a tool-free turn: make a final, tools-off call.
        assistant_message, _ = self.client.chat_with_tools(
            messages + [
                {
                    "role": "user",
                    "content": "Use what you have now and give your final answer as JSON only.",
                }
            ],
            tools=None,
            temperature=self.temperature,
        )
        return AgentResult(
            content=assistant_message.get("content"),
            trace=trace,
            iterations=self.max_iters,
        )
