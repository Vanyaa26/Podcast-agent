"""Deterministic tests (no OpenAI calls).

Run: python -m tests.test_agentic   (from backend/, venv active)
"""
from __future__ import annotations

import json

from app.models import OutlineSegment, PaperBrief, ParsedDocument, PodcastOutline
from app.services.pipeline import DialogueService, PaperKB, PlannerService
from app.services.summary_service import OpenAIClient


class _DisabledClient(OpenAIClient):
    def __init__(self) -> None:
        self.enabled = False


class _StubKB(PaperKB):
    def __init__(self) -> None:
        pass

    def search(self, paper_id, queries, top_k=3):
        return [{"chunk_id": "c0", "score": 0.9, "text": "Evidence text from the paper."}]


def _make_inputs():
    doc = ParsedDocument(
        paper_id="p1", filename="x.pdf", title="Test Paper", text="...", page_count=1
    )
    brief = PaperBrief(
        paper_id="p1",
        title="Test Paper",
        one_liner="A one liner.",
        problem="The problem.",
        method="The method.",
        results="The results.",
        limitations="The limits.",
    )
    segs = [
        OutlineSegment(segment_id="s1", title="Why it matters", goal="Introduce", retrieval_queries=["q"]),
        OutlineSegment(segment_id="s2", title="Method", goal="Explain method", retrieval_queries=["q"]),
        OutlineSegment(segment_id="s3", title="Results", goal="Cover results", retrieval_queries=["q"]),
        OutlineSegment(segment_id="s4", title="Takeaways", goal="Wrap up", retrieval_queries=["q"]),
    ]
    outline = PodcastOutline(
        paper_id="p1",
        episode_title="Understanding Test Paper",
        hook="Here's why this matters.",
        segments=segs,
        closing="Thanks for listening.",
    )
    return doc, brief, outline


def test_single_greeting_and_roles():
    doc, brief, outline = _make_inputs()
    service = DialogueService(client=_DisabledClient(), kb=_StubKB())
    script = service.generate(brief, outline)

    all_text = " ".join(t.text.lower() for s in script.segments for t in s.turns)
    greetings = all_text.count("welcome")
    assert greetings == 1, f"expected exactly one greeting, found {greetings}"

    # First segment opens with a greeting; middle segments must not.
    first_open = script.segments[0].turns[0].text.lower()
    assert "welcome" in first_open, "opening beat should greet"
    for seg in script.segments[1:]:
        assert "welcome" not in seg.turns[0].text.lower(), "non-opening beats must not greet"

    # Every segment carries a recap (running memory feed-forward).
    assert all(s.recap for s in script.segments), "each beat should emit a recap"
    print("test_single_greeting_and_roles: OK")


def test_chunk_overlap_keeps_sentence_boundaries():
    from app.services.document_parser import chunk_text

    text = (
        "First sentence explains the setup. "
        "Second sentence lists components. "
        "Third sentence gives results. "
        "Fourth sentence gives the takeaway."
    )
    chunks = chunk_text(text, "p1", chunk_size=70, overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert not chunk.text.startswith("entence"), chunk.text
        assert not chunk.text.startswith("omponents"), chunk.text
    print("test_chunk_overlap_keeps_sentence_boundaries: OK")


class _OneShotToolClient(OpenAIClient):
    """Asks for one tool call, then returns a final JSON answer."""

    def __init__(self, final_content: str) -> None:
        self.enabled = True
        self.calls = 0
        self._final = final_content

    def chat_with_tools(self, messages, tools=None, temperature=0.4, tool_choice="auto"):
        self.calls += 1
        if self.calls == 1:
            assistant = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_paper_kb",
                            "arguments": '{"queries": ["what is X"]}',
                        },
                    }
                ],
            }
            return assistant, [
                {"id": "call_1", "name": "search_paper_kb", "arguments": {"queries": ["what is X"]}}
            ]
        return {"role": "assistant", "content": self._final}, []


class _AlwaysToolClient(OpenAIClient):
    """Always wants a tool call when tools are offered (tests the budget guard)."""

    def __init__(self, final_content: str) -> None:
        self.enabled = True
        self.tool_rounds = 0
        self._final = final_content

    def chat_with_tools(self, messages, tools=None, temperature=0.4, tool_choice="auto"):
        if tools:
            self.tool_rounds += 1
            assistant = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{self.tool_rounds}",
                        "type": "function",
                        "function": {"name": "search_paper_kb", "arguments": '{"queries": ["q"]}'},
                    }
                ],
            }
            return assistant, [
                {"id": f"call_{self.tool_rounds}", "name": "search_paper_kb", "arguments": {"queries": ["q"]}}
            ]
        return {"role": "assistant", "content": self._final}, []


class _ToolChoiceClient(OpenAIClient):
    """Records whether the runner forced the first tool call."""

    def __init__(self, final_content: str) -> None:
        self.enabled = True
        self.calls = 0
        self.first_tool_choice = None
        self._final = final_content

    def chat_with_tools(self, messages, tools=None, temperature=0.4, tool_choice="auto"):
        self.calls += 1
        if self.calls == 1:
            self.first_tool_choice = tool_choice
            assistant = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_paper_kb", "arguments": '{"queries": ["forced"]}'},
                    }
                ],
            }
            return assistant, [
                {"id": "call_1", "name": "search_paper_kb", "arguments": {"queries": ["forced"]}}
            ]
        return {"role": "assistant", "content": self._final}, []


class _StubWebSearch:
    def search(self, query, max_results=3):
        return [
            {
                "title": "Retrieval-Augmented Generation",
                "snippet": "RAG combines retrieval with language generation.",
                "url": "https://example.com/rag",
                "source": "web",
            }
        ]


class _KbThenWebClient(OpenAIClient):
    """KB first, then web, then final answer."""

    def __init__(self, final_content: str) -> None:
        self.enabled = True
        self.round = 0
        self._final = final_content

    def chat_with_tools(self, messages, tools=None, temperature=0.4, tool_choice="auto"):
        self.round += 1
        if self.round == 1:
            return (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "search_paper_kb", "arguments": '{"queries": ["RAG"]}'},
                        }
                    ],
                },
                [{"id": "c1", "name": "search_paper_kb", "arguments": {"queries": ["RAG"]}}],
            )
        if tools and self.round == 2:
            return (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query": "what is RAG"}'},
                        }
                    ],
                },
                [{"id": "c2", "name": "web_search", "arguments": {"query": "what is RAG"}}],
            )
        return {"role": "assistant", "content": self._final}, []


class _FallbackAnswerClient(OpenAIClient):
    def __init__(self) -> None:
        self.enabled = True

    def generate_json(self, system, user, temperature=0.3):
        return {
            "turns": [
                {
                    "speaker": "guest",
                    "text": (
                        "An ablation is a controlled test where parts of the system are "
                        "removed or changed to see which component is responsible for the result."
                    ),
                    "source_chunk_ids": ["auto_c2"],
                },
                {
                    "speaker": "host",
                    "text": "So for AutoScientist, it helps explain what each agent or mechanism contributes.",
                    "source_chunk_ids": ["auto_c2"],
                },
            ]
        }


class _RawChunkKB(PaperKB):
    def __init__(self) -> None:
        pass

    def search(self, paper_id, queries, top_k=3):
        return [
            {
                "chunk_id": "auto_c2",
                "score": 0.91,
                "text": (
                    "t be tracked to avoid repeated exploration. The ablation studies "
                    "remove individual AutoScientist components to test their contribution."
                ),
            }
        ]


class _BeatSearchClient(OpenAIClient):
    def __init__(self, payload: dict) -> None:
        self.enabled = True
        self._payload = payload
        self.calls = 0
        self.first_tool_choice = None

    def chat_with_tools(self, messages, tools=None, temperature=0.4, tool_choice="auto"):
        self.calls += 1
        if self.calls == 1:
            self.first_tool_choice = tool_choice
            return (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "beat_search",
                            "type": "function",
                            "function": {
                                "name": "search_paper_kb",
                                "arguments": '{"queries": ["AUTOSCIENTISTS ablations no analyst cross-agent feedback"], "top_k": 4}',
                            },
                        }
                    ],
                },
                [
                    {
                        "id": "beat_search",
                        "name": "search_paper_kb",
                        "arguments": {
                            "queries": [
                                "AUTOSCIENTISTS ablations no analyst cross-agent feedback"
                            ],
                            "top_k": 4,
                        },
                    }
                ],
            )
        return {"role": "assistant", "content": json.dumps(self._payload)}, []


class _AblationKB(PaperKB):
    def __init__(self) -> None:
        pass

    def search(self, paper_id, queries, top_k=3):
        self.queries = queries
        return [
            {
                "chunk_id": "auto_c8",
                "score": 0.96,
                "text": (
                    "The ablations remove one component at a time: No analyst, "
                    "No cross-agent feedback, No self-organization, and Independent agents. "
                    "Removing the analyst is most damaging on TDC-hERG, while independent "
                    "agents are most damaging on Cell-Cell Communication."
                ),
            }
        ]


def test_web_search_tool():
    from app.services.tools import make_web_search

    tool = make_web_search(_StubWebSearch())
    result = tool.handler({"query": "what is RAG", "max_results": 2})
    summary, _ = tool.summarize(result)
    assert result[0]["url"] == "https://example.com/rag"
    assert "web hit" in summary
    print("test_web_search_tool: OK")


def test_agent_kb_then_web():
    from app.services.agent_runner import AgentRunner, extract_json
    from app.services.tools import ToolRegistry, make_search_paper_kb, make_web_search

    final = '{"grounded": false, "turns": [{"speaker": "guest", "text": "RAG is retrieval plus generation.", "source_chunk_ids": []}]}'
    client = _KbThenWebClient(final)
    registry = (
        ToolRegistry()
        .add(make_search_paper_kb(_StubKB(), "p1"))
        .add(make_web_search(_StubWebSearch()))
    )
    runner = AgentRunner(client, registry, max_iters=4, required_first_tool="search_paper_kb")

    result = runner.run("system", "user")

    assert [t.tool for t in result.trace] == ["search_paper_kb", "web_search"]
    assert result.trace[1].source_urls == ["https://example.com/rag"]
    payload = extract_json(result.content)
    assert "RAG" in payload["turns"][0]["text"]
    print("test_agent_kb_then_web: OK")


def test_interrupt_fallback_synthesizes_evidence():
    from app.services.interaction_service import InterruptAgent

    agent = InterruptAgent(client=_FallbackAnswerClient(), kb=_RawChunkKB())
    grounded, turns, trace = agent._fallback(
        "p1",
        "I still don't understand the meaning of ablations of AutoScientist.",
        None,
        [],
    )

    assert grounded
    assert trace[0].tool == "search_paper_kb"
    answer = " ".join(turn.text for turn in turns)
    assert "controlled test" in answer
    assert "From the paper: t be tracked" not in answer
    print("test_interrupt_fallback_synthesizes_evidence: OK")


def test_agentic_script_beat_uses_specific_paper_evidence():
    brief = PaperBrief(
        paper_id="p1",
        title="AUTOSCIENTISTS",
        one_liner="A self-organizing agent team for scientific experimentation.",
        problem="Long-running experiments require coordination.",
        method="Agents collaborate through shared state and feedback.",
        results="Ablations show components contribute differently.",
        limitations="See paper.",
    )
    outline = PodcastOutline(
        paper_id="p1",
        episode_title="Ablations of AUTOSCIENTISTS",
        hook="Let's unpack which parts actually matter.",
        segments=[
            OutlineSegment(
                segment_id="ablations",
                title="Ablations of AUTOSCIENTISTS",
                goal="Explain the ablation setup, removed components, and task-specific findings.",
                retrieval_queries=[
                    "AUTOSCIENTISTS ablations no analyst cross-agent feedback independent agents"
                ],
                estimated_seconds=120,
            )
        ],
        closing="That is what the ablations show.",
    )
    payload = {
        "evidence_notes": [
            "The ablation setup removes one component at a time while holding other factors fixed."
        ],
        "turns": [
            {
                "speaker": "host",
                "text": "Here, ablation means removing one part of AUTOSCIENTISTS at a time to see what breaks.",
                "source_chunk_ids": ["auto_c8"],
            },
            {
                "speaker": "guest",
                "text": "The four cuts are No analyst, No cross-agent feedback, No self-organization, and Independent agents.",
                "source_chunk_ids": ["auto_c8"],
            },
            {
                "speaker": "guest",
                "text": "The failures differ by task: removing the analyst hurts TDC-hERG most, while independent agents hurt Cell-Cell Communication most.",
                "source_chunk_ids": ["auto_c8"],
            },
        ],
        "recap": "Explained the ablation setup, four component removals, and task-specific findings.",
    }
    kb = _AblationKB()
    client = _BeatSearchClient(payload)
    service = DialogueService(client=client, kb=kb)

    script = service.generate(brief, outline)

    all_text = " ".join(turn.text for turn in script.segments[0].turns)
    assert client.first_tool_choice == {
        "type": "function",
        "function": {"name": "search_paper_kb"},
    }
    assert "ablations" in " ".join(kb.queries).lower()
    assert "No analyst" in all_text
    assert "Cell-Cell Communication" in all_text
    assert "fascinating" not in all_text.lower()
    print("test_agentic_script_beat_uses_specific_paper_evidence: OK")


def test_agent_executes_tool_then_answers():
    from app.services.agent_runner import AgentRunner, extract_json
    from app.services.tools import ToolRegistry, make_search_paper_kb

    final = '{"grounded": true, "turns": [{"speaker": "guest", "text": "Here is the answer.", "source_chunk_ids": ["c0"]}]}'
    client = _OneShotToolClient(final)
    registry = ToolRegistry().add(make_search_paper_kb(_StubKB(), "p1"))
    runner = AgentRunner(client, registry, max_iters=3)

    result = runner.run("system", "user")

    assert len(result.trace) == 1, "agent should have made exactly one tool call"
    assert result.trace[0].tool == "search_paper_kb"
    assert result.trace[0].chunk_ids == ["c0"], "trace should capture retrieved chunk ids"

    payload = extract_json(result.content)
    assert payload["turns"][0]["text"] == "Here is the answer."
    print("test_agent_executes_tool_then_answers: OK")


def test_required_first_tool_choice():
    from app.services.agent_runner import AgentRunner
    from app.services.tools import ToolRegistry, make_search_paper_kb

    final = '{"grounded": true, "turns": [{"speaker": "guest", "text": "Done.", "source_chunk_ids": ["c0"]}]}'
    client = _ToolChoiceClient(final)
    registry = ToolRegistry().add(make_search_paper_kb(_StubKB(), "p1"))
    runner = AgentRunner(client, registry, max_iters=3, required_first_tool="search_paper_kb")

    result = runner.run("system", "user")

    assert client.first_tool_choice == {
        "type": "function",
        "function": {"name": "search_paper_kb"},
    }
    assert result.trace, "required first tool should produce a search trace"
    print("test_required_first_tool_choice: OK")


def test_agent_budget_guard():
    from app.services.agent_runner import AgentRunner, extract_json
    from app.services.tools import ToolRegistry, make_search_paper_kb

    final = '{"grounded": false, "turns": [{"speaker": "host", "text": "Final.", "source_chunk_ids": []}]}'
    client = _AlwaysToolClient(final)
    registry = ToolRegistry().add(make_search_paper_kb(_StubKB(), "p1"))
    runner = AgentRunner(client, registry, max_iters=3)

    result = runner.run("system", "user")

    # Tools are disabled on the final iteration, so the model is forced to stop
    # calling tools. With max_iters=3 that means at most 2 tool rounds.
    assert len(result.trace) <= 2, f"budget guard should cap tool rounds, got {len(result.trace)}"
    assert client.tool_rounds <= 2
    payload = extract_json(result.content)
    assert payload["turns"][0]["text"] == "Final."
    print("test_agent_budget_guard: OK")


def test_extract_json_tolerates_fences():
    from app.services.agent_runner import extract_json

    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} done') == {"a": 2}
    assert extract_json("not json") == {}
    print("test_extract_json_tolerates_fences: OK")


class _ReviseClient(OpenAIClient):
    """Returns a fixed revised plan, simulating the planner agent."""

    def __init__(self, payload: dict) -> None:
        self.enabled = True
        self._payload = payload
        self.last_user = None

    def generate_json(self, system, user, temperature=0.3):
        self.last_user = user
        return self._payload


class _RecordingKB(PaperKB):
    def __init__(self) -> None:
        pass

    def search(self, paper_id, queries, top_k=3):
        self.paper_id = paper_id
        self.queries = queries
        self.top_k = top_k
        return [
            {
                "chunk_id": "p1_c7",
                "score": 0.92,
                "text": "The paper includes task-specific test cases and qualitative examples.",
            }
        ]


class _ReviseToolClient(OpenAIClient):
    """Forces the planner to search first, then returns a revised outline."""

    def __init__(self, payload: dict) -> None:
        self.enabled = True
        self._payload = payload
        self.calls = 0
        self.first_tool_choice = None

    def chat_with_tools(self, messages, tools=None, temperature=0.4, tool_choice="auto"):
        self.calls += 1
        if self.calls == 1:
            self.first_tool_choice = tool_choice
            return (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "revise_search",
                            "type": "function",
                            "function": {
                                "name": "search_paper_kb",
                                "arguments": '{"queries": ["test cases specific researches"], "top_k": 4}',
                            },
                        }
                    ],
                },
                [
                    {
                        "id": "revise_search",
                        "name": "search_paper_kb",
                        "arguments": {"queries": ["test cases specific researches"], "top_k": 4},
                    }
                ],
            )
        return {"role": "assistant", "content": json.dumps(self._payload)}, []


def test_revise_applies_user_instruction():
    _, brief, outline = _make_inputs()
    revised_payload = {
        "episode_title": "Focused: Case Study",
        "hook": "Straight to the case study.",
        "closing": "That's the case study.",
        "segments": [
            {
                "segment_id": "s_case",
                "title": "Case study only",
                "goal": "Walk through the case study in depth",
                "retrieval_queries": ["case study", "validation"],
                "estimated_seconds": 120,
            }
        ],
    }
    client = _ReviseClient(revised_payload)
    planner = PlannerService(client=client)

    result = planner.revise(brief, outline, "Remove everything except the case study.")

    assert len(result.segments) == 1
    assert result.segments[0].title == "Case study only"
    assert result.paper_id == outline.paper_id
    # The user's instruction must actually reach the agent.
    assert "case study" in (client.last_user or "").lower()
    print("test_revise_applies_user_instruction: OK")


def test_revise_searches_paper_for_user_focus():
    _, brief, outline = _make_inputs()
    revised_payload = {
        "episode_title": "Focused: Test Cases",
        "hook": "Let's focus on how the paper tests the idea.",
        "closing": "That's the evidence from the test cases.",
        "segments": [
            {
                "segment_id": "s_tests",
                "title": "Task-specific test cases",
                "goal": "Use retrieved paper evidence to discuss the requested test cases",
                "retrieval_queries": ["task-specific test cases", "qualitative examples"],
                "estimated_seconds": 120,
            }
        ],
    }
    kb = _RecordingKB()
    client = _ReviseToolClient(revised_payload)
    planner = PlannerService(client=client, kb=kb)

    result = planner.revise(brief, outline, "I want more test cases specific researches.")

    assert client.first_tool_choice == {
        "type": "function",
        "function": {"name": "search_paper_kb"},
    }
    assert "test cases" in " ".join(kb.queries)
    assert planner.last_revision_trace[0].tool == "search_paper_kb"
    assert result.segments[0].title == "Task-specific test cases"
    print("test_revise_searches_paper_for_user_focus: OK")


def test_revise_noops_without_client_or_instruction():
    _, brief, outline = _make_inputs()

    # No instruction -> unchanged, even with a capable client.
    capable = _ReviseClient({"segments": [{"title": "x", "goal": "y"}]})
    assert PlannerService(client=capable).revise(brief, outline, "   ") is outline

    # Disabled client -> unchanged.
    disabled = _DisabledClient()
    assert PlannerService(client=disabled).revise(brief, outline, "change it") is outline
    print("test_revise_noops_without_client_or_instruction: OK")


def test_cue_timing_math():
    from app.models import CueTrack, TurnCue

    cues = [
        TurnCue(index=0, segment_id="s1", segment_title="A", speaker="host", text="hi", start_ms=0, end_ms=1000),
        TurnCue(index=1, segment_id="s1", segment_title="A", speaker="guest", text="yo", start_ms=1350, end_ms=2350),
    ]
    track = CueTrack(paper_id="p1", total_ms=2700, cues=cues)
    for i in range(1, len(track.cues)):
        assert track.cues[i].start_ms >= track.cues[i - 1].end_ms, "cues must not overlap"
    assert track.total_ms >= track.cues[-1].end_ms, "total must cover last cue"
    print("test_cue_timing_math: OK")


if __name__ == "__main__":
    test_chunk_overlap_keeps_sentence_boundaries()
    test_single_greeting_and_roles()
    test_agent_executes_tool_then_answers()
    test_required_first_tool_choice()
    test_agent_budget_guard()
    test_extract_json_tolerates_fences()
    test_revise_applies_user_instruction()
    test_revise_searches_paper_for_user_focus()
    test_revise_noops_without_client_or_instruction()
    test_web_search_tool()
    test_agent_kb_then_web()
    test_interrupt_fallback_synthesizes_evidence()
    test_agentic_script_beat_uses_specific_paper_evidence()
    test_cue_timing_math()
    print("ALL DETERMINISTIC TESTS PASSED")
