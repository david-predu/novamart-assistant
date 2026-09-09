"""Offline tests for runtime helpers and ask() against a fake runner emitting real ADK events."""

import json

import pytest
from google.adk.events import Event
from google.genai import errors, types

from novamart_agent import config
from novamart_agent.runtime import (
    MissingApiKey,
    ToolCall,
    ToolResponse,
    ask,
    friendly_error,
    parse_sources,
    require_api_key,
    retrieved_doc_ids,
    top_score,
)

OK_SEARCH = ToolResponse(
    "search_policies",
    {
        "status": "ok",
        "query": "return window",
        "results": [
            {"doc_id": "POL-RET-001", "score": 0.712, "text": "a"},
            {"doc_id": "POL-EMP-004", "score": 0.655, "text": "b"},
            {"doc_id": "POL-RET-001", "score": 0.601, "text": "c"},
        ],
    },
)
NO_CONTENT = ToolResponse(
    "search_policies", {"status": "NO_RELEVANT_CONTENT", "top_score": 0.31, "message": "m"}
)
ORDER = ToolResponse("get_order_status", {"found": True, "status": "SHIPPED"})


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Ninety days.\nSources: POL-RET-001, POL-EMP-004", ["POL-EMP-004", "POL-RET-001"]),
        ("**Sources:** POL-GC-006.", ["POL-GC-006"]),
        ("Sources: [POL-OPS-007#peak-season]", ["POL-OPS-007"]),
        ("Sources: none", []),
        ("Sources: none.", []),
        ("Sources: N/A", []),
        ("An answer with no sources line at all.", []),
        ("sources: pol-ret-001", ["POL-RET-001"]),
        ("Sources: POL-RET-001\nActually...\nSources: POL-GC-006", ["POL-GC-006"]),
        ("Sources: POL-RET-001, POL-RET-001", ["POL-RET-001"]),
    ],
)
def test_parse_sources(answer, expected):
    assert parse_sources(answer) == expected


def test_retrieved_doc_ids_sorted_unique_search_only():
    assert retrieved_doc_ids([ORDER, OK_SEARCH, NO_CONTENT]) == ["POL-EMP-004", "POL-RET-001"]
    assert retrieved_doc_ids([ORDER]) == []
    assert retrieved_doc_ids([]) == []


def test_top_score():
    assert top_score([OK_SEARCH]) == 0.712
    assert top_score([NO_CONTENT]) == 0.31
    assert top_score([NO_CONTENT, OK_SEARCH]) == 0.712
    assert top_score([ORDER]) is None
    assert top_score([]) is None


@pytest.mark.parametrize(
    ("err", "needle"),
    [
        ("429: You exceeded your current quota", "aistudio.google.com/rate-limit"),
        ("ClientError: RESOURCE_EXHAUSTED", "midnight Pacific"),
        ("400: API key not valid. Please pass a valid API key.", "aistudio.google.com/apikey"),
        ("ClientError: API_KEY_INVALID", "fresh key"),
        ("RuntimeError: boom", "RuntimeError: boom"),
    ],
)
def test_friendly_error(err, needle):
    assert needle in friendly_error(err)


def test_require_api_key_raises_without_keys(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingApiKey, match="aistudio.google.com/apikey"):
        require_api_key()


def test_require_api_key_accepts_gemini_var_and_warns_on_backend_flag(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "set-by-test")
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    require_api_key()
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    with pytest.warns(UserWarning, match="GOOGLE_GENAI_USE_VERTEXAI"):
        require_api_key()


def test_import_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from novamart_agent.agent import INSTRUCTION, root_agent

    assert len(root_agent.tools) == 2
    assert {t.__name__ for t in root_agent.tools} == {"search_policies", "get_order_status"}
    assert "NOVAMART-SYS-CANARY-4187" in INSTRUCTION
    assert root_agent.name == "novamart_assistant"


def _model_event(*parts: types.Part, partial: bool = False, usage=None) -> Event:
    return Event(
        author="novamart_assistant",
        content=types.Content(role="model", parts=list(parts)),
        partial=partial,
        usage_metadata=usage,
    )


def _tool_event(name: str, response: dict) -> Event:
    part = types.Part(function_response=types.FunctionResponse(name=name, response=response))
    return Event(author="novamart_assistant", content=types.Content(role="user", parts=[part]))


class FakeRunner:
    def __init__(self, events: list[Event] = (), exc: Exception | None = None) -> None:
        self.events = list(events)
        self.exc = exc
        self.seen = []

    async def run_async(self, *, user_id, session_id, new_message):
        self.seen.append((user_id, session_id, new_message.parts[0].text))
        if self.exc:
            raise self.exc
        for event in self.events:
            yield event


async def test_ask_folds_events_into_turn():
    usage = types.GenerateContentResponseUsageMetadata(prompt_token_count=10, total_token_count=15)
    call = types.Part(
        function_call=types.FunctionCall(name="get_order_status", args={"order_id": "10432"})
    )
    events = [
        _model_event(call),
        _tool_event("get_order_status", {"found": True, "status": "SHIPPED"}),
        _model_event(types.Part(text="Order NM-10432 has "), partial=True),
        _model_event(types.Part(text="Order NM-10432 has SHIPPED.\nSources: none"), usage=usage),
    ]
    runner = FakeRunner(events)
    turn = await ask(runner, "sess1", "Where is 10432?")
    assert runner.seen == [("store_employee", "sess1", "Where is 10432?")]
    assert turn.tool_calls == [ToolCall("get_order_status", {"order_id": "10432"})]
    assert turn.tool_responses == [
        ToolResponse("get_order_status", {"found": True, "status": "SHIPPED"})
    ]
    assert turn.answer == "Order NM-10432 has SHIPPED.\nSources: none"
    assert turn.sources == [] and turn.retrieved_doc_ids == [] and turn.top_score is None
    assert turn.model_calls == 2
    assert turn.usage == {"prompt_token_count": 10, "total_token_count": 15}
    assert turn.error is None and turn.model == config.MODEL and turn.latency_s >= 0
    as_dict = turn.to_dict()
    assert as_dict["tool_calls"] == [{"name": "get_order_status", "args": {"order_id": "10432"}}]
    json.dumps(as_dict)


async def test_ask_captures_api_error_and_generic_error():
    api_error = errors.APIError(
        429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}
    )
    turn = await ask(FakeRunner(exc=api_error), "s", "hi")
    assert turn.error == "429: quota" and turn.answer == ""
    turn = await ask(FakeRunner(exc=RuntimeError("boom")), "s", "hi")
    assert turn.error == "RuntimeError: boom"
