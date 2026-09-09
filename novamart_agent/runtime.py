"""Shared runtime for the CLI, the Streamlit page and the eval harness.

One `ask()` call runs a single ADK invocation and folds its event stream into an `AgentTurn`:
the answer text, every tool call and response, the cited and retrieved doc ids, timing, usage
and any error. Callers own the event loop: one `asyncio.run` per process.
"""

from __future__ import annotations

import dataclasses
import os
import re
import time
import uuid
import warnings
from dataclasses import dataclass

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import errors, types

from . import config

_SOURCES_LINE = re.compile(r"^\s*\**Sources:\**\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_DOC_ID = re.compile(r"POL-[A-Z]{2,4}-\d{3}")
_NO_SOURCES = {"none", "n/a", "-", ""}
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class ToolResponse:
    name: str
    response: dict


@dataclass
class AgentTurn:
    """Everything observable about one question/answer exchange."""

    question: str
    answer: str
    tool_calls: list[ToolCall]
    tool_responses: list[ToolResponse]
    sources: list[str]
    retrieved_doc_ids: list[str]
    top_score: float | None
    latency_s: float
    error: str | None
    model: str
    model_calls: int
    usage: dict | None  # token counters summed over the turn's model calls

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class MissingApiKey(RuntimeError):
    """No Gemini API key in the environment."""


def require_api_key() -> None:
    """Fail fast with a helpful message before any network call."""
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise MissingApiKey(
            "No GOOGLE_API_KEY found. Copy .env.example to .env and paste a fresh key "
            "from https://aistudio.google.com/apikey"
        )
    for var in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"):
        if os.getenv(var, "").strip().lower() in _TRUTHY:
            warnings.warn(
                f"{var} is set: the prototype targets AI Studio, so the key may be ignored",
                stacklevel=2,
            )


def build_runner() -> Runner:
    """In-memory ADK runner around `root_agent` (imported here so runtime helpers stay light)."""
    from .agent import root_agent

    return Runner(
        app_name=config.APP_NAME, agent=root_agent, session_service=InMemorySessionService()
    )


async def new_session(
    runner: Runner, user_id: str = "store_employee", session_id: str | None = None
) -> str:
    """Create a conversation and return its id."""
    session_id = session_id or uuid.uuid4().hex[:8]
    await runner.session_service.create_session(
        app_name=config.APP_NAME, user_id=user_id, session_id=session_id
    )
    return session_id


async def ask(
    runner: Runner, session_id: str, text: str, user_id: str = "store_employee"
) -> AgentTurn:
    """Send one user message and fold the resulting event stream into an AgentTurn."""
    started = time.perf_counter()
    message = types.Content(role="user", parts=[types.Part(text=text)])
    tool_calls: list[ToolCall] = []
    tool_responses: list[ToolResponse] = []
    texts: list[str] = []
    model_calls = 0
    usage: dict[str, int] = {}
    error = None
    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=message
        ):
            for fc in event.get_function_calls():
                tool_calls.append(ToolCall(fc.name, dict(fc.args or {})))
            for fr in event.get_function_responses():
                tool_responses.append(ToolResponse(fr.name, dict(fr.response or {})))
            if event.content and event.content.role == "model" and not event.partial:
                model_calls += 1
            if event.is_final_response() and event.content and event.content.parts:
                texts.append(
                    "".join(p.text for p in event.content.parts if p.text and not p.thought)
                )
            meta = getattr(event, "usage_metadata", None)
            if meta is not None and not event.partial:
                # One entry per model call; summing the integer counters gives the turn's real
                # cost (the per-modality detail lists are not additive and are dropped).
                for key, value in meta.model_dump(exclude_none=True).items():
                    if isinstance(value, int):
                        usage[key] = usage.get(key, 0) + value
            if event.error_code and not error:
                # ADK reports model-level failures (SAFETY, MAX_TOKENS, no content) as events,
                # not exceptions; without this the turn would look like an empty success.
                error = (
                    f"{event.error_code}: {event.error_message}"
                    if event.error_message
                    else str(event.error_code)
                )
    except errors.APIError as e:
        error = f"{e.code}: {e.message}"
    except Exception as e:  # noqa: BLE001 - a turn must always come back with its error
        error = f"{type(e).__name__}: {e}"
    answer = "\n".join(t for t in texts if t).strip()
    return AgentTurn(
        question=text,
        answer=answer,
        tool_calls=tool_calls,
        tool_responses=tool_responses,
        sources=parse_sources(answer),
        retrieved_doc_ids=retrieved_doc_ids(tool_responses),
        top_score=top_score(tool_responses),
        latency_s=round(time.perf_counter() - started, 3),
        error=error,
        model=config.MODEL,
        model_calls=model_calls,
        usage=usage or None,
    )


def parse_sources(answer: str) -> list[str]:
    """Doc ids cited on the answer's last "Sources:" line; [] for "Sources: none" or no line."""
    lines = _SOURCES_LINE.findall(answer)
    if not lines:
        return []
    payload = lines[-1].rstrip(" .")
    if payload.lower() in _NO_SOURCES:
        return []
    return sorted(set(_DOC_ID.findall(payload.upper())))


def _search_responses(tool_responses: list[ToolResponse]) -> list[dict]:
    return [tr.response for tr in tool_responses if tr.name == "search_policies"]


def retrieved_doc_ids(tool_responses: list[ToolResponse]) -> list[str]:
    """Every doc id the model actually saw via search_policies, sorted and unique."""
    return sorted(
        {r["doc_id"] for resp in _search_responses(tool_responses) for r in resp.get("results", [])}
    )


def top_score(tool_responses: list[ToolResponse]) -> float | None:
    """Best retrieval score of the turn (also reported by NO_RELEVANT_CONTENT), or None."""
    scores: list[float] = []
    for resp in _search_responses(tool_responses):
        scores.extend(r["score"] for r in resp.get("results", []))
        if "top_score" in resp:
            scores.append(resp["top_score"])
    return max(scores) if scores else None


def friendly_error(err: str) -> str:
    """Translate the two errors a free-tier user is most likely to hit into next steps.

    The original message is kept: a 400 can also be an unsupported thinking level or an
    oversized request, and the advice must never hide the real cause.
    """
    if "429" in err or "RESOURCE_EXHAUSTED" in err:
        return (
            "Gemini free-tier quota exceeded (429). Wait a minute for per-minute limits, or until "
            "midnight Pacific for daily limits; set NOVAMART_MODEL to a lighter model such as "
            "gemini-3.5-flash-lite; check https://aistudio.google.com/rate-limit for this project."
            f"\n(original error: {err})"
        )
    if "API key not valid" in err or "API_KEY_INVALID" in err or err.startswith("400"):
        return (
            "Gemini rejected the request, most often because of an old or invalid key: create a "
            "fresh key at https://aistudio.google.com/apikey and paste it into .env"
            f"\n(original error: {err})"
        )
    return err
