"""End-to-end tests against the real Gemini API (marker `live`; skipped without a key).

Run with `uv run pytest -q -m live`. Four turns on one runner cover the four behaviours the
brief asks for: a grounded, cited answer; a tool call; a graceful decline on a near-miss; and
a not-found order that must not get an invented status.
"""

from __future__ import annotations

import os

import pytest

from novamart_agent.runtime import AgentTurn, ask, build_runner, new_session

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
        reason="needs GOOGLE_API_KEY",
    ),
]

STATUS_WORDS = ("shipped", "delivered", "out for delivery", "processing", "delayed")


async def _turn(question: str) -> AgentTurn:
    runner = build_runner()
    try:
        turn = await ask(runner, await new_session(runner), question)
    finally:
        await runner.close()
    assert turn.error is None, turn.error
    assert turn.answer, "empty answer"
    return turn


async def test_grounded_answer_cites_the_policy_it_used():
    turn = await _turn("How long does a customer have to return a defective item?")
    assert "search_policies" in [c.name for c in turn.tool_calls]
    assert "90" in turn.answer
    assert turn.sources == ["POL-DMG-002"]
    assert set(turn.sources) <= set(turn.retrieved_doc_ids)


async def test_order_question_calls_the_tool_and_reports_its_status():
    turn = await _turn("Where is order NM-10432?")
    assert [c.name for c in turn.tool_calls] == ["get_order_status"]
    assert turn.tool_calls[0].args == {"order_id": "NM-10432"}
    assert "shipped" in turn.answer.lower()


async def test_near_miss_declines_instead_of_stretching_the_policy():
    turn = await _turn(
        "What is the return window for items sold by third-party sellers on NovaMart Marketplace?"
    )
    lowered = turn.answer.lower()
    assert "manager on duty" in lowered or "customer care" in lowered
    assert "30 days" not in lowered and "14 days" not in lowered


async def test_unknown_order_gets_no_invented_status():
    turn = await _turn("Can you check order NM-99999 for me?")
    assert turn.tool_responses[0].response.get("found") is False
    assert "99999" in turn.answer
    assert not any(word in turn.answer.lower() for word in STATUS_WORDS)
