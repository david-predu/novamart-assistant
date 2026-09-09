"""NovaMart Store Assistant - a minimal Streamlit chat page over novamart_agent.runtime.
One event loop, one ADK runner and one session persist in st.session_state per browser tab.
Run from the repo root: uv run streamlit run ui/app.py"""

import asyncio

import streamlit as st

from novamart_agent.config import MIN_SCORE, MODEL
from novamart_agent.runtime import (
    AgentTurn,
    MissingApiKey,
    ask,
    build_runner,
    friendly_error,
    new_session,
    require_api_key,
)

EXAMPLES = [
    "How long does a customer have to return a defective item?",
    "Where is order NM-10433 and what should I tell the customer?",
    "What's the return window for items sold by third-party sellers on NovaMart Marketplace?",
]

st.set_page_config(page_title="NovaMart Store Assistant", page_icon="🛒")
state = st.session_state


def run_blocking(coro):
    """Drive the session's loop from the current script thread, refusing to re-enter it.

    Streamlit starts a fresh script thread per rerun (runner.fastReruns), so a click or a second
    question during a slow turn would re-enter the one shared loop and asyncio would raise
    "This event loop is already running". Warn and return None instead of a red traceback.
    """
    if state.loop.is_running():
        coro.close()  # avoid "coroutine was never awaited"
        st.warning("A turn is still running - wait for the answer, then try again.")
        return None
    try:
        return state.loop.run_until_complete(coro)
    except RuntimeError as exc:  # lost the (microseconds-wide) check-then-call race
        coro.close()
        st.error(friendly_error(f"RuntimeError: {exc}"))
        return None


if "runner" not in state:
    try:
        require_api_key()
    except MissingApiKey as exc:
        st.error(str(exc))
        st.stop()
    # The genai async client binds to the loop it first runs on, so one loop lives for the session.
    state.loop = asyncio.new_event_loop()
    state.runner = build_runner()
    state.session_id = run_blocking(new_session(state.runner))
    state.history = []


def render_turn(turn: AgentTurn) -> None:
    """Show the assistant's reply and, when tools ran, an expander with each call and response."""
    if turn.error:
        st.error(friendly_error(turn.error))
    else:
        st.text(turn.answer)  # not Markdown: "$" amounts in answers would render as LaTeX
    if turn.tool_calls:
        with st.expander(f"{len(turn.tool_calls)} tool call(s) · {turn.latency_s:.1f}s"):
            for call, response in zip(turn.tool_calls, turn.tool_responses):
                st.code(f"{call.name}({call.args})")
                st.json(response.response, expanded=False)


with st.sidebar:
    st.caption(f"Model {MODEL} · retrieval gate {MIN_SCORE:.2f}")
    if st.button("New conversation"):
        sid = run_blocking(new_session(state.runner))
        if sid is not None:  # None: a turn is still running, keep the current conversation
            state.session_id = sid
            state.history = []
            st.rerun()
    st.write("Try asking:")
    for example in EXAMPLES:
        st.text(example)

st.title("NovaMart Store Assistant")

for turn in state.history:
    with st.chat_message("user"):
        st.text(turn.question)
    with st.chat_message("assistant"):
        render_turn(turn)

if question := st.chat_input("Ask about a NovaMart policy or an order number"):
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking policies and orders..."):
            turn = run_blocking(ask(state.runner, state.session_id, question))
        if turn is not None:
            render_turn(turn)
    if turn is not None:
        state.history.append(turn)
