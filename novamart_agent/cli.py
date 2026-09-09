"""Typer CLI: `novamart ask | chat | search | index`."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import NoReturn

import typer

from . import config, runtime
from .runtime import AgentTurn, MissingApiKey

app = typer.Typer(add_completion=False, help="NovaMart store-employee assistant (Gemini + ADK).")


def _preflight() -> None:
    """Configure logging and fail with exit code 2 when no API key is available."""
    logging.basicConfig(level=os.getenv("NOVAMART_LOG_LEVEL", "WARNING").upper())
    try:
        runtime.require_api_key()
    except MissingApiKey as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None


def _summary(name: str, response: dict) -> str:
    """One-line digest of a tool response for the trace lines."""
    if "error" in response:
        return f"error={response['error']}"
    if name == "search_policies":
        hits = ", ".join(f"{r['doc_id']} {r['score']:.3f}" for r in response.get("results", []))
        return f"{response.get('status')} {hits or 'top_score=' + str(response.get('top_score'))}"
    status = response.get("status")
    return f"found={response.get('found')}" + (f" status={status}" if status else "")


def _print_turn(turn: AgentTurn) -> None:
    for call in turn.tool_calls:
        args = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
        typer.secho(f"  -> {call.name}({args})", dim=True)
    for resp in turn.tool_responses:
        typer.secho(f"  <- {resp.name}: {_summary(resp.name, resp.response)}", dim=True)
    if turn.answer:
        typer.echo(turn.answer)
    typer.secho(
        f"[{turn.latency_s:.1f}s | model={turn.model} | calls={turn.model_calls}]", dim=True
    )


def _report_error(turn: AgentTurn) -> None:
    typer.secho(runtime.friendly_error(turn.error or ""), fg=typer.colors.RED, err=True)


def _fail(exc: Exception) -> NoReturn:
    """Report a failure as one red line and exit 1, like `ask` does; never a traceback."""
    message = runtime.friendly_error(f"{type(exc).__name__}: {exc}")
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(1) from None


async def _one_shot(question: str) -> AgentTurn:
    runner = runtime.build_runner()
    try:
        return await runtime.ask(runner, await runtime.new_session(runner), question)
    finally:
        await runner.close()


@app.command()
def ask(
    question: str,
    json_out: bool = typer.Option(False, "--json", help="Print the full AgentTurn as JSON."),
) -> None:
    """Ask one question in a fresh session."""
    _preflight()
    turn = asyncio.run(_one_shot(question))
    if json_out:
        typer.echo(json.dumps(turn.to_dict(), indent=2))
    else:
        _print_turn(turn)
    if turn.error:
        _report_error(turn)
        raise typer.Exit(1)


async def _repl() -> None:
    runner = runtime.build_runner()
    session_id = await runtime.new_session(runner)
    typer.secho(f"NovaMart assistant ({config.MODEL}). /reset starts over, /quit exits.", dim=True)
    try:
        while True:
            try:
                text = input("you> ").strip()
            except EOFError:
                break
            if text == "/quit":
                break
            if text == "/reset":
                session_id = await runtime.new_session(runner)
                typer.secho("  (new conversation)", dim=True)
            elif text:
                turn = await runtime.ask(runner, session_id, text)
                _print_turn(turn)
                if turn.error:
                    _report_error(turn)
    finally:
        await runner.close()


@app.command()
def chat() -> None:
    """Multi-turn conversation in one session (/reset, /quit, Ctrl-C)."""
    _preflight()
    try:
        asyncio.run(_repl())
    except KeyboardInterrupt:
        typer.echo()


@app.command()
def search(
    query: str, k: int = typer.Option(5, "-k", "--k", help="Number of chunks to show.")
) -> None:
    """Score policy chunks against a query and show where the confidence gate sits."""
    _preflight()
    from .retrieval import get_index

    try:  # a stale index, a rejected key or an exhausted embedding quota all land here
        hits = get_index().search(query, k=k)
    except Exception as e:  # noqa: BLE001
        _fail(e)
    for hit in hits:
        typer.echo(f"{hit.score:.3f}  {hit.chunk.chunk_id}  {hit.chunk.section}")
    top = float(hits[0].score) if hits else 0.0
    side = "above" if top >= config.MIN_SCORE else "below"
    typer.secho(
        f"top score {top:.3f} is {side} the gate MIN_SCORE={config.MIN_SCORE:.2f}", dim=True
    )


@app.command()
def index(
    force: bool = typer.Option(False, "--force", help="Re-embed even when the cache is fresh."),
) -> None:
    """Build or verify the embedding cache for data/policies."""
    _preflight()
    from .retrieval import get_index

    try:
        idx = get_index(force_rebuild=force)
    except Exception as e:  # noqa: BLE001
        _fail(e)
    n_docs = len({chunk.doc_id for chunk in idx.chunks})
    typer.echo(f"{len(idx.chunks)} chunks from {n_docs} docs -> {config.INDEX_PATH}")
