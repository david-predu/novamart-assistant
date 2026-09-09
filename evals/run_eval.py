"""Run the golden set through the agent and the LLM judge, then write and print the report.

Usage:
    uv run python -m evals.run_eval [--cases G-01,T-06] [--no-judge] [--refresh] [--report-only]
                                    [--sleep SECONDS] [--agent-model X] [--judge-model Y]

Flags:
    --cases         comma-separated case ids to run (default: all)
    --no-judge      deterministic checks only; judge-dependent metrics and gates are skipped
    --refresh       ignore evals/.cache and call the APIs again
    --report-only   re-render results/latest.md from results/latest.json with zero API calls
    --sleep         seconds to pause after every live call (free-tier rate limits)
    --agent-model   Gemini model for the agent (sets NOVAMART_MODEL)
    --judge-model   Gemini model for the judge (sets NOVAMART_JUDGE_MODEL)

Exit codes: 0 every gate passed or was skipped; 1 a quality gate failed; 2 inconclusive
(n_error > 0: a turn or a judge call failed, so the numbers cannot be trusted either way).
An inconclusive run, or a partial one (--cases, --no-judge), prints its report and leaves
evals/results untouched, so a 429 storm cannot overwrite the committed evidence.

--agent-model / --judge-model work by writing the environment variables before novamart_agent
is imported, because config.py reads the environment at import time; that is why every project
import lives inside main() and run() rather than at module level.

Cache: evals/.cache/agent/<key>.json keyed by (case_id, question, agent model, instruction hash,
corpus hash, behaviour hash), where the behaviour hash covers MIN_SCORE, TOP_K, data/orders.json
and the two tool docstrings; evals/.cache/judge/<key>.json keyed by (case_id, judge model, prompt
hash, answer, context, reference). Changing the judge prompt re-judges cached answers; changing
the corpus, the instruction, the retrieval gate, the order data or a tool description re-runs the
agent (and then the judge) automatically. Failures are never cached.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / "golden.json"
CACHE_DIR = EVALS_DIR / ".cache"
RESULTS_DIR = EVALS_DIR / "results"
LATEST_JSON = RESULTS_DIR / "latest.json"
LATEST_MD = RESULTS_DIR / "latest.md"

_MEANS = {"mean_groundedness", "tool_arg_match", "latency_p50_s", "latency_max_s"}
_COUNTS = {"partially_correct_count", "n_error", "n_pass", "n_fail"}
_CASE_COLUMNS = (
    "case | category | tools called | outcome | correctness | groundedness | latency s | status"
    " | fail reasons"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cache_key(*parts: str) -> str:
    return _sha("|".join(parts))


def cache_read(kind: str, key: str) -> dict | None:
    path = CACHE_DIR / kind / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else None


def cache_write(kind: str, key: str, data: dict) -> None:
    path = CACHE_DIR / kind / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def git_sha() -> str:
    """Short HEAD sha, "-dirty" when tracked files differ from HEAD; "unknown" without git."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=EVALS_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=EVALS_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    sha = done.stdout.strip()
    return f"{sha}-dirty" if status.stdout.strip() else sha


async def run(args: argparse.Namespace) -> dict:
    """Run every selected case once (agent, checks, judge) and return the latest.json payload."""
    import google.adk
    import google.genai

    from evals import checks, judge
    from novamart_agent import agent, config, corpus, runtime, tools

    try:
        runtime.require_api_key()
    except runtime.MissingApiKey as exc:
        sys.exit(f"{exc} (refusing to overwrite evals/results with errors)")
    golden_bytes = GOLDEN_PATH.read_bytes()
    cases = [checks.GoldenCase.model_validate(c) for c in json.loads(golden_bytes)]
    if args.cases:
        wanted = set(args.cases.split(","))
        if unknown := wanted - {c.case_id for c in cases}:
            sys.exit(f"unknown case ids: {sorted(unknown)}")
        cases = [c for c in cases if c.case_id in wanted]
    instruction_hash = _sha(agent.INSTRUCTION)
    corpus_hash = corpus.corpus_sha256(corpus.load_chunks())
    # Everything else that changes what the agent sees or gets back but is not hashed above.
    behaviour_hash = _sha(
        f"min_score={config.MIN_SCORE}|top_k={config.TOP_K}|"
        f"{config.ORDERS_PATH.read_text(encoding='utf-8')}|"
        f"{tools.search_policies.__doc__}|{tools.get_order_status.__doc__}"
    )
    prompt_hash = _sha(judge.JUDGE_PROMPT)
    judge_on = not args.no_judge
    meta = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent_model": config.MODEL,
        "judge_model": config.JUDGE_MODEL if judge_on else None,
        "adk_version": google.adk.__version__,
        "genai_version": google.genai.__version__,
        "corpus_sha256": corpus_hash,
        "instruction_hash": instruction_hash,
        "min_score": config.MIN_SCORE,
        "top_k": config.TOP_K,
        "behaviour_hash": behaviour_hash,
        "golden_hash": hashlib.sha256(golden_bytes).hexdigest(),
        "git_sha": git_sha(),
        "n_cases": len(cases),
    }
    counters = {"agent_model_calls": 0, "judge_calls": 0, "cache_hits": 0}
    rows: list[dict] = []
    runner = runtime.build_runner()
    try:
        for case in cases:
            key = cache_key(
                case.case_id,
                case.question,
                config.MODEL,
                instruction_hash,
                corpus_hash,
                behaviour_hash,
            )
            turn = None if args.refresh else cache_read("agent", key)
            if turn is not None:
                counters["cache_hits"] += 1
            else:
                session_id = await runtime.new_session(runner)
                turn = (await runtime.ask(runner, session_id, case.question)).to_dict()
                counters["agent_model_calls"] += turn["model_calls"]
                if not turn["error"]:
                    cache_write("agent", key, turn)
                await asyncio.sleep(args.sleep)

            verdict = None
            if judge_on and not turn["error"]:
                context = judge.render_context(turn["tool_calls"], turn["tool_responses"])
                jkey = cache_key(
                    case.case_id,
                    config.JUDGE_MODEL,
                    prompt_hash,
                    turn["answer"],
                    context,
                    case.reference_answer,
                )
                verdict = None if args.refresh else cache_read("judge", jkey)
                if verdict is not None:
                    counters["cache_hits"] += 1
                else:
                    counters["judge_calls"] += 1
                    result = judge.judge(
                        case.question, context, case.reference_answer, turn["answer"]
                    )
                    if result is not None:
                        verdict = result.model_dump()
                        cache_write("judge", jkey, verdict)
                    await asyncio.sleep(args.sleep)

            row = checks.make_row(case, turn, verdict, judge_enabled=judge_on)
            rows.append(row)
            print(f"{case.case_id:<6} {row['status']:<5} {turn['latency_s']:.1f}s", file=sys.stderr)
    finally:
        await runner.close()

    summary = checks.aggregate(rows)
    meta.update(counters, n_error=summary["n_error"]["num"])
    return {"meta": meta, "summary": summary, "gates": checks.gates(summary), "cases": rows}


def _num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return str(value) if isinstance(value, int) else f"{value:.{digits}f}"


def _metric(name: str, metric: dict) -> str:
    """Rates as "num/den (value)", means as the value, counters as the count, empty as n/a."""
    if metric["value"] is None:
        return "n/a"
    if name in _COUNTS:
        return str(metric["num"])
    if name in _MEANS:
        return f"{metric['value']:.2f}"
    return f"{metric['num']}/{metric['den']} ({metric['value']:.2f})"


def _case_line(row: dict) -> str:
    derived, verdict = row["derived"], row["judge"] or {}
    reasons = ", ".join(row["fail_reasons"])
    if row["flags"]:
        reasons = f"{reasons} [flags: {', '.join(row['flags'])}]".strip()
    # An aborted turn has no meaningful deterministic reasons: show the cause instead.
    if row.get("error"):
        reasons = f"agent error: {row['error'][:60]}"
    elif row.get("judge_error"):
        reasons = "judge error: no verdict (safety block, truncation or API failure; see stderr)"
    cells = [
        row["case_id"],
        row["category"],
        ", ".join(call["name"] for call in row["tool_calls"]) or "none",
        derived["abstention_outcome"] or "-",
        verdict.get("correctness", "-"),
        _num(derived["groundedness"]),
        _num(row["latency_s"], 1),
        row["status"],
        reasons or "-",
    ]
    return "| " + " | ".join(cells) + " |"


def render_markdown(result: dict) -> str:
    """Render the latest.json payload as the markdown report (pure; also used by --report-only)."""
    meta, summary, gate_rows, cases = (
        result["meta"],
        result["summary"],
        result["gates"],
        result["cases"],
    )
    lines = [
        "# NovaMart assistant — eval report",
        "",
        f"- run: {meta['run_at']} (git {meta['git_sha']}) · cases: {meta['n_cases']}",
        (
            f"- agent model: `{meta['agent_model']}` · judge model: "
            f"`{meta['judge_model'] or 'disabled (--no-judge)'}` · retrieval gate min_score "
            f"{meta.get('min_score', '?')} · top_k {meta.get('top_k', '?')}"
        ),
        f"- google-adk {meta['adk_version']} · google-genai {meta['genai_version']}",
        (
            f"- corpus sha256 `{meta['corpus_sha256'][:12]}` · instruction sha256 "
            f"`{meta['instruction_hash'][:12]}` · golden sha256 `{meta['golden_hash'][:12]}`"
        ),
        (
            f"- agent model calls: {meta['agent_model_calls']} · judge calls: {meta['judge_calls']}"
            f" · cache hits: {meta['cache_hits']} · errors: {meta['n_error']}"
        ),
        "",
        "## Metrics",
        "",
        "| metric | value | n |",
        "|---|---|---|",
        *(
            f"| {name} | {_metric(name, m)} | {m['den']} |"
            for name, m in summary.items()
            if name != "top_score"
        ),
        "",
        "## Gates",
        "",
        "| gate | value | threshold | status |",
        "|---|---|---|---|",
        *(
            f"| {g['name']} | {_num(g['value'])} | {g['threshold']} | {g['status']} |"
            for g in gate_rows
        ),
        "",
        "## Cases",
        "",
        f"| {_CASE_COLUMNS} |",
        "|" + "---|" * 9,
        *(_case_line(row) for row in cases),
        "",
        "## Top-1 retrieval score",
        "",
        "| group | n | min | median | max |",
        "|---|---|---|---|---|",
    ]
    for group, label in (
        ("grounded", "grounded (expected_doc_ids non-empty)"),
        ("should_decline", "should_decline"),
    ):
        s = summary["top_score"][group]
        lines.append(
            f"| {label} | {s['n']} | {_num(s['min'])} | {_num(s['median'])} | {_num(s['max'])} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run_eval", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--no-judge", action="store_true", help="deterministic checks only")
    parser.add_argument("--refresh", action="store_true", help="ignore the response cache")
    parser.add_argument(
        "--report-only", action="store_true", help="re-render latest.md from latest.json"
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="seconds to pause after each live call"
    )
    parser.add_argument("--agent-model", help="sets NOVAMART_MODEL before importing the agent")
    parser.add_argument(
        "--judge-model", help="sets NOVAMART_JUDGE_MODEL before importing the judge"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.agent_model:
        os.environ["NOVAMART_MODEL"] = args.agent_model
    if args.judge_model:
        os.environ["NOVAMART_JUDGE_MODEL"] = args.judge_model
    from evals import checks  # after the env vars: importing novamart_agent freezes config

    if args.report_only:
        if not LATEST_JSON.exists():
            sys.exit(f"{LATEST_JSON} does not exist yet; run once without --report-only")
        result = json.loads(LATEST_JSON.read_text())
    else:
        result = asyncio.run(run(args))
        # Persist only a clean full-suite run: an errored or partial run must never replace
        # the committed evidence (the missing-key case is refused earlier, in run()).
        skip = None
        if result["meta"]["n_error"]:
            skip = f"{result['meta']['n_error']} case(s) errored (inconclusive)"
        elif args.cases or args.no_judge:
            skip = "partial run (--cases/--no-judge)"
        if skip:
            print(render_markdown(result))
            print(
                f"{skip}: left {LATEST_JSON.name} and {LATEST_MD.name} untouched; "
                "re-run the full suite to refresh them.",
                file=sys.stderr,
            )
            return checks.exit_code(result["gates"])
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        LATEST_JSON.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    markdown = render_markdown(result)
    LATEST_MD.write_text(markdown)
    print(markdown)
    return checks.exit_code(result["gates"])


if __name__ == "__main__":
    sys.exit(main())
