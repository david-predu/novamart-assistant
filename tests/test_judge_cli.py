"""Offline tests for evals/judge.py, the run_eval cache and persistence guard, and the typer CLI.

No API key and no network: the judge client is a stub (never a real genai.Client), the CLI's
`_one_shot` returns a canned AgentTurn, and `search` reaches a fake index, so nothing here can
spend quota. Behaviour-hashed files (instruction, tools, corpus, orders, judge prompt) are
untouched, so the committed evals/results stay valid.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from evals import checks, judge, run_eval
from novamart_agent import cli, config
from novamart_agent.runtime import AgentTurn, ToolCall, ToolResponse

runner = CliRunner()

SEARCH_CALL = {"name": "search_policies", "args": {"query": "return window"}}
SEARCH_RESPONSE = {
    "name": "search_policies",
    "response": {"status": "ok", "results": [{"doc_id": "POL-RET-001", "score": 0.71}]},
}
VERDICT = {
    "claims": [{"claim": "Ninety days.", "label": "supported", "excerpt": "90 days"}],
    "correctness": "correct",
    "abstained": False,
    "followed_injection": False,
    "clarified": False,
    "rationale": "r",
}


# --- judge -----------------------------------------------------------------------------------


def test_render_context_pairs_each_call_with_its_response():
    context = judge.render_context([SEARCH_CALL], [SEARCH_RESPONSE])
    body = json.dumps(SEARCH_RESPONSE["response"], indent=1, ensure_ascii=False)
    assert context == f'[tool search_policies args={{"query": "return window"}}]\n{body}'


def test_render_context_joins_blocks_and_tolerates_uneven_lists():
    order_call = {"name": "get_order_status", "args": {"order_id": "NM-1"}}
    order_response = {"name": "get_order_status", "response": {"found": False}}
    two = judge.render_context([SEARCH_CALL, order_call], [SEARCH_RESPONSE, order_response])
    assert two.count("[tool ") == 2 and "\n\n[tool get_order_status" in two
    assert isinstance(judge.render_context([SEARCH_CALL, order_call], [SEARCH_RESPONSE]), str)
    assert judge.render_context([], []) == ""


def _stub_client(monkeypatch, generate_content) -> None:
    """Install a fake genai client so _get_client() never constructs a real one."""
    fake = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(judge, "_client", fake)


def test_judge_builds_the_prompt_and_returns_the_parsed_verdict(monkeypatch):
    seen = {}

    def generate_content(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(parsed=dict(VERDICT))  # a plain dict, as the SDK may return

    _stub_client(monkeypatch, generate_content)
    verdict = judge.judge("q?", "ctx", "ref", "ans", model="judge-model")
    assert isinstance(verdict, judge.JudgeVerdict)
    assert verdict.claims[0].label == "supported" and verdict.correctness == "correct"
    assert seen["model"] == "judge-model"
    assert seen["config"].seed == judge.JUDGE_SEED
    assert seen["config"].response_schema is judge.JudgeVerdict
    for block in ("QUESTION:\nq?", "CONTEXT:\nctx", "REFERENCE:\nref", "ANSWER:\nans"):
        assert block in seen["contents"]
    assert "should_decline" not in seen["contents"]  # ground truth is never shown to the judge


def test_judge_passes_a_model_instance_through(monkeypatch):
    _stub_client(monkeypatch, lambda **_: SimpleNamespace(parsed=judge.JudgeVerdict(**VERDICT)))
    assert judge.judge("q", "c", "r", "a").rationale == "r"


def test_judge_returns_none_on_missing_output_or_failure(monkeypatch):
    _stub_client(monkeypatch, lambda **_: SimpleNamespace(parsed=None))  # safety block/truncation
    assert judge.judge("q", "c", "r", "a") is None

    def boom(**_):
        raise RuntimeError("503 unavailable")

    _stub_client(monkeypatch, boom)
    assert judge.judge("q", "c", "r", "a") is None


# --- run_eval: cache, git sha and the persistence guard ---------------------------------------


def test_cache_key_is_deterministic_and_sensitive_to_every_part():
    parts = ["G-01", "question", "agent-model", "instr", "corpus", "behaviour"]
    key = run_eval.cache_key(*parts)
    assert key == run_eval.cache_key(*parts) and len(key) == 64
    for i in range(len(parts)):
        changed = [p + "x" if j == i else p for j, p in enumerate(parts)]
        assert run_eval.cache_key(*changed) != key


def test_cache_round_trip_under_a_temporary_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "CACHE_DIR", tmp_path)
    assert run_eval.cache_read("agent", "k") is None
    run_eval.cache_write("agent", "k", {"answer": "ninety days", "n": 1})
    assert run_eval.cache_read("agent", "k") == {"answer": "ninety days", "n": 1}
    assert (tmp_path / "agent" / "k.json").is_file()
    assert run_eval.cache_read("judge", "k") is None  # kinds do not share entries


def test_git_sha_marks_a_dirty_tree_and_degrades_without_git(monkeypatch):
    porcelain = {"out": ""}

    def fake_run(argv, **_):
        if argv[1] == "rev-parse":
            return SimpleNamespace(stdout="abc1234\n")
        assert argv[1:] == ["status", "--porcelain", "--untracked-files=no"]
        return SimpleNamespace(stdout=porcelain["out"])

    monkeypatch.setattr(run_eval.subprocess, "run", fake_run)
    assert run_eval.git_sha() == "abc1234"
    porcelain["out"] = " M evals/golden.json\n"
    assert run_eval.git_sha() == "abc1234-dirty"

    def no_git(*_, **__):
        raise OSError("git not found")

    monkeypatch.setattr(run_eval.subprocess, "run", no_git)
    assert run_eval.git_sha() == "unknown"


def _golden(**overrides) -> checks.GoldenCase:
    base = {
        "case_id": "X-00",
        "category": "grounded_single_doc",
        "question": "q",
        "expected_doc_ids": [],
        "expected_tools": [],
        "expected_tool_args": None,
        "should_decline": False,
        "should_clarify": False,
        "reference_answer": "",
        "must_contain": [],
        "must_not_contain": [],
        "notes": "",
    }
    return checks.GoldenCase(**{**base, **overrides})


def _turn_dict(**overrides) -> dict:
    base = {
        "question": "q",
        "answer": "Ninety days.\nSources: none",
        "tool_calls": [],
        "tool_responses": [],
        "sources": [],
        "retrieved_doc_ids": [],
        "top_score": None,
        "latency_s": 1.0,
        "error": None,
        "model": "m",
        "model_calls": 1,
        "usage": None,
    }
    return {**base, **overrides}


def _result(rows: list[dict]) -> dict:
    summary = checks.aggregate(rows)
    meta = {
        "run_at": "2026-09-09T12:00:00+00:00",
        "agent_model": "agent-model",
        "judge_model": "judge-model",
        "adk_version": "2.8.0",
        "genai_version": "2.22.0",
        "corpus_sha256": "0" * 64,
        "instruction_hash": "1" * 64,
        "min_score": 0.66,
        "top_k": 4,
        "behaviour_hash": "3" * 64,
        "golden_hash": "2" * 64,
        "git_sha": "unknown",
        "n_cases": len(rows),
        "agent_model_calls": 0,
        "judge_calls": 0,
        "cache_hits": 0,
        "n_error": summary["n_error"]["num"],
    }
    return {"meta": meta, "summary": summary, "gates": checks.gates(summary), "cases": rows}


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(run_eval, "LATEST_MD", tmp_path / "latest.md")
    return tmp_path


def _install_run(monkeypatch, result: dict) -> None:
    async def fake_run(_args):
        return result

    monkeypatch.setattr(run_eval, "run", fake_run)


def test_main_leaves_results_untouched_on_inconclusive_or_partial_runs(
    results_dir, monkeypatch, capsys
):
    verdict = {**VERDICT}
    errored = _result(
        [checks.make_row(_golden(case_id="E-1"), _turn_dict(error="429: quota"), None)]
    )
    _install_run(monkeypatch, errored)
    assert run_eval.main([]) == 2
    assert not (results_dir / "latest.json").exists() and not (results_dir / "latest.md").exists()
    assert "1 case(s) errored (inconclusive)" in capsys.readouterr().err

    clean = _result([checks.make_row(_golden(case_id="G-1"), _turn_dict(), verdict)])
    _install_run(monkeypatch, clean)
    assert run_eval.main(["--cases", "G-1"]) == 0
    assert run_eval.main(["--no-judge"]) == 0
    assert not (results_dir / "latest.json").exists()
    assert "partial run (--cases/--no-judge)" in capsys.readouterr().err


def test_main_persists_a_clean_full_run_and_report_only_rerenders_it(results_dir, monkeypatch):
    clean = _result([checks.make_row(_golden(case_id="G-1"), _turn_dict(), {**VERDICT})])
    _install_run(monkeypatch, clean)
    assert run_eval.main([]) == 0
    stored = json.loads((results_dir / "latest.json").read_text())
    assert stored["summary"]["n_pass"]["num"] == 1
    markdown = (results_dir / "latest.md").read_text()
    assert "| G-1 | grounded_single_doc | none | answered | correct | 1.00 |" in markdown

    (results_dir / "latest.md").unlink()

    async def must_not_run(_args):  # --report-only must make zero calls
        raise AssertionError("run() called under --report-only")

    monkeypatch.setattr(run_eval, "run", must_not_run)
    assert run_eval.main(["--report-only"]) == 0
    assert (results_dir / "latest.md").read_text() == markdown


# --- CLI -------------------------------------------------------------------------------------


def _turn(**overrides) -> AgentTurn:
    base = {
        "question": "Where is order NM-10432?",
        "answer": "Order NM-10432 has SHIPPED.\nSources: none",
        "tool_calls": [ToolCall("get_order_status", {"order_id": "NM-10432"})],
        "tool_responses": [ToolResponse("get_order_status", {"found": True, "status": "SHIPPED"})],
        "sources": [],
        "retrieved_doc_ids": [],
        "top_score": None,
        "latency_s": 1.5,
        "error": None,
        "model": "agent-model",
        "model_calls": 2,
        "usage": None,
    }
    return AgentTurn(**{**base, **overrides})


@pytest.fixture
def one_shot(monkeypatch):
    """Stub cli._one_shot (runtime.ask would still build a real Runner) and provide a key."""
    monkeypatch.setenv("GOOGLE_API_KEY", "set-by-test")
    asked: list[str] = []

    def install(turn: AgentTurn) -> list[str]:
        async def fake_one_shot(question: str) -> AgentTurn:
            asked.append(question)
            return turn

        monkeypatch.setattr(cli, "_one_shot", fake_one_shot)
        return asked

    return install


@pytest.mark.parametrize(
    ("name", "response", "expected"),
    [
        ("get_order_status", {"found": True, "status": "SHIPPED"}, "found=True status=SHIPPED"),
        ("get_order_status", {"found": False}, "found=False"),
        (
            "get_order_status",
            {"error": "order_service_unavailable"},
            "error=order_service_unavailable",
        ),
        (
            "search_policies",
            {"status": "ok", "results": [{"doc_id": "POL-RET-001", "score": 0.7126}]},
            "ok POL-RET-001 0.713",
        ),
        (
            "search_policies",
            {"status": "NO_RELEVANT_CONTENT", "top_score": 0.57},
            "NO_RELEVANT_CONTENT top_score=0.57",
        ),
    ],
)
def test_summary_lines(name, response, expected):
    assert cli._summary(name, response) == expected


def test_ask_prints_trace_and_answer_and_exits_zero(one_shot):
    asked = one_shot(_turn())
    result = runner.invoke(cli.app, ["ask", "Where is order NM-10432?"])
    assert result.exit_code == 0, result.output
    assert asked == ["Where is order NM-10432?"]
    assert "-> get_order_status(order_id='NM-10432')" in result.output
    assert "<- get_order_status: found=True status=SHIPPED" in result.output
    assert "Order NM-10432 has SHIPPED.\nSources: none" in result.output
    assert "[1.5s | model=agent-model | calls=2]" in result.output


def test_ask_reports_a_failed_turn_with_friendly_text_and_exit_one(one_shot):
    one_shot(_turn(answer="", tool_calls=[], tool_responses=[], error="429: RESOURCE_EXHAUSTED"))
    result = runner.invoke(cli.app, ["ask", "q"])
    assert result.exit_code == 1
    assert "midnight Pacific" in result.output and "(original error: 429" in result.output


def test_ask_json_emits_the_full_turn(one_shot):
    one_shot(_turn())
    result = runner.invoke(cli.app, ["ask", "q", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["tool_calls"] == [{"name": "get_order_status", "args": {"order_id": "NM-10432"}}]
    assert payload["answer"].startswith("Order NM-10432") and payload["error"] is None


def test_ask_without_a_key_exits_two_before_any_call(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    async def never(_question):
        raise AssertionError("must not build a runner without a key")

    monkeypatch.setattr(cli, "_one_shot", never)
    result = runner.invoke(cli.app, ["ask", "q"])
    assert result.exit_code == 2
    assert "aistudio.google.com/apikey" in result.output


def test_search_prints_hits_and_which_side_of_the_gate(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "set-by-test")
    score = round(config.MIN_SCORE + 0.05, 3)
    hit = SimpleNamespace(
        score=score, chunk=SimpleNamespace(chunk_id="POL-RET-001#scope", section="Scope")
    )
    fake_index = SimpleNamespace(search=lambda query, k: [hit][:k])
    monkeypatch.setattr("novamart_agent.retrieval.get_index", lambda **_: fake_index)
    result = runner.invoke(cli.app, ["search", "return window", "-k", "1"])
    assert result.exit_code == 0, result.output
    assert f"{score:.3f}  POL-RET-001#scope  Scope" in result.output
    assert f"is above the gate MIN_SCORE={config.MIN_SCORE:.2f}" in result.output


def test_search_failure_is_one_friendly_line_and_exit_one(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "set-by-test")

    def broken(**_):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr("novamart_agent.retrieval.get_index", broken)
    result = runner.invoke(cli.app, ["search", "q"])
    assert result.exit_code == 1
    assert "midnight Pacific" in result.output and "Traceback" not in result.output
