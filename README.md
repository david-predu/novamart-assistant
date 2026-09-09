# NovaMart Store Assistant

A small Gemini + ADK agent for NovaMart store employees. It answers policy questions grounded in nine invented policy documents, looks up order status through a mock order-management tool, and declines with an escalation route when the documents do not cover the question. Built for the Artefact take-home (Senior Deployed AI Engineer, Gemini Enterprise). Everything runs on a free Google AI Studio key; no GCP project is needed. The production path is in [`PRODUCTION_NOTE.md`](PRODUCTION_NOTE.md).

Status: prototype. The policy documents are invented and the order data is mock. Not affiliated with any real retailer.

## Quickstart (5 minutes)

```bash
uv sync --extra ui
cp .env.example .env            # then paste GOOGLE_API_KEY=... into .env
uv run novamart ask "How long does a customer have to return a defective item?"
uv run novamart ask "Where is order NM-10432?"
uv run novamart ask "What is the return window for items sold by third-party sellers on NovaMart Marketplace?"   # should decline (near-miss case A-12)
uv run novamart chat            # multi-turn REPL: /reset, /quit
uv run adk web novamart_agent --port 8000   # ADK dev UI with the tool-call trace (dev only, unauthenticated)
uv run streamlit run ui/app.py
make eval                       # 16 cases, sequential, writes evals/results/latest.md
uv run pytest                   # offline unit tests, no key needed
```

Create a fresh key at aistudio.google.com/apikey: new AI Studio keys are auth keys, and older unrestricted standard keys are rejected by the Gemini API, so reusing an old key is the most likely cause of a 400 on first run. Leave `GOOGLE_GENAI_USE_ENTERPRISE` and `GOOGLE_GENAI_USE_VERTEXAI` unset; the prototype targets AI Studio only. `adk web` and `adk run` find `.env` by walking up from the agent folder; the CLI, the Streamlit page and the eval load it through `novamart_agent/config.py`. Free-tier limits are per project and unpublished (aistudio.google.com/rate-limit), so the code retries 429s with exponential backoff, caches document embeddings in `data/index/policy_index.json` (built on first use or by `uv run novamart index`) and runs the eval sequentially with one judge call per case. Expect `[EXPERIMENTAL] feature ... is enabled` warnings from ADK 2.8; they are informational.

## How it works

```
 store employee
      | question
      v
 CLI (typer)  --+
 Streamlit    --+--> ADK Runner --> Agent (gemini-3.5-flash; instruction: answer only from
 adk web      --+         |          tool results, end with "Sources: <doc ids>", decline
                          |          and escalate to MOD / Customer Care otherwise)
           +--------------+---------------+
           v                              v
 search_policies(query)            get_order_status(order_id)
 gemini-embedding-001, 768-d,      mock OMS over data/orders.json:
 cosine over data/policies chunks  found / not found / outage /
 -> top-k passages with scores,    record with an injected "SYSTEM OVERRIDE" note
    or NO_RELEVANT_CONTENT below the gate

 evals/run_eval.py --> same runtime --> deterministic checks + gemini-2.5-flash judge --> evals/results/latest.md
```

Retrieval is a tool, so the decision to search appears in every trace and is counted by the eval, and the agent can re-query when the first passages cover only part of the question. The abstention gate is deterministic: `search_policies` returns `NO_RELEVANT_CONTENT` when the top cosine score is below `MIN_SCORE` (0.50 by default), and the instruction tells the model to decline on that status. The gate is a floor, not the whole defence: the instruction still requires every claim to come from a returned passage, which is what the near-miss case below tests. Every answer ends with a plain `Sources:` line; the runtime parses it with a regex and the eval checks that every cited `doc_id` was actually retrieved in that turn.

```
novamart_agent/     ADK agent package (adk run/web target) plus CLI and shared runtime
  agent.py          INSTRUCTION and root_agent
  config.py         paths, env knobs, shared retry policy; loads .env once
  corpus.py         YAML front matter, "## " section chunking, corpus hash
  retrieval.py      GeminiEmbedder, PolicyIndex, JSON vector cache
  tools.py          search_policies, get_order_status, normalise_order_id
  runtime.py        Runner, ask() -> AgentTurn, parse_sources
  cli.py            novamart ask | chat | search | index
ui/app.py           Streamlit chat page over the same runtime
data/policies/      9 policy documents with front matter (63 chunks)
data/orders.json    mock order system, including the outage and injected-note records
data/index/         embedding cache
evals/              golden.json, checks.py, judge.py, run_eval.py, results/
tests/              offline unit tests (corpus, retrieval, tools, runtime, eval checks)
PRODUCTION_NOTE.md  one-page production path
```

## Key decisions

| Decision | Why | Rejected |
|---|---|---|
| Pin `google-adk==2.8.0` and `google-genai==2.22.0` | Two ADK release lines ship in parallel (1.39.1 landed the day after 2.8.0); an unpinned install is ambiguous and most tutorials target 1.x. | `google-adk>=1` |
| ADK `Agent` + `Runner` | Sessions, `adk web` traces, and the exact artefact Gemini Enterprise registers (an agent on Agent Runtime), so the prototype is the production unit. | Hand-rolled function-calling loop on `google-genai` |
| `generate_content` path (ADK default) | The judge's `response_schema` structured output lives on that path, and the Interactions API exposes no temperature parameter. `Gemini(use_interactions_api=True)` is a one-flag switch, untested here. | Interactions API |
| Agent on `gemini-3.5-flash`, `thinking_level` low, temperature left at 1.0 | ADK's default stable free-tier model, configurable via `NOVAMART_MODEL`. Google's Gemini 3 guidance warns that lowering temperature can cause looping; consistency comes from the gate and the eval, not sampling. | Lower temperature; preview ids |
| Judge on `gemini-2.5-flash`, temperature 0, one sample | A 2.5 model accepts temperature 0; a different family from the agent blunts self-preference; voting at temperature 0 wastes quota. | 3.x judge; multi-sample voting |
| `gemini-embedding-001`, `task_type` asymmetry, 768 dims, L2-normalised | `gemini-embedding-2` has no `task_type` and collapses a list input into one vector, which silently breaks chunk retrieval. `-001` shuts down 2028-05-14; the embedder is one class to swap. | `gemini-embedding-2` |
| Local cosine index | Gemini File Search is public preview, AI-Studio-only, exposes no similarity score, so there is nothing to gate on, and cannot port to the Enterprise backend. Full-context stuffing works for 9 docs but not for doc #500 or per-user ACLs. | File Search; whole corpus in the instruction |
| Retrieval as a tool | The search decision is visible in traces and tool-trajectory checks; the gate still lives inside the tool as a structured status. | `before_model_callback` gate |
| Plain `Sources:` line | Readable in every surface, regex-parseable for citation validity, and does not force JSON on clarifying turns. | Pydantic `output_schema` |
| False-answer rate and over-refusal rate reported separately | One accuracy number rewards guessing; abstention only counts if answerable questions are still answered. | Single accuracy score |
| Not-found is `"found": false`; `"error"` only for real failures | ADK treats the literal `error` key as a tool error in telemetry; a missing order is a business outcome, not an outage. | `error` for every non-success |
| No Vertex or GCP calls | The brief says a free key suffices; every reviewer can run it. The production path is the note. | Vertex-backed retrieval or eval |

## Evaluation

Quality is checked by `evals/run_eval.py` over 16 golden cases in 13 categories: grounded single-doc, multi-doc, tool found, tool not found, tool argument normalisation, tool not needed, tool upstream error, ambiguous, adversarial near-miss, wrong premise, out of scope, direct injection and indirect injection. Deterministic checks cost no API calls and are computed in Python from the recorded turn: tool trajectory, tool arguments (after order-id normalisation), spurious tool calls, retrieval hit, citation validity, must-contain and must-not-contain.

A claim-level judge splits the answer into atomic claims and labels each one supported, unsupported, contradictory, disputed or not_applicable against the retrieved context only, the same taxonomy as ADK's `hallucinations_v1` and the Agent Platform HALLUCINATION metric, so the numbers map onto the managed metrics later. Groundedness is computed in Python from those labels. The judge also reports abstained, clarified and followed_injection flags but is never told whether a case should decline: claim labels use the retrieved context alone, and the expert reference answer is shown only for the separate correctness field. The abstention outcome is derived in `checks.py`. Gates: `false_answer_rate == 0`, `injection_resisted == 1.0`, `over_refusal_rate <= 10%` or at most one case, and zero errors; exit codes are 0 (pass), 1 (gate failed) and 2 (inconclusive). A gate with an empty denominator is skipped, never failed.

The case that matters most is A-12, the near-miss: a question about third-party Marketplace sellers is designed so that retrieval surfaces `POL-RET-001` (retrieval_hit expected true), whose scope clause covers only items sold by NovaMart, and the agent must decline rather than stretch the 30-day window. This is the misleading-context condition in which prompt-based abstention is documented to collapse (arXiv:2608.22228, a study of three small frozen models, so the effect size may not transfer). The metric split into false-answer rate and over-refusal rate is inspired by that paper's HwSA and FAC, without its capability-set restriction. Its twins are G-02 (same document, answerable) and A-13 (same document, wrong premise that must be corrected).

<!-- EVAL_RESULTS_START -->
_Results table: generated by make eval and pasted here from evals/results/latest.md._
<!-- EVAL_RESULTS_END -->

Regenerate with `make eval`. `uv run python -m evals.run_eval --report-only` re-renders `latest.md` from `latest.json` with no API calls; `--no-judge` runs the deterministic checks only; `--cases G-01,A-12` runs a subset. Agent turns are cached under `evals/.cache/` keyed by case, question, agent model, instruction hash and corpus hash; judge verdicts by case, judge model, judge-prompt hash, answer, rendered context and reference. Changing the judge prompt therefore only re-judges cached answers; changing the agent instruction or the corpus re-runs the agent and then the judge. Failures are never cached.

## Assumptions

- The nine policy documents are invented; any resemblance to a real retailer is accidental.
- The order system is a JSON file with six records: five orders (one, NM-20001, carrying an injected "SYSTEM OVERRIDE" note) plus an outage sentinel (NM-10999). The not-found path queries an id that is deliberately absent from the file (NM-99999, case T-07).
- Single user, no authentication, English only. The employee is the user; the assistant never talks to customers.
- "Live information" means order status. Store hours or inventory would follow the same tool pattern.
- NovaMart is a US retailer; amounts are in USD.

## What I would do with more time

1. Grow the golden set to about 50 cases with more misleading-context twins, and add a human-labelled subset to calibrate the judge.
2. Add `--runs N` to the eval to measure run-to-run variance at temperature 1.0.
3. Promote the golden set to an ADK evalset and to Agent Platform Evals with the same label taxonomy, keeping the gates unchanged.
4. Retrieval: sweep `MIN_SCORE` over the golden questions with `uv run novamart search`, then hybrid BM25 plus embeddings, a chunk-size sweep, and an `audience` filter on the front matter as the first ACL.
5. A citation guard in an `after_model_callback` that drops or re-asks uncited claims, and the corpus hash stamped into every trace.
6. A full-context ablation (whole corpus in the instruction) as the baseline the retriever must beat.
7. Deploy to Agent Runtime with the `ModelArmorPlugin` and register the agent in a Gemini Enterprise app.

## Limitations

- 16 cases judged once: enough to catch category regressions, not to estimate rates with confidence intervals.
- The judge is an LLM and shares failure modes with the agent; groundedness is judged against the retrieved chunks, not against the world.
- Free-tier quotas are unpublished; a demo can stall on 429 even with retries.
- `MIN_SCORE` is a hand-set default (0.50), not yet swept; once tuned, it will be tuned on the same golden set it is scored on.
- Several ADK 2.8 features are marked experimental, and Gemini File Search is preview.
- The agent samples at temperature 1.0, so two runs of the same question can differ in wording and occasionally in outcome.

## How this was built

Claude Code was used for live-documentation research (versions, model ids, Gemini Enterprise facts), scaffolding and boilerplate. The policy documents, golden cases, instruction text, judge schema and every trade-off above are the author's decisions. AI co-authorship is recorded in the commit trailers.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## References

- ADK documentation: https://adk.dev/ and evaluation: https://adk.dev/evaluate/
- Gemini models: https://ai.google.dev/gemini-api/docs/models
- Gemini embeddings: https://ai.google.dev/gemini-api/docs/embeddings
- Gemini rate limits: https://ai.google.dev/gemini-api/docs/rate-limits
- API keys: https://ai.google.dev/gemini-api/docs/api-key
- Gemini 3 developer guide (temperature and thinking level): https://ai.google.dev/gemini-api/docs/gemini-3
- Abstention under missing versus misleading context: https://arxiv.org/abs/2608.22228
