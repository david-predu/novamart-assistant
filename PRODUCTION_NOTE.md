# Production note: NovaMart store-employee assistant

**Recommendation.** Buy the Gemini Enterprise app, Standard edition (or Pay-as-you-go), as surface and governance plane; build only the order-status agent (this repo minus its retriever) on Agent Runtime (Agent Platform, formerly Vertex AI) via `adk deploy agent_engine`, then register the `reasoningEngines/ID` resource into the app. Business is out: Google's edition table leaves full-code agents built outside Gemini Enterprise, the full connector ecosystem and "enterprise-grade security and compliance" blank. Assumed: internal OMS with an HTTP API; headcount in the low thousands; US-only.

## 1. Gemini Enterprise versus custom build

- Two products: the seat-licensed app and the Agent Platform (ADK, Agent Runtime, Gateway, Evals, Observability); "custom build" is this stack minus seats and the app.
- Pure buy fails on order status whatever the connector catalogue holds: ingestion syncs no faster than every three hours, useless for an ETA; order status must be a live tool. Pure build re-implements commodity: ACL-aware search, app shell, feedback, analytics.
- Costs: Agent Compute is free to 50 vCPU-hours and Agent Memory to 100 GiB-hours monthly, then $0.085/vCPU-h and $0.009/GiB-h; Agent Gateway bills on Agent Compute (about 15,000 calls per vCPU-hour), metering §2's OMS calls. Seats dominate; store staff are the largest, lightest-usage population. Levers: the Frontline add-on (Standard/Plus, 150+ users, admin-provisioned agents only) or Pay-as-you-go's $0 seat fee (limited rollout, invoiced billing); Frontline's price is unpublished: price both before day 90's seat mix.
- 30 days: Standard pilot, US multi-region, agent registered, golden set in CI; 60: ACL-scoped stores, Model Armor, Online Monitors; 90: seat mix from measured usage.

## 2. Connecting real data

- Policy documents: connector into a Discovery Engine data store, ingested (not federated) for quality; ACLs via `acl_info`; identity via Workforce Identity Federation or Google Identity.
- Policy Q&A therefore moves to the app's ACL-aware search; the agent keeps only `get_order_status`, so nothing outside those ACLs can leak manager-only text; §4's policy cases re-target that search, order-status cases stay in agent CI.
- Identity syncs as often as every 30 minutes; a revoked employee keeps access until then, so run it at 30.
- Order status: an OMS MCP server in Agent Registry behind Agent Gateway (my judgement); contract as shipped in `tools.py`: `found: true|false`, `"error"` only on a real outage.

## 3. Security and governance

- VPC Service Controls perimeter. Take US multi-region: CMEK, Access Transparency and Model Armor are unavailable in `global`, which Google recommends for the latest features; a retailer needs those controls more than first-day models, and Model Armor below depends on it.
- Enable console Model Armor; it "doesn't automatically protect ADK agents", so the agent carries ADK's `ModelArmorPlugin` (fail-closed).
- Paid Google Cloud carries the Training Restriction; the free AI Studio tier is marked "Used to improve our products: Yes" and Interactions stores requests (1 day free, 55 paid) unless `store=false`: the prototype key never touches the real corpus.

## 4. Evaluating and monitoring quality over time

- CI (Test Case Evaluation): run `evals/golden.json` on Agent Platform Evals with pinned metric versions. Managed GROUNDING (hard-zero) and HALLUCINATION (ratio) score policy answers; the release gates are reference-free rubrics over custom judge fields, so they move as custom `RubricMetric`s. Gates as shipped in `checks.py`: false-answer 0, injection resisted 100%, over-refusal at most 10% or one case, zero judge errors (else inconclusive, never pass). A corpus hash in traces separates document from model changes.
- Production (Online Monitoring): monitors sample traces and logs every ~10 minutes, score via Evals into Cloud Monitoring; restrict OnlineEvaluator creation: it runs as the project service account, so anyone who can create one can attach it to any agent in the project. Observability adds p50/p95/p99 latency, per-tool error rate and how often no tool was called (answering without a lookup). Monthly, the assistant owner promotes clustered thumbs-down turns and monitor failures into the golden set; that is how 16 cases becomes a regression suite.

Sources: [editions](https://docs.cloud.google.com/gemini/enterprise/docs/editions) · [register an ADK agent](https://docs.cloud.google.com/gemini/enterprise/docs/register-and-manage-an-adk-agent) · [connectors and data stores](https://docs.cloud.google.com/gemini/enterprise/docs/connectors/introduction-to-connectors-and-data-stores) · [security controls](https://docs.cloud.google.com/gemini/enterprise/docs/compliance-security-controls) · [locations](https://docs.cloud.google.com/gemini/enterprise/docs/locations) · [evaluate online](https://docs.cloud.google.com/gemini-enterprise-agent-platform/optimize/evaluation/evaluate-online) · [observability](https://docs.cloud.google.com/gemini-enterprise-agent-platform/optimize/observability/overview) · [Agent Platform pricing](https://cloud.google.com/products/gemini-enterprise-agent-platform/pricing)
