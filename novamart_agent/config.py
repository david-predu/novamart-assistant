"""Single place for paths, environment knobs and the shared retry policy.

Every entry point (ADK CLI, our CLI, Streamlit, the eval harness, pytest) imports this
module first, so `.env` is loaded exactly once. Shell environment wins over `.env`.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

DOCS_DIR = ROOT / "data" / "policies"
ORDERS_PATH = ROOT / "data" / "orders.json"
INDEX_PATH = ROOT / "data" / "index" / "policy_index.json"

# Must equal the package folder name, otherwise ADK's Runner logs an "App name mismatch" warning.
APP_NAME = "novamart_agent"

# Stable, free-tier model ids only. Never "-latest" aliases or dated previews (non-reproducible evals).
MODEL = os.getenv("NOVAMART_MODEL", "gemini-3.5-flash")
JUDGE_MODEL = os.getenv("NOVAMART_JUDGE_MODEL", "gemini-2.5-flash")
EMBED_MODEL = "gemini-embedding-001"  # supports task_type + output_dimensionality; shutdown 2028-05-14
EMBED_DIMS = 768

TOP_K = int(os.getenv("NOVAMART_TOP_K", "4"))
MIN_SCORE = float(os.getenv("NOVAMART_MIN_SCORE", "0.50"))

# google-genai does NOT retry by default; share one policy across agent, embedder and judge clients.
RETRY = types.HttpRetryOptions(
    attempts=5,
    initial_delay=2.0,
    max_delay=60.0,
    exp_base=2.0,
    jitter=0.5,
    http_status_codes=[429, 500, 502, 503, 504],
)
