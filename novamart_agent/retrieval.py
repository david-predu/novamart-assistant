"""Embedding retrieval over policy chunks with a committed JSON vector cache.

Nothing here touches the network at import time: the Gemini client is created on first use,
and `get_index()` only embeds when the cache is missing or stale.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from google import genai
from google.genai import types

from . import config
from .corpus import Chunk, corpus_sha256, load_chunks

DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"
MAX_BATCH = 100  # embed_content accepts at most 100 inputs per request
NO_KEY_HINT = (
    "policy index is stale or missing and no API key is available; run `uv run novamart index`"
)

_INDEX: PolicyIndex | None = None


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk with its cosine score."""

    chunk: Chunk
    score: float


class Embedder(Protocol):
    """Anything that turns texts into unit-normalised row vectors and knows its own identity."""

    model: str
    dims: int

    def embed(self, texts: list[str], task_type: str) -> np.ndarray: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # keep an all-zero row zero instead of NaN
    return vectors / norms


class GeminiEmbedder:
    """gemini-embedding-001 via google-genai; L2-normalises because dims != 3072 are not unit."""

    def __init__(self, model: str = config.EMBED_MODEL, dims: int = config.EMBED_DIMS) -> None:
        self.model = model
        self.dims = dims
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:  # lazy: importing this module must not need an API key
            self._client = genai.Client(http_options=types.HttpOptions(retry_options=config.RETRY))
        return self._client

    def embed(self, texts: list[str], task_type: str) -> np.ndarray:
        client = self._get_client()
        cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dims)
        rows: list[list[float]] = []
        for start in range(0, len(texts), MAX_BATCH):
            batch = texts[start : start + MAX_BATCH]
            resp = client.models.embed_content(model=self.model, contents=batch, config=cfg)
            embeddings = resp.embeddings or []
            assert len(embeddings) == len(batch), (
                f"{self.model} returned {len(embeddings)} embeddings for {len(batch)} inputs; "
                "gemini-embedding-2 collapses a list into one vector, use gemini-embedding-001"
            )
            rows.extend(e.values or [] for e in embeddings)
        # reshape doubles as a check that every vector has the requested dimensionality
        return _normalise(np.asarray(rows, dtype=np.float32).reshape(len(texts), self.dims))


class PolicyIndex:
    """Cosine search over unit vectors: one matmul per query."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder

    def search(self, query: str, k: int = config.TOP_K) -> list[Hit]:
        """Top-k chunks for `query`, best first."""
        scores = self.vectors @ self.embedder.embed([query], QUERY_TASK)[0]
        top = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]


def save_index(path: Path, chunks: list[Chunk], vectors: np.ndarray, model: str, dims: int) -> None:
    """Write the cache as JSON; vectors are rounded to 6 dp so the committed file stays diffable."""
    payload = {
        "model": model,
        "dims": dims,
        "task_type": DOCUMENT_TASK,
        "corpus_sha256": corpus_sha256(chunks),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "chunks": [asdict(chunk) for chunk in chunks],
        "vectors": np.round(vectors.astype(np.float64), 6).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def load_index(path: Path) -> tuple[dict[str, Any], list[Chunk], np.ndarray] | None:
    """Read the cache back as (meta, chunks, unit vectors); None when the file does not exist."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in data["chunks"]]
    vectors = _normalise(np.asarray(data["vectors"], dtype=np.float32))  # rounding denormalises
    meta = {key: value for key, value in data.items() if key not in ("chunks", "vectors")}
    return meta, chunks, vectors


def is_fresh(
    meta: dict[str, Any],
    cached_chunks: list[Chunk],
    current_chunks: list[Chunk],
    model: str,
    dims: int,
) -> bool:
    """True when the cache was built by the same embedder over byte-identical chunks."""
    same_corpus = [(c.chunk_id, c.sha256) for c in cached_chunks] == [
        (c.chunk_id, c.sha256) for c in current_chunks
    ]
    return same_corpus and meta.get("model") == model and meta.get("dims") == dims


def build_index(chunks: list[Chunk], embedder: Embedder, path: Path) -> PolicyIndex:
    """Embed every chunk as a document, persist the cache and return the searchable index."""
    vectors = embedder.embed([chunk.text for chunk in chunks], DOCUMENT_TASK)
    save_index(path, chunks, vectors, embedder.model, embedder.dims)
    return PolicyIndex(chunks, vectors, embedder)


def get_index(force_rebuild: bool = False, embedder: Embedder | None = None) -> PolicyIndex:
    """Process-wide index: served from the cache when fresh, otherwise rebuilt (needs a key)."""
    global _INDEX
    if _INDEX is not None and not force_rebuild:
        return _INDEX
    embedder = embedder or GeminiEmbedder()
    chunks = load_chunks()
    cached = None if force_rebuild else load_index(config.INDEX_PATH)
    if cached is not None and is_fresh(cached[0], cached[1], chunks, embedder.model, embedder.dims):
        _INDEX = PolicyIndex(chunks, cached[2], embedder)
        return _INDEX
    try:
        _INDEX = build_index(chunks, embedder, config.INDEX_PATH)
    except ValueError as exc:  # genai.Client() raises ValueError when no API key is configured
        raise RuntimeError(NO_KEY_HINT) from exc
    return _INDEX


def reset_index() -> None:
    """Drop the process-wide index so the next get_index() re-reads the cache (tests)."""
    global _INDEX
    _INDEX = None
