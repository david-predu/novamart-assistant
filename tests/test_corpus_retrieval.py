"""Offline tests for corpus parsing, chunking and the cached embedding index (no API key)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from novamart_agent import config, retrieval
from novamart_agent.corpus import Chunk, chunk_doc, corpus_sha256, load_chunks, load_docs, slug
from novamart_agent.retrieval import (
    GeminiEmbedder,
    PolicyIndex,
    build_index,
    get_index,
    is_fresh,
    load_index,
    reset_index,
    save_index,
)

DIMS = 64

RETURNS_DOC = """---
doc_id: POL-RET-001
title: Returns & Refunds
owner: Customer Care Operations
last_reviewed: 2026-08-15
audience: store_employees
unknown_key: ignored
---
# Returns & Refunds

## Standard return window
Customers may return unused items within 30 days of receipt with tags attached.

## Electronics
Electronics may be returned within 14 days.
"""

GIFT_CARD_DOC = """---
doc_id: POL-GC-006
title: Gift Cards
owner: Customer Care Operations
last_reviewed: 2026-07-01
audience: store_employees
---
# Gift Cards
Gift cards never expire and carry no fees.

## Lost cards
Lost cards are replaced only with the receipt and card number.
"""


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder that counts document-side calls."""

    model = "fake-bow"
    dims = DIMS

    def __init__(self) -> None:
        self.document_calls = 0

    def embed(self, texts: list[str], task_type: str) -> np.ndarray:
        if task_type == retrieval.DOCUMENT_TASK:
            self.document_calls += 1
        rows = np.zeros((len(texts), DIMS), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                rows[i, int(hashlib.sha256(token.encode()).hexdigest(), 16) % DIMS] += 1
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def make_chunk(chunk_id: str, text: str) -> Chunk:
    doc_id, _, section = chunk_id.partition("#")
    return Chunk(chunk_id, doc_id, "T", section, text, hashlib.sha256(text.encode()).hexdigest())


@pytest.fixture(autouse=True)
def _fresh_singleton():
    reset_index()
    yield
    reset_index()


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    (tmp_path / "POL-RET-001.md").write_text(RETURNS_DOC)
    (tmp_path / "POL-GC-006.md").write_text(GIFT_CARD_DOC)
    return tmp_path


def test_front_matter_is_parsed(docs_dir: Path):
    docs = load_docs(docs_dir)
    assert [d.doc_id for d in docs] == ["POL-GC-006", "POL-RET-001"]  # sorted by path
    ret = docs[1]
    assert ret.title == "Returns & Refunds"
    assert ret.owner == "Customer Care Operations"
    assert ret.last_reviewed == "2026-08-15"
    assert ret.audience == "store_employees"
    assert ret.path == docs_dir / "POL-RET-001.md"
    assert ret.body.startswith("# Returns & Refunds")


def test_front_matter_requires_doc_id_and_title(tmp_path: Path):
    (tmp_path / "bad.md").write_text("---\ntitle: No id\n---\n# No id\n")
    with pytest.raises(ValueError, match="doc_id"):
        load_docs(tmp_path)


def test_slug():
    assert slug("Standard Return Window!") == "standard-return-window"
    assert slug("  Price -- Match & Adjust ") == "price-match-adjust"


def test_chunk_doc_splits_on_h2(docs_dir: Path):
    ret, gift = load_docs(docs_dir)[1], load_docs(docs_dir)[0]
    chunks = chunk_doc(ret)
    assert [c.chunk_id for c in chunks] == [
        "POL-RET-001#standard-return-window",
        "POL-RET-001#electronics",
    ]  # blank intro under the title produces no overview chunk
    assert [c.section for c in chunks] == ["Standard return window", "Electronics"]
    assert (
        chunks[1].text
        == "Returns & Refunds > Electronics\nElectronics may be returned within 14 days."
    )
    assert chunks[1].sha256 == hashlib.sha256(chunks[1].text.encode()).hexdigest()
    gift_chunks = chunk_doc(gift)
    assert gift_chunks[0].chunk_id == "POL-GC-006#overview"
    assert (
        gift_chunks[0].text == "Gift Cards > overview\nGift cards never expire and carry no fees."
    )


def test_load_chunks_rejects_duplicate_ids(tmp_path: Path):
    (tmp_path / "dup.md").write_text(
        "---\ndoc_id: POL-X-001\ntitle: Dup\n---\n# Dup\n\n## Same\na\n\n## Same\nb\n"
    )
    with pytest.raises(AssertionError, match="duplicate"):
        load_chunks(tmp_path)


def test_corpus_sha256_is_stable_and_content_sensitive(docs_dir: Path):
    chunks = load_chunks(docs_dir)
    assert corpus_sha256(chunks) == corpus_sha256(load_chunks(docs_dir))
    changed = [replace(chunks[0], text=chunks[0].text + "!", sha256="0" * 64)] + chunks[1:]
    assert corpus_sha256(changed) != corpus_sha256(chunks)


def test_policy_index_ranks_by_shared_tokens():
    chunks = [
        make_chunk("A#one", "alpha beta gamma"),
        make_chunk("B#two", "delta epsilon"),
        make_chunk("C#three", "alpha beta zeta"),
    ]
    embedder = FakeEmbedder()
    index = PolicyIndex(
        chunks, embedder.embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT"), embedder
    )
    hits = index.search("alpha beta gamma", k=3)
    assert [h.chunk.chunk_id for h in hits] == ["A#one", "C#three", "B#two"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score > hits[1].score > hits[2].score
    assert len(index.search("alpha", k=2)) == 2


def test_save_and_load_index_round_trip(tmp_path: Path, docs_dir: Path):
    chunks = load_chunks(docs_dir)
    vectors = FakeEmbedder().embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT")
    path = tmp_path / "index" / "policy_index.json"
    save_index(path, chunks, vectors, "fake-bow", DIMS)
    meta, cached_chunks, cached_vectors = load_index(path)
    assert meta["model"] == "fake-bow" and meta["dims"] == DIMS
    assert meta["task_type"] == "RETRIEVAL_DOCUMENT"
    assert meta["corpus_sha256"] == corpus_sha256(chunks)
    assert meta["built_at"].endswith("+00:00")
    assert cached_chunks == chunks
    assert cached_vectors.dtype == np.float32
    np.testing.assert_allclose(cached_vectors, vectors, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(cached_vectors, axis=1), 1.0, atol=1e-6)
    assert load_index(tmp_path / "missing.json") is None


def test_is_fresh_flips_on_text_model_or_dims(docs_dir: Path):
    chunks = load_chunks(docs_dir)
    meta = {"model": "fake-bow", "dims": DIMS}
    assert is_fresh(meta, chunks, chunks, "fake-bow", DIMS)
    edited = [replace(chunks[0], text="x", sha256="0" * 64)] + chunks[1:]
    assert not is_fresh(meta, chunks, edited, "fake-bow", DIMS)
    assert not is_fresh(meta, chunks, chunks[:-1], "fake-bow", DIMS)
    assert not is_fresh(meta, chunks, chunks, "gemini-embedding-001", DIMS)
    assert not is_fresh(meta, chunks, chunks, "fake-bow", 768)


def test_get_index_uses_fresh_cache_without_embedding_documents(
    tmp_path: Path, docs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    index_path = tmp_path / "index" / "policy_index.json"
    monkeypatch.setattr(config, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(config, "INDEX_PATH", index_path)
    builder = FakeEmbedder()
    build_index(load_chunks(), builder, index_path)
    assert builder.document_calls == 1

    reader = FakeEmbedder()
    index = get_index(embedder=reader)
    assert reader.document_calls == 0
    assert get_index() is index  # process-wide singleton
    assert index.search("return window electronics", k=1)[0].chunk.doc_id == "POL-RET-001"

    (docs_dir / "POL-GC-006.md").write_text(GIFT_CARD_DOC.replace("no fees", "no fees at all"))
    reset_index()
    rebuilder = FakeEmbedder()
    get_index(embedder=rebuilder)
    assert rebuilder.document_calls == 1  # stale cache -> rebuilt
    assert is_fresh(*load_index(index_path)[:2], load_chunks(), "fake-bow", DIMS)

    get_index(force_rebuild=True, embedder=rebuilder)
    assert rebuilder.document_calls == 2


def test_get_index_without_key_and_cache_raises_actionable_error(
    tmp_path: Path, docs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(config, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(config, "INDEX_PATH", tmp_path / "missing.json")
    GeminiEmbedder()  # constructing the embedder never needs a key
    with pytest.raises(RuntimeError, match="uv run novamart index"):
        get_index()


def test_gemini_embedder_batches_and_normalises():
    calls: list[tuple[str, int, str, int]] = []

    def embed_content(*, model, contents, config):
        calls.append((model, len(contents), config.task_type, config.output_dimensionality))
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[float(i + 1), 0.0, 0.0, 0.0]) for i in range(len(contents))
            ]
        )

    embedder = GeminiEmbedder(model="m", dims=4)
    embedder._client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    vectors = embedder.embed([f"text {i}" for i in range(150)], "RETRIEVAL_DOCUMENT")
    assert vectors.shape == (150, 4) and vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert calls == [("m", 100, "RETRIEVAL_DOCUMENT", 4), ("m", 50, "RETRIEVAL_DOCUMENT", 4)]


def test_gemini_embedder_rejects_collapsed_response():
    def embed_content(*, model, contents, config):
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0])])

    embedder = GeminiEmbedder(model="m", dims=2)
    embedder._client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    with pytest.raises(AssertionError, match="gemini-embedding-2"):
        embedder.embed(["a", "b"], "RETRIEVAL_DOCUMENT")


@pytest.mark.skipif(not any(config.DOCS_DIR.glob("*.md")), reason="data/policies not written yet")
def test_real_corpus_parses_with_unique_ids_and_well_formed_chunks():
    docs = load_docs()
    doc_ids = [d.doc_id for d in docs]
    assert len(set(doc_ids)) == len(doc_ids)
    for doc in docs:
        assert re.fullmatch(r"POL-[A-Z]{2,4}-\d{3}", doc.doc_id), doc.path
        for key in ("title", "owner", "last_reviewed", "audience"):
            assert getattr(doc, key), f"{doc.path}: front matter missing {key}"
    chunks = load_chunks()
    assert chunks
    for chunk in chunks:
        assert chunk.chunk_id == f"{chunk.doc_id}#{slug(chunk.section)}"
        assert re.fullmatch(r"POL-[A-Z]{2,4}-\d{3}#[a-z0-9]+(?:-[a-z0-9]+)*", chunk.chunk_id)


@pytest.mark.skipif(not config.INDEX_PATH.exists(), reason="no committed index yet")
def test_committed_index_matches_corpus():
    meta, cached_chunks, vectors = load_index(config.INDEX_PATH)
    chunks = load_chunks()
    assert is_fresh(meta, cached_chunks, chunks, config.EMBED_MODEL, config.EMBED_DIMS)
    assert vectors.shape == (len(chunks), config.EMBED_DIMS)
