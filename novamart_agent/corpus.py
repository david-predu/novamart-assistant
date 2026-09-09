"""Policy corpus: YAML front matter, `## ` section chunking and content hashing. No network."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import config

# gemini-embedding-001 truncates input at 2,048 tokens; 6,000 characters keeps a wide margin.
MAX_CHUNK_CHARS = 6000

_TITLE_LINE = re.compile(r"\A# .*\n?")
_SECTION_START = re.compile(r"^## ", re.MULTILINE)


@dataclass(frozen=True)
class Doc:
    """One policy document: front-matter fields plus the markdown body."""

    doc_id: str
    title: str
    owner: str
    last_reviewed: str
    audience: str
    body: str
    path: Path


@dataclass(frozen=True)
class Chunk:
    """One `## ` section of a document, the unit that is embedded, retrieved and cited."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    sha256: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_doc(path: Path) -> Doc:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path}: file must start with a '---' front-matter line")
    end = next((i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end is None:
        raise ValueError(f"{path}: front matter is not closed by a second '---' line")
    meta = yaml.safe_load("\n".join(lines[1:end])) or {}
    missing = [key for key in ("doc_id", "title") if not meta.get(key)]
    if missing:
        raise ValueError(f"{path}: front matter is missing {missing}")
    return Doc(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        owner=str(meta.get("owner", "")),
        last_reviewed=str(meta.get("last_reviewed", "")),  # YAML parses bare dates as date objects
        audience=str(meta.get("audience", "")),
        body="\n".join(lines[end + 1 :]),
        path=path,
    )


def load_docs(docs_dir: Path | None = None) -> list[Doc]:
    """Parse every `*.md` in `docs_dir` (default: config.DOCS_DIR, resolved at call time)."""
    docs_dir = docs_dir or config.DOCS_DIR
    return [_parse_doc(path) for path in sorted(docs_dir.glob("*.md"))]


def slug(s: str) -> str:
    """Lower-case, non-alphanumerics collapsed to single dashes, no leading/trailing dash."""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def chunk_doc(doc: Doc) -> list[Chunk]:
    """Split the body on `## ` headings; intro text after the `# ` title becomes `overview`."""
    body = _TITLE_LINE.sub("", doc.body.lstrip(), count=1)
    overview, *sections = _SECTION_START.split(body)
    parts = [("overview", overview)]
    for section in sections:
        heading, _, section_body = section.partition("\n")
        parts.append((heading.strip(), section_body))
    chunks: list[Chunk] = []
    for heading, section_body in parts:
        section_body = section_body.strip()
        if not section_body:
            continue  # a heading with no text has nothing to retrieve
        text = f"{doc.title} > {heading}\n{section_body}"
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{slug(heading)}",
                doc_id=doc.doc_id,
                title=doc.title,
                section=heading,
                text=text,
                sha256=_sha256(text),
            )
        )
    return chunks


def load_chunks(docs_dir: Path | None = None) -> list[Chunk]:
    """All chunks of all docs in path order; enforces unique ids and the embedding size cap."""
    chunks = [chunk for doc in load_docs(docs_dir) for chunk in chunk_doc(doc)]
    ids = [chunk.chunk_id for chunk in chunks]
    duplicates = sorted({chunk_id for chunk_id in ids if ids.count(chunk_id) > 1})
    assert not duplicates, f"duplicate chunk ids (repeated section heading?): {duplicates}"
    for chunk in chunks:
        assert len(chunk.text) <= MAX_CHUNK_CHARS, (
            f"{chunk.chunk_id} is {len(chunk.text)} chars (> {MAX_CHUNK_CHARS}); split the section"
        )
    return chunks


def corpus_sha256(chunks: list[Chunk]) -> str:
    """Fingerprint of the whole corpus: sha256 over the ordered chunk hashes."""
    return _sha256("".join(chunk.sha256 for chunk in chunks))
