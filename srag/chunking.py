"""Semantically-coherent chunking (doc section 4.3).

Prefer paragraph/section-aware chunks over fixed-token windows, with a little
overlap, and carry rich metadata (source title, section, timestamp) that feeds
both citations and later conflict resolution.

This is a dependency-free implementation: it splits on blank lines into
paragraphs, then greedily packs paragraphs into chunks up to a target word
budget, keeping a small overlap of trailing sentences between consecutive
chunks. A document may declare sections with markdown-style `## Heading` lines.
"""

from __future__ import annotations

import re
from typing import Iterable

from .state import Chunk

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$")


def _sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_document(
    text: str,
    source: str,
    *,
    doc_id: str | None = None,
    timestamp: str = "",
    target_words: int = 120,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Split one document into section/paragraph-aware chunks.

    Args:
        text: raw document text. Markdown `#` headings start new sections.
        source: human-readable source title (stored on every chunk).
        doc_id: stable id prefix for chunk ids; defaults to a slug of `source`.
        timestamp: optional ISO date carried into chunk metadata.
        target_words: soft upper bound on chunk size in words.
        overlap_sentences: trailing sentences repeated at the start of the next
            chunk to preserve cross-boundary context.
    """
    doc_id = doc_id or _slug(source)
    section = ""
    chunks: list[Chunk] = []
    idx = 0

    # Group the document into (section, paragraph) units first.
    units: list[tuple[str, str]] = []
    for block in _split_blocks(text):
        heading = _HEADING.match(block)
        if heading:
            section = heading.group(1).strip()
            continue
        units.append((section, block.strip()))

    buf: list[str] = []
    buf_section = ""
    buf_words = 0

    def flush() -> None:
        nonlocal buf, buf_words, idx
        if not buf:
            return
        body = "\n\n".join(buf).strip()
        if body:
            chunks.append(
                Chunk(
                    id=f"{doc_id}::c{idx}",
                    text=body,
                    source=source,
                    section=buf_section,
                    timestamp=timestamp,
                )
            )
            idx += 1
        buf = []
        buf_words = 0

    for sec, para in units:
        # A section change forces a boundary so a chunk never mixes sections.
        if buf and sec != buf_section:
            flush()
        if not buf:
            buf_section = sec

        para_words = _word_count(para)

        # A single oversized paragraph is split on sentence boundaries.
        if para_words > target_words * 1.5:
            flush()
            buf_section = sec
            for piece in _pack_sentences(_sentences(para), target_words):
                chunks.append(
                    Chunk(
                        id=f"{doc_id}::c{idx}",
                        text=piece,
                        source=source,
                        section=sec,
                        timestamp=timestamp,
                    )
                )
                idx += 1
            continue

        if buf_words + para_words > target_words and buf:
            # Carry overlap sentences from the tail of the current buffer.
            tail = _sentences("\n\n".join(buf))[-overlap_sentences:] if overlap_sentences else []
            flush()
            buf_section = sec
            if tail:
                buf.append(" ".join(tail))
                buf_words += _word_count(" ".join(tail))

        buf.append(para)
        buf_words += para_words

    flush()
    return chunks


def chunk_corpus(documents: Iterable[dict], **kwargs) -> list[Chunk]:
    """Chunk many documents.

    Each document is a dict with keys: `text`, `source` (or `title`), optional
    `id`, optional `timestamp`. Returns a flat list of chunks.
    """
    out: list[Chunk] = []
    for doc in documents:
        source = doc.get("source") or doc.get("title") or doc.get("id") or "untitled"
        out.extend(
            chunk_document(
                doc["text"],
                source=source,
                doc_id=doc.get("id"),
                timestamp=doc.get("timestamp", ""),
                **kwargs,
            )
        )
    return out


def _pack_sentences(sentences: list[str], target_words: int) -> list[str]:
    pieces: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for s in sentences:
        w = _word_count(s)
        if cur and cur_words + w > target_words:
            pieces.append(" ".join(cur))
            cur = []
            cur_words = 0
        cur.append(s)
        cur_words += w
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _split_blocks(text: str) -> list[str]:
    # Split on blank lines, but keep heading lines as their own blocks.
    raw = re.split(r"\n\s*\n", text)
    blocks: list[str] = []
    for r in raw:
        lines = r.splitlines()
        cur: list[str] = []
        for ln in lines:
            if _HEADING.match(ln):
                if cur:
                    blocks.append("\n".join(cur))
                    cur = []
                blocks.append(ln)
            else:
                cur.append(ln)
        if cur:
            blocks.append("\n".join(cur))
    return [b for b in blocks if b.strip()]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "doc"
