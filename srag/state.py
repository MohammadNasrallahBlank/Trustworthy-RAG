"""Typed state objects that flow through the pipeline.

These mirror section 5 of the architecture doc. Stage 1 populates the input /
planning / retrieval / generation slices; the verification and control fields
are declared on RAGState so the schema is stable as later stages wire in the
correction loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """A retrievable unit of text with provenance and per-stage scores."""

    id: str
    text: str
    source: str = ""
    section: str = ""
    timestamp: str = ""
    sparse_score: float = 0.0
    dense_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0

    def short(self, n: int = 80) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= n else t[: n - 1] + "…"


@dataclass
class SubQuery:
    """One hop / sub-question produced by the planner.

    For a bridge hop whose phrasing depends on an earlier hop's answer, the
    sub-query carries an unfilled `template` and points at its prerequisite via
    `depends_on`; `resolved` is False until the controller (stage 4) fills it.
    `text` always holds a best-effort standalone phrasing so retrieval/coverage
    can run even before the slot is filled.
    """

    id: str
    text: str
    hop_index: int = 0
    resolved: bool = True
    depends_on: Optional[str] = None   # id of the hop that resolves this one's slot
    template: Optional[str] = None     # e.g. "What nationality is {entity}?"

    def fill(self, entity: str) -> "SubQuery":
        """Return a resolved copy with the template slot filled by `entity`."""
        text = self.template.format(entity=entity) if self.template else self.text
        return SubQuery(
            id=self.id,
            text=text,
            hop_index=self.hop_index,
            resolved=True,
            depends_on=self.depends_on,
            template=self.template,
        )


@dataclass
class Claim:
    """An atomic factual statement plus the chunk ids that should support it."""

    text: str
    citations: list[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return len(self.citations) > 0


@dataclass
class GenerationResult:
    """The generator's structured, extract-first output (doc section 4.5).

    `answer` is *always present* and separate from any free-text explanation so
    a short answer survives even when the explanation is long or truncated.
    """

    answer: str
    claims: list[Claim] = field(default_factory=list)
    unsupported_facts: list[str] = field(default_factory=list)
    answerable: bool = True
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "claims": [{"text": c.text, "citations": c.citations} for c in self.claims],
            "unsupported_facts": self.unsupported_facts,
            "answerable": self.answerable,
        }


@dataclass
class RAGState:
    """Single typed state object for the whole graph (doc section 5)."""

    # Input
    question: str
    question_type: str = "single-hop"

    # Planning
    sub_queries: list[SubQuery] = field(default_factory=list)

    # Retrieval (accumulated across passes)
    evidence: list[Chunk] = field(default_factory=list)
    seen_chunk_ids: set[str] = field(default_factory=set)

    # Generation
    answer: str = ""
    claims: list[Claim] = field(default_factory=list)
    answerable: bool = True
    explanation: str = ""

    # Verification (stage 2+)
    confidence: Optional[float] = None
    diagnosis: Optional[str] = None
    failing_hops: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    conflict: bool = False
    suggested_queries: list[str] = field(default_factory=list)

    # Control / bookkeeping (stage 4+)
    correction_count: int = 0
    new_chunks_last_pass: int = 0
    trace: list[dict] = field(default_factory=list)

    # Output
    final_answer: str = ""
    abstained: bool = False
    answer_status: str = "answered"   # answered | hedged | abstained
    missing: list[str] = field(default_factory=list)   # what could not be verified
    citations: list[str] = field(default_factory=list)

    def log(self, event: str, **data) -> None:
        self.trace.append({"event": event, **data})
