"""Planner — query decomposition (doc section 4.2).

For multi-hop questions, decompose into complementary, self-contained
sub-queries — one per hop or per compared entity — and tag the question type
(bridge / comparison / yes-no / single-hop). Rules from the doc:

  * Each sub-query names its entity explicitly (no pronouns) so embeddings
    spread across the corpus.
  * Sub-queries must be complementary, not paraphrases — dedup near-identical.
  * The type drives both decomposition and the later coverage check.
  * The planner NEVER answers the question; it only restructures it.

Output is structured (`question_type`, `list[SubQuery]`) so the coverage check
(verifier Check 4) can map evidence to hops.

Primary path: an LLM that returns `{"type", "sub_queries": [...]}` JSON.
Fallback (offline / no weights): a rule-based decomposer that handles the common
bridge and comparison shapes. The fallback is intentionally conservative — for
anything it can't confidently decompose it returns a single-hop plan rather than
inventing hops. A capable LLM planner is recommended for general multi-hop, per
the doc; `.backend` reports which path produced the plan.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .state import SubQuery

# An LLM planner: takes the question, returns {"type": str, "sub_queries": [str|dict]}.
PlannerLLM = Callable[[str], dict]

_YESNO_STARTS = (
    "is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ",
    "has ", "have ", "had ", "will ", "would ", "should ", "are there ",
)
_COMPARISON_MARKERS = (" or ", " versus ", " vs ", " vs. ", " than ", " compared to ", " between ")
_COMPARATIVE_WORDS = (
    "older", "younger", "taller", "shorter", "larger", "bigger", "smaller",
    "longer", "earlier", "later", "more", "less", "greater", "higher", "lower",
    "first", "better", "faster",
)

# Relations whose object is itself an entity to be resolved (bridge links).
_BRIDGE_ROLES = (
    "director", "author", "writer", "founder", "president", "ceo", "creator",
    "inventor", "composer", "designer", "producer", "painter", "architect",
    "owner", "captain", "governor", "mayor", "leader", "star", "actor",
)


def detect_question_type(question: str) -> str:
    q = question.strip().lower()
    # Comparison: a comparison marker AND either a wh-word or a comparative term.
    has_marker = any(m in f" {q} " for m in _COMPARISON_MARKERS)
    if has_marker and (
        q.startswith(("which", "who", "what"))
        or any(w in q for w in _COMPARATIVE_WORDS)
    ):
        return "comparison"
    # Bridge: "<attr> of the <role> of <entity>" or possessive nesting.
    if _bridge_match(question) is not None:
        return "bridge"
    if q.startswith(_YESNO_STARTS):
        return "yes-no"
    return "single-hop"


def _bridge_match(question: str):
    """Return (attribute, role, inner_entity) if the question is a bridge.

    Handles two shapes:
      A) "What <attr> is the <role> of <entity>?"  e.g. nationality / director / Ran
      B) "Where/when was the <role> of <entity> <verb>?" (born/founded/...).
    """
    q = question.strip().rstrip("?")
    # Shape A: What/which <attr> is/was the <role> of <entity>
    m = re.match(
        r"^(?:what|which)\s+(\w+)\s+(?:is|was|are|were)\s+the\s+(\w+)\s+of\s+(.+)$",
        q, re.IGNORECASE,
    )
    if m and m.group(2).lower() in _BRIDGE_ROLES:
        return (m.group(1).lower(), m.group(2).lower(), m.group(3).strip())
    # Shape A': What is the <attr> of the <role> of <entity>
    m = re.match(
        r"^what\s+(?:is|was)\s+the\s+(\w+)\s+of\s+the\s+(\w+)\s+of\s+(.+)$",
        q, re.IGNORECASE,
    )
    if m and m.group(2).lower() in _BRIDGE_ROLES:
        return (m.group(1).lower(), m.group(2).lower(), m.group(3).strip())
    # Shape B: Where/when was the <role> of <entity> <verb>
    m = re.match(
        r"^(where|when)\s+(?:was|were|is|are)\s+the\s+(\w+)\s+of\s+(.+?)\s+(born|founded|created|made|established|filmed)$",
        q, re.IGNORECASE,
    )
    if m and m.group(2).lower() in _BRIDGE_ROLES:
        attr = "birthplace" if m.group(1).lower() == "where" else "date"
        return (attr, m.group(2).lower(), m.group(3).strip())
    return None


class Planner:
    def __init__(self, llm: Optional[PlannerLLM] = None):
        self.llm = llm
        self.backend = "llm" if llm else "rule-based"

    def plan(self, question: str) -> tuple[str, list[SubQuery]]:
        """Return (question_type, sub_queries). Never answers the question."""
        if self.llm is not None:
            return self._plan_llm(question)
        return self._plan_rules(question)

    # ------------------------------------------------------------------ #
    # Rule-based decomposition
    # ------------------------------------------------------------------ #
    def _plan_rules(self, question: str) -> tuple[str, list[SubQuery]]:
        qtype = detect_question_type(question)
        if qtype == "comparison":
            subs = self._decompose_comparison(question)
            if len(subs) >= 2:
                return qtype, subs
            return "single-hop", [SubQuery("hop_0", question, 0)]
        if qtype == "bridge":
            subs = self._decompose_bridge(question)
            if subs:
                return qtype, subs
            return "single-hop", [SubQuery("hop_0", question, 0)]
        # yes-no and single-hop both retrieve once.
        return qtype, [SubQuery("hop_0", question, 0)]

    def _decompose_bridge(self, question: str) -> list[SubQuery]:
        parsed = _bridge_match(question)
        if parsed is None:
            return []
        attr, role, inner = parsed
        inner = re.sub(r"^the\s+", "", inner, flags=re.IGNORECASE).strip()
        hop0 = SubQuery(
            id="hop_0",
            text=f"Who is the {role} of {inner}?",
            hop_index=0,
        )
        # Hop 1 is templated on hop 0's answer; text is a best-effort standalone.
        hop1 = SubQuery(
            id="hop_1",
            text=f"What is the {attr} of the {role} of {inner}?",
            hop_index=1,
            resolved=False,
            depends_on="hop_0",
            template=f"What is the {attr} of {{entity}}?",
        )
        return [hop0, hop1]

    def _decompose_comparison(self, question: str) -> list[SubQuery]:
        entities, predicate = _split_comparison(question)
        if len(entities) < 2:
            return []
        subs: list[SubQuery] = []
        for i, ent in enumerate(entities):
            text = f"{predicate} {ent}".strip() if predicate else ent
            subs.append(SubQuery(id=f"hop_{i}", text=_capitalize(text), hop_index=i))
        return _dedup(subs)

    # ------------------------------------------------------------------ #
    # LLM path
    # ------------------------------------------------------------------ #
    def _plan_llm(self, question: str) -> tuple[str, list[SubQuery]]:
        raw = self.llm(question) or {}
        qtype = str(raw.get("type", "single-hop"))
        subs: list[SubQuery] = []
        for i, item in enumerate(raw.get("sub_queries", []) or []):
            if isinstance(item, str):
                subs.append(SubQuery(id=f"hop_{i}", text=item, hop_index=i))
            elif isinstance(item, dict):
                subs.append(
                    SubQuery(
                        id=item.get("id", f"hop_{i}"),
                        text=str(item.get("text", "")),
                        hop_index=int(item.get("hop_index", i)),
                        resolved=bool(item.get("resolved", True)),
                        depends_on=item.get("depends_on"),
                        template=item.get("template"),
                    )
                )
        subs = _dedup([s for s in subs if s.text.strip()])
        if not subs:
            subs = [SubQuery("hop_0", question, 0)]
            qtype = qtype or "single-hop"
        return qtype, subs


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
_PREDICATE_STOP = {
    "which", "who", "what", "is", "was", "are", "were", "the", "of", "a", "an",
    "more", "most", "older", "younger", "bigger", "larger", "taller", "first",
    "earlier", "later", "between", "did", "does", "do", "has", "have",
}


def _split_comparison(question: str) -> tuple[list[str], str]:
    """Split a comparison question into (entities, shared_predicate).

    Heuristic: the entities are separated by a comparison marker; the predicate
    is the residual content words (e.g. 'directed films' from 'who directed more
    films, A or B'). Entities are kept verbatim so they stay name-explicit.
    """
    q = question.strip().rstrip("?")
    # Normalize separators to a single token.
    marker = None
    for m in (" versus ", " vs. ", " vs ", " or ", " than ", " compared to ", " between "):
        if m in f" {q} ".lower():
            marker = m
            break
    if marker is None:
        return [], ""
    # Find the segment containing the entities (after a comma if present).
    head, _, tail = q.partition(",")
    segment = tail if tail and marker.strip() in tail.lower() else q
    # Predicate words come from the head/wh-clause.
    predicate_src = head if tail else ""
    parts = re.split(re.escape(marker.strip()), segment, flags=re.IGNORECASE)
    entities = [_clean_entity(p) for p in parts if _clean_entity(p)]
    predicate = " ".join(
        w for w in re.findall(r"[A-Za-z0-9']+", predicate_src)
        if w.lower() not in _PREDICATE_STOP
    )
    return entities, predicate


def _clean_entity(s: str) -> str:
    s = s.strip().strip(",")
    # Drop a leading wh / comparative scaffold if it leaked into the entity.
    s = re.sub(r"^(?:which|who|what|is|was|the)\s+", "", s, flags=re.IGNORECASE)
    return s.strip()


def _capitalize(s: str) -> str:
    s = s.strip()
    return s[0].upper() + s[1:] if s else s


def _dedup(subs: list[SubQuery]) -> list[SubQuery]:
    """Remove near-identical sub-queries (doc: complementary, not paraphrases)."""
    seen: list[set[str]] = []
    out: list[SubQuery] = []
    for s in subs:
        terms = {t for t in re.findall(r"[a-z0-9]+", s.text.lower()) if len(t) > 2}
        if any(_jaccard(terms, prev) >= 0.8 for prev in seen):
            continue
        seen.append(terms)
        out.append(s)
    # Reindex ids/hops after dedup.
    for i, s in enumerate(out):
        s.id = f"hop_{i}"
        s.hop_index = i
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
