"""Dataset loaders for the eval harness (doc section 7).

A dataset is a list of dicts with the fields the harness consumes:
  question: str
  answers: list[str]            # acceptable gold answers ([] for unanswerable)
  supporting_chunk_ids: list[str]   # gold passages, for retrieval recall/MRR
  answerable: bool
  type: str                     # bridge | comparison | single-hop | unanswerable

`load_eval_set` reads the bundled JSONL over the sample corpus. `load_hotpot_style`
adapts a HotpotQA / 2WikiMultiHopQA-style record so the *same* harness can run on
the real datasets locally (where HuggingFace is reachable) without code changes.

Corpus discipline (doc section 7): state your setting. The bundled set is a
distractor setting over `data/sample_corpus.jsonl`; do NOT index only the gold
contexts of your eval questions and call it open retrieval -- that leaks.
"""

from __future__ import annotations

import json
from typing import Iterable


def load_eval_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_hotpot_style(records: Iterable[dict], *, chunk_id_for=None) -> list[dict]:
    """Adapt HotpotQA/2Wiki-style records to the harness schema.

    Each input record is expected to have at least `question` and `answer`, and
    optionally `supporting_facts` (list of [title, sent_id]) and `type`. Provide
    `chunk_id_for(title, sent_id) -> chunk_id` to map gold supporting facts onto
    your indexed chunk ids; without it, retrieval recall/MRR are skipped.
    """
    out: list[dict] = []
    for r in records:
        ans = r.get("answer", "")
        answerable = bool(ans) and ans.lower() not in {"", "noanswer", "unanswerable"}
        gold_ids: list[str] = []
        if chunk_id_for is not None:
            for sf in r.get("supporting_facts", []) or []:
                try:
                    title, sent_id = sf
                    cid = chunk_id_for(title, sent_id)
                    if cid:
                        gold_ids.append(cid)
                except (ValueError, TypeError):
                    continue
        out.append({
            "question": r["question"],
            "answers": [ans] if answerable else [],
            "supporting_chunk_ids": gold_ids,
            "answerable": answerable,
            "type": r.get("type", "unknown"),
        })
    return out


# ---------------------------------------------------------------------- #
# HotpotQA distractor-setting loader
# ---------------------------------------------------------------------- #
def load_hotpot_distractor(records):
    """Load HotpotQA / 2Wiki distractor records into (documents, dataset).

    Each record has the official schema fields: `question`, `answer`,
    `supporting_facts` ([[title, sent_id], ...]), and `context`
    ([[title, [sent0, sent1, ...]], ...]). This builds:

      * documents: one chunk per context sentence, id = f"{title}::s{sent_id}",
        so retrieval runs in the real *distractor* setting (gold + distractors),
        not a leaked gold-only corpus (doc section 7 corpus discipline);
      * dataset:   harness records with gold answers and the gold chunk ids
        mapped from `supporting_facts`, for recall@k / MRR.

    Works identically on the bundled sample and on the full HotpotQA dev set
    (`hotpot_dev_distractor_v1.json`).
    """
    documents: list[dict] = []
    seen_ids: set[str] = set()
    dataset: list[dict] = []

    for r in records:
        # Chunk at the PARAGRAPH level (standard HotpotQA distractor setting):
        # one coherent passage per Wikipedia intro, id = slug(title). The same
        # title across questions is the same passage, so it dedups in the pool.
        title_to_id: dict[str, str] = {}
        for title, sentences in r.get("context", []):
            cid = _slug(title)
            title_to_id[title] = cid
            if cid not in seen_ids:
                seen_ids.add(cid)
                documents.append({
                    "id": cid, "source": title,
                    "text": (f"{title}. " + " ".join(s.strip() for s in sentences)).strip(),
                })

        # Gold = the paragraphs (titles) that contain a supporting fact.
        gold_ids = []
        for sf in r.get("supporting_facts", []) or []:
            try:
                title, _sent_id = sf
            except (ValueError, TypeError):
                continue
            cid = title_to_id.get(title)
            if cid and cid not in gold_ids:
                gold_ids.append(cid)

        answer = r.get("answer", "")
        answerable = bool(answer) and answer.lower() not in {"", "noanswer", "unanswerable"}
        dataset.append({
            "question": r["question"],
            "answers": [answer] if answerable else [],
            "supporting_chunk_ids": gold_ids,
            "answerable": answerable,
            "type": r.get("type", "unknown"),
        })

    return documents, dataset


def _slug(title: str) -> str:
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "-", str(title).lower()).strip("-") or "doc"


# ---------------------------------------------------------------------- #
# Scale + unanswerable-subset construction
# ---------------------------------------------------------------------- #
def make_unanswerable_holdout(documents, dataset, *, frac=0.25, seed=0):
    """Turn a fraction of answerable items into genuine *unanswerable* ones.

    For each chosen question we remove its gold supporting chunks from the
    indexed corpus, but only chunks that are NOT gold for any *kept* answerable
    question (so we never sabotage another question's evidence). The chosen
    question then has its gold ids and answers cleared and `answerable=False`,
    giving a principled no-answer subset for abstention evaluation without a
    separate dataset.

    Returns (documents', dataset') — both new lists.
    """
    import random as _random

    rng = _random.Random(seed)
    answerable_idx = [i for i, d in enumerate(dataset) if d.get("answerable")]
    n_hold = int(round(len(answerable_idx) * frac))
    hold = set(rng.sample(answerable_idx, n_hold)) if n_hold else set()

    # Gold ids that must be preserved (used by questions we keep answerable).
    keep_gold: set[str] = set()
    for i, d in enumerate(dataset):
        if d.get("answerable") and i not in hold:
            keep_gold.update(d.get("supporting_chunk_ids", []) or [])

    remove_ids: set[str] = set()
    new_dataset = []
    for i, d in enumerate(dataset):
        if i in hold:
            for cid in d.get("supporting_chunk_ids", []) or []:
                if cid not in keep_gold:
                    remove_ids.add(cid)
            new_dataset.append({
                **d, "answers": [], "supporting_chunk_ids": [],
                "answerable": False, "type": "unanswerable",
            })
        else:
            new_dataset.append(dict(d))

    new_documents = [doc for doc in documents if doc["id"] not in remove_ids]
    return new_documents, new_dataset


def sample_dataset(documents, dataset, *, n=None, seed=0):
    """Sample up to `n` questions (and keep the full corpus). Stable by seed."""
    if not n or n >= len(dataset):
        return documents, dataset
    import random as _random
    rng = _random.Random(seed)
    idx = sorted(rng.sample(range(len(dataset)), n))
    return documents, [dataset[i] for i in idx]


# ---------------------------------------------------------------------- #
# 2WikiMultiHopQA — identical distractor schema to HotpotQA.
# ---------------------------------------------------------------------- #
def load_2wiki(records):
    """2WikiMultiHopQA uses the same fields as HotpotQA (context +
    supporting_facts), so the HotpotQA distractor loader applies directly."""
    return load_hotpot_distractor(records)


# ---------------------------------------------------------------------- #
# MuSiQue — paragraphs with is_supporting flags + native unanswerables.
# ---------------------------------------------------------------------- #
def load_musique(records):
    """Load MuSiQue (-Ans or -Full) records into (documents, dataset).

    MuSiQue is harder and less leak-prone than HotpotQA, and MuSiQue-Full
    includes genuinely unanswerable questions (answerable=False) — ideal for the
    abstention axis without any synthetic holdout. Each record carries its own
    ~20 paragraphs (not shared across questions), so chunk ids are namespaced by
    the record id.
    """
    documents = []
    seen = set()
    dataset = []
    for r in records:
        rid = str(r.get("id", "q"))
        gold = []
        for p in r.get("paragraphs", []):
            idx = p.get("idx", len(gold))
            cid = f"mq-{rid}::p{idx}"
            title = p.get("title", "")
            text = (f"{title}. " + p.get("paragraph_text", "")).strip()
            if cid not in seen:
                seen.add(cid)
                documents.append({"id": cid, "source": title, "text": text})
            if p.get("is_supporting"):
                gold.append(cid)
        answerable = bool(r.get("answerable", True)) and bool(r.get("answer"))
        answers = []
        if answerable:
            answers = [r["answer"]] + list(r.get("answer_aliases", []) or [])
        dataset.append({
            "question": r["question"],
            "answers": answers,
            "supporting_chunk_ids": gold if answerable else [],
            "answerable": answerable,
            "type": "musique",
        })
    return documents, dataset


def load_auto(records):
    """Sniff the record schema and dispatch to the right loader.

    HotpotQA/2Wiki records have a `context` field; MuSiQue records have
    `paragraphs`. Returns (documents, dataset).
    """
    sample = records[0] if records else {}
    if "paragraphs" in sample:
        return load_musique(records)
    return load_hotpot_distractor(records)


# ---------------------------------------------------------------------- #
# Practitioner docs/manual dataset (plan v2 section 3.0 / Task 6b)
# ---------------------------------------------------------------------- #
def load_docs_eval(path: str):
    """Load the bundled docs/manual eval set into (documents, dataset).

    The JSONL mixes two record types:
      {"type":"doc","id":...,"text":...,"source":...}
      {"type":"q","question":...,"answers":[...],"answerable":bool}

    Answerable questions have their answer present in some doc; unanswerable
    questions are authored out-of-corpus (answers == []). Returns the harness
    schema so the same trust eval runs on real-document Q&A.
    """
    documents: list[dict] = []
    dataset: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("type") == "doc":
                documents.append({"id": r["id"], "text": r["text"],
                                  "source": r.get("source", "")})
            elif r.get("type") == "q":
                answerable = bool(r.get("answerable", bool(r.get("answers"))))
                dataset.append({
                    "question": r["question"],
                    "answers": list(r.get("answers", []) or []),
                    "supporting_chunk_ids": [],
                    "answerable": answerable,
                    "type": "docs",
                })
    return documents, dataset
