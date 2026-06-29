"""Watch the loop catch a retrieval gap and fix it — step by step.

    python examples/show_self_correction.py

A bridge question ("What nationality is the director of the film Ran?") needs two
facts: who directed Ran, and that person's nationality. The first pass retrieves
film/director passages but NOT the nationality evidence — because you can't
search for "the nationality of Akira Kurosawa" until you know the director is
Akira Kurosawa. The verifier flags the gap (`planning_fault`), the controller
resolves the bridge entity, RE-RETRIEVES, finds the nationality passage it
missed, and answers. This prints exactly that: the wrong/incomplete retrieval
first, the diagnosis, the correction, the new evidence, and the fixed answer.

Runs on the deterministic offline stack (no weights), so it reproduces exactly.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srag import Controller, Planner, Stage1Pipeline, Verifier  # noqa: E402

# A corpus where the director's *biography* (with the nationality) shares no words
# with "the film Ran", so it can only be found AFTER the bridge entity is resolved.
CORPUS = [
    {"id": "ran-film", "source": "Ran (1985 film)",
     "text": "Ran is a 1985 epic samurai film directed by Akira Kurosawa, set in Sengoku-era Japan."},
    {"id": "kurosawa-bio", "source": "Akira Kurosawa",
     "text": "Akira Kurosawa was a Japanese filmmaker and painter, born in Tokyo in 1910."},
    # distractors that match "film / director / nationality" but are NOT the bio:
    {"id": "welles", "source": "Orson Welles",
     "text": "Orson Welles was an American film director famous for Citizen Kane."},
    {"id": "kagemusha", "source": "Kagemusha (1980 film)",
     "text": "Kagemusha is a 1980 jidaigeki film about a feudal lord, a samurai-era epic."},
    {"id": "fellini", "source": "Federico Fellini",
     "text": "Federico Fellini was an Italian film director known for La Dolce Vita."},
]
ID2TEXT = {d["id"]: (d["source"], d["text"]) for d in CORPUS}


def snippet(cid):
    src, text = ID2TEXT.get(cid, (cid, ""))
    t = text if len(text) <= 90 else text[:89] + "…"
    return f"[{cid}] {src}: {t}"


def chunk_root(cid):  # chunk ids look like "ran-film::c0"
    return cid.split("::")[0]


def main():
    pipe = Stage1Pipeline(planner=Planner(), retrieve_k=4, rerank_top_n=4)
    pipe.index_documents(CORPUS)
    ctrl = Controller(pipe, Verifier(), tau=0.55, max_corrections=2)
    q = "What nationality is the director of the film Ran?"

    # --- A/B: the SAME question, loop OFF vs loop ON ----------------------
    baseline = pipe.run(q)                 # single retrieve -> answer (no loop)
    state = ctrl.run(q)                    # with the self-correcting loop
    bar = "=" * 70
    print(f"\n{bar}\nQ: {q}\n{bar}")
    print(f"  WITHOUT self-correction (single pass): {baseline.answer!r}"
          "   <- wrong: that's the director, not the nationality")
    print(f"  WITH self-correction (the loop):       {state.final_answer!r}"
          "   <- correct")
    print(f"{bar}\nHow the loop got there:")
    tr = state.trace

    def evidence_at(rerank_event):
        return [chunk_root(i) for i in rerank_event.get("ids", [])]

    reranks = [e for e in tr if e["event"] == "rerank"]
    gens = [e for e in tr if e["event"] == "generate"]
    verifies = [e for e in tr if e["event"] == "verify"]
    fills = [e for e in tr if e["event"] == "fill"]
    plan = next(e for e in tr if e["event"] == "plan")

    line = "─" * 70
    print("PLAN (decomposed into hops):")
    for h in plan["sub_queries"]:
        print(f"   • {h}")

    print(f"\n{line}\n① FIRST ATTEMPT")
    seen1 = set(evidence_at(reranks[0])) if reranks else set()
    print("   retrieved (top evidence given to the model):")
    for cid in (dict.fromkeys(evidence_at(reranks[0])) if reranks else []):
        print(f"      {snippet(cid)}")
    if "kurosawa-bio" not in seen1:
        print("   ⚠  the nationality passage (Akira Kurosawa bio) was NOT retrieved.")
    print(f"   model answered: {gens[0]['answer']!r}")
    diag = verifies[0]["diagnosis"] if verifies else "?"
    fails = verifies[0].get("failing_hops", []) if verifies else []
    print(f"   ✗ verifier diagnosis: {diag}" + (f"  (uncovered hop: {fails})" if fails else ""))

    if fills:
        f = fills[0]
        print(f"\n{line}\n② CORRECTION  (fill-and-retrieve)")
        print(f"   resolved the bridge entity → \"{f['entity']}\"")
        print(f"   re-retrieved with: \"{f['query']}\"")
        newroots = [chunk_root(i) for i in f.get("new_ids", [])]
        for cid in dict.fromkeys(newroots):
            print(f"   ✚ NEW evidence found: {snippet(cid)}")

    if len(reranks) > 1:
        print(f"\n{line}\n③ SECOND ATTEMPT (after correction)")
        print(f"   model now answers: {state.final_answer!r}")
        last_diag = verifies[-1]["diagnosis"] if verifies else "?"
        print(f"   ✓ verifier diagnosis: {last_diag}")

    print(f"\n{line}\nFINAL: {ctrl.finalizer.message(state)}")
    print(f"(status={state.answer_status}, corrections={state.correction_count})\n")


if __name__ == "__main__":
    main()
