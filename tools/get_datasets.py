"""Fetch multi-hop QA datasets into data/raw/ for the benchmark matrix.

Runs on your machine (real internet). HotpotQA is fetched reliably via the
`datasets` library. 2WikiMultiHopQA and MuSiQue are attempted from a few known
Hugging Face mirrors; whatever can't be fetched is reported with a note so you
can drop the official file into data/raw/ manually.
"""
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(OUT, exist_ok=True)


def _save(name, recs):
    path = os.path.join(OUT, name)
    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    else:
        json.dump(recs, open(path, "w", encoding="utf-8"))
    print(f"  saved {len(recs)} -> data/raw/{name}", flush=True)


def get_hotpotqa(limit=None):
    print("HotpotQA (distractor, validation) via `datasets` ...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    recs = []
    for r in ds:
        ctx = list(zip(r["context"]["title"], r["context"]["sentences"]))
        sf = list(zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]))
        recs.append({"_id": r["id"], "question": r["question"], "answer": r["answer"],
                     "type": r.get("type", "unknown"),
                     "supporting_facts": [[t, int(s)] for t, s in sf],
                     "context": [[t, list(s)] for t, s in ctx]})
        if limit and len(recs) >= limit:
            break
    _save("hotpotqa_dev.json", recs)


def _try_hf(candidates, split="validation"):
    from datasets import load_dataset
    for cand in candidates:
        try:
            name, config = (cand if isinstance(cand, tuple) else (cand, None))
            ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
            print(f"  loaded {cand} ({len(ds)} rows)", flush=True)
            return ds
        except Exception as e:  # noqa: BLE001
            print(f"  [miss] {cand}: {str(e)[:120]}", flush=True)
    return None


def get_2wiki(limit=None):
    print("2WikiMultiHopQA — attempting HF mirrors ...", flush=True)
    ds = _try_hf(["voidful/2WikiMultihopQA", "scholarly-shapes/2wiki",
                  ("xanhho/2WikiMultihopQA", None)])
    if ds is None:
        print("  -> not found. Put the official dev.json (HotpotQA schema) at "
              "data/raw/2wiki_dev.json", flush=True)
        return
    recs = []
    for r in ds:
        ctx = r.get("context")
        sf = r.get("supporting_facts")
        # Normalize the two common HF shapes (dict-of-lists vs list-of-pairs).
        if isinstance(ctx, dict):
            ctx = list(zip(ctx["title"], ctx["sentences"]))
        if isinstance(sf, dict):
            sf = list(zip(sf["title"], sf["sent_id"]))
        recs.append({"_id": r.get("_id", r.get("id", len(recs))),
                     "question": r["question"], "answer": r["answer"],
                     "type": r.get("type", "unknown"),
                     "supporting_facts": [[t, int(s)] for t, s in sf],
                     "context": [[t, list(s)] for t, s in ctx]})
        if limit and len(recs) >= limit:
            break
    _save("2wiki_dev.json", recs)


def get_musique(limit=None):
    print("MuSiQue — attempting HF mirrors ...", flush=True)
    ds = _try_hf(["dgslibisey/MuSiQue", "allenai/musique", "musique"])
    if ds is None:
        print("  -> not found. Put musique_ans_v1.0_dev.jsonl at "
              "data/raw/musique_dev.jsonl (from github.com/StonyBrookNLP/musique)",
              flush=True)
        return
    recs = []
    for r in ds:
        recs.append(dict(r))
        if limit and len(recs) >= limit:
            break
    _save("musique_dev.jsonl", recs)


if __name__ == "__main__":
    which = sys.argv[1:] or ["hotpotqa", "2wiki", "musique"]
    if "hotpotqa" in which:
        try:
            get_hotpotqa()
        except Exception as e:  # noqa: BLE001
            print("HotpotQA FAILED:", e, flush=True)
    if "2wiki" in which:
        try:
            get_2wiki()
        except Exception as e:  # noqa: BLE001
            print("2Wiki FAILED:", e, flush=True)
    if "musique" in which:
        try:
            get_musique()
        except Exception as e:  # noqa: BLE001
            print("MuSiQue FAILED:", e, flush=True)
    print("done.", flush=True)
