"""Command-line interface for Self-Correcting RAG (plan v2 Task 10).

    srag index   mydocs.jsonl --out index.pkl       # build a reusable index
    srag ask     "your question" --index index.pkl  # ask against a saved index
    srag ask     "your question" --docs mydocs.jsonl # or build on the fly
    srag evaluate --dataset docs                     # run the trust benchmark

Documents come from a `.jsonl` file (one {"text","id"?,"source"?} per line) or a
`.txt` file (one document per line). By default everything runs on the offline,
no-weights components so the CLI works instantly; bring a real model with
--openai-model / --ollama-model / --hf-model.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

from .api import SelfCorrectingRAG


# --------------------------------------------------------------------------- #
# Corpus loading + LLM selection
# --------------------------------------------------------------------------- #
def load_corpus_file(path: str) -> list:
    if path.endswith(".jsonl"):
        docs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        return docs
    # plain text: one document per non-empty line
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_llm(args):
    if getattr(args, "openai_model", None):
        from .adapters import OpenAIChat
        return OpenAIChat(model=args.openai_model)
    if getattr(args, "ollama_model", None):
        from .adapters import OllamaChat
        return OllamaChat(model=args.ollama_model)
    if getattr(args, "hf_model", None):
        from .adapters import TransformersChat
        return TransformersChat(args.hf_model)
    return None


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_index(args) -> int:
    documents = load_corpus_file(args.corpus)
    rag = SelfCorrectingRAG.from_documents(documents, mode=args.mode)
    with open(args.out, "wb") as f:
        pickle.dump(rag, f)
    print(f"Indexed {len(documents)} documents -> {args.out}")
    return 0


def cmd_ask(args) -> int:
    if args.index:
        with open(args.index, "rb") as f:
            rag = pickle.load(f)
    elif args.docs:
        documents = load_corpus_file(args.docs)
        llm = build_llm(args)
        rag = SelfCorrectingRAG.from_documents(
            documents, llm=llm, real_models=bool(args.real_models), mode=args.mode)
    else:
        print("error: provide --index <file> or --docs <corpus>", file=sys.stderr)
        return 2

    answer = rag.ask(args.question)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
    else:
        print(f"status:   {answer.status}")
        print(f"answer:   {answer.message}")
        if answer.citations:
            print(f"citations: {', '.join(answer.citations)}")
        print(f"corrections: {answer.corrections}")
    return 0


def cmd_evaluate(args) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "examples"))
    from run_trust import run_trust
    run_trust(args.dataset, backend=args.backend, n=args.n,
              out_dir=args.out, seed=args.seed)
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="srag",
                                description="Self-Correcting / Trustworthy RAG CLI")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="build a reusable index from a corpus")
    pi.add_argument("corpus", help=".jsonl or .txt corpus file")
    pi.add_argument("--out", default="srag_index.pkl")
    pi.add_argument("--mode", choices=["trustworthy", "self_correcting"],
                    default="trustworthy")
    pi.set_defaults(func=cmd_index)

    pa = sub.add_parser("ask", help="ask a question")
    pa.add_argument("question")
    pa.add_argument("--index", help="a saved index .pkl")
    pa.add_argument("--docs", help="a corpus file to index on the fly")
    pa.add_argument("--mode", choices=["trustworthy", "self_correcting"],
                    default="trustworthy")
    pa.add_argument("--real-models", action="store_true",
                    help="use real sentence-transformers + NLI")
    pa.add_argument("--openai-model", help="answer with an OpenAI model (OPENAI_API_KEY)")
    pa.add_argument("--ollama-model", help="answer with a local Ollama model")
    pa.add_argument("--hf-model", help="answer with a local Hugging Face model")
    pa.add_argument("--json", action="store_true", help="print JSON")
    pa.set_defaults(func=cmd_ask)

    pe = sub.add_parser("evaluate", help="run the trust benchmark")
    pe.add_argument("--dataset", choices=["docs", "hotpotqa"], default="docs")
    pe.add_argument("--backend", choices=["offline", "real"], default="offline")
    pe.add_argument("--n", type=int, default=None)
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--out", default=None)
    pe.set_defaults(func=cmd_evaluate)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
