"""Interactive demo — watch self-correcting RAG decide, diagnose, and correct.

    pip install -e ".[demo]"
    python app.py                 # opens a local Gradio app

Type a question and watch the live decision trace: plan -> retrieve -> rerank ->
generate -> verify (with the *diagnosis*) -> targeted correction(s) -> answer or
calibrated abstention, with evidence and citations. Runs on the deterministic
offline stack by default (no weights, Spaces-friendly); tick "real models" on a
GPU box to use sentence-transformers + a local LLM.

Designed to drop straight onto Hugging Face Spaces (this file is `app.py`).
"""

from __future__ import annotations

import json
import os

import gradio as gr

from srag import (
    Controller, CrossEncoderReranker, Embedder, EntailmentModel,
    GroundedGenerator, HybridRetriever, Planner, Stage1Pipeline, Verifier,
    calibrate_thresholds, collect_points,
)
from srag.trace_view import render_trace_html

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_jsonl(name):
    with open(os.path.join(HERE, "data", name), "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


SAMPLE_DOCS = _load_jsonl("sample_corpus.jsonl")
DEV = _load_jsonl("dev_set.jsonl")

EXAMPLES = [
    "What nationality is the director of the film Ran?",
    "Who directed Seven Samurai?",
    "In what year was the Kintai Bridge built?",
    "Who won the 2050 World Cup?",
    "What is the capital of France?",
]

_CACHE = {}


def _build_controller(real: bool):
    key = ("real" if real else "offline")
    if key in _CACHE:
        return _CACHE[key]
    prefer_fallback = not real
    chat = None
    generator = GroundedGenerator()
    planner = Planner()
    if real:
        from srag import TransformersChat, make_llm_generator_fn, make_llm_planner_fn
        chat = TransformersChat(os.environ.get("DEMO_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
        generator = GroundedGenerator(llm=make_llm_generator_fn(chat))
        planner = Planner(llm=make_llm_planner_fn(chat))
    pipe = Stage1Pipeline(
        retriever=HybridRetriever(Embedder(prefer_fallback=prefer_fallback)),
        reranker=CrossEncoderReranker(prefer_fallback=prefer_fallback),
        generator=generator, planner=planner,
    ).index_documents(SAMPLE_DOCS)
    ver = Verifier(EntailmentModel(prefer_fallback=prefer_fallback))
    base = Controller(pipe, ver, tau=0.55, max_corrections=3)
    cal = calibrate_thresholds(collect_points(base, DEV))
    ctrl = Controller(pipe, ver, tau=cal.tau_answer, tau_abstain=cal.tau_abstain,
                      max_corrections=3)
    _CACHE[key] = ctrl
    return ctrl


def run(question, real):
    if not question or not question.strip():
        return "<i>Enter a question.</i>"
    try:
        ctrl = _build_controller(bool(real))
        state = ctrl.run(question.strip())
        return render_trace_html(state)
    except Exception as exc:  # noqa: BLE001
        return f"<pre style='color:#dc2626'>Error: {exc}</pre>"


with gr.Blocks(title="Self-Correcting RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Self-Correcting RAG — live demo\n"
        "Type a question over a small built-in corpus (Kurosawa films, the Kintai "
        "Bridge, an IR primer) and watch the system **plan → retrieve → rerank → "
        "generate → verify (diagnose) → correct → answer or abstain**. The corpus is "
        "deliberately small so you can see the unanswerable questions get refused.")
    with gr.Row():
        q = gr.Textbox(label="Question", scale=4,
                       placeholder="e.g. What nationality is the director of the film Ran?")
        real = gr.Checkbox(label="Real models (GPU)", value=False, scale=1)
    gr.Examples(EXAMPLES, inputs=q)
    btn = gr.Button("Run", variant="primary")
    out = gr.HTML()
    btn.click(run, inputs=[q, real], outputs=out)
    q.submit(run, inputs=[q, real], outputs=out)


if __name__ == "__main__":
    demo.launch()
