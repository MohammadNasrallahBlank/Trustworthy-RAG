"""Real open-model LLM adapters for the generator and planner.

The embedder and cross-encoder reranker already auto-upgrade to
`sentence-transformers` when its weights are reachable (pass
`prefer_fallback=False`). This module supplies the missing concrete pieces: a
local instruction-tuned LLM, wrapped to return the strict JSON the generator and
planner contracts expect.

  * `TransformersChat` — a thin wrapper over a Hugging Face text-generation
    pipeline (`transformers`), lazily constructed so importing this module never
    requires the dependency or any weights.
  * `make_llm_generator_fn(chat)` — adapts a chat callable into the generator's
    `llm(question, evidence) -> dict` contract.
  * `make_llm_planner_fn(chat)`  — adapts it into the planner's
    `llm(question) -> {type, sub_queries}` contract.

Both adapters accept ANY `chat(prompt: str) -> str` callable, so they are tested
offline against a fake chat (no network, no weights). Swap `TransformersChat` in
on a machine where the model is available.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Sequence

from .generator import SYSTEM_PROMPT, GroundedGenerator
from .planner import Planner
from .state import Chunk

ChatFn = Callable[[str], str]


# ---------------------------------------------------------------------- #
# Local transformers chat wrapper (lazy)
# ---------------------------------------------------------------------- #
class TransformersChat:
    """Local Hugging Face causal-LM chat wrapper (GPU, bf16, greedy by default).

    Loads the tokenizer + model once (lazily), applies the model's chat template,
    and returns the assistant's reply. Designed for a single 4090-class GPU; no
    `accelerate` required (weights are moved with `.to(device)`).

        chat = TransformersChat("Qwen/Qwen2.5-3B-Instruct")
        gen = GroundedGenerator(llm=make_llm_generator_fn(chat))
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", *,
                 max_new_tokens: int = 320, temperature: float = 0.0,
                 device: str = "cuda", dtype: str = "bfloat16"):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.dtype = dtype
        self._tok = None
        self._model = None

    def _ensure(self):  # pragma: no cover - requires transformers + weights + GPU
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                  "float32": torch.float32}.get(self.dtype, torch.bfloat16)
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dt)
            self._model.to(self.device)
            self._model.eval()
        return self._tok, self._model

    def __call__(self, prompt: str) -> str:  # pragma: no cover - needs weights
        import torch

        tok, model = self._ensure()
        messages = [{"role": "user", "content": prompt}]
        if getattr(tok, "chat_template", None):
            text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
        else:
            text = prompt
        inputs = tok(text, return_tensors="pt").to(model.device)
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens,
                          do_sample=self.temperature > 0,
                          pad_token_id=tok.eos_token_id)
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------- #
# Generator adapter
# ---------------------------------------------------------------------- #
def make_llm_generator_fn(chat: ChatFn, prompt_mode: str = "grounded") -> Callable[[str, Sequence[Chunk]], dict]:
    """Adapt a chat callable into the generator's llm(question, evidence)->dict.

    `prompt_mode` selects which system prompt is sent to the model: "grounded"
    (default) or "abstain" (the prompted-abstain baseline, which appends the
    "say I don't know if unsupported" instruction). This MUST match the wrapping
    GroundedGenerator's prompt_mode, or the model never sees the abstain rule.
    """
    helper = GroundedGenerator(prompt_mode=prompt_mode)  # reuse its prompt builder

    def llm(question: str, evidence: Sequence[Chunk]) -> dict:
        prompt = helper.build_prompt(question, evidence)
        raw = chat(prompt)
        return _loads_json(raw)

    return llm


# ---------------------------------------------------------------------- #
# Planner adapter
# ---------------------------------------------------------------------- #
_PLANNER_PROMPT = """Decompose the QUESTION into complementary, self-contained
sub-queries for multi-hop retrieval. Return ONLY JSON:
{"type": "bridge|comparison|yes-no|single-hop",
 "sub_queries": [{"text": "<sub-question, name entities explicitly>",
                  "depends_on": "<id of a prior hop or null>",
                  "template": "<e.g. 'What nationality is {entity}?' or null>"}]}
Rules: name entities explicitly (no pronouns); sub-queries must be complementary,
not paraphrases; for a bridge, the second hop may be templated on the first hop's
answer with depends_on set. Do NOT answer the question.

QUESTION: %s

JSON:"""


def make_llm_planner_fn(chat: ChatFn) -> Callable[[str], dict]:
    """Adapt a chat callable into the planner's llm(question)->dict."""

    def llm(question: str) -> dict:
        raw = chat(_PLANNER_PROMPT % question)
        data = _loads_json(raw)
        # Assign ids if the model omitted them.
        subs = data.get("sub_queries") or []
        for i, item in enumerate(subs):
            if isinstance(item, dict) and "id" not in item:
                item["id"] = f"hop_{i}"
                item.setdefault("hop_index", i)
                item.setdefault("resolved", item.get("template") in (None, ""))
        return data

    return llm


# ---------------------------------------------------------------------- #
def _loads_json(s: str) -> dict:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]
    try:
        return json.loads(s)
    except Exception:
        return {}


# Re-export for convenience / discoverability.
__all__ = [
    "TransformersChat",
    "make_llm_generator_fn",
    "make_llm_planner_fn",
    "SYSTEM_PROMPT",
]
