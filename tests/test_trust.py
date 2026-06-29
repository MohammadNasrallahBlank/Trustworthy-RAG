"""Trustworthy-RAG evaluation tests (plan v2 Tasks 1 & 3).

Task 1 here: the baseline finalizer modes, the prompted-abstain generator, and
the five trust configs. Task 3 metrics live further down (added in that task).
"""

from __future__ import annotations

from srag.finalizer import Finalizer, ANSWERED, ABSTAINED
from srag.generator import GroundedGenerator
from srag.state import Chunk, RAGState


# --------------------------------------------------------------------------- #
# Task 1 — Finalizer baseline modes
# --------------------------------------------------------------------------- #
def test_always_answer_never_abstains_with_a_span():
    f = Finalizer(always_answer=True)
    s = RAGState(question="q")
    s.answer = "Paris"
    s.answerable = True
    s.confidence = 0.0  # would normally abstain
    f.finalize(s)
    assert s.answer_status == ANSWERED
    assert s.final_answer == "Paris"
    assert s.abstained is False


def test_always_answer_abstains_only_with_no_span():
    f = Finalizer(always_answer=True)
    s = RAGState(question="q")
    s.answer = ""
    s.answerable = False
    s.confidence = 0.9
    f.finalize(s)
    assert s.answer_status == ABSTAINED
    assert s.abstained is True


def test_score_only_abstains_below_threshold():
    f = Finalizer(score_only=True, tau_abstain=0.5)
    s = RAGState(question="q")
    s.answer = "Paris"
    s.answerable = True
    s.confidence = 0.2
    f.finalize(s)
    assert s.abstained is True
    assert s.answer_status == ABSTAINED


def test_score_only_answers_at_or_above_threshold():
    f = Finalizer(score_only=True, tau_abstain=0.5)
    s = RAGState(question="q")
    s.answer = "Paris"
    s.answerable = True
    s.confidence = 0.7
    f.finalize(s)
    assert s.abstained is False
    assert s.answer_status == ANSWERED
    assert s.final_answer == "Paris"


# --------------------------------------------------------------------------- #
# Task 1 — prompted-abstain generator
# --------------------------------------------------------------------------- #
def test_abstain_prompt_mode_coerces_idk_to_unanswerable():
    # A model that replies with a frozen abstention phrase -> not answerable.
    def llm(question, evidence):
        return {"answer": "I don't know", "claims": [], "answerable": True}

    g = GroundedGenerator(llm=llm, prompt_mode="abstain")
    out = g.generate("q", [Chunk(id="a", text="irrelevant")])
    assert out.answerable is False
    assert out.answer == ""


def test_abstain_prompt_mode_recognizes_controlled_phrases():
    for phrase in [
        "The context does not contain the answer.",
        "Insufficient information.",
        "Cannot answer from the provided context.",
    ]:
        def llm(question, evidence, _p=phrase):
            return {"answer": _p, "claims": [], "answerable": True}

        g = GroundedGenerator(llm=llm, prompt_mode="abstain")
        out = g.generate("q", [Chunk(id="a", text="x")])
        assert out.answerable is False, phrase


def test_abstain_prompt_mode_passes_through_a_real_answer():
    def llm(question, evidence):
        return {"answer": "Paris", "claims": [{"text": "Paris", "citations": ["a"]}],
                "answerable": True}

    g = GroundedGenerator(llm=llm, prompt_mode="abstain")
    out = g.generate("q", [Chunk(id="a", text="The capital is Paris.")])
    assert out.answerable is True
    assert out.answer == "Paris"


def test_abstain_prompt_template_mentions_idk():
    g = GroundedGenerator(prompt_mode="abstain")
    prompt = g.build_prompt("q", [Chunk(id="a", text="x")])
    assert "I don't know" in prompt


# --------------------------------------------------------------------------- #
# Task 1 — the five trust configs
# --------------------------------------------------------------------------- #
_DOCS = [
    {"id": "d0", "text": "The Eiffel Tower is located in Paris, the capital of France."},
    {"id": "d1", "text": "Mount Everest is the highest mountain on Earth."},
    {"id": "d2", "text": "The Great Barrier Reef lies off the coast of Australia."},
]


def test_trust_configs_registry():
    from srag.evaluation.configs import TRUST_CONFIGS
    assert TRUST_CONFIGS == [
        "plain_rag_always_answer",
        "prompted_abstain_rag",
        "retrieval_score_abstain",
        "guarded",
        "guarded+correct",
    ]


def test_build_each_trust_config_runs_and_sets_status():
    from srag.evaluation.configs import build_trust_config, TRUST_CONFIGS
    for name in TRUST_CONFIGS:
        runner = build_trust_config(name, _DOCS, prefer_fallback=True)
        state = runner.run("Where is the Eiffel Tower located?")
        assert state.answer_status in {"answered", "hedged", "abstained"}
        assert isinstance(state.abstained, bool)


def test_plain_always_answer_does_not_abstain_when_evidence_supports():
    from srag.evaluation.configs import build_trust_config
    runner = build_trust_config("plain_rag_always_answer", _DOCS, prefer_fallback=True)
    state = runner.run("Where is the Eiffel Tower located?")
    assert state.abstained is False
    assert state.answer_status == "answered"


# --------------------------------------------------------------------------- #
# Task 3 — categorization
# --------------------------------------------------------------------------- #
def test_categorize():
    from srag.evaluation.trust import categorize
    assert categorize(answerable=False, abstained=False, correct=False) == "uca"
    assert categorize(answerable=False, abstained=True,  correct=False) == "refusal"
    assert categorize(answerable=True,  abstained=True,  correct=False) == "missed"
    assert categorize(answerable=True,  abstained=False, correct=True)  == "correct"
    assert categorize(answerable=True,  abstained=False, correct=False) == "wrong"


# --------------------------------------------------------------------------- #
# Task 3 — trust + selective-risk metrics
# --------------------------------------------------------------------------- #
def test_trust_metrics_basic_and_selective_risk():
    from srag.evaluation.trust import trust_metrics
    # 4 answerable (2 correct, 1 wrong, 1 missed) + 2 unanswerable (1 uca, 1 refusal)
    records = [
        {"answerable": True,  "abstained": False, "correct": True},
        {"answerable": True,  "abstained": False, "correct": True},
        {"answerable": True,  "abstained": False, "correct": False},  # wrong
        {"answerable": True,  "abstained": True,  "correct": False},  # missed
        {"answerable": False, "abstained": False, "correct": False},  # uca
        {"answerable": False, "abstained": True,  "correct": False},  # refusal
    ]
    m = trust_metrics(records)
    # coverage = answered answerable / answerable = 3/4
    assert abs(m["coverage"] - 0.75) < 1e-9
    # uca_rate = uca / unanswerable = 1/2
    assert abs(m["uca_rate"] - 0.5) < 1e-9
    assert abs(m["uca_among_unanswerable"] - 0.5) < 1e-9
    assert abs(m["far"] - 0.5) < 1e-9
    # frr = missed / answerable = 1/4
    assert abs(m["frr"] - 0.25) < 1e-9
    # attempted answerable = 3 (2 correct, 1 wrong)
    assert abs(m["attempted_accuracy"] - (2 / 3)) < 1e-9
    assert abs(m["selective_risk"] - (1 / 3)) < 1e-9
    # trustworthy_rate = 1 - (wrong + uca)/N = 1 - 2/6
    assert abs(m["trustworthy_rate"] - (1 - 2 / 6)) < 1e-9


def test_trust_metrics_handles_empty_and_zero_division():
    from srag.evaluation.trust import trust_metrics
    m = trust_metrics([])
    for k in ("coverage", "uca_rate", "selective_risk", "attempted_accuracy", "frr"):
        assert m[k] == 0.0


# --------------------------------------------------------------------------- #
# Task 3 — calibration metrics
# --------------------------------------------------------------------------- #
def test_calibration_auroc_perfect_separation():
    from srag.evaluation.trust import calibration_metrics
    m = calibration_metrics(scores=[0.9, 0.8, 0.2, 0.1], labels=[1, 1, 0, 0])
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0


def test_calibration_auroc_random_is_half():
    from srag.evaluation.trust import calibration_metrics
    # symmetric score distributions across classes -> no separation (AUROC 0.5)
    m = calibration_metrics(scores=[0.6, 0.6, 0.4, 0.4], labels=[1, 0, 1, 0])
    assert abs(m["auroc"] - 0.5) < 1e-9


def test_calibration_ece_is_zero_for_perfect_confidence():
    from srag.evaluation.trust import calibration_metrics
    # confidence exactly matches empirical accuracy in each bucket
    m = calibration_metrics(scores=[0.0, 0.0, 1.0, 1.0], labels=[0, 0, 1, 1], n_bins=10)
    assert m["ece"] < 1e-9


# --------------------------------------------------------------------------- #
# Task 3 — evaluate_trust end-to-end (offline)
# --------------------------------------------------------------------------- #
def test_evaluate_trust_offline_smoke():
    from srag.evaluation.configs import build_trust_config
    from srag.evaluation.trust import evaluate_trust
    docs = [
        {"id": "d0", "text": "The Eiffel Tower is located in Paris, France."},
        {"id": "d1", "text": "Mount Everest is the highest mountain on Earth."},
    ]
    dataset = [
        {"question": "Where is the Eiffel Tower located?", "answers": ["Paris"],
         "answerable": True},
        {"question": "Who composed the opera Carmen?", "answers": ["Georges Bizet"],
         "answerable": False},  # not in corpus
    ]
    runner = build_trust_config("guarded", docs, prefer_fallback=True)
    report = evaluate_trust(runner, dataset, n_boot=200)
    assert set(report.metrics) >= {
        "coverage", "uca_rate", "trustworthy_rate", "selective_risk",
        "attempted_accuracy", "frr", "far", "auroc", "ece",
    }
    assert len(report.records) == 2
    for r in report.records:
        assert r["category"] in {"correct", "wrong", "missed", "refusal", "uca"}


# --------------------------------------------------------------------------- #
# Task 9 — regression: the prompted-abstain baseline must actually send the
# abstain instruction to the model (bug found in the first real run, where it
# silently sent the plain grounded prompt and so never abstained).
# --------------------------------------------------------------------------- #
def test_prompted_abstain_config_sends_abstain_prompt_to_the_model():
    captured = {}

    def fake_chat(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"answer":"I don\'t know","claims":[],"answerable":false}'

    from srag.evaluation.configs import build_trust_config
    runner = build_trust_config("prompted_abstain_rag", _DOCS,
                                prefer_fallback=True, chat=fake_chat)
    state = runner.run("What is the population of Mars?")
    # The model must have seen the abstain instruction...
    assert "I don't know" in captured["prompt"]
    assert "does not contain the answer" in captured["prompt"]
    # ...and its refusal must propagate to an abstention.
    assert state.abstained is True


def test_make_llm_generator_fn_grounded_prompt_has_no_abstain_rule():
    from srag.llm import make_llm_generator_fn
    seen = {}
    fn = make_llm_generator_fn(lambda p: seen.update(p=p) or '{"answer":"x"}')
    fn("q", [])
    assert "reply with exactly" not in seen["p"]  # grounded mode: no abstain rule
