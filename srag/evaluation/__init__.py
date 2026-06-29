"""Stage 6 evaluation harness (doc section 7).

Honest evaluation is part of the architecture: correctness, faithfulness,
retrieval quality, abstention, and cost are reported *separately*, and the
headline result is an accuracy-vs-cost curve, not a single number. The full
loop is held to the standard of beating the reranked baseline within CIs.
"""

from .metrics import (
    normalize_answer,
    exact_match,
    token_f1,
    retrieval_recall_at_k,
    reciprocal_rank,
    bootstrap_ci,
    paired_bootstrap_pvalue,
)
from .datasets import (
    load_eval_set, load_hotpot_style, load_hotpot_distractor,
    make_unanswerable_holdout, sample_dataset,
    load_2wiki, load_musique, load_auto,
)
from .configs import CONFIGS, build_config
from .harness import evaluate_config, QuestionRecord, ConfigReport
from .report import compare_configs, render_markdown_report
from .figures import accuracy_cost_curve, diagnosis_routing_diagram

__all__ = [
    "normalize_answer",
    "exact_match",
    "token_f1",
    "retrieval_recall_at_k",
    "reciprocal_rank",
    "bootstrap_ci",
    "paired_bootstrap_pvalue",
    "load_eval_set",
    "load_hotpot_style",
    "load_hotpot_distractor",
    "make_unanswerable_holdout",
    "sample_dataset",
    "load_2wiki",
    "load_musique",
    "load_auto",
    "CONFIGS",
    "build_config",
    "evaluate_config",
    "QuestionRecord",
    "ConfigReport",
    "compare_configs",
    "render_markdown_report",
    "accuracy_cost_curve",
    "diagnosis_routing_diagram",
]
