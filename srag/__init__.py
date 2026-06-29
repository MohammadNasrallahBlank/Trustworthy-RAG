"""Self-Correcting RAG package.

Stage 1 (strong static pipeline): chunking -> hybrid retrieval (BM25 + dense,
RRF) -> cross-encoder reranker -> grounded generation with claim-level
citations and an extract-first schema.

Stage 2 (verifier): standalone Checks 1-5 + an aggregator producing a
confidence scalar and a structured diagnosis.

Stage 3 (planner): multi-hop query decomposition feeding the per-hop coverage
check.

Stage 4 (controller): diagnosis -> targeted correction with a bounded control
loop (confidence / budget / no-new-evidence stops). Abstention + calibration
(stage 5) and the eval harness (stage 6) are still to come.
"""

from .state import Chunk, SubQuery, Claim, GenerationResult, RAGState
from .chunking import chunk_document, chunk_corpus
from .embeddings import Embedder
from .retrieval import HybridRetriever
from .reranker import CrossEncoderReranker
from .generator import GroundedGenerator
from .planner import Planner, detect_question_type
from .pipeline import Stage1Pipeline
from .entailment import EntailmentModel
from .verifier import Verifier, VerificationResult, CheckResult
from .controller import Controller, Budget
from .api import SelfCorrectingRAG, Answer
from .gate import RetrievalGate, GateDecision
from .llm import TransformersChat, make_llm_generator_fn, make_llm_planner_fn
from .adapters import OpenAIChat, OllamaChat, VLLMChat
from .finalizer import Finalizer, ANSWERED, HEDGED, ABSTAINED
from .calibration import calibrate_thresholds, collect_points, CalibrationResult

__all__ = [
    "Chunk",
    "SubQuery",
    "Claim",
    "GenerationResult",
    "RAGState",
    "chunk_document",
    "chunk_corpus",
    "Embedder",
    "HybridRetriever",
    "CrossEncoderReranker",
    "GroundedGenerator",
    "Planner",
    "detect_question_type",
    "Stage1Pipeline",
    "EntailmentModel",
    "Verifier",
    "VerificationResult",
    "CheckResult",
    "Controller",
    "Budget",
    "SelfCorrectingRAG",
    "Answer",
    "RetrievalGate",
    "GateDecision",
    "TransformersChat",
    "OpenAIChat",
    "OllamaChat",
    "VLLMChat",
    "make_llm_generator_fn",
    "make_llm_planner_fn",
    "Finalizer",
    "ANSWERED",
    "HEDGED",
    "ABSTAINED",
    "calibrate_thresholds",
    "collect_points",
    "CalibrationResult",
]

__version__ = "1.0.0"
