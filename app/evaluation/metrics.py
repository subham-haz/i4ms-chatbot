"""RAG evaluation metrics (LLM-as-judge).

Implements the core RAGAS-style metrics from first principles so the logic is
transparent and interview-defensible:

  - faithfulness:        is the answer grounded in retrieved context?
  - answer_relevancy:    does the answer address the question?
  - context_precision:   are the retrieved contexts actually relevant?
  - context_recall:      does the context cover the ground-truth answer?

Each metric returns a 0..1 score. Designed to also run via the official `ragas`
package when installed (see evaluate.py), but works standalone.
"""
from __future__ import annotations

import json
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.schemas import EvalResult, EvalSample


@lru_cache
def _judge() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model, temperature=0.0, api_key=settings.openai_api_key
    )


def _ask_score(prompt: str) -> tuple[float, str]:
    """Ask the judge for a JSON {score, reason}; robust to formatting noise."""
    raw = _judge().invoke(prompt).content
    text = str(raw).strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        data = json.loads(text)
        return float(data.get("score", 0.0)), str(data.get("reason", ""))
    except Exception:
        return 0.0, f"unparseable judge output: {text[:120]}"


def faithfulness(sample: EvalSample) -> EvalResult:
    context = "\n".join(sample.contexts)
    prompt = (
        "You are evaluating FAITHFULNESS: whether every claim in the ANSWER is "
        "supported by the CONTEXT. Return JSON {\"score\": 0..1, \"reason\": str}. "
        "1.0 = fully grounded, 0.0 = hallucinated.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{sample.answer}"
    )
    score, reason = _ask_score(prompt)
    return EvalResult(metric="faithfulness", score=score, detail={"reason": reason})


def answer_relevancy(sample: EvalSample) -> EvalResult:
    prompt = (
        "You are evaluating ANSWER RELEVANCY: whether the ANSWER directly addresses "
        "the QUESTION. Return JSON {\"score\": 0..1, \"reason\": str}.\n\n"
        f"QUESTION:\n{sample.question}\n\nANSWER:\n{sample.answer}"
    )
    score, reason = _ask_score(prompt)
    return EvalResult(metric="answer_relevancy", score=score, detail={"reason": reason})


def context_precision(sample: EvalSample) -> EvalResult:
    context = "\n---\n".join(sample.contexts)
    prompt = (
        "You are evaluating CONTEXT PRECISION: what fraction of the retrieved "
        "CONTEXT chunks are relevant to answering the QUESTION. Return JSON "
        "{\"score\": 0..1, \"reason\": str}.\n\n"
        f"QUESTION:\n{sample.question}\n\nCONTEXT CHUNKS:\n{context}"
    )
    score, reason = _ask_score(prompt)
    return EvalResult(metric="context_precision", score=score, detail={"reason": reason})


def context_recall(sample: EvalSample) -> EvalResult:
    if not sample.ground_truth:
        return EvalResult(
            metric="context_recall", score=0.0, detail={"reason": "no ground_truth"}
        )
    context = "\n".join(sample.contexts)
    prompt = (
        "You are evaluating CONTEXT RECALL: whether the CONTEXT contains the "
        "information present in the GROUND TRUTH answer. Return JSON "
        "{\"score\": 0..1, \"reason\": str}.\n\n"
        f"GROUND TRUTH:\n{sample.ground_truth}\n\nCONTEXT:\n{context}"
    )
    score, reason = _ask_score(prompt)
    return EvalResult(metric="context_recall", score=score, detail={"reason": reason})


ALL_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def evaluate_sample(sample: EvalSample) -> list[EvalResult]:
    return [metric(sample) for metric in ALL_METRICS]
