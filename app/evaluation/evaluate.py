"""Offline evaluation harness wired to Langfuse datasets + scores.

Runs the agent over a test set, computes RAG metrics, and:
  - prints an aggregate report
  - attaches per-trace scores in Langfuse (so eval metrics show on the trace)
  - optionally creates/links a Langfuse dataset run for experiment tracking
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from app.agents.graph import run_agent
from app.core.logging_config import get_logger
from app.core.schemas import EvalResult, EvalSample
from app.evaluation.metrics import evaluate_sample
from app.observability.langfuse_client import flush, get_langfuse, score_trace

logger = get_logger(__name__)


def _retrieve_contexts(query: str) -> list[str]:
    from app.rag.retriever import hybrid_retrieve

    return [c.content for c in hybrid_retrieve(query)]


def run_evaluation(dataset: list[dict]) -> dict[str, float]:
    """dataset: list of {"question": str, "ground_truth": str (optional)}."""
    per_metric: dict[str, list[float]] = defaultdict(list)

    for row in dataset:
        question = row["question"]
        ground_truth = row.get("ground_truth")

        response = run_agent(question, session_id="eval")
        contexts = _retrieve_contexts(question)

        sample = EvalSample(
            question=question,
            ground_truth=ground_truth,
            contexts=contexts,
            answer=response.answer,
        )
        results: list[EvalResult] = evaluate_sample(sample)

        for r in results:
            per_metric[r.metric].append(r.score)
            if response.trace_id:
                score_trace(response.trace_id, r.metric, r.score, r.detail.get("reason"))

        logger.info(
            "Q: %s | %s",
            question[:60],
            ", ".join(f"{r.metric}={r.score:.2f}" for r in results),
        )

    aggregate = {m: round(statistics.mean(v), 3) for m, v in per_metric.items() if v}
    logger.info("Aggregate: %s", aggregate)
    flush()
    return aggregate


def push_dataset_to_langfuse(name: str, dataset: list[dict]) -> None:
    """Create a Langfuse dataset for repeatable experiment runs."""
    client = get_langfuse()
    if client is None:
        logger.warning("Langfuse not configured; skipping dataset push.")
        return
    client.create_dataset(name=name)
    for row in dataset:
        client.create_dataset_item(
            dataset_name=name,
            input={"question": row["question"]},
            expected_output=row.get("ground_truth"),
        )
    logger.info("Pushed %d items to Langfuse dataset '%s'.", len(dataset), name)
