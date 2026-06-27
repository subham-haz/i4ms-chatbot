"""Run the offline evaluation suite over i4MS questions (policy + data)."""
from __future__ import annotations

import json

from app.core.logging_config import configure_logging, get_logger
from app.evaluation.evaluate import push_dataset_to_langfuse, run_evaluation

logger = get_logger(__name__)

# Mix of policy (RAG) and data (DB) questions to exercise both agent paths.
SAMPLE_DATASET = [
    {
        "question": "How long before a quarry lease expires must a renewal application be filed?",
        "ground_truth": "At least 90 days before the lease expires.",
    },
    {
        "question": "Can an e-transit pass be generated against an exhausted e-permit?",
        "ground_truth": "No. A pass can only be generated against a valid, non-exhausted permit.",
    },
    {
        "question": "How many days before expiry must a crusher unit apply to renew its registration?",
        "ground_truth": "At least 30 days before the registration expires.",
    },
    {
        "question": "What happens to permits and passes when a lease is suspended?",
        "ground_truth": "Fresh permits and passes cannot be issued against a suspended lease.",
    },
]


def main() -> None:
    configure_logging()
    push_dataset_to_langfuse("i4ms-policy-eval", SAMPLE_DATASET)
    aggregate = run_evaluation(SAMPLE_DATASET)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
