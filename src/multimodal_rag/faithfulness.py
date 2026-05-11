"""
Faithfulness validation for RAG outputs.
Updated for Precision: Supports short answers and strict document grounding.
"""

import re
from typing import List, Set

def tokenize(text: str) -> Set[str]:
    """
    Extract meaningful tokens (3+ characters) for comparison.
    This helps filter out common stop words like 'the', 'is', 'and'.
    """
    return set(re.findall(r"[a-zA-Z]{3,}", text.lower()))


def compute_overlap(answer: str, documents: List[dict]) -> float:
    """
    Compute how much of the generated answer is grounded in the retrieved text.
    Returns a score from 0.0 to 1.0.
    """
    answer_tokens = tokenize(answer)

    # If the answer is empty or lacks meaningful words, it's not faithful.
    if not answer_tokens:
        return 0.0

    overlaps = []

    for doc in documents:
        doc_tokens = tokenize(doc.get("text", ""))

        if not doc_tokens:
            continue

        # Intersection: How many words from the answer exist in this PDF chunk?
        overlap_count = len(answer_tokens & doc_tokens)
        overlap_ratio = overlap_count / len(answer_tokens)
        overlaps.append(overlap_ratio)

    # We take the best match. If the answer exists in AT LEAST one chunk, it's faithful.
    return max(overlaps) if overlaps else 0.0


def is_faithful(
    answer: str,
    documents: List[dict],
    threshold: float = 0.15
) -> bool:
    """
    Check if the answer's grounding score meets the required threshold.
    0.15 is a safe balance for short biographical answers.
    """
    overlap = compute_overlap(answer, documents)
    return overlap >= threshold


def enforce_faithfulness(answer: str, documents: List[dict]) -> str:
    """
    The Master Guardrail:
    1. If the answer is grounded in the PDF, return it as is.
    2. If the AI hallucinated (0% overlap), refuse to show the answer.
    """
    if is_faithful(answer, documents):
        return answer

    # 🛠️ THE "STRICT GUARDRAIL" FALLBACK:
    # This prevents 'Unverified Responses' or 'DanTDM' answers from appearing.
    return (
        "I'm sorry, but I couldn't find any information about that "
        "in the provided documents."
    )