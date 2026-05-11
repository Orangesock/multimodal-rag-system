"""Query expansion for multi-query retrieval."""

import re
import logging
from typing import List

from multimodal_rag.resources import get_resources

logger = logging.getLogger(__name__)


def _clean_lines(text: str) -> List[str]:
    """Clean seq2seq model output into a list of usable queries."""
    # Remove markdown lists or numbers (e.g., "1.", "-", "*") at the start of lines
    text = re.sub(r"(?m)^\s*(?:[-*]|\d+\.)\s*", "", text)
    
    # T5 might return comma-separated or newline-separated strings
    items = re.split(r"[,;\n]+", text)

    cleaned = []
    for item in items:
        # Strip whitespace and trailing/leading quotes
        item = item.strip().strip('"\'')
        if item and len(item) > 3:
            cleaned.append(item)

    return cleaned


def generate_query_variations(query: str, n: int = 3) -> List[str]:
    """Generate multiple variations of a query using a seq2seq LLM."""
    if not query.strip():
        return [query]

    resources = get_resources()
    generator = resources.get_generator_model()

    # Flan-T5 responds much better to direct, single-sentence seq2seq commands 
    # rather than complex conversational personas.
    prompt = f"Write {n} alternative versions of this search query, separated by commas: '{query}'"

    try:
        # Enable sampling to ensure we get diverse wording
        output = generator(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1
        )[0]["generated_text"]

        variations = _clean_lines(output)

    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        variations = []

    # Deduplicate (case-insensitive) and ensure the original is included first
    seen = {query.lower()}
    final = [query]

    for q in variations:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            final.append(q)

    return final[: n + 1]