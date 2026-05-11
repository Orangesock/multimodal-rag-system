"""Document reranking using cross-encoder models."""

from multimodal_rag.resources import get_resources


def prepare_pairs(query: str, documents: list) -> list:
    """Prepare query-document pairs for cross-encoder scoring.

    Args:
        query: Search query
        documents: List of documents

    Returns:
        List of (query, text) pairs
    """
    return [(query, doc["text"]) for doc in documents]


def rerank(query: str, documents: list, top_k: int = 5) -> list:
    """Rerank documents using cross-encoder similarity scores.

    Args:
        query: Search query
        documents: List of retrieved documents
        top_k: Number of top results to return

    Returns:
        List of reranked documents sorted by relevance score
    """
    if not documents:
        return []

    pairs = prepare_pairs(query, documents)

    resources = get_resources()
    reranker = resources.get_reranker_model()
    scores = reranker.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    ranked = sorted(
        documents,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked[:top_k]