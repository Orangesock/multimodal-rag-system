"""Hybrid retrieval with multi-query expansion."""
from collections import defaultdict

from multimodal_rag.config import settings
from multimodal_rag.resources import get_resources
from multimodal_rag.retrieval.vector import retrieve as vector_retrieve
from multimodal_rag.retrieval.bm25 import search_bm25
from multimodal_rag.retrieval.query_expansion import generate_query_variations

RRF_K = 60


def reciprocal_rank_fusion(results_list: list[list]) -> list:
    """Fuse multiple ranked lists using RRF."""
    scores = defaultdict(float)

    for results in results_list:
        for rank, doc in enumerate(results):
            key = (doc["source"], doc["chunk_index"])
            scores[key] += 1 / (RRF_K + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {"source": source, "chunk_index": chunk_index, "score": score}
        for (source, chunk_index), score in ranked
    ]


def hybrid_search(query: str, top_k: int = 5) -> list:
    """Hybrid retrieval using multi-query expansion."""
    resources = get_resources()

    # Cached resources
    index, metadata, bm25, corpus_texts = resources.get_retrieval_resources()

    metadata_map = {
        (m["source"], m["chunk_index"]): m for m in metadata
    }

    # 🔥 STEP 3: Generate multiple queries
    queries = generate_query_variations(query, n=3)

    all_results = []

    for q in queries:
        vector_results = vector_retrieve(q, index, metadata, top_k=top_k)

        bm25_results = search_bm25(
            q, bm25, corpus_texts, metadata, top_k=top_k
        )

        combined = vector_results + bm25_results
        all_results.append(combined)

    # Fuse all results
    fused = reciprocal_rank_fusion(all_results)

    # Enrich
    enriched = []
    for item in fused:
        key = (item["source"], item["chunk_index"])
        meta = metadata_map.get(key, {})

        enriched.append({
            **item,
            "text": meta.get("text", ""),
            "page": meta.get("page")
        })

    return enriched[:top_k]