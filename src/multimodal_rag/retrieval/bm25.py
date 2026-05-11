"""BM25 keyword-based retrieval."""

from typing import List, Tuple
import pickle
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    if not text:
        return []
    return text.lower().split()


def load_corpus(metadata_path: str) -> Tuple[List[str], List[dict]]:
    """Load corpus texts and metadata."""
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    texts = [m["text"] for m in metadata]
    return texts, metadata


def build_bm25(corpus_texts: List[str]):
    """Build BM25 index."""
    if not corpus_texts:
        raise ValueError("Cannot build BM25 from empty corpus")

    tokenized_corpus = [tokenize(doc) for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, tokenized_corpus


def search_bm25(
    query: str,
    bm25,
    corpus_texts: List[str],
    metadata: List[dict],
    top_k: int = 5
) -> List[dict]:
    """Search corpus using BM25 scoring."""
    if not query or not query.strip():
        return []

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []
    for idx in ranked_indices[:top_k]:
        meta = metadata[idx]

        results.append({
            "score": float(scores[idx]),
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "page": meta.get("page"),
            "text": meta["text"],
        })

    return results