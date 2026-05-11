"""Vector-based semantic search using FAISS (IVF + nprobe tuned)."""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from multimodal_rag.config import settings
from multimodal_rag.resources import get_resources

logger = logging.getLogger(__name__)


# =========================
# LOADERS
# =========================

def load_index(index_path: Path):
    """Load FAISS index safely and configure IVF parameters."""
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    try:
        index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path}")

        # 🔥 CRITICAL: Set nprobe for IVF indexes
        try:
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", 10)
            logger.info("Set FAISS nprobe = 10")
        except Exception:
            logger.warning("Could not set nprobe (index may not be IVF)")

        return index

    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise RuntimeError("Could not load FAISS index") from e


def load_metadata(meta_path: Path):
    """Load metadata safely."""
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    try:
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        logger.info(f"Loaded metadata with {len(metadata)} entries")
        return metadata

    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise RuntimeError("Could not load metadata") from e


# =========================
# EMBEDDING
# =========================

def embed_query(query: str) -> np.ndarray:
    """Embed query into normalized vector."""
    if not query.strip():
        raise ValueError("Query cannot be empty")

    resources = get_resources()
    model = resources.get_embedding_model()

    try:
        vector = model.encode([query])
        vector = np.array(vector).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(vector)

        return vector

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise RuntimeError("Query embedding failed") from e


# =========================
# SEARCH
# =========================

def search(index, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Search FAISS index."""
    if index is None:
        raise ValueError("FAISS index is not loaded")

    if query_vector is None or len(query_vector) == 0:
        raise ValueError("Invalid query vector")

    try:
        scores, indices = index.search(query_vector, top_k)
        return scores[0], indices[0]

    except Exception as e:
        logger.error(f"FAISS search failed: {e}")
        raise RuntimeError("Vector search failed") from e


# =========================
# RETRIEVAL PIPELINE
# =========================

def retrieve(query: str, index, metadata: List[dict], top_k: int = 5) -> list:
    """Retrieve top-k relevant documents using vector similarity."""
    if not query.strip():
        return []

    query_vector = embed_query(query)
    scores, indices = search(index, query_vector, top_k)

    results = []

    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]

        results.append({
            "score": float(score),
            "source": meta.get("source"),
            "chunk_index": meta.get("chunk_index"),
            "page": meta.get("page"),
            "text": meta.get("text", "")
        })

    return results