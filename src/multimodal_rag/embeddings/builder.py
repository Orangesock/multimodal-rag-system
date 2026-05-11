"""Build FAISS index + metadata from PDFs using Parent-Child summarization logic."""

import logging
import os
from pathlib import Path
from typing import Literal

import pickle
import faiss
import numpy as np

from multimodal_rag.config import settings
from multimodal_rag.ingestion.semantic_chunker import load_and_semantic_chunk_pdfs
from multimodal_rag.resources import get_resources
from multimodal_rag.retrieval.bm25 import build_bm25

logger = logging.getLogger(__name__)

# Aggressive macOS stability settings to prevent segmentation faults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

IndexType = Literal["flat", "ivf"]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of text chunks into vectors - Mac-safe version."""
    if not texts:
        raise ValueError("Cannot embed empty text list")

    try:
        resources = get_resources()
        model = resources.get_embedding_model()

        # Very conservative settings for Mac to ensure stability
        vectors = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=8,              # Very small batch for CPU/MPS stability
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return np.array(vectors).astype("float32")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise


def build_index(
    data_dir: str | Path | None = None,
    index_type: IndexType = "flat",
    progress=None  # 🛠️ Added progress argument here
) -> int:
    """
    Orchestrates the build process:
    1. Extracts structural blocks and generates LLM summaries.
    2. Embeds the summaries (Child Vectors).
    3. Stores raw text in metadata (Parent Chunks).
    """
    # Diagnostic Log: Immediate visibility into which model is being used
    logger.info(f"🚀 Starting build process using generator: {settings.generator_model}")

    data_dir = Path(data_dir or settings.data_dir)
    output_dir = settings.embeddings_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extraction + Summarization (The time-intensive step)
    # 🛠️ Pass progress down to the chunker
    chunks = load_and_semantic_chunk_pdfs(str(data_dir), progress=progress)
    
    if not chunks:
        logger.warning("No chunks created. Check if your data directory contains PDFs.")
        return 0

    # 2. Map Multi-Vector Logic
    # We embed the LLM-generated summaries (search_text)
    chunk_texts = [c["search_text"] for c in chunks]
    
    # We store the raw structural text in metadata for the LLM to read (text)
    metadata = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "page": c.get("page"),
            "text": c["text"],
        }
        for c in chunks
    ]

    # 3. Create Embeddings
    vectors = embed_texts(chunk_texts)

    # 4. Build FAISS index
    dimension = vectors.shape[1]
    faiss.omp_set_num_threads(1)
    
    num_vectors = len(vectors)

    # Prevent FAISS crash on small datasets when using IVF
    if index_type == "ivf":
        if num_vectors < 1000:
            logger.warning(
                f"Only {num_vectors} vectors found. IVF requires a larger dataset. "
                "Falling back to 'flat' index."
            )
            index_type = "flat"

    if index_type == "ivf":
        # Heuristic for nlist: roughly square root of dataset size
        nlist = max(int(num_vectors ** 0.5), 1)
        
        # Ensure nlist isn't too large for the dataset to prevent training crash
        max_allowed_nlist = max(num_vectors // 39, 1)
        nlist = min(nlist, max_allowed_nlist)
        
        logger.info(f"Training IVF index with nlist={nlist}")
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
    else:
        logger.info("Building Flat index (Exact Search)")
        index = faiss.IndexFlatIP(dimension)

    faiss.normalize_L2(vectors)
    index.add(vectors)

    # 5. Build BM25 index (Keyword search also uses the summaries)
    logger.info("Building BM25 keyword index...")
    bm25_index, _ = build_bm25(chunk_texts)

    # 6. Save to disk
    faiss.write_index(index, str(output_dir / "faiss.index"))
    
    with open(output_dir / "meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
        
    with open(output_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)

    logger.info(f"✅ Successfully indexed {len(chunks)} structural chunks.")
    return len(chunks)