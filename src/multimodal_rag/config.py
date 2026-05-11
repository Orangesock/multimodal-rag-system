"""Configuration management for the multimodal RAG system."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# 🔥 PROFESSIONAL FIX: This line tells Python to actually read your .env file
load_dotenv()

def _env_bool(name: str, default: bool) -> bool:
    """Parse boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    """Parse integer environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Parse float environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def configure_logging() -> None:
    """Configure application-wide logging with structured format."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


@dataclass(frozen=True)
class Settings:
    """Centralized settings for the RAG application."""

    base_dir: Path
    data_dir: Path
    embeddings_dir: Path
    embedding_model: str
    reranker_model: str
    generator_model: str
    retrieval_top_k: int
    rerank_top_k: int
    rerank_score_fraction: float
    min_rerank_score: float
    max_new_tokens: int
    num_beams: int
    no_repeat_ngram_size: int
    answer_min_words: int
    answer_retry_min_words: int
    min_chunk_words: int
    min_alpha_ratio: float
    heading_max_words: int
    evidence_max_items: int
    evidence_words_per: int
    overview_strategy: str
    overview_max_pages: int
    overview_max_chunks: int
    app_host: str
    app_port: int
    app_share: bool

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        # Detect project root
        base_dir = Path(
            os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])
        ).resolve()
        
        data_dir = Path(os.getenv("DATA_DIR", base_dir / "data")).resolve()
        embeddings_dir = Path(
            os.getenv("EMBEDDINGS_DIR", base_dir / "embeddings")
        ).resolve()

        return cls(
            base_dir=base_dir,
            data_dir=data_dir,
            embeddings_dir=embeddings_dir,
            embedding_model=os.getenv("EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2"),
            reranker_model=os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            # This will now correctly pull 'large' from your .env
            generator_model=os.getenv("GENERATOR_MODEL", "google/flan-t5-large"),
            retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 30),
            rerank_top_k=_env_int("RERANK_TOP_K", 8),
            rerank_score_fraction=_env_float("RERANK_SCORE_FRACTION", 0.6),
            min_rerank_score=_env_float("MIN_RERANK_SCORE", -1e9),
            max_new_tokens=_env_int("MAX_NEW_TOKENS", 512),
            num_beams=_env_int("NUM_BEAMS", 4),
            no_repeat_ngram_size=_env_int("NO_REPEAT_NGRAM_SIZE", 3),
            answer_min_words=_env_int("ANSWER_MIN_WORDS", 120),
            answer_retry_min_words=_env_int("ANSWER_RETRY_MIN_WORDS", 160),
            min_chunk_words=_env_int("MIN_CHUNK_WORDS", 20),
            min_alpha_ratio=_env_float("MIN_ALPHA_RATIO", 0.6),
            heading_max_words=_env_int("HEADING_MAX_WORDS", 12),
            evidence_max_items=_env_int("EVIDENCE_MAX_ITEMS", 5),
            evidence_words_per=_env_int("EVIDENCE_WORDS_PER", 28),
            overview_strategy=os.getenv("OVERVIEW_STRATEGY", "full_doc"),
            overview_max_pages=_env_int("OVERVIEW_MAX_PAGES", 5),
            overview_max_chunks=_env_int("OVERVIEW_MAX_CHUNKS", 16),
            app_host=os.getenv("APP_HOST", "0.0.0.0"),
            app_port=_env_int("APP_PORT", 7860),
            app_share=_env_bool("APP_SHARE", False),
        )


settings = Settings.from_env()