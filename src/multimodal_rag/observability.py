"""Observability and tracing for RAG pipeline."""

import time
import logging
from typing import Any, Dict

logger = logging.getLogger("rag_observability")


class Trace:
    """Simple trace object for measuring pipeline steps."""

    def __init__(self):
        self.steps = []
        self.start_time = time.time()

    def log_step(self, name: str, data: Dict[str, Any] = None):
        """Log a pipeline step."""
        now = time.time()
        self.steps.append({
            "step": name,
            "timestamp": now,
            "elapsed": now - self.start_time,
            "data": data or {}
        })

    def summary(self):
        """Return full trace."""
        return {
            "total_time": time.time() - self.start_time,
            "steps": self.steps
        }


def log_query(query: str):
    logger.info(f"[QUERY] {query}")


def log_retrieval(results: list):
    logger.info(f"[RETRIEVAL] {len(results)} docs")

    for r in results[:3]:  # only log top 3 to avoid noise
        logger.info(f"  -> {r.get('text', '')[:100]}")


def log_rerank(results: list):
    logger.info(f"[RERANK] {len(results)} docs")

    for r in results[:3]:
        logger.info(f"  -> score={r.get('rerank_score')} | {r.get('text', '')[:100]}")


def log_answer(answer: str):
    logger.info(f"[ANSWER] {answer[:200]}")