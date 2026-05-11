"""
Singleton resource manager with Thread-Safe Precision Qwen2 (GGUF).
Prevents segmentation faults on Intel Macs by locking the model during inference.
"""

import logging
import threading
import pickle
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

from multimodal_rag.config import settings
from multimodal_rag.cache import SimpleCache

logger = logging.getLogger(__name__)

class ResourceManager:
    """Singleton manager for heavy local resources."""

    _instance = None
    _lock = threading.Lock()
    
    # 🧠 THE BRAIN LOCK: 
    # Ensures only one process (Indexing or Querying) can use the LLM at a time.
    # This prevents the 'Segmentation Fault' on Intel Macs.
    model_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Models (lazy loaded)
        self._embedding_model = None
        self._reranker_model = None
        self._generator_model = None

        # Retrieval resources
        self._faiss_index = None
        self._metadata = None
        self._bm25 = None
        self._corpus_texts = None

        # Disk caches
        cache_dir = settings.base_dir / "cache"
        self.embedding_cache = SimpleCache(cache_dir / "embeddings")
        self.retrieval_cache = SimpleCache(cache_dir / "retrieval")
        self.response_cache = SimpleCache(cache_dir / "responses")

        self._initialized = True

    def _get_device(self) -> str:
        """Detect best hardware: MPS (Mac), CUDA (Nvidia), or CPU."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # =========================
    # GENERATOR (QWEN2-1.5B GGUF)
    # =========================

    def get_generator_model(self):
        """Loads local Qwen2-1.5B via llama-cpp-python with thread safety."""
        if self._generator_model is None:
            model_path = settings.base_dir / "models" / "qwen2-1_5b-instruct-q8_0.gguf"
            
            if model_path.exists():
                logger.info(f"🧠 Loading Thread-Safe Precision Qwen2-1.5B: {model_path}")
                from llama_cpp import Llama

                # Configured for Intel Mac stability
                llm = Llama(
                    model_path=str(model_path),
                    n_gpu_layers=0, 
                    n_ctx=3072,  # Breathing room for large PDF chunks
                    verbose=False
                )

                class LlamaCPPWrapper:
                    def __init__(self, model, lock):
                        self.model = model
                        self.lock = lock

                    def __call__(self, prompt, **kwargs):
                        # 🔐 THE LOCK:
                        # If the indexer is using the brain, the query will wait.
                        # This stops the memory collision (Segmentation Fault).
                        with self.lock:
                            formatted_prompt = (
                                f"<|im_start|>system\n"
                                f"You are a precise research assistant. Your goal is to answer questions "
                                f"directly and accurately. Use ONLY the provided context. Do not add fluff "
                                f"or irrelevant preamble. Match the length of your answer to the "
                                f"complexity of the question.<|im_end|>\n"
                                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                                f"<|im_start|>assistant\n"
                            )
                            
                            response = self.model(
                                formatted_prompt,
                                max_tokens=kwargs.get("max_new_tokens", 512),
                                temperature=0.1,
                                stop=["<|im_end|>", "<|im_start|>"]
                            )
                            return [{"generated_text": response["choices"][0]["text"].strip()}]

                self._generator_model = LlamaCPPWrapper(llm, self.model_lock)

            else:
                logger.warning("⚠️ Local GGUF not found. Falling back to pipeline.")
                device = self._get_device()
                self._generator_model = pipeline(
                    "text2text-generation",
                    model=settings.generator_model,
                    device=device
                )
                
        return self._generator_model

    # =========================
    # OTHER RESOURCES
    # =========================

    def get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            device = self._get_device()
            self._embedding_model = SentenceTransformer(settings.embedding_model, device=device)
        return self._embedding_model

    def get_reranker_model(self) -> CrossEncoder:
        if self._reranker_model is None:
            device = self._get_device()
            self._reranker_model = CrossEncoder(settings.reranker_model, device=device)
        return self._reranker_model

    def get_retrieval_resources(self):
        if self._faiss_index is None:
            index_path = settings.embeddings_dir / "faiss.index"
            meta_path = settings.embeddings_dir / "meta.pkl"
            bm25_path = settings.embeddings_dir / "bm25.pkl"

            if not index_path.exists() or not meta_path.exists() or not bm25_path.exists():
                raise FileNotFoundError("Retrieval indexes not found. Please build the index first.")

            self._faiss_index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                self._metadata = pickle.load(f)
            with open(bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)
            self._corpus_texts = [m["text"] for m in self._metadata]

        return self._faiss_index, self._metadata, self._bm25, self._corpus_texts

    def reload_indexes(self) -> None:
        self._faiss_index = self._metadata = self._bm25 = self._corpus_texts = None

_resource_manager = None

def get_resources() -> ResourceManager:
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager