"""Caching layer for RAG system."""

import hashlib
import pickle
from pathlib import Path


class SimpleCache:
    """Disk-based cache."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        hashed = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed}.pkl"

    def get(self, key: str):
        path = self._key_to_path(key)
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def set(self, key: str, value):
        path = self._key_to_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f)