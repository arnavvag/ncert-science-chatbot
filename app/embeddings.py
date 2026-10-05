"""Embedding backends. FastEmbed (ONNX, CPU, ~130 MB model, no API cost) is the default, so a
cache hit never leaves the machine. A hash-based fake backend exists only for unit tests."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings

from .config import get_settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _norm(v) -> list[float]:
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_norm(v) for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Question -> textbook passage retrieval (asymmetric, uses the BGE query prefix)."""
        return _norm(next(iter(self.model.embed([BGE_QUERY_PREFIX + text]))))

    def embed_sym(self, text: str) -> list[float]:
        """Question <-> question comparison for the cache (symmetric, no prefix)."""
        return _norm(next(iter(self.model.embed([text]))))


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embedder. TESTS ONLY."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return _norm(v)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def embed_sym(self, text):
        return self._vec(text)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    s = get_settings()
    if s.embed_backend == "fake":
        return FakeEmbeddings()
    return FastEmbedEmbeddings(s.embed_model, s.embed_cache_dir)
