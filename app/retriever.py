"""Textbook retriever: a LangChain FAISS vector store built by scripts/ingest.py."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document


class Retriever:
    def __init__(self, embeddings, index_dir: Path):
        index_dir = Path(index_dir)
        if not (index_dir / "faiss_index" / "index.faiss").exists():
            raise FileNotFoundError(
                f"No FAISS index in {index_dir}. Put the NCERT PDFs in ./data and run: python scripts/ingest.py")
        # Embeddings are L2-normalised, so inner product == cosine similarity (higher = closer).
        self.store = FAISS.load_local(
            str(index_dir / "faiss_index"), embeddings,
            allow_dangerous_deserialization=True,  # we load only the index we built ourselves
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        self.meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        self.index_version: str = self.meta["index_version"]
        self.chapters: list[str] = self.meta["chapters"]

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        return [(d, float(s)) for d, s in self.store.similarity_search_with_score(query, k=k)]
