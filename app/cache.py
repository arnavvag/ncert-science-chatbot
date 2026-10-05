"""Semantic answer cache: SQLite (durable) + in-memory FAISS (fast lookup).

A cache *hit* = embedding candidate  AND  deterministic guards (textutils.validate)  AND
same context key. It never touches an LLM and typically costs a few milliseconds.

Context key
  - standalone questions live in the "GLOBAL" context
  - conversation-dependent follow-ups live in a context derived from the cache entries of the
    previous turns, so "what about its laws?" is only reused after the *same* preceding turns.
Entries are also tied to the textbook index version, so re-indexing invalidates old answers.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass

import faiss
import numpy as np

from .textutils import canonical, validate

GLOBAL = "GLOBAL"


@dataclass
class Entry:
    id: str
    question: str
    context_id: str
    answer: str
    citations: list[str]


@dataclass
class Hit:
    entry: Entry
    similarity: float


def make_context_id(chain: list) -> str | None:
    """Context key from the cache-entry ids of the last two turns (None => not cacheable)."""
    last = chain[-2:]
    if not last or any(x is None for x in last):
        return None
    return hashlib.sha1("|".join(last).encode()).hexdigest()[:16]


class SemanticCache:
    def __init__(self, embeddings, db_path, index_version: str, sim_threshold: float, search_k: int = 50):
        self.emb = embeddings
        self.threshold = sim_threshold
        self.search_k = search_k
        self.index_version = index_version
        self._lock = threading.Lock()
        self._entries: list[Entry] = []
        self._index = None
        self.stats = {"lookups": 0, "hits": 0, "stores": 0, "vetoed": 0}
        db_path = str(db_path)
        import os
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("""CREATE TABLE IF NOT EXISTS entries(
            id TEXT PRIMARY KEY, question TEXT, context_id TEXT, answer TEXT, citations TEXT,
            index_version TEXT, embedding BLOB, created REAL, hits INTEGER DEFAULT 0)""")
        self._db.commit()
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self):
        rows = self._db.execute(
            "SELECT id, question, context_id, answer, citations, embedding FROM entries WHERE index_version=?",
            (self.index_version,)).fetchall()
        vecs = []
        for id_, q, ctx, ans, cit, blob in rows:
            self._entries.append(Entry(id_, q, ctx, ans, json.loads(cit)))
            vecs.append(np.frombuffer(blob, dtype="float32"))
        if vecs:
            self._index = faiss.IndexFlatIP(len(vecs[0]))
            self._index.add(np.vstack(vecs))

    def __len__(self):
        return len(self._entries)

    # ------------------------------------------------------------------ API
    def lookup(self, question: str, context_id: str) -> Hit | None:
        self.stats["lookups"] += 1
        if self._index is None or not self._entries:
            return None
        vec = np.asarray([self.emb.embed_sym(question)], dtype="float32")
        with self._lock:
            k = min(self.search_k, len(self._entries))
            scores, ids = self._index.search(vec, k)
            for score, i in zip(scores[0], ids[0]):
                if i < 0 or score < self.threshold:
                    break  # results are sorted, nothing better follows
                e = self._entries[i]
                if e.context_id != context_id:
                    continue
                ok, _reason = validate(question, e.question)
                if ok:
                    self.stats["hits"] += 1
                    self._db.execute("UPDATE entries SET hits=hits+1 WHERE id=?", (e.id,))
                    self._db.commit()
                    return Hit(e, float(score))
                self.stats["vetoed"] += 1  # looked similar, but a guard said no
        return None

    def store(self, question: str, context_id: str, answer: str, citations: list[str]) -> str:
        entry_id = hashlib.sha1(f"{context_id}|{canonical(question)}".encode()).hexdigest()[:16]
        with self._lock:
            if any(e.id == entry_id for e in self._entries):
                return entry_id
            vec = np.asarray([self.emb.embed_sym(question)], dtype="float32")
            if self._index is None:
                self._index = faiss.IndexFlatIP(vec.shape[1])
            self._index.add(vec)
            self._entries.append(Entry(entry_id, question, context_id, answer, citations))
            self._db.execute(
                "INSERT OR IGNORE INTO entries VALUES (?,?,?,?,?,?,?,?,0)",
                (entry_id, question, context_id, answer, json.dumps(citations), self.index_version,
                 vec[0].tobytes(), time.time()))
            self._db.commit()
            self.stats["stores"] += 1
        return entry_id
