"""Chat orchestration: classify turn -> (maybe) cache -> retrieve -> generate -> (maybe) store."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from .cache import GLOBAL, SemanticCache, make_context_id
from .config import Settings
from .textutils import CLARIFY, CONTEXTUAL, META, STANDALONE, classify

DECLINE = ("I'm sorry, but I couldn't find that in the NCERT Class 10 Science textbook, so I can't answer it "
           "reliably. Please ask me something from the book - for example about light, electricity, acids and "
           "bases, life processes or heredity.")
CLARIFY_MSG = ("I'm not sure what you'd like me to explain yet. Please ask a science question from the NCERT "
               "Class 10 book first (for example \"What is refraction?\"), and then I can simplify it, give an "
               "example, or answer follow-ups.")


@dataclass
class Session:
    id: str
    history: list = field(default_factory=list)   # [{"role","content"}]
    chain: list = field(default_factory=list)     # cache-entry id (or None) per turn
    topic: str = ""                               # running retrieval topic for follow-ups


class ChatService:
    def __init__(self, retriever, cache: SemanticCache, generator, settings: Settings):
        self.retriever, self.cache, self.generator, self.s = retriever, cache, generator, settings
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ sessions
    def new_session(self) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self.sessions[sid] = Session(sid)
        return sid

    def get_session(self, sid: str) -> Session | None:
        return self.sessions.get(sid)

    # ------------------------------------------------------------------ chat
    def chat(self, sid: str, message: str) -> dict:
        t0 = time.perf_counter()
        sess = self.sessions[sid]
        message = message.strip()[: self.s.max_message_chars]
        kind = classify(message, has_history=bool(sess.history))

        def done(reply, citations, hit, entry_id=None):
            sess.history += [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
            sess.chain.append(entry_id)
            if kind == STANDALONE:
                sess.topic = message[-300:]
            elif kind == CONTEXTUAL:
                sess.topic = f"{sess.topic} {message}"[-300:]
            return {"reply": reply, "citations": citations, "cache_hit": hit,
                    "latency_ms": max(1, int((time.perf_counter() - t0) * 1000))}

        if kind == CLARIFY:
            return done(CLARIFY_MSG, [], False)

        # ---- 1. which cache context may this turn use? ----
        if kind == STANDALONE:
            ctx = GLOBAL
        elif kind == CONTEXTUAL:
            ctx = make_context_id(sess.chain)        # None if a recent turn was uncacheable
        else:                                         # META: tied to this student's last answer
            ctx = None

        # ---- 2. cache lookup (no LLM call on this path) ----
        if ctx is not None:
            hit = self.cache.lookup(message, ctx)
            if hit:
                return done(hit.entry.answer, hit.entry.citations, True, hit.entry.id)

        # ---- 3. fresh answer: retrieve from the textbook ----
        rq = message if kind == STANDALONE else f"{sess.topic} {message}"
        results = self.retriever.search(rq, self.s.top_k)
        best = results[0][1] if results else 0.0
        if best < self.s.min_relevance:
            return done(DECLINE, [], False)

        docs = [d for d, _ in results]
        history = sess.history[-2 * self.s.max_history_turns:]
        gen = self.generator.generate(message, history, docs, self.retriever.chapters)
        if gen.out_of_scope:
            return done(DECLINE, [], False)

        # ---- 4. store only safe, well-grounded answers ----
        entry_id = None
        if ctx is not None and best >= self.s.cache_min_relevance:
            entry_id = self.cache.store(message, ctx, gen.text, gen.sources)
        return done(gen.text, gen.sources, False, entry_id)
