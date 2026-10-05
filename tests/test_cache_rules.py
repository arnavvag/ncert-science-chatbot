"""Offline tests for the cache rules. They use a fake embedder and a fake LLM, so they need no
API key, no internet and no textbook PDFs:   python -m pytest -q
"""
import os
os.environ["EMBED_BACKEND"] = "fake"

import pytest
from fastapi.testclient import TestClient
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from app.cache import SemanticCache
from app.config import Settings
from app.embeddings import FakeEmbeddings
from app.llm import GenResult
from app.service import DECLINE, ChatService
from app.textutils import CLARIFY, CONTEXTUAL, META, STANDALONE, classify, validate

CHAPTER = "Light – Reflection and Refraction"
TEXTS = [
    "Refraction of light is the bending of light when it passes from one medium to another. Laws of refraction.",
    "Reflection of light from mirrors. Concave mirror and convex mirror form images. Focal length f = R/2.",
    "Acids bases and salts. pH scale and litmus.",
]


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, question, history, docs, valid_chapters):
        self.calls += 1
        return GenResult(f"ANSWER#{self.calls} to: {question}", False, [docs[0].metadata["chapter"]])


@pytest.fixture()
def env(tmp_path):
    emb = FakeEmbeddings()
    docs = [Document(page_content=t, metadata={"chapter": CHAPTER}) for t in TEXTS]
    store = FAISS.from_documents(docs, emb, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)

    class R:
        chapters = [CHAPTER]
        index_version = "test"

        def search(self, q, k):
            return [(d, float(s)) for d, s in store.similarity_search_with_score(q, k=k)]

    s = Settings()
    s.min_relevance = s.cache_min_relevance = 0.05
    s.cache_sim_threshold = 0.5   # fake embeddings are crude; the guards decide, as in production
    cache = SemanticCache(emb, tmp_path / "c.sqlite3", "test", s.cache_sim_threshold)
    gen = FakeGenerator()
    return ChatService(R(), cache, gen, s), gen


def ask(svc, sid, msg):
    return svc.chat(sid, msg)


# ---------------------------------------------------------------- the five situations in the brief
def test_same_doubt_different_wording_hits_without_llm(env):
    svc, gen = env
    s = svc.new_session()
    first = ask(svc, s, "What is refraction?")
    assert not first["cache_hit"] and gen.calls == 1
    second = ask(svc, svc.new_session(), "What does refraction mean?")
    assert second["cache_hit"] and gen.calls == 1            # no LLM call on a hit
    assert second["reply"] == first["reply"] and second["latency_ms"] < 500
    assert second["citations"] == [CHAPTER]


def test_similar_words_different_question_is_not_served(env):
    svc, gen = env
    ask(svc, svc.new_session(), "Image by a concave mirror")
    r = ask(svc, svc.new_session(), "Image by a convex mirror")
    assert not r["cache_hit"] and gen.calls == 2


def test_same_question_different_numbers_is_not_served(env):
    svc, gen = env
    ask(svc, svc.new_session(), "Focal length when R = 20 cm")
    r30 = ask(svc, svc.new_session(), "Focal length when R = 30 cm")
    assert not r30["cache_hit"] and gen.calls == 2
    again = ask(svc, svc.new_session(), "focal length when R=20 centimetres")
    assert again["cache_hit"] and gen.calls == 2


def test_followup_is_only_reused_in_the_same_conversation_context(env):
    svc, gen = env
    a = svc.new_session()
    ask(svc, a, "What is refraction?")
    ask(svc, a, "What about its laws?")
    calls = gen.calls
    b = svc.new_session()                      # same conversation again -> reuse is correct
    ask(svc, b, "What is refraction?")
    hit = ask(svc, b, "What about its laws?")
    assert hit["cache_hit"] and gen.calls == calls
    c = svc.new_session()                      # different earlier topic -> must NOT reuse
    ask(svc, c, "Reflection of light from mirrors")
    miss = ask(svc, c, "What about its laws?")
    assert not miss["cache_hit"]


def test_requests_tied_to_the_conversation_are_never_cached(env):
    svc, gen = env
    for _ in range(2):
        s = svc.new_session()
        ask(svc, s, "What is refraction?")
        r = ask(svc, s, "Explain it more simply")
        assert not r["cache_hit"]
    assert gen.calls == 1 + 2                   # 1 for the cached question + both 'simply' requests


def test_turn_after_an_uncacheable_turn_is_not_cached(env):
    svc, gen = env
    for _ in range(2):
        s = svc.new_session()
        ask(svc, s, "What is refraction?")
        ask(svc, s, "Explain it more simply")
        r = ask(svc, s, "What about its laws?")
        assert not r["cache_hit"]


# ---------------------------------------------------------------- safety around the edges
def test_out_of_scope_is_declined_without_llm_and_not_cached(env):
    svc, gen = env
    r = ask(svc, svc.new_session(), "cricket champion trophy")
    assert r["reply"] == DECLINE and r["citations"] == [] and gen.calls == 0
    assert len(svc.cache) == 0


def test_first_message_that_needs_context_asks_for_clarification(env):
    svc, gen = env
    r = ask(svc, svc.new_session(), "Explain it more simply")
    assert gen.calls == 0 and not r["cache_hit"] and len(svc.cache) == 0


def test_llm_failure_is_not_cached(env):
    svc, gen = env

    def boom(*a, **k):
        raise RuntimeError("quota")
    gen.generate = boom
    with pytest.raises(RuntimeError):
        ask(svc, svc.new_session(), "What is refraction?")
    assert len(svc.cache) == 0


def test_cache_survives_restart_but_not_a_new_index_version(env, tmp_path):
    svc, gen = env
    ask(svc, svc.new_session(), "What is refraction?")
    same = SemanticCache(FakeEmbeddings(), tmp_path / "c.sqlite3", "test", 0.5)
    other = SemanticCache(FakeEmbeddings(), tmp_path / "c.sqlite3", "new-index", 0.5)
    assert len(same) == 1 and len(other) == 0


# ---------------------------------------------------------------- guards in isolation
@pytest.mark.parametrize("q,c,ok", [
    ("What is refraction?", "What does refraction mean?", True),
    ("Define refraction", "What is refraction", True),
    ("Difference between mitosis and meiosis", "Differentiate meiosis and mitosis", True),
    ("Image by a concave mirror", "Image by a convex mirror", False),
    ("Image by a mirror", "Image by a concave mirror", False),
    ("Focal length when R = 20 cm", "Focal length when R = 30 cm", False),
    ("Focal length when R = 20 cm", "focal length if R=20 centimetres", True),
    ("u = 20 cm and f = 30 cm, find v", "u = 30 cm and f = 20 cm, find v", False),
    ("convert ice to water", "convert water to ice", False),
    ("What is the formula of NaOH", "What is the formula of NaCl", False),
    ("What is H2O", "What is H2O2", False),
    ("What is Newton's first law", "What is Newton's second law", False),
    ("Why does the sky look blue", "How does the sky look blue", False),
])
def test_validate(q, c, ok):
    assert validate(q, c)[0] is ok


@pytest.mark.parametrize("msg,hist,kind", [
    ("What is refraction?", False, STANDALONE),
    ("What are the laws of refraction?", True, STANDALONE),
    ("What about its laws?", True, CONTEXTUAL),
    ("and for convex mirrors?", True, CONTEXTUAL),
    ("Why does it bend towards the normal?", True, CONTEXTUAL),
    ("Explain it more simply", True, META),
    ("Give me an example", True, META),
    ("Explain it more simply", False, CLARIFY),
])
def test_classify(msg, hist, kind):
    assert classify(msg, hist) == kind


# ---------------------------------------------------------------- HTTP contract
def test_api_contract(env, monkeypatch):
    import app.api as api
    svc, _ = env
    monkeypatch.setattr(api, "_service", svc)
    client = TestClient(api.app)           # no lifespan: service is injected
    sid = client.post("/session").json()["session_id"]
    body = client.post("/chat", json={"session_id": sid, "message": "What is refraction?"}).json()
    assert set(body) == {"reply", "citations", "cache_hit", "latency_ms"}
    assert client.post("/chat", json={"session_id": "nope", "message": "hi"}).status_code == 404
    assert client.post("/chat", json={"session_id": sid, "message": ""}).status_code == 422
