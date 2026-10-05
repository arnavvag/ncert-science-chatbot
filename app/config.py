"""Central configuration. Everything can be overridden with environment variables
(or a local .env file, or Streamlit secrets - see README)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # ---- LLM (any OpenAI-compatible endpoint; default = Gemini) ----
    llm_api_key: str = field(default_factory=lambda: (
        os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""))
    llm_base_url: str = field(default_factory=lambda: _env(
        "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gemini-2.5-flash"))

    # ---- Embeddings / index ----
    embed_backend: str = field(default_factory=lambda: _env("EMBED_BACKEND", "fastembed"))  # or "fake" (tests)
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    embed_cache_dir: str = field(default_factory=lambda: _env("EMBED_CACHE_DIR", str(ROOT / ".cache" / "fastembed")))
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", str(ROOT / "index"))))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))

    # ---- Retrieval gates ----
    # Below this best-chunk score the question is treated as "not in the book" (no LLM call).
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.45")))
    # Only answers whose best chunk scores at least this are allowed into the cache.
    cache_min_relevance: float = field(default_factory=lambda: float(_env("CACHE_MIN_RELEVANCE", "0.55")))

    # ---- Cache ----
    cache_db: Path = field(default_factory=lambda: Path(_env("CACHE_DB", str(ROOT / ".cache" / "cache.sqlite3"))))
    # Embedding similarity floor for a cache candidate. The lexical/number guards in
    # app/textutils.py are the real safety net; this just filters obvious non-matches.
    cache_sim_threshold: float = field(default_factory=lambda: float(_env("CACHE_SIM_THRESHOLD", "0.75")))
    cache_search_k: int = 50

    # ---- Chat ----
    max_history_turns: int = field(default_factory=lambda: int(_env("MAX_HISTORY_TURNS", "6")))
    max_message_chars: int = 1000


def get_settings() -> Settings:
    return Settings()
