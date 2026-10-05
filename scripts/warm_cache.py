"""OPTIONAL: pre-fill the cache with common questions (uses your Gemini quota, ~1 call each).

    python scripts/warm_cache.py
The cache lives in ./.cache/cache.sqlite3 (git-ignored). If you want a pre-seeded cache on Streamlit
Cloud, remove `.cache/` from .gitignore and commit the file - but note a fresh deployment will then
show cache hits right away, which may make demos less obvious. Default: don't.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api import build_service  # noqa: E402

QUESTIONS = """What is refraction?
What are the laws of refraction?
What is a concave mirror?
What is a convex mirror?
What is the mirror formula?
What is the lens formula?
What is power of a lens?
What is Ohm's law?
What is electric current?
What is resistance?
What is an electric fuse?
What is a chemical reaction?
What is a balanced chemical equation?
What is pH?
What are acids and bases?
What is photosynthesis?
What is respiration?
What is a reflex action?
What is heredity?
What is an ecosystem?""".splitlines()

if __name__ == "__main__":
    svc = build_service()
    for q in QUESTIONS:
        r = svc.chat(svc.new_session(), q)
        print(f"{'HIT ' if r['cache_hit'] else 'NEW '} {r['latency_ms']:>6} ms  {q}")
