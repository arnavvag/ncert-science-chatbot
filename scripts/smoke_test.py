"""End-to-end check against a RUNNING backend (local or deployed). Prints a table you can paste
into your README / explainer.

    python scripts/smoke_test.py --api http://127.0.0.1:8000

It needs a working Gemini key (fresh answers are real LLM calls) and uses a few API calls.
"""
import argparse
import sys

import requests


def ask(api, sid, msg):
    r = requests.post(f"{api}/chat", json={"session_id": sid, "message": msg}, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    api = ap.parse_args().api.rstrip("/")
    new = lambda: requests.post(f"{api}/session", timeout=15).json()["session_id"]  # noqa: E731

    # (label, [messages in one conversation], index of the turn to judge, expected cache_hit)
    # Run twice: the first pass fills the cache, the second pass is judged.
    scenarios = [
        ("Same doubt, different wording", ["What is refraction?"], ["What does refraction mean?"], True),
        ("Similar words, different question", ["Image formed by a concave mirror"],
         ["Image formed by a convex mirror"], False),
        ("Same question, different numbers", ["Focal length of a spherical mirror when R = 20 cm"],
         ["Focal length of a spherical mirror when R = 30 cm"], False),
        ("Same question, same numbers (reworded)", ["Focal length of a spherical mirror when R = 20 cm"],
         ["focal length of spherical mirror if R=20 centimetres"], True),
        ("Follow-up, same conversation", ["What is refraction?", "What about its laws?"],
         ["What is refraction?", "What about its laws?"], True),
        ("Follow-up, different conversation", ["What is the reflection of light?"],
         ["What is the reflection of light?", "What about its laws?"], None),
        ("Tied to conversation", ["What is refraction?", "Explain it more simply"],
         ["What is refraction?", "Explain it more simply"], False),
        ("Out of scope", ["Who is the prime minister of India?"], ["Who is the prime minister of India?"], False),
    ]
    print(f"{'Scenario':<38}{'cache_hit':<11}{'expected':<10}{'ms':<7}result")
    bad = 0
    for name, seed, probe, expected in scenarios:
        sid = new()
        for m in seed:
            ask(api, sid, m)
        sid = new()
        res = None
        for m in probe:
            res = ask(api, sid, m)
        ok = "-" if expected is None else ("PASS" if res["cache_hit"] == expected else "FAIL")
        bad += ok == "FAIL"
        print(f"{name:<38}{str(res['cache_hit']):<11}{str(expected):<10}{res['latency_ms']:<7}{ok}")
    stats = requests.get(f"{api}/cache/stats", timeout=10).json()
    print("\ncache stats:", stats)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
