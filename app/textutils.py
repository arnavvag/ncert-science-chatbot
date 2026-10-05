"""Text analysis used by the cache.

The cache must only serve an answer when it *correctly answers the current question as-is*.
Embedding similarity alone cannot guarantee that ("concave" vs "convex" mirror embed almost
identically), so a cached candidate must ALSO pass these deterministic guards:

  1. numbers (with units) must be identical and in the same order
  2. the set of meaningful terms must be identical (after removing filler words like
     "what/does/mean/define", lower-casing and stemming)
  3. if the question has direction words (to/from/than/vs) the term ORDER must match
  4. turns that depend on the conversation are never matched globally (see classify())

Precision is deliberately preferred over recall: a missed hit costs one LLM call, a wrong
hit teaches a student something wrong.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import snowballstemmer

_stemmer = snowballstemmer.stemmer("english")


def _stem(w: str) -> str:
    return _stemmer.stemWord(w)


# --------------------------------------------------------------------------- vocab
STOP = set("""a an the is are was were be been being am do does did done can could will would should shall may
might must of in on at by for with as and or but if so than then there here which what who whom whose
has have had having i me my we our you your us between""".split()) - {"than", "then"}

# Words that change the *style* of the question but not what is being asked.
FILLER = set("""mean means meaning meant define definition explain explanation tell describe description state give
please pls kindly know want need understand about regarding concept term called briefly""".split())

# Direction / order words: "ice to water" != "water to ice"
DIRECTIONAL = {"to", "from", "into", "onto", "than", "versus", "vs"}

# Tiny synonym map (question phrasing + unit spellings) applied before stemming.
SYN = {
    "differentiate": "difference", "distinguish": "difference", "distinction": "difference",
    "contrast": "difference", "differences": "difference",
    "applications": "use", "application": "use", "uses": "use", "usage": "use",
    "significance": "importance", "purpose": "function", "role": "function",
    "centimetre": "cm", "centimetres": "cm", "centimeter": "cm", "centimeters": "cm",
    "metre": "m", "metres": "m", "meter": "m", "meters": "m",
    "millimetre": "mm", "millimetres": "mm", "millimeter": "mm", "millimeters": "mm",
    "kilometre": "km", "kilometres": "km", "kilometer": "km", "kilometers": "km",
    "second": "s", "seconds": "s", "sec": "s", "secs": "s",
    "ohms": "ohm", "volts": "v", "volt": "v", "ampere": "a", "amperes": "a", "amp": "a", "amps": "a",
    "kilogram": "kg", "kilograms": "kg", "gram": "g", "grams": "g",
}
UNITS = {"cm", "m", "mm", "km", "s", "ms", "min", "h", "hr", "kg", "g", "mg", "n", "j", "w", "v", "a", "ma",
         "ohm", "hz", "k", "c", "mol", "l", "ml", "d", "%", "°c", "°", "kj", "kw", "kwh", "wh", "pa", "ev"}

CONDITIONAL = {"when", "given", "suppose", "assume", "assuming", "having"}

WORD_NUMBERS = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
                "nine": "9", "ten": "10"}

# Words that refer to something said earlier.
REF_WORDS = set("""it its itself this that these those they them their theirs he she him her his above previous
earlier former latter same such another again also further""".split())

# Aspects that are NOT a topic by themselves ("its laws", "give an example", "the formula").
GENERIC_ASPECTS = {_stem(w) for w in """law use type example formula unit importance advantage disadvantage property
characteristic diagram equation function cause effect difference feature kind fact summary reason step process
numerical question answer problem derivation""".split()}
WH = {"how", "why", "when", "where", "who"}

STARTER_RE = re.compile(
    r"^\s*(and|also|but|so|then|now|okay|ok|what about|how about|what if|why not|for|in case|in the case|"
    r"in this case|similarly|likewise|next|continue|go on|more)\b", re.I)
FRAME_RE = re.compile(
    r"^\s*(what|why|how|when|where|which|who|whom|whose|define|explain|state|describe|derive|list|name|write|"
    r"differentiate|distinguish|give|calculate|find|compute|determine|draw|is|are|do|does|did|can|could|tell|"
    r"show|prove|compare)\b", re.I)
# Requests to transform / repeat the previous answer.
META_RE = re.compile(
    r"\b(simpl(e|er|y|ify|ified)|easier|easy (words|terms|language)|eli5|layman|shorter|briefer|brief|short(ly)?|"
    r"summar(y|i[sz]e)|tl;?dr|repeat|again|rephrase|reword|elaborate|expand|in detail|more detail|detailed|"
    r"examples?|analogy|clearer|clarify|don'?t understand|do not understand|didn'?t get|confus\w*|translate|"
    r"hindi|bullet|points|step by step|one more|another)\b", re.I)

_NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])(-?\d+(?:\.\d+)?)(?:\s*([A-Za-zµΩ°%]+))?")
_WORDNUM_RE = re.compile(r"\b(" + "|".join(WORD_NUMBERS) + r")\b", re.I)
_TOKEN_RE = re.compile(r"[a-z][a-z0-9]*")


# --------------------------------------------------------------------------- analysis
@dataclass
class Analysis:
    raw: str
    numbers: list = field(default_factory=list)   # ordered [(value, unit)]
    seq: list = field(default_factory=list)       # ordered content stems
    content: frozenset = frozenset()              # set of content stems
    has_direction: bool = False
    refs: bool = False
    starter: bool = False
    frame: bool = False
    meta: bool = False
    topic: list = field(default_factory=list)     # content stems that identify a subject


def _extract_numbers(text: str) -> list:
    t = text.replace("−", "-").replace("–", "-")
    found = []
    for m in _NUM_RE.finditer(t):
        val = format(float(m.group(1)), "g")
        unit = (m.group(2) or "").lower()
        unit = SYN.get(unit, unit)
        found.append((m.start(), val, unit if unit in UNITS else ""))
    for m in _WORDNUM_RE.finditer(t):
        found.append((m.start(), WORD_NUMBERS[m.group(1).lower()], ""))
    found.sort()
    return [(v, u) for _, v, u in found]


def analyze(text: str) -> Analysis:
    t = text.lower().replace("’", "'").replace("–", "-").replace("−", "-")
    t = re.sub(r"'s\b", "", t).replace("'", "")
    t = re.sub(r"\bnon[\s-]+(?=[a-z])", "non", t)
    t = t.replace("-", " ")
    raw_tokens = _TOKEN_RE.findall(t)

    refs = any(w in REF_WORDS for w in raw_tokens)
    numbers = _extract_numbers(text)
    # In a numerical ("... when R = 20 cm" / "... if R = 20 cm") these words only introduce the givens.
    conditional = CONDITIONAL if numbers else set()
    seq, dirs = [], False
    for w in raw_tokens:
        if w in WORD_NUMBERS or w in STOP or w in FILLER or w in conditional:
            continue
        if w in DIRECTIONAL:
            dirs = True
            continue
        seq.append(_stem(SYN.get(w, w)))
    topic = [s for s in seq if s not in GENERIC_ASPECTS and s not in WH
             and s not in {_stem(r) for r in REF_WORDS}]
    return Analysis(
        raw=text, numbers=numbers, seq=seq, content=frozenset(seq), has_direction=dirs,
        refs=refs, starter=bool(STARTER_RE.match(text)), frame=bool(FRAME_RE.match(text)),
        meta=bool(META_RE.search(text)), topic=topic,
    )


def canonical(text: str) -> str:
    a = analyze(text)
    return " ".join(sorted(a.content)) + "|" + ",".join(f"{v}{u}" for v, u in a.numbers)


# --------------------------------------------------------------------------- guards
def validate(question: str, candidate: str) -> tuple[bool, str]:
    """May a cached answer for `candidate` be served for `question`? Returns (ok, reason)."""
    a, b = analyze(question), analyze(candidate)
    if a.numbers != b.numbers:
        return False, "numbers/units differ"
    if a.content != b.content:
        return False, "meaningful terms differ"
    if (a.has_direction or b.has_direction) and a.seq != b.seq:
        return False, "term order differs"
    if not a.content and not a.numbers:
        return False, "nothing to match on"
    return True, "ok"


# --------------------------------------------------------------------------- turn type
STANDALONE, CONTEXTUAL, META, CLARIFY = "standalone", "contextual", "meta", "clarify"


def classify(message: str, has_history: bool) -> str:
    """Decide how a turn may interact with the cache.

    standalone  - fully self-contained: global cache is allowed
    contextual  - depends on earlier turns ("what about its laws?"): cached only inside the
                  exact same conversation context (see service.py)
    meta        - "explain it more simply", "give an example": never cached
    clarify     - nothing to anchor on ("explain it more simply" as the first message)
    """
    a = analyze(message)
    needs_context = a.refs or a.starter or not a.topic or (len(a.seq) <= 2 and not a.frame)
    if not needs_context:
        return STANDALONE
    if not has_history:
        # No history -> pronouns cannot refer to anything outside the message itself.
        return CLARIFY if (not a.topic or (a.meta and a.refs)) else STANDALONE
    if a.meta and (a.refs or not a.topic):
        return META
    return CONTEXTUAL
