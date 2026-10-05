"""Answer generation: LangChain (prompt | ChatOpenAI | parser) against any OpenAI-compatible API.
Default endpoint is Gemini's OpenAI-compatible URL; switch provider via LLM_BASE_URL / LLM_MODEL."""
from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import Settings

OUT_OF_SCOPE = "[[OUT_OF_SCOPE]]"

SYSTEM_PROMPT = """You are a friendly, accurate science tutor for students of NCERT Class 10 Science.

Answer ONLY from the textbook excerpts in CONTEXT below.

Rules:
1. If CONTEXT does not contain what is needed, or the question is not about the Class 10 science textbook,
   reply with exactly {oos} and nothing else.
2. Explain clearly for a Class 10 student. For numericals show the formula, substitution and the final
   answer with units, using the sign convention given in the textbook.
3. Chat history is only for understanding follow-up questions. Facts must come from CONTEXT.
4. Never mention "context" or "excerpts". Do not invent facts, values or chapter names.
5. For mathematical expressions, use LaTeX with $...$ for inline math and $$...$$ for display equations. Never wrap equations in square brackets. Do not output raw LaTeX commands outside math delimiters.
6. Finish with ONE last line in exactly this format, listing only chapters from CONTEXT that you used,
   written exactly as in the [Chapter: ...] labels:
   SOURCES: <chapter name>; <chapter name>

CONTEXT:
{context}"""


@dataclass
class GenResult:
    text: str
    out_of_scope: bool
    sources: list[str]


class LLMError(RuntimeError):
    pass


def format_context(docs) -> str:
    return "\n\n---\n\n".join(f"[Chapter: {d.metadata['chapter']}]\n{d.page_content}" for d in docs)

def normalize_math(raw: str) -> str:
    raw = re.sub(
        r"\\\[\s*(.*?)\s*\\\]",
        lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n",
        raw,
        flags=re.S,
    )

    raw = re.sub(
        r"\[\s*([^\[\]]+?)\s*\]",
        lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n"
        if ("=" in m.group(1) or "\\" in m.group(1))
        else m.group(0),
        raw,
    )

    return raw

def parse_output(raw: str, valid_chapters: list[str], fallback: list[str]) -> GenResult:
    raw = (raw or "").strip()
    raw = normalize_math(raw)
    if not raw or OUT_OF_SCOPE in raw:
        return GenResult("", True, [])
    sources: list[str] = []
    m = re.search(r"^\s*\**SOURCES:?\**\s*(.+)$", raw, re.I | re.M)
    if m:
        raw = (raw[:m.start()] + raw[m.end():]).strip()
        for part in re.split(r"[;|]", m.group(1)):
            name = part.strip().strip("*`'\" .")
            match = next((c for c in valid_chapters if c.lower() == name.lower()), None)
            if match and match not in sources:
                sources.append(match)
    return GenResult(raw, False, sources or fallback[:1])


class Generator:
    def __init__(self, settings: Settings):
        from langchain_openai import ChatOpenAI
        if not settings.llm_api_key:
            raise LLMError("No LLM API key found. Set LLM_API_KEY - see README.")
        llm = ChatOpenAI(model=settings.llm_model, api_key=settings.llm_api_key,
                         base_url=settings.llm_base_url, temperature=0.2, timeout=90, max_retries=2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ])
        self.chain = prompt | llm | StrOutputParser()

    def generate(self, question: str, history: list[dict], docs, valid_chapters: list[str]) -> GenResult:
        msgs = [HumanMessage(h["content"]) if h["role"] == "user" else AIMessage(h["content"][:1500])
                for h in history]
        try:
            raw = self.chain.invoke({"context": format_context(docs), "history": msgs,
                                     "question": question, "oos": OUT_OF_SCOPE})
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise LLMError(f"LLM call failed: {e}") from e
        fallback = [docs[0].metadata["chapter"]] if docs else []
        return parse_output(raw, valid_chapters, fallback)
