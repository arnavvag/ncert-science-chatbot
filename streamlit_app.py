"""Streamlit frontend for the NCERT Class 10 Science chatbot.

The frontend talks to the FastAPI backend over HTTP.

- Local / Streamlit Cloud: if API_URL is not set, FastAPI starts in a background
  thread inside this same process, so one deployment serves both.
- Separate backend: set API_URL=http://host:8000 and the app only calls the API.
- Conversation history is maintained in the Streamlit session and supports
  multiple ChatGPT-style conversations in the sidebar.
"""

import os
import threading
import time
from datetime import datetime

import requests
import streamlit as st


st.set_page_config(
    page_title="NCERT Class 10 Science",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.18);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .app-brand {
            padding: 0.25rem 0 1.1rem 0;
        }

        .app-brand-title {
            font-size: 1.18rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
        }

        .app-brand-subtitle {
            font-size: 0.78rem;
            opacity: 0.62;
            margin-top: 0.18rem;
        }

        .hero {
            text-align: center;
            padding: 1.6rem 0 1.2rem 0;
        }

        .hero-icon {
            font-size: 2.4rem;
            margin-bottom: 0.25rem;
        }

        .hero-title {
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 750;
            letter-spacing: -0.045em;
            line-height: 1.05;
            margin: 0;
        }

        .hero-subtitle {
            margin-top: 0.65rem;
            opacity: 0.62;
            font-size: 0.98rem;
        }

        .empty-card {
            max-width: 720px;
            margin: 1.5rem auto 2rem auto;
            padding: 1.35rem 1.4rem;
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 16px;
            text-align: center;
            background: rgba(128, 128, 128, 0.035);
        }

        .empty-card-title {
            font-size: 1.08rem;
            font-weight: 650;
            margin-bottom: 0.35rem;
        }

        .empty-card-text {
            font-size: 0.9rem;
            opacity: 0.62;
        }

        .section-label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.52;
            margin: 1.15rem 0 0.45rem 0;
        }

        .conversation-title {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .cache-note {
            font-size: 0.73rem;
            opacity: 0.55;
            line-height: 1.35;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 14px;
        }

        .meta {
            font-size: 0.75rem;
            opacity: 0.58;
            margin-top: 0.35rem;
        }

        [data-testid="stChatInput"] {
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Streamlit Cloud secrets -> environment variables
# ---------------------------------------------------------------------------

for _key in (
    "GEMINI_API_KEY",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "API_URL",
):
    try:
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@st.cache_resource(
    show_spinner="Starting the backend (first start may download the embedding model)..."
)
def start_backend() -> str:
    if os.getenv("API_URL"):
        return os.environ["API_URL"].rstrip("/")

    import uvicorn
    from app.api import app as fastapi_app

    port = int(os.getenv("API_PORT", "8000"))
    server = uvicorn.Server(
        uvicorn.Config(
            fastapi_app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    )
    threading.Thread(target=server.run, daemon=True).start()

    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 240

    while time.time() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=2).ok:
                return url
        except requests.RequestException:
            time.sleep(1)

    raise RuntimeError(
        "Backend did not start. Check the app logs for a missing index or API key."
    )


def new_session(api: str) -> str:
    response = requests.post(f"{api}/session", timeout=15)
    response.raise_for_status()
    return response.json()["session_id"]


try:
    API = start_backend()
except Exception as exc:
    st.error(f"Backend failed to start: {exc}")
    st.stop()


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

def make_title(text: str) -> str:
    title = " ".join(text.strip().split())
    if not title:
        return "New conversation"

    replacements = {
        "What is ": "",
        "What are ": "",
        "What does ": "",
        "Explain ": "",
        "Define ": "",
        "How does ": "",
        "How do ": "",
    }

    lowered = title.lower()
    for prefix, replacement in replacements.items():
        if lowered.startswith(prefix.lower()):
            title = replacement + title[len(prefix):]
            break

    title = title.rstrip("?.!")
    if len(title) > 34:
        title = title[:34].rsplit(" ", 1)[0] + "…"

    return title or "New conversation"


def create_conversation(api: str) -> str:
    conversation_id = new_session(api)
    st.session_state.conversations[conversation_id] = {
        "id": conversation_id,
        "title": "New conversation",
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }
    return conversation_id


if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = create_conversation(API)

if "pending" not in st.session_state:
    st.session_state.pending = None


def active_conversation() -> dict:
    conversation_id = st.session_state.active_conversation_id
    return st.session_state.conversations[conversation_id]


def switch_conversation(conversation_id: str) -> None:
    st.session_state.active_conversation_id = conversation_id


def render_meta(meta: dict) -> None:
    chapters = " · ".join(meta.get("citations", [])) or "No chapter citation"
    kind = "⚡ Cache hit" if meta.get("cache_hit") else "🧠 Fresh answer"
    latency = meta.get("latency_ms", "—")

    st.markdown(
        f'<div class="meta">📖 {chapters} &nbsp;·&nbsp; {kind} &nbsp;·&nbsp; ⏱ {latency} ms</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="app-brand">
            <div class="app-brand-title">🔬 NCERT Science</div>
            <div class="app-brand-subtitle">Class 10 · Textbook-grounded AI tutor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋  New conversation", use_container_width=True, type="primary"):
        st.session_state.active_conversation_id = create_conversation(API)
        st.session_state.pending = None
        st.rerun()

    st.markdown('<div class="section-label">Conversations</div>', unsafe_allow_html=True)

    conversation_items = list(st.session_state.conversations.values())

    for conversation in reversed(conversation_items):
        is_active = (
            conversation["id"] == st.session_state.active_conversation_id
        )
        label = (
            f"●  {conversation['title']}"
            if is_active
            else f"   {conversation['title']}"
        )

        if st.button(
            label,
            key=f"conversation_{conversation['id']}",
            use_container_width=True,
        ):
            switch_conversation(conversation["id"])
            st.session_state.pending = None
            st.rerun()

    st.divider()

    # Cache statistics
    try:
        stats = requests.get(f"{API}/cache/stats", timeout=5).json()

        st.markdown(
            '<div class="section-label">Cache performance</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        c1.metric("Entries", stats.get("entries", 0))
        c2.metric("Hits", stats.get("hits", 0))

        st.markdown(
            f"""
            <div class="cache-note">
                {stats.get("lookups", 0)} lookups ·
                {stats.get("vetoed", 0)} unsafe similar-match rejections
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    # Suggested questions
    current_messages = active_conversation()["messages"]

    if not current_messages:
        st.markdown(
            '<div class="section-label">Try these</div>',
            unsafe_allow_html=True,
        )

        examples = [
            "What is refraction?",
            "What does refraction mean?",
            "Image formed by a concave mirror",
            "Image formed by a convex mirror",
            "Focal length of a mirror when R = 20 cm",
            "Focal length of a mirror when R = 30 cm",
            "What about its laws?",
            "Explain it more simply",
        ]

        for example in examples:
            if st.button(
                example,
                key=f"example_{example}",
                use_container_width=True,
            ):
                st.session_state.pending = example
                st.rerun()

    st.divider()

    st.caption(
        "Answers are grounded in the NCERT Class 10 Science textbook. "
        "Cache hits bypass the LLM."
    )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

conversation = active_conversation()
messages = conversation["messages"]

if not messages:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-icon">🔬</div>
            <div class="hero-title">NCERT Class 10 Science</div>
            <div class="hero-subtitle">
                Ask a doubt. Get answers grounded only in the textbook,
                with chapter citations and smart caching.
            </div>
        </div>

        <div class="empty-card">
            <div class="empty-card-title">What would you like to learn?</div>
            <div class="empty-card-text">
                Try a concept, numerical, definition, or follow-up question
                from any chapter of Class 10 Science.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div style="padding: 0.7rem 0 0.35rem 0;">
            <div style="font-size:1.45rem;font-weight:700;letter-spacing:-0.03em;">
                {conversation["title"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_meta(message["meta"])


# ---------------------------------------------------------------------------
# Chat input and API call
# ---------------------------------------------------------------------------

prompt = st.chat_input("Ask a Class 10 Science doubt...")

if prompt is None:
    prompt = st.session_state.pop("pending", None)

if prompt:
    conversation = active_conversation()

    if not conversation["messages"]:
        conversation["title"] = make_title(prompt)

    user_message = {
        "role": "user",
        "content": prompt,
    }
    conversation["messages"].append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API}/chat",
                    json={
                        "session_id": conversation["id"],
                        "message": prompt,
                    },
                    timeout=120,
                )

                # Backend restart can invalidate an old session.
                if response.status_code == 404:
                    new_backend_session = new_session(API)
                    conversation["id"] = new_backend_session
                    st.session_state.conversations[new_backend_session] = conversation
                    del st.session_state.conversations[
                        st.session_state.active_conversation_id
                    ]
                    st.session_state.active_conversation_id = new_backend_session

                    response = requests.post(
                        f"{API}/chat",
                        json={
                            "session_id": new_backend_session,
                            "message": prompt,
                        },
                        timeout=120,
                    )

                response.raise_for_status()
                data = response.json()

            except Exception as exc:
                detail = (
                    getattr(getattr(exc, "response", None), "text", "")
                    or str(exc)
                )
                st.error(f"Sorry, something went wrong: {detail[:400]}")
                st.stop()

        st.markdown(data["reply"])
        render_meta(data)

    conversation["messages"].append(
        {
            "role": "assistant",
            "content": data["reply"],
            "meta": data,
        }
    )

    st.rerun()
