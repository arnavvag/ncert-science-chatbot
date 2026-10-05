# NCERT Class 10 Science Chatbot with Smart Caching

🔴 **Live Demo:** https://ncert-science-chatbot-class10th.streamlit.app

A doubt-solving chatbot for the **current NCERT Class 10 Science textbook**.

The chatbot:
- Answers questions using only retrieved NCERT textbook content.
- Cites the chapter used for the answer.
- Politely declines questions outside the textbook.
- Supports multi-turn follow-ups within a conversation.
- Uses a **semantic cache with safety guards** so safe repeated/paraphrased questions can be answered without another LLM call.
- Displays cache status and response latency in the Streamlit UI.

> **Current LLM provider:** Groq through LangChain's OpenAI-compatible `ChatOpenAI` interface.

---

## Tech Stack

**Python · LangChain · FAISS · FastAPI · Streamlit · Groq · FastEmbed · SQLite**

### Architecture

```text
Student
   │
   ▼
Streamlit UI
   │
   ▼
FastAPI /chat
   │
   ├── Turn / context classification
   │
   ├── Semantic cache lookup
   │       │
   │       ├── HIT ──► cached reply
   │       │             (no LLM call)
   │       │
   │       └── MISS
   │
   ▼
FAISS retrieval over NCERT
   │
   ▼
Groq LLM via LangChain
   │
   ▼
Safety validation + cache store
   │
   ▼
Reply + chapter + cache status + latency
```



---

## 1. Project Layout

```text
ncert-science-chatbot/
│
├── streamlit_app.py        Streamlit frontend
│
├── app/
│   ├── api.py              FastAPI: /session, /chat, /health, /cache/stats
│   ├── service.py          Main orchestration: classify → cache → retrieve → generate
│   ├── cache.py            Semantic cache using SQLite + in-memory FAISS
│   ├── textutils.py        Cache safety guards + turn classification
│   ├── llm.py              LangChain LLM chain, citation parsing, math normalization
│   ├── retriever.py        FAISS-based NCERT retriever
│   ├── embeddings.py       FastEmbed embeddings
│   └── config.py           Environment/configuration settings
│
├── scripts/
│   ├── ingest.py           NCERT PDFs → chunks → FAISS index
│   ├── smoke_test.py       End-to-end cache validation
│   └── warm_cache.py       Optional cache pre-fill
│
├── tests/                  Offline tests
├── docs/                   Explainer and supporting files
├── data/                   NCERT PDFs (git-ignored)
├── index/                  Generated FAISS index (commit this folder)
│
├── .env.example
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 2. Run Locally

### Prerequisites

- Python **3.11 or 3.12**
- Git
- A Groq API key

### Step 1 — Get a Groq API Key

Create an API key from the Groq Console:

<https://console.groq.com/keys>

**Never commit or publicly share your API key.**

---

### Step 2 — Download the NCERT Textbook

1. Go to <https://ncert.nic.in>.
2. Open **Textbooks → Class X → Science**.
3. Download the complete current Science textbook.
4. Put the chapter PDFs inside the project's `data/` folder.

Subfolders inside `data/` are also supported.

---

### Step 3 — Create the Virtual Environment

```bash
cd ncert-science-chatbot
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

### Step 4 — Configure Environment Variables

Create `.env` from the example:

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Set the Groq configuration in `.env`:

```env
LLM_API_KEY=your_groq_api_key
LLM_MODEL=your_groq_model
LLM_BASE_URL=https://api.groq.com/openai/v1
```

The application uses `ChatOpenAI`, so Groq works through its OpenAI-compatible API.

> **Important:** Never put the API key in source code, GitHub, screenshots, README files, or other public material. Use Streamlit Secrets for deployment.

---

### Step 5 — Build the NCERT FAISS Index

First verify the chapter mapping:

```bash
python scripts/ingest.py --list
```

Then build the index:

```bash
python scripts/ingest.py
```

The first run downloads the local embedding model and creates the `index/` directory.

If a chapter is mapped incorrectly, create `data/chapters.json` using `data/chapters.json.example`, correct the mapping, and run ingestion again.

---

### Step 6 — Optional Offline Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

These tests do not require the LLM API.

---

### Step 7 — Start the Application

#### Option A — Streamlit

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

#### Option B — Run FastAPI separately

**Terminal 1:**

```powershell
.venv\Scripts\Activate.ps1
uvicorn app.api:app --port 8000
```

**Terminal 2:**

```powershell
.venv\Scripts\Activate.ps1
$env:API_URL="http://127.0.0.1:8000"
streamlit run streamlit_app.py
```

---

## 3. Verify the Smart Cache

With FastAPI running:

```bash
python scripts/smoke_test.py --api http://127.0.0.1:8000
```

The smoke test checks:
- Same doubt with different wording
- Similar wording but different question
- Different numerical values
- Reworded identical questions
- Follow-ups in the same conversation
- Follow-ups in a different conversation
- Conversation-dependent questions
- Out-of-scope questions

A successful run should show `PASS` for the applicable scenarios.

### Latest validation

The cache has been validated with real NCERT data. Recent cache-hit timings were:

```text
22 ms
42 ms
23 ms
```

These are well below the assignment requirement of **500 ms for cache hits**.

A cache hit makes **no LLM call**.

---

## 4. API

### Create a Session

```http
POST /session
```

Example:

```json
{
  "session_id": "3f2c..."
}
```

### Ask a Question

```http
POST /chat
Content-Type: application/json
```

Example request:

```json
{
  "session_id": "3f2c...",
  "message": "What is refraction?"
}
```

Example response:

```json
{
  "reply": "...",
  "citations": ["Light – Reflection and Refraction"],
  "cache_hit": false,
  "latency_ms": 1386
}
```

`latency_ms` is the measured request latency displayed by the UI.

---

## 5. Smart Cache Design

The cache is deliberately **safe first, fast second**.

A cached response is reused only when the current question is sufficiently equivalent to the cached question and the conversation context is compatible.

| Safety check | Example |
|---|---|
| Standalone or valid same-conversation follow-up | `"What about its laws?"` |
| Semantic similarity | Reject unrelated questions |
| Numbers and units must match | `R = 20 cm` vs `R = 30 cm` |
| Important terms must match | `concave` vs `convex` |
| Entity/chemical terms must match | `NaOH` vs `NaCl` |
| Direction/order-sensitive terms are protected | `ice → water` vs `water → ice` |

### Not blindly cached

The system avoids caching:
- Refusals
- Clarification prompts
- LLM errors
- `"explain more simply"` / transformation requests
- Turns dependent on an uncached previous turn
- Answers associated with an older/incompatible index

This prevents superficially similar questions from receiving the wrong cached answer.

---

## 6. Response Formatting

The LLM output is normalized before display.

Math normalization handles common textbook equations so expressions such as:

```text
V = IR
R = V / I
```

render cleanly in Streamlit instead of exposing raw LaTeX escape sequences.

---

## 7. Deploy to Streamlit Community Cloud

### A. Push to GitHub

Create a **public GitHub repository** and push the project:

```bash
git init
git add .
git status
git commit -m "NCERT Class 10 Science chatbot with smart caching"
git branch -M main
git remote add origin https://github.com/<your-username>/ncert-science-chatbot.git
git push -u origin main
```

Before committing, verify:

```text
.env       MUST NOT be committed
API keys   MUST NOT appear in the repository
index/     MUST be committed
```

---

### B. Deploy

1. Open <https://share.streamlit.io>.
2. Sign in with GitHub.
3. Create a new app from the repository.
4. Select:
   - Repository: `<your-username>/ncert-science-chatbot`
   - Branch: `main`
   - Main file: `streamlit_app.py`
5. Add the following Streamlit secrets:

```toml
LLM_API_KEY = "your_groq_api_key"
LLM_MODEL = "your_groq_model"
LLM_BASE_URL = "https://api.groq.com/openai/v1"
```

Use the same model configured in your local `.env`.

6. Deploy the app.

After deployment, test:

```text
What is refraction?
```

Then:

```text
What does refraction mean?
```

The second request should be eligible for a **Cache hit** when the first answer has been safely cached.

---

## 8. Deployment Notes

- `index/` must be committed because the deployed app needs the generated FAISS index.
- The original NCERT PDFs do not need to be committed if the generated index is included.
- API keys belong in Streamlit Secrets, never in source code.
- The local cache may start empty after a restart/redeploy; this is expected.
- The first question of a new cache entry may call the LLM; safe equivalents can then be served from cache.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `No FAISS index in .../index` | Run `python scripts/ingest.py` and commit `index/` |
| `No PDFs found` | Put the NCERT chapter PDFs inside `data/` |
| Wrong chapter mapping | Create/fix `data/chapters.json` and re-run ingestion |
| `No LLM API key found` | Set `LLM_API_KEY` in `.env` / Streamlit Secrets |
| LLM authentication error | Check the Groq API key and `LLM_BASE_URL` |
| Model not found | Check `LLM_MODEL` against the model available in Groq |
| `429` / rate-limit error | Wait for the provider limit to reset or use an account/project with sufficient quota |
| Cache hit is slow | Confirm the request is actually a cache hit; cache hits should not call the LLM |
| Valid question is rejected | Check retrieval relevance thresholds such as `MIN_RELEVANCE` |
| Raw LaTeX appears in the UI | Verify math normalization in `app/llm.py` |
| Streamlit deployment fails | Check app logs, secrets, dependencies, and that `index/` is committed |

---

## 10. Final Submission Checklist

- [ ] `python scripts/ingest.py --list` maps chapters correctly
- [ ] `python scripts/ingest.py` completes successfully
- [ ] FastAPI starts without errors
- [ ] Streamlit starts successfully
- [ ] In-scope questions return NCERT-grounded answers
- [ ] Chapter citations are displayed
- [ ] Out-of-scope questions are declined
- [ ] Same-conversation follow-ups work
- [ ] Different numerical values do not incorrectly hit the cache
- [ ] Cache hits do not call the LLM
- [ ] Cache-hit latency is below 500 ms
- [ ] Math expressions render correctly
- [ ] `.env` is not committed
- [ ] Groq credentials are stored as deployment secrets
- [ ] `index/` is committed
- [ ] GitHub repository is public
- [ ] Streamlit live URL works
- [ ] `docs/EXPLAINER.pdf` is included

---

## License / Assignment Use

This project was developed as an AI internship assignment demonstrating textbook-grounded question answering, retrieval, conversation handling, and safe semantic caching.
