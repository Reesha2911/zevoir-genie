# Zevoir Genie — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers **only** from
Zevoir Technologies website content, powered by **Ollama** (local LLM).

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | 3.10 +  |
| Ollama      | latest  |

---

## Step 1 — Install Ollama

Download from https://ollama.com/download and follow your OS installer.

---

## Step 2 — Pull required models

Open a terminal and run:

```bash
# Embedding model (for RAG retrieval)
ollama pull nomic-embed-text

# Chat model (for answer generation) — pick one you have:
ollama pull llama3
# OR:  ollama pull mistral
# OR:  ollama pull phi3
```

> **Tip:** If you use a different chat model, set the `CHAT_MODEL` env var:
> `export CHAT_MODEL=mistral`

---

## Step 3 — Install Python dependencies

```bash
cd zevoir_rag
pip install -r requirements.txt
```

---

## Step 4 — Run ingestion (build the knowledge base)

This downloads the Zevoir website pages, chunks them, embeds them, and saves
a FAISS vector index locally.

```bash
python ingest.py
```

Expected output:
```
📥  Fetching and chunking pages …
   → https://zevoirtechnologies.com.au/
   → https://zevoirtechnologies.com.au/services.html
   → https://zevoirtechnologies.com.au/contactus.html
🔢  Embedding N chunks with 'nomic-embed-text' …
✅  Index saved to 'vector_index/'
```

Only needs to be re-run when the website content changes.

---

## Step 5 — Start the backend

```bash
uvicorn app:app --reload --port 8000
```

> If port 8000 is busy: `uvicorn app:app --reload --port 8080`

---

## Step 6 — Open the UI

Navigate to http://localhost:8000 in your browser.

---

## Quick test

Try these in the chat:

| Type of question | Example |
|-----------------|---------|
| ✅ In-scope     | "What services does Zevoir offer?" |
| ✅ In-scope     | "How do I contact Zevoir?" |
| ❌ Out-of-scope  | "How do I build a neural network?" |
| ❌ Out-of-scope  | "What is the capital of France?" |

A full test question set is in `test_questions.txt`.

---

## Project structure

```
zevoir_rag/
├── app.py              ← FastAPI backend (RAG pipeline + /chat endpoint)
├── ingest.py           ← One-time ingestion script
├── requirements.txt    ← Python dependencies
├── test_questions.txt  ← 20 test questions (10 in-scope, 10 out-of-scope)
├── run_tests.py        ← sends those questions to the running server
├── tests/              ← unit tests for the keyword layer (no Ollama needed)
├── README.md           ← This file
├── static/
│   └── index.html      ← Chat UI (served by FastAPI at /)
└── vector_index/       ← Created by ingest.py (FAISS index + chunk metadata)
    ├── index.faiss
    ├── chunks.json
    └── meta.json
```

---

## Environment variables (all optional)

| Variable         | Default                       | Description               |
|-----------------|-------------------------------|---------------------------|
| `OLLAMA_BASE_URL`| `http://localhost:11434`      | Ollama server URL         |
| `EMBED_MODEL`   | `nomic-embed-text`            | Ollama embedding model    |
| `CHAT_MODEL`    | `llama3`                      | Ollama chat model         |
| `INDEX_DIR`     | `vector_index`                | Directory for FAISS index |
| `TOP_K`         | `5`                           | Chunks to retrieve        |
| `SIM_THRESHOLD` | `0.30`                        | Min cosine similarity     |

---

## API

### `POST /query` (alias: `POST /chat`)
```json
Request:  { "message": "What does Zevoir do?" }
Response: { "reply": "Zevoir Technologies…", "sources": [{"title": "…", "url": "…"}] }
```

### `GET /health`
```json
{ "status": "ok", "index": true, "ollama": true, "n_chunks": 42 }
```

---

## How a message is handled

1. An email address in the message gets a "team will contact you" reply.
2. A short message (up to 3 words) that matches the built-in keyword replies (greetings, thanks, services, and so on) gets a canned reply.
3. Everything else goes through retrieval: embed the question, find the closest website chunks in FAISS, and refuse if the best match scores below `SIM_THRESHOLD`. Otherwise `llama3` answers using only those chunks.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q                 # keyword-layer unit tests, no Ollama needed

# With Ollama running and the server started:
python run_tests.py       # sends test_questions.txt to the bot
```

`run_tests.py` reports PASS or FAIL for the out-of-scope questions (the bot should refuse) and CHECK for the in-scope ones (read the answers yourself). Model answers vary, so treat it as a smoke test.

## Known limitations
- The keyword layer is a simple lookup table. It only handles short messages, so a longer message that contains a keyword goes to retrieval instead.
- One similarity threshold (default `0.30`) is used for every question. Raising it makes the bot refuse more, and lowering it makes it answer more. Re-run the test questions after changing it.
- The index covers three pages of the Zevoir website, so questions about other pages are refused.
- There is no conversation memory, login or rate limiting. It is built for local use.
- The knowledge base is built from the public Zevoir website. Re-run `ingest.py` when the site changes.

## Milestone summary

| Milestone | What was built |
|-----------|---------------|
| **1** — RAG core         | `ingest.py`: scrape → chunk → embed → FAISS index; `app.py`: embed query → retrieve → LLM prompt → answer |
| **2** — Backend + UI     | FastAPI `/chat` endpoint; updated `static/index.html` with sources display |
| **3** — Guardrails       | Similarity threshold refusal; Ollama health check; graceful error handling; `test_questions.txt` |
