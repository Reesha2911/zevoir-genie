"""
app.py — Zevoir Genie RAG Chatbot backend (FastAPI + Ollama + FAISS)
Keeps original keyword/chatbot responses from the TodoBot,
adds full RAG pipeline for Zevoir website questions.
"""

import os
import re
import json
import logging
import traceback

import requests
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("zevoir-rag")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "nomic-embed-text")
CHAT_MODEL      = os.getenv("CHAT_MODEL",      "llama3")
INDEX_DIR       = os.getenv("INDEX_DIR",        "vector_index")
TOP_K           = int(os.getenv("TOP_K", "5"))
SIM_THRESHOLD   = float(os.getenv("SIM_THRESHOLD", "0.30"))

REFUSAL_MSG = (
    "I can only answer questions about Zevoir Technologies based on our website content.\n"
    "You can ask me about:\n"
    "• Our services and solutions\n"
    "• What Zevoir Technologies does\n"
    "• How to contact us"
)

SYSTEM_PROMPT = """You are Zevoir Genie, the official AI assistant for Zevoir Technologies.

STRICT RULES:
1. Answer ONLY using the context snippets provided below.
2. If the context does not contain the answer, say:
   "I can only answer questions about Zevoir Technologies based on our website content. You can ask about our services, solutions, or how to contact us."
3. Do NOT invent services, pricing, or capabilities not present in the context.
4. Keep answers concise and friendly.

Context:
{context}
"""

# ─── Original keyword system (preserved exactly) ──────────────────────────────
SPELLING_FIXES = {
    "u": "you", "ur": "your", "r": "are", "wanna": "want to",
    "gonna": "going to", "gotta": "got to", "kinda": "kind of",
    "lol": "haha", "ngl": "not going to lie", "tbh": "to be honest",
    "imo": "in my opinion", "idk": "i don't know", "omg": "oh my god",
    "brb": "be right back", "btw": "by the way",
    "buisness": "business", "bussiness": "business", "busines": "business",
    "busness": "business", "artifical": "artificial",
    "inteligence": "intelligence", "intellegence": "intelligence",
    "automaton": "automation", "automatiom": "automation",
    "analyitics": "analytics", "analatics": "analytics",
    "analitics": "analytics", "developement": "development",
    "developmnt": "development", "websit": "website", "webiste": "website",
    "sevices": "services", "servises": "services", "servces": "services",
    "intrested": "interested", "intersted": "interested",
    "securtiy": "security", "securty": "security",
    "clod": "cloud", "coud": "cloud",
    "thnks": "thanks", "thnk": "thanks", "thanx": "thanks",
    "byee": "bye", "byeee": "bye", "bbye": "bye", "gooodbye": "goodbye",
    "helo": "hello", "hallo": "hello", "heey": "hey", "heyy": "hey", "heyyy": "hey",
}

CHATBOT_RESPONSES = {
    ("hi", "hello", "hey", "hiya", "howdy", "hai", "hii", "hiii",
     "good morning", "good evening", "good afternoon", "good night",
     "gm", "morning", "evening", "afternoon"): (
        "Hey there! 👋 Welcome to Zevoir Technologies.\n"
        "What brings you here today?"
    ),
    ("help", "i need help", "assist", "support", "i need assistance",
     "can you help", "i need some help"): (
        "Sure! I'm here to help 😊\n"
        "You can ask me about:\n\n"
        "🤖 Our AI & Automation services\n"
        "📊 Data Analytics\n"
        "☁️ Cloud Services\n"
        "💻 Web & Mobile Development\n"
        "📞 How to contact us\n\n"
        "Or ask anything about Zevoir Technologies!"
    ),
    ("i need help with my business", "help with my business", "my business",
     "business", "my company", "for my business"): (
        "That's great! 😊 Are you looking for a website, AI solution, or something else?"
    ),
    ("ai", "maybe ai", "ai solution", "artificial intelligence",
     "i want ai", "we need ai", "i need ai"): (
        "Awesome choice 🚀 AI can really boost efficiency.\n"
        "Are you thinking about automation, chatbots, or data insights?"
    ),
    ("automation", "chatbot automation", "data insights",
     "ai automation", "i want automation"): (
        "Great choice! 💡 We've helped many businesses save time and costs with AI automation.\n"
        "Would you like to see how it works for your industry?"
    ),
    ("where is my order", "order status", "track order",
     "my order", "order update"): (
        "I can help with that! 📦\n"
        "Please enter your Order ID to check the latest status."
    ),
    ("i can't login", "cant login", "i cant login", "login issue",
     "login issues", "login problem", "login error", "can't log in",
     "cant log in", "cannot login", "can't sign in", "cant sign in",
     "sign in issue", "sign in problem", "forgot password", "reset password",
     "password issue", "password problem", "i forgot my password"): (
        "No worries! 🔐\n"
        "You can reset your password using the 'Forgot Password' option.\n"
        "Would you like me to send you the reset link?"
    ),
    ("otp", "i didn't receive otp", "no otp", "resend otp",
     "didn't get otp", "otp not received"): (
        "Sometimes OTPs take a few seconds to arrive. ⏳\n"
        "Please wait 30 seconds and try again.\n"
        "Would you like me to resend the OTP?"
    ),
    ("services", "what services do you offer", "tell me about your services",
     "what do you offer", "what services", "your services", "list services"): (
        "We help businesses grow with:\n\n"
        "🤖 AI & Automation\n"
        "📊 Data Analytics\n"
        "☁️ Cloud Services\n"
        "💻 Web & Mobile Development\n\n"
        "What are you most interested in?"
    ),
    ("web development", "web dev", "website development", "website",
     "build website", "make a website", "i need a website"): (
        "Nice 👍 Are you planning a new website or upgrading an existing one?"
    ),
    ("new website", "build a website", "create a website",
     "i want a new website"): (
        "Perfect! We can design and build a fast, modern site for you.\n"
        "Would you like a quick consultation?"
    ),
    ("can you build a chatbot", "build a chatbot", "chatbot",
     "create a chatbot", "i need a chatbot", "build me a chatbot"): (
        "Absolutely! 🤖 We create smart chatbots for customer support, automation, and sales."
    ),
    ("will it work like you", "like you", "work like you",
     "similar to you", "will it work like this"): (
        "Pretty close 😄 It can answer questions, guide users, and even capture leads."
    ),
    ("sounds good", "sounds great", "that sounds good", "nice",
     "great", "perfect", "awesome", "wow"): (
        "Great! Want me to connect you with our AI team for a demo?"
    ),
    ("i have a lot of data", "a lot of data", "data but no insights",
     "no insights", "raw data", "we have data", "tons of data"): (
        "That's where we come in 📊 We turn raw data into clear, actionable insights."
    ),
    ("how", "how?", "how do you do it", "how does it work",
     "how does this work"): (
        "We build dashboards, reports, and predictive models so you can make smarter decisions."
    ),
    ("interesting", "that's interesting", "thats interesting",
     "cool", "i see", "ok i see", "makes sense"): (
        "Want to see a sample dashboard or discuss your use case?"
    ),
    ("data analytics", "analytics", "data analysis", "data"): (
        "📊 We turn raw data into clear, actionable insights.\n"
        "We build dashboards, reports, and predictive models so you can make smarter decisions.\n\n"
        "Want to see a sample dashboard?"
    ),
    ("do you provide cloud services", "cloud services", "cloud", "aws",
     "azure", "google cloud", "cloud hosting", "i need cloud"): (
        "Yes ☁️ We help businesses move to cloud platforms like AWS, Azure, and Google Cloud."
    ),
    ("is it secure", "secure", "security", "is it safe",
     "safe", "how secure is it", "data security"): (
        "Absolutely 🔐 We focus on security, scalability, and reliability."
    ),
    ("i'm interested", "im interested", "i am interested", "interested",
     "i want to know more", "tell me more"): (
        "That's great to hear! 😊 Can I have your name and email so our team can reach out?"
    ),
    ("i'm just exploring", "im just exploring", "just exploring",
     "exploring", "just looking", "just browsing", "i'm browsing"): (
        "No problem at all 😊 Take your time. I'm here if you need anything!"
    ),
    ("what makes you different", "makes you different", "why choose you",
     "why zevoir", "what's different", "why should i choose you"): (
        "Great question! 🚀 We focus on smart solutions, fast delivery, and real business impact."
    ),
    ("demo", "i want a demo", "book a demo", "schedule demo",
     "free demo", "can i see a demo", "show me a demo"): (
        "Great! 🚀\n"
        "Please share your name, email, and company name,\n"
        "and our team will schedule a free demo for you."
    ),
    ("human", "talk to a person", "real person", "agent",
     "speak to someone", "talk to someone", "connect me to a human",
     "i want to talk to a person"): (
        "Sure 👍\n"
        "I'm connecting you with one of our support specialists.\n"
        "Please wait a moment..."
    ),
    ("thanks", "thank you", "thankyou", "thx", "ty", "thank u",
     "many thanks", "thanks a lot", "thank you so much"): (
        "You're welcome! 😊\n"
        "If you need anything else, just ask.\n"
        "Let's build something amazing together!"
    ),
    ("yes", "yeah", "yep", "yup", "ok", "okay", "no", "nope", "nah",
     "alright", "sure", "got it", "understood"): (
        "Got it! 😊\n"
        "Feel free to ask me anything about Zevoir Technologies!"
    ),
    ("bye", "goodbye", "ok bye", "okay bye", "see you", "see ya",
     "cya", "take care", "good bye", "bbye", "tata", "later", "talk later"): (
        "Thank you for visiting! 👋\n"
        "Have a wonderful day. Let's build something amazing together!"
    ),
    ("hmm", "hm", "hmmm", "hmmmm", "umm", "uhm", "uh", "uhh",
     "well", "let me think"): (
        "Take your time! 🤔 No rush at all.\n"
        "Whenever you're ready, I'm here to help.\n"
        "You can ask about our services, AI, cloud, or anything about Zevoir!"
    ),
}

HELP_RESPONSE = (
    "Sure! I'm here to help 😊\n"
    "You can ask me about:\n\n"
    "🤖 Our AI & Automation services\n"
    "📊 Data Analytics\n"
    "☁️ Cloud Services\n"
    "💻 Web & Mobile Development\n"
    "📞 How to contact us"
)


def normalize(text: str) -> str:
    text = text.lower().strip()
    words = [SPELLING_FIXES.get(w, w) for w in text.split()]
    return " ".join(words)


def check_chatbot_response(user_input: str):
    if re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', user_input):
        return (
            "Thanks! 😊 A team member will connect with you shortly.\n"
            "We'll reach out to you at the email you provided. 📧"
        )
    text = normalize(user_input)
    for keywords, response in CHATBOT_RESPONSES.items():
        if text in keywords:
            return response
    # Loose word matching only for short messages. Longer questions go to the
    # RAG pipeline; otherwise a stray "how" or "data" hijacks the whole question.
    if len(text.split()) > 3:
        return None
    for keywords, response in CHATBOT_RESPONSES.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                return response
    if "help" in text:
        return HELP_RESPONSE
    return None


# ─── FAISS index ──────────────────────────────────────────────────────────────
_index  = None
_chunks = None


def _load_index():
    global _index, _chunks
    if _index is not None:
        return True
    try:
        import faiss
        idx_path    = os.path.join(INDEX_DIR, "index.faiss")
        chunks_path = os.path.join(INDEX_DIR, "chunks.json")
        if not os.path.exists(idx_path) or not os.path.exists(chunks_path):
            log.warning("Vector index not found — run ingest.py first.")
            return False
        _index = faiss.read_index(idx_path)
        with open(chunks_path, encoding="utf-8") as f:
            _chunks = json.load(f)
        log.info(f"Loaded index: {_index.ntotal} chunks")
        return True
    except Exception as e:
        log.error(f"Failed to load index: {e}")
        return False


def _embed_query(text: str):
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as e:
        log.error(f"Embed error: {e}")
        return None


def _ollama_chat(messages: list):
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": CHAT_MODEL, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        log.error(f"Chat error: {e}")
        return None


def _retrieve(query: str, k: int = TOP_K):
    if not _load_index():
        return [], []
    vec = _embed_query(query)
    if vec is None:
        return [], []
    arr = np.array([vec], dtype="float32")
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm
    scores, indices = _index.search(arr, k)
    results, result_scores = [], []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(_chunks):
            results.append(_chunks[idx])
            result_scores.append(float(score))
    return results, result_scores


def rag_answer(query: str) -> dict:
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).raise_for_status()
    except Exception:
        return {
            "reply": "⚠️ I can't reach the AI model right now. Please make sure Ollama is running.",
            "sources": [],
        }

    chunks, scores = _retrieve(query)

    if not chunks or (scores and max(scores) < SIM_THRESHOLD):
        log.info(f"Weak retrieval → refusal")
        return {"reply": REFUSAL_MSG, "sources": []}

    context_lines = []
    sources = []
    seen_urls = set()
    for chunk, score in zip(chunks, scores):
        context_lines.append(
            f"[Source: {chunk['page_title']} | {chunk['url']}]\n{chunk['chunk_text']}"
        )
        if chunk["url"] not in seen_urls:
            sources.append({"title": chunk["page_title"], "url": chunk["url"]})
            seen_urls.add(chunk["url"])

    context  = "\n\n---\n\n".join(context_lines)
    system   = SYSTEM_PROMPT.format(context=context)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": query},
    ]
    answer = _ollama_chat(messages)

    if answer is None:
        return {
            "reply": "⚠️ The AI model didn't respond. Please try again in a moment.",
            "sources": [],
        }

    return {"reply": answer, "sources": sources}


# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Zevoir Genie RAG API")


@app.on_event("startup")
async def startup():
    _load_index()


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>UI not found</h1>", status_code=404)


@app.post("/query")
async def query_endpoint(request: Request):
    """POST /query — same endpoint name as original TodoBot."""
    try:
        body    = await request.json()
        message = (body.get("message") or "").strip()

        if not message:
            return JSONResponse({
                "reply": "Looks like you forgot to type something! 😊 Please ask me about Zevoir Technologies.",
                "sources": []
            })

        # 1. Keyword responses first (hi, bye, services, etc.)
        keyword_reply = check_chatbot_response(message)
        if keyword_reply:
            return JSONResponse({"reply": keyword_reply, "sources": []})

        # 2. RAG pipeline for website questions
        result = rag_answer(message)
        return JSONResponse(result)

    except Exception:
        log.error(traceback.format_exc())
        return JSONResponse(
            {"reply": "⚠️ Something went wrong on my end. Please try again!", "sources": []},
            status_code=200,
        )


@app.post("/chat")
async def chat_alias(request: Request):
    return await query_endpoint(request)


@app.get("/health")
async def health():
    index_ok = _load_index()
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).raise_for_status()
        ollama_ok = True
    except Exception:
        ollama_ok = False
    return {
        "status":   "ok" if (index_ok and ollama_ok) else "degraded",
        "index":    index_ok,
        "ollama":   ollama_ok,
        "n_chunks": _index.ntotal if _index else 0,
    }
