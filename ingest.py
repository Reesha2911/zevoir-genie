"""
ingest.py — Milestone 1: Ingest Zevoir website pages, chunk, embed, save FAISS index.

Run once (or anytime you want to refresh the knowledge base):
    python ingest.py
"""

import os
import json
import re
import requests
from bs4 import BeautifulSoup

# ─── Config ───────────────────────────────────────────────────────────────────
URLS = [
    "https://zevoirtechnologies.com.au/",
    "https://zevoirtechnologies.com.au/services.html",
    "https://zevoirtechnologies.com.au/contactus.html",
]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
INDEX_DIR       = os.getenv("INDEX_DIR", "vector_index")

CHUNK_SIZE      = 400   # characters
CHUNK_OVERLAP   = 80

# ─── Helpers ──────────────────────────────────────────────────────────────────

def fetch_page(url: str) -> dict:
    """Download a page and return {url, title, text}."""
    headers = {"User-Agent": "Mozilla/5.0 (ZevoirBot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove nav / footer / script / style noise
    for tag in soup(["nav", "footer", "script", "style", "noscript", "header"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else url

    # Gather visible text with heading structure preserved
    lines = []
    for el in soup.find_all(["h1","h2","h3","h4","p","li","td","th","span","div"]):
        t = el.get_text(separator=" ", strip=True)
        if t and len(t) > 10:
            lines.append(t)

    # Deduplicate consecutive identical lines
    deduped = []
    prev = None
    for line in lines:
        if line != prev:
            deduped.append(line)
        prev = line

    text = "\n".join(deduped)
    return {"url": url, "title": title, "text": text}


def chunk_text(page: dict) -> list[dict]:
    """Split page text into overlapping chunks."""
    text  = page["text"]
    url   = page["url"]
    title = page["title"]

    chunks = []
    start  = 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            # Try to find a section heading in the chunk
            heading_match = re.search(r"(?m)^(.{5,60})$", chunk)
            heading = heading_match.group(1).strip() if heading_match else ""
            chunks.append({
                "url":             url,
                "page_title":      title,
                "section_heading": heading,
                "chunk_text":      chunk,
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(text: str) -> list[float]:
    """Call Ollama embeddings endpoint."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    try:
        import faiss
        import numpy as np
    except ImportError:
        raise SystemExit("Install faiss-cpu and numpy:  pip install faiss-cpu numpy")

    os.makedirs(INDEX_DIR, exist_ok=True)

    all_chunks: list[dict] = []

    print("📥  Fetching and chunking pages …")
    for url in URLS:
        print(f"   → {url}")
        try:
            page   = fetch_page(url)
            chunks = chunk_text(page)
            all_chunks.extend(chunks)
            print(f"      {len(chunks)} chunks from '{page['title']}'")
        except Exception as e:
            print(f"   ⚠️  Could not fetch {url}: {e}")

    if not all_chunks:
        raise SystemExit("No chunks produced — check URLs and network.")

    print(f"\n🔢  Embedding {len(all_chunks)} chunks with '{EMBED_MODEL}' …")
    embeddings = []
    for i, chunk in enumerate(all_chunks, 1):
        vec = embed(chunk["chunk_text"])
        embeddings.append(vec)
        if i % 10 == 0 or i == len(all_chunks):
            print(f"   {i}/{len(all_chunks)}", end="\r")
    print()

    dim     = len(embeddings[0])
    matrix  = np.array(embeddings, dtype="float32")

    # L2-normalise for cosine similarity via inner product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix /= norms

    index = faiss.IndexFlatIP(dim)   # inner product = cosine on unit vectors
    index.add(matrix)

    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=1)

    with open(os.path.join(INDEX_DIR, "meta.json"), "w") as f:
        json.dump({"dim": dim, "n": len(all_chunks), "model": EMBED_MODEL}, f)

    print(f"\n✅  Index saved to '{INDEX_DIR}/'  ({len(all_chunks)} chunks, dim={dim})")


if __name__ == "__main__":
    main()
