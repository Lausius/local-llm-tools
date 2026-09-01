"""
code_search.py

A repo-grounding search layer for this project.

The project starts with transparent keyword matching, but for better
"how does it work?" answers we also support a lightweight local RAG-style
retrieval flow: split source files into chunks, embed them via Ollama's
embedding endpoint, store the index in a small JSON cache, and retrieve the
most relevant chunks for a question before answering.

If embeddings are unavailable or the model is offline, the code falls back to
plain keyword search so the bot remains usable.
"""

import json
import math
import os
import re
from typing import List, Optional

import requests

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCHABLE_EXTENSIONS = (".py", ".md", ".yml", ".yaml")
CONTEXT_LINES = 6   # lines of surrounding context to include around a match
MAX_SNIPPETS = 4    # cap how many snippets get sent to the model per query
INDEX_CACHE_FILE = os.path.join(os.getenv("DATA_DIR", PROJECT_DIR), "repo_code_index.json")

# Files/dirs to skip -- generated data, not source
SKIP_NAMES = {"chat_memory.json", "remembered_facts.json", "schedules.json", "__pycache__"}


def _normalize_token(token: str) -> str:
    token = token.lower().strip()
    if len(token) <= 2:
        return ""

    replacements = {
        "extraction": "extract",
        "extracting": "extract",
        "remembered": "remember",
        "remembering": "remember",
        "remembers": "remember",
        "reminder": "remember",
        "memorized": "remember",
        "memorize": "remember",
        "stored": "store",
        "persist": "store",
        "persisted": "store",
        "schedules": "schedule",
        "scheduling": "schedule",
        "scheduled": "schedule",
    }
    token = replacements.get(token, token)

    if token.endswith("ing") and len(token) > 6:
        token = token[:-3]
    if token.endswith("ed") and len(token) > 5:
        token = token[:-2]
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        token = token[:-1]
    if token.endswith("tion") and len(token) > 5:
        token = token[:-3] + "t"
    if token.endswith("ion") and len(token) > 5:
        token = token[:-2]

    return token


def _tokenize(text: str) -> set:
    tokens = set()
    normalized = text.replace("_", " ")
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized)
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9]*", normalized):
        norm = _normalize_token(raw)
        if norm:
            tokens.add(norm)
    return tokens


def _ollama_base_url() -> str:
    raw = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    if "/api/" in raw:
        return raw.rsplit("/api/", 1)[0]
    return raw.rstrip("/")


def _split_text_into_chunks(text: str, max_lines: int = 80, overlap: int = 20) -> List[dict]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks = []
    step = max_lines - overlap
    for start_idx in range(0, len(lines), step):
        window = lines[start_idx:start_idx + max_lines]
        if not window:
            continue
        text_chunk = "\n".join(window)
        start_line = start_idx + 1
        end_line = start_idx + len(window)
        chunks.append({
            "text": text_chunk,
            "start_line": start_line,
            "end_line": end_line,
        })
    return chunks


def _embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    if not texts:
        return []
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    try:
        response = requests.post(
            f"{_ollama_base_url()}/api/embed",
            json={"model": embed_model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            return None
        if embeddings and isinstance(embeddings[0], list):
            return embeddings
        return None
    except Exception:
        return None


def _iter_source_files():
    for root, dirs, files in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES and not d.startswith(".")]
        for name in files:
            if name in SKIP_NAMES:
                continue
            if name.endswith(SEARCHABLE_EXTENSIONS):
                yield os.path.join(root, name)


def _load_index() -> List[dict]:
    try:
        if not os.path.exists(INDEX_CACHE_FILE):
            return []
        with open(INDEX_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("chunks", [])
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_index(chunks: List[dict]):
    os.makedirs(os.path.dirname(INDEX_CACHE_FILE), exist_ok=True)
    with open(INDEX_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, indent=2)


def build_repo_index(force: bool = False) -> List[dict]:
    """Build a local repo index using Ollama embeddings when available.
    Always writes a cache file, even when embeddings are unavailable, so the
    repo index can be inspected and debugged without silently failing.
    """
    if not force:
        cached = _load_index()
        if cached:
            return cached

    chunks = []
    for filepath in _iter_source_files():
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue

        file_chunks = _split_text_into_chunks(text)
        if not file_chunks:
            continue

        rel_path = os.path.relpath(filepath, PROJECT_DIR)
        embedding_list = _embed_texts([c["text"] for c in file_chunks])

        for chunk, embedding in zip(file_chunks, embedding_list or [None] * len(file_chunks)):
            entry = {
                "file": rel_path,
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "text": chunk["text"],
            }
            if embedding is not None:
                entry["embedding"] = embedding
            chunks.append(entry)

    _save_index(chunks)
    return chunks


def _dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vector_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    denom = _vector_norm(a) * _vector_norm(b)
    if denom == 0:
        return 0.0
    return _dot_product(a, b) / denom


def _vector_search(query: str) -> list:
    query_embedding = _embed_texts([query])
    index = _load_index()
    if not index:
        index = build_repo_index()
    if not query_embedding or not index:
        return []

    query_vec = query_embedding[0]
    scored = []
    for chunk in index:
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list):
            continue
        score = _cosine_similarity(query_vec, embedding)
        if score <= 0:
            continue
        scored.append({
            "score": score,
            "file": chunk["file"],
            "line_number": chunk["start_line"],
            "snippet": chunk["text"],
        })

    scored.sort(key=lambda m: m["score"], reverse=True)

    results = []
    seen = set()
    for match in scored:
        key = (match["file"], match["line_number"])
        if key in seen:
            continue
        seen.add(key)
        results.append(match)
        if len(results) >= MAX_SNIPPETS:
            break
    return results


def _keyword_search(query: str) -> list:
    """Original transparent keyword search; kept as a fallback when embeddings aren't available."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored_matches = []

    for filepath in _iter_source_files():
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError:
            continue

        file_priority = 3 if filepath.endswith(".py") else 1 if filepath.endswith(".md") else 0

        for i, line in enumerate(lines):
            line_tokens = _tokenize(line)
            overlap = query_tokens & line_tokens
            if not overlap:
                continue

            start = max(0, i - CONTEXT_LINES // 2)
            end = min(len(lines), i + CONTEXT_LINES // 2 + 1)
            snippet = "".join(lines[start:end])

            def_bonus = 6 if "def " in line or "class " in line else 0
            identifier_bonus = 3 if any(token in line.lower() for token in query_tokens) and ("_" in line or "(" in line) else 0
            scored_matches.append({
                "score": len(overlap) * 5 + def_bonus + identifier_bonus + file_priority,
                "file": os.path.relpath(filepath, PROJECT_DIR),
                "line_number": i + 1,
                "snippet": snippet,
            })

    scored_matches.sort(key=lambda m: m["score"], reverse=True)

    results = []
    seen_regions = set()
    for match in scored_matches:
        region_key = (match["file"], match["line_number"] // (CONTEXT_LINES + 1))
        if region_key in seen_regions:
            continue
        seen_regions.add(region_key)
        results.append(match)
        if len(results) >= MAX_SNIPPETS:
            break

    return results


def search_codebase(query: str) -> list:
    """Search all project source files for lines relevant to the query.
    Prefers a local vector/RAG index when available, then falls back to keyword
    matching for compatibility and offline safety.
    """
    if query and query.strip():
        results = _vector_search(query)
        if results:
            return results
    return _keyword_search(query)


def format_snippets_for_prompt(matches: list) -> str:
    """Format search results as readable context to inject into a prompt."""
    if not matches:
        return "No relevant code found for this question."

    blocks = []
    for m in matches:
        blocks.append(f"--- {m['file']} (around line {m['line_number']}) ---\n{m['snippet']}")
    return "\n\n".join(blocks)