"""
code_search.py

A lightweight, keyword-based search over this project's own source files.
No embeddings or vector database needed -- the codebase is small enough that
simple keyword scoring works well, and it's fully transparent (you can read
exactly how a snippet was chosen).

Used to ground the bot's answers about its own implementation in real code,
rather than letting the model guess/hallucinate about how it works.
"""

import os
import re

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCHABLE_EXTENSIONS = (".py", ".md", ".yml", ".yaml")
CONTEXT_LINES = 6   # lines of surrounding context to include around a match
MAX_SNIPPETS = 4    # cap how many snippets get sent to the model per query

# Files/dirs to skip -- generated data, not source
SKIP_NAMES = {"chat_memory.json", "remembered_facts.json", "schedules.json", "__pycache__"}


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower()))


def _iter_source_files():
    for root, dirs, files in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES and not d.startswith(".")]
        for name in files:
            if name in SKIP_NAMES:
                continue
            if name.endswith(SEARCHABLE_EXTENSIONS):
                yield os.path.join(root, name)


def search_codebase(query: str) -> list:
    """Search all project source files for lines relevant to the query.
    Returns a list of dicts: {file, line_number, snippet}, best matches first.
    """
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

        for i, line in enumerate(lines):
            line_tokens = _tokenize(line)
            overlap = query_tokens & line_tokens
            if not overlap:
                continue

            start = max(0, i - CONTEXT_LINES // 2)
            end = min(len(lines), i + CONTEXT_LINES // 2 + 1)
            snippet = "".join(lines[start:end])

            scored_matches.append({
                "score": len(overlap),
                "file": os.path.relpath(filepath, PROJECT_DIR),
                "line_number": i + 1,
                "snippet": snippet,
            })

    # Sort by score, then take the best few, avoiding near-duplicate
    # snippets from the same file/area
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


def format_snippets_for_prompt(matches: list) -> str:
    """Format search results as readable context to inject into a prompt."""
    if not matches:
        return "No relevant code found for this question."

    blocks = []
    for m in matches:
        blocks.append(f"--- {m['file']} (around line {m['line_number']}) ---\n{m['snippet']}")
    return "\n\n".join(blocks)