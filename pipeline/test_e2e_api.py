#!/usr/bin/env python3
"""Index the Hermes Kanban PDF and run query tests via HTTP API."""
import urllib.request, json, urllib.parse

BASE = "http://localhost:8766"

def api_get(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as r:
        return json.loads(r.read())

def api_post_form(path, data_dict):
    body = urllib.parse.urlencode(data_dict).encode()
    req = urllib.request.Request(BASE + path, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

# ── 1. Parse PDF ────────────────────────────────────────────────────────────
print("=== PARSE ===")
from liteparse import LiteParse
parser = LiteParse(quiet=True)
result = parser.parse('/home/sumit/.hermes/hermes-agent/docs/hermes-kanban-v1-spec.pdf')
full_text = '\n'.join(p.text for p in result.pages if p.text)
print(f"PDF: {len(full_text)} chars, {len(result.pages)} pages")

# ── 2. Index ────────────────────────────────────────────────────────────────
print("\n=== INDEXING ===")
idx = api_post_form("/index", {"text_corpus": full_text})
print(json.dumps(idx, indent=2))

# ── 3. Run queries ──────────────────────────────────────────────────────────
print("\n=== QUERIES ===")
queries = [
    ("multi-agent collaboration", 3),
    ("kanban board", 3),
    ("sqlite queue", 3),
    ("hermes agent coordination", 3),
    ("what is the kanban architecture", 5),
]

for query, top_k in queries:
    try:
        r = api_get(f"/search?query={urllib.parse.quote(query)}&top_k={top_k}")
        print(f"\nQ: {query}")
        for item in r.get('results', []):
            score = item.get('score', 0)
            text = item.get('text', item.get('chunk_text', ''))[:200]
            print(f"  [{score:.4f}] {text}")
    except Exception as e:
        print(f"\nQ: {query} → ERROR: {e}")

print("\n=== DONE ===")