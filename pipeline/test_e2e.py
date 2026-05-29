#!/usr/bin/env python3
"""End-to-end test of GraphQ pipeline via HTTP API."""
import urllib.request, json, sys

base = "http://localhost:8766"

# 1. Health
with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
    print("HEALTH:", r.read().decode())

# 2. Parse PDF and index it
from liteparse import LiteParse
import urllib.parse

parser = LiteParse(quiet=True)
result = parser.parse('/home/sumit/.hermes/hermes-agent/docs/hermes-kanban-v1-spec.pdf')
full_text = '\n'.join(p.text for p in result.pages if p.text)
print(f"\nPDF: {len(full_text)} chars, {len(result.pages)} pages")

body = f"text_corpus={urllib.parse.quote(full_text)}".encode()
req = urllib.request.Request(f"{base}/index", data=body,
    headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        idx = json.loads(r.read())
        print("INDEX:", json.dumps(idx, indent=2))
except Exception as e:
    print("INDEX FAILED:", e)

# 3. Search
try:
    with urllib.request.urlopen(f"{base}/search?query=multi-agent%20collaboration&top_k=3", timeout=10) as r:
        sr = json.loads(r.read())
        print("\nSEARCH RESULTS:")
        for item in sr['results']:
            print(f"  [{item['score']:.4f}] {item['text'][:150]}")
except Exception as e:
    print("SEARCH FAILED:", e)