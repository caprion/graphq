# GraphQ — Zero-Chunking Document Search with NPMI + GEE + LIDER

Sub-millisecond semantic document search without neural networks, embeddings APIs, or vector databases.

**Repo:** [github.com/caprion/graphq](https://github.com/caprion/graphq) | **Live demo:** [graphq.pages.dev](https://915fefef.graphq.pages.dev) — upload a PDF, build the index, search interactively with D3 graph visualization and pipeline trace mode.


**Pipeline:** Raw text → NPMI co-occurrence graph → Graph Encoder Embedding (GEE) → LIDER learned index → sub-ms retrieval.

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  RAW     │     │  NPMI    │     │   GEE    │     │  LIDER   │
│  TEXT    │ ──▶ │  GRAPH   │ ──▶ │ EMBEDDING│ ──▶ │  INDEX   │ ──▶ query in <1ms
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                      │                │                │
                  edge weights     dense vectors    piecewise-RMI
                  NPMI ∈ [-1,1]    Z = D^α·A·Y     predicts position
```

## Why This Exists

Most semantic search stacks require: chunking strategy → embedding API → vector database → ANN index. This pipeline replaces all of that with three compute-only stages that run on commodity hardware:

- **No embedding API costs** (no OpenAI/Cohere/Jina bills)
- **No GPU needed** (pure NumPy, CPU-only)
- **No chunking hyperparameter tuning** (the graph structure captures cross-document relationships natively)
- **No ANN approximation drift** (LIDER is a learned index, not an approximate nearest-neighbor tree)

## Quick Start

### Prerequisites

```bash
pip install numpy scikit-learn scipy fastapi uvicorn liteparse python-multipart
```

### 5-Minute Demo

```python
from pipeline import NPMGEEPipeline

docs = [
    "Machine learning algorithms process large datasets to identify patterns.",
    "Graph theory applications include social network analysis.",
    "Natural language processing enables computers to understand text.",
    "Computer vision systems analyze images and videos.",
    "Information retrieval systems help users find relevant documents.",
]

pipeline = NPMGEEPipeline(npmi_window=3)
pipeline.build_from_text_corpus(docs)

results = pipeline.query("neural networks deep learning", top_k=3)
for idx, score, text in results:
    print(f"[{score:.4f}] {text[:80]}...")
```

### Start the API Server

```bash
GRAPHQ_PORT=8766 python pipeline/app.py
```

Then open `http://localhost:8766/docs` for the Swagger UI.

### Upload + Search Flow

```bash
# 1. Upload a PDF
curl -F "file=@document.pdf" http://localhost:8766/documents/upload
# → {"doc_id": "a1b2c3d4", "status": "uploaded", ...}

# 2. Build the search index
curl -X POST http://localhost:8766/documents/a1b2c3d4/index
# → {"status": "indexed", "stats": {"vocab_size": 1234, ...}}

# 3. Search
curl "http://localhost:8766/search?query=your+question&top_k=5"
# → {"results": [{"idx": 0, "score": 0.87, "text": "..."}, ...]}

# 4. Developer trace (step-by-step pipeline execution)
curl "http://localhost:8766/search/trace?query=your+question"
```

## Architecture

### Stage 1: NPMI — Edge Weighting

Normalized Pointwise Mutual Information converts raw co-occurrence counts into meaningful edge weights.

$$NPMI(i,j) = \frac{\ln\frac{P(i,j)}{P(i) \cdot P(j)}}{-\ln P(i,j)}$$

- **Range:** [-1, +1]
- **+1:** Perfect co-occurrence (always together)
- **0:** Statistical independence
- **-1:** Never co-occur (anti-correlated)

**Key property:** NPMI suppresses high-frequency hub words (like "the", "is", "and") that dominate raw co-occurrence but carry no semantic signal. It amplifies rare but meaningful word pairs.

### Stage 2: GEE — Graph Encoder Embedding

Model-free, non-iterative graph embedding — no eigen-decomposition, no gradient descent.

$$Z = D^{\alpha} \cdot A \cdot Y$$

Where:
- **A:** n×n NPMI-weighted adjacency matrix (shifted to [0,1])
- **D^α:** Degree normalization matrix (α=0.5 gives square-root normalization)
- **Y:** n×K one-hot label matrix (class assignments)
- **Z:** n×K embedding matrix (K-dimensional vectors for each vertex)

**Intuition:** Multiply the graph structure by one-hot class labels. Community structure is encoded directly through label propagation, without spectral decomposition or iterative training. K is typically the number of document chunks or word categories.

### Stage 3: LIDER — Learned Index for Dense Retrieval

Instead of tree/graph-based ANN traversal, LIDER *predicts* where a query vector's match lives.

**Components:**
1. **SK-LSH:** Projects K-dimensional embeddings to a 1D sorting key using the dominant principal component (power iteration on Z^T·Z, no full SVD)
2. **RMI (Recursive Model Index):** Partitions the sorted keys into bins and fits a piecewise polynomial model per bin. At query time: predict position → binary search locally around the prediction
3. **Search:** `O(log n_bins + search_radius)` per query instead of `O(n·K)` for brute force

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — `{"status": "ready", "pipeline": "ready"}` |
| `POST` | `/documents/upload` | Upload PDF (multipart form) → returns `doc_id` |
| `GET` | `/documents` | List all uploaded documents |
| `GET` | `/documents/{id}` | Get document metadata |
| `POST` | `/documents/{id}/index` | Build NPMI→GEE→LIDER index for a document |
| `DELETE` | `/documents/{id}` | Delete document + index |
| `GET` | `/search?query=...&top_k=5` | Search the index |
| `GET` | `/search/trace?query=...&top_k=5` | Search with full pipeline trace |
| `GET` | `/graph?limit=100` | Get co-occurrence graph for D3 visualization |
| `GET` | `/stats` | Pipeline statistics |
| `POST` | `/index/stream` | Build index with SSE progress streaming |
| `POST` | `/parse` | Parse PDF → page texts (no index) |
| `POST` | `/index` | Legacy: build index from text corpus or directory |

## Project Structure

```
graphq/
├── pipeline/
│   ├── pipeline.py        # Core: NPMI + GEE + LIDER classes
│   ├── app.py             # FastAPI server wrapping LiteParse + pipeline
│   ├── jina_search.py     # Jina Reader URL→markdown + multi-tier web search
│   ├── test_e2e.py        # End-to-end test (HTTP)
│   ├── test_e2e_api.py    # API integration test
│   └── SKILL.md           # Original design doc
├── frontend/
│   ├── index.html         # Browser UI (upload, search, D3 graph viz)
│   ├── app.js             # Frontend logic (SSE, canvas animations, D3)
│   ├── style.css          # Dark-theme styles
│   ├── deploy.sh          # One-command deploy: backend + tunnel + Cloudflare Pages
│   └── README.md          # Frontend deployment guide
├── .gitignore
└── README.md
```

## Running the Frontend

The frontend is a static HTML/JS app that talks to the FastAPI backend. It features:

- **PDF drop zone** — drag-and-drop upload
- **Live indexing progress** — animated canvas showing each pipeline phase
- **Developer mode** — toggle to see full NPMI→GEE→LIDER trace per query
- **D3 force graph** — interactive co-occurrence network visualization

```bash
# One-command deploy (starts backend + tunnel + deploys to Cloudflare Pages)
bash frontend/deploy.sh
```

See `frontend/README.md` for the full deployment guide.

## Web Search Integration

`jina_search.py` adds two capabilities:

1. **Jina Reader** — converts any URL to clean markdown: `fetch_url("https://example.com/article")`
2. **Multi-tier search** — automatic fallback chain: Tavily → Brave Search → DuckDuckGo

```python
from jina_search import fetch_url, search_web, build_from_weburls

# Fetch a URL as markdown
md = fetch_url("https://github.com/run-llama/liteparse")

# Search the web
results = search_web("Graph Encoder Embedding")

# Build a pipeline directly from URLs
pipeline = build_from_weburls(["https://example.com/doc1", "https://example.com/doc2"])
pipeline.query("your question")
```

## Mathematical Guarantees

### NPMI
- **Bounded:** NPMI ∈ [-1, 1] by construction
- **Symmetric:** NPMI(i,j) = NPMI(j,i)
- **Scale-invariant:** Insensitive to document length (unlike raw PMI)
- **Hub suppression:** High-frequency words receive NPMI close to 0 when co-occurrence is proportional to frequency

### GEE
- **No spectral decomposition:** O(n²·K) vs O(n³) for spectral methods
- **Deterministic given labels:** Same input always produces same embeddings
- **Label-consistent:** Vertices with the same label are embedded nearby
- **Degree-normalized:** α parameter controls the influence of high-degree hub nodes

### LIDER
- **Exact at bin boundaries:** Error bounded by bin width
- **Worst-case query:** O(n_bins + search_radius) — sublinear in n
- **No training:** Piecewise polynomial fit is O(n) one-time cost
- **Cache-friendly:** Sorted keys + sequential bin access = excellent CPU cache utilization

## Limitations

- **Vocabulary-based:** Only captures word-level co-occurrence, no phrase or sentence-level semantics
- **K depends on labeling:** Unsupervised mode uses random labels → weaker embeddings. Best results with pseudo-labels from document structure (pages, sections, chunks)
- **No cross-lingual:** Works per-language (co-occurrence is language-specific)
- **LIDER approximate:** search_radius trades speed vs recall; exact search is available as fallback

## Performance Characteristics

| Stage | Complexity | Typical time (10K words) |
|-------|-----------|--------------------------|
| Tokenization | O(total_chars) | <100ms |
| Co-occurrence | O(n_words · window) | ~500ms |
| NPMI matrix | O(n²) | ~200ms |
| GEE embedding | O(n²·K) | ~300ms |
| LIDER fit | O(n · n_bins) | ~100ms |
| **Query (LIDER)** | **O(log n_bins + radius)** | **<1ms** |
| Query (exact) | O(n·K) | ~5ms |

## Citation

If you use this in research:

```
@software{graphq2026,
  title = {GraphQ: Zero-Chunking Document Search with NPMI + GEE + LIDER},
  year = {2026},
  url = {https://github.com/kapsy/graphq}
}
```

## License

MIT
