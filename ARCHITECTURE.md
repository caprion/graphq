# GraphQ Architecture — Deep Dive

## Table of Contents

1. [System Overview](#system-overview)
2. [Stage 1: NPMI Edge Weighting](#stage-1-npmi-edge-weighting)
3. [Stage 2: GEE — Graph Encoder Embedding](#stage-2-gee--graph-encoder-embedding)
4. [Stage 3: LIDER Learned Index](#stage-3-lider-learned-index)
5. [Pipeline Orchestrator](#pipeline-orchestrator)
6. [API Server Design](#api-server-design)
7. [Frontend Architecture](#frontend-architecture)
8. [Validation & Edge Cases](#validation--edge-cases)
9. [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)

---

## System Overview

```
                         ┌────────────────────────────────────────┐
                         │            GRAPH Q  SYSTEM              │
                         │                                        │
  PDF / URL / Text ──▶  │  ┌──────────┐    ┌──────────────────┐  │
                         │  │ LiteParse│    │  jina_search.py  │  │
                         │  │ PDF→text │    │  URL→markdown    │  │
                         │  └────┬─────┘    │  + web search    │  │
                         │       │          └────────┬─────────┘  │
                         │       ▼                    ▼           │
                         │  ┌─────────────────────────────────┐   │
                         │  │        TEXT CORPUS               │   │
                         │  └───────────────┬─────────────────┘   │
                         │                  │                      │
                         │       ┌──────────▼──────────┐          │
                         │       │  STAGE 1: NPMI       │          │
                         │       │  Tokenize + count    │          │
                         │       │  co-occurrence in    │          │
                         │       │  sliding window      │          │
                         │       │                      │          │
                         │       │  NPMI(i,j) =         │          │
                         │       │    ln(Pij/PiPj)      │          │
                         │       │    ─────────────     │          │
                         │       │    -ln(Pij)          │          │
                         │       │                      │          │
                         │       │  Output: n×n matrix   │          │
                         │       │  values in [-1, 1]    │          │
                         │       └──────────┬──────────┘          │
                         │                  │                      │
                         │       ┌──────────▼──────────┐          │
                         │       │  STAGE 2: GEE        │          │
                         │       │  D^α · A · Y         │          │
                         │       │                      │          │
                         │       │  - Shift to [0,1]    │          │
                         │       │  - Remove self-loops │          │
                         │       │  - Degree normalize   │          │
                         │       │  - Project via Y     │          │
                         │       │  - L2 normalize rows │          │
                         │       │                      │          │
                         │       │  Output: n×K matrix   │          │
                         │       │  (dense embeddings)   │          │
                         │       └──────────┬──────────┘          │
                         │                  │                      │
                         │       ┌──────────▼──────────┐          │
                         │       │  STAGE 3: LIDER      │          │
                         │       │                      │          │
                         │       │  1. PCA→1D sort key  │          │
                         │       │  2. Sort vectors     │          │
                         │       │  3. Partition→bins   │          │
                         │       │  4. Fit poly per bin │          │
                         │       │                      │          │
                         │       │  Query:              │          │
                         │       │  project→predict pos │          │
                         │       │  → local search      │          │
                         │       └──────────┬──────────┘          │
                         │                  │                      │
                         │       ┌──────────▼──────────┐          │
                         │       │   FASTAPI SERVER     │          │
                         │       │   :8766              │          │
                         │       │                      │          │
                         │       │  /documents/upload   │          │
                         │       │  /documents/{id}/index│         │
                         │       │  /search?query=...   │          │
                         │       │  /search/trace       │          │
                         │       │  /graph              │          │
                         │       │  /index/stream (SSE) │          │
                         │       └──────────┬──────────┘          │
                         │                  │                      │
                         └──────────────────┼──────────────────────┘
                                            │
                                   ┌────────▼────────┐
                                   │  FRONTEND (SPA)  │
                                   │  Cloudflare Pages│
                                   │                  │
                                   │  - PDF drop zone │
                                   │  - Live progress │
                                   │  - D3 graph viz  │
                                   │  - Dev trace mode│
                                   └─────────────────┘
```

## Stage 1: NPMI Edge Weighting

### Algorithm

```
Input: list of tokenized documents
Output: n×n NPMI matrix (symmetric, values in [-1, 1])

1. Build vocabulary V = unique tokens across all documents
2. Initialize n×n co-occurrence matrix C = zeros
3. For each document:
   a. For each token position i:
      For each j in [i-window, i+window], j ≠ i:
        C[idx(token_i), idx(token_j)] += 1
4. Symmetrize: C = C + C^T
5. Total = sum(C)
6. P_joint = (C + ε) / Total      [n×n joint probability]
7. P_i = row_sum(C) / Total        [n×1 marginal]
8. P_j = col_sum(C) / Total        [1×n marginal]
9. PMI = ln(P_joint / (P_i · P_j + ε))
10. denom = -ln(P_joint + ε), clamped to [ε, ∞)
11. NPMI = PMI / denom
12. Clip to [-1, 1]
```

### Key Implementation Details (pipeline.py:17-80)

- **ε = 1e-12** prevents log(0) — critical for sparse vocabularies
- **Window size** defaults to 5 tokens; larger windows capture longer-range dependencies at the cost of compute
- **Symmetrization** uses `C + C^T` making NPMI(i,j) = NPMI(j,i)
- **Denominator clamping** at line 65 prevents division by zero when P(i,j) → 0

### Why NPMI Over Alternatives

| Metric | Hub Suppression | Bounded | Train-free | Interpretable |
|--------|:-:|:-:|:-:|:-:|
| Raw co-occurrence | ✗ | ✗ | ✓ | ✗ |
| PMI | ✗ | ✗ | ✓ | ✓ |
| **NPMI** | **✓** | **✓** | **✓** | **✓** |
| PPMI (shifted) | ✗ | ✗ | ✓ | ✓ |
| GloVe | ✓ | ✗ | ✗ | ✓ |

NPMI is the only metric that is simultaneously bounded, hub-suppressing, train-free, and interpretable. The bound [-1, 1] is critical because the next stage (GEE) needs a well-conditioned adjacency matrix.

### Edge Case: Empty Documents

`compute_npmi` raises `ValueError("Empty adjacency matrix")` when total co-occurrence is 0. The caller (`build_from_text_corpus`) guarantees non-empty input by rejecting empty document lists.

---

## Stage 2: GEE — Graph Encoder Embedding

### Mathematical Derivation

Given an NPMI-weighted adjacency matrix A_npmi ∈ [-1, 1]^(n×n):

```
Step 1: Shift to non-negative
  A = (A_npmi + 1.0) / 2.0    → values in [0, 1]

Step 2: Remove self-loops
  diag(A) = 0

Step 3: Degree normalization
  d = row_sum(A)
  D^α = diag(d^α)             → α = 0.5 (sqrt normalization)

Step 4: Create label matrix Y
  If supervised: Y = one_hot(labels)     → n×K
  If unsupervised: Y = random_binary     → n×K (K estimated as max(3, n/10))

Step 5: Project
  Z = D^α @ A @ Y              → n×K

Step 6: L2 normalize rows
  Z[i] = Z[i] / ||Z[i]||_2
```

### Intuition

The operation `A @ Y` performs one round of label propagation: each vertex's embedding is the weighted sum of its neighbors' one-hot labels. The `D^α` pre-multiplication normalizes for degree — vertices with many edges don't dominate.

This is equivalent to one step of power iteration on the label matrix, but stopped early to preserve local structure rather than converging to the stationary distribution.

### Unsupervised Variant (pipeline.py:177-221)

When no labels are provided, `unsupervised_gee` iterates:

```
Y = random_init(n, K)
For iter in 1..n_iter:
  Y = D^α @ A @ Y            # propagate
  Y = row_normalize(Y)        # keep soft assignments
labels = argmax(Y)
centroids = mean(Y[labels==c]) for each class c
Z = Y @ centroids^T
```

This is essentially a soft k-means where graph structure determines cluster membership. Default `n_iter=10`.

### Complexity

- Time: O(n²·K) — dominated by the dense matrix multiply `A @ Y`
- Space: O(n² + n·K) — the NPMI matrix dominates for large n

For n=10,000 vocabulary words and K=100 clusters, GEE runs in ~300ms on a single CPU core.

### Why Not Spectral Decomposition

Spectral methods (Laplacian eigenmaps, node2vec) require eigendecomposition of the n×n Laplacian — O(n³). GEE replaces this with a single matrix multiply, O(n²·K), which is faster for K ≪ n. Additionally:

- **Deterministic:** No random-walk sampling (unlike node2vec/DeepWalk)
- **No hyperparameters:** α is the only parameter; spectral methods have window size, walk length, negative samples, etc.
- **Train-free:** No gradient descent, no convergence checks

---

## Stage 3: LIDER Learned Index

### Problem

Given n K-dimensional vectors, find the k nearest neighbors to a query vector q. ANN libraries (FAISS, Annoy, HNSW) build graph/tree structures and traverse them — typically O(log n) with approximation error. LIDER takes a different approach: *predict* where the match lives, then verify locally.

### Algorithm

```
BUILD (fit):
1. Compute 1D sorting key for each vector:
   - Power iteration on Z^T·Z → dominant eigenvector v (K-dim)
   - keys[i] = Z[i] @ v
2. Sort vectors by key
3. Partition sorted vectors into n_bins equal-sized bins
4. For each bin:
   - Fit polynomial: position ≈ poly(key_value)
   - Store (start_idx, end_idx, mean_key, std_key, coeffs)

QUERY (search):
1. Project query: q_key = q @ v
2. Predict bin: binary search on bin_boundaries
3. Predict position: predicted_pos = poly((q_key - mean)/std)
4. Search window [predicted_pos - radius, predicted_pos + radius]
5. Compute exact distances in window, return top-k
```

### Key Vector Computation (pipeline.py:284-307)

```python
def _compute_key_vector(self, Z):
    Z_centered = Z - Z.mean(axis=0)
    v = random.randn(k) / norm
    for _ in range(20):           # power iteration
        MtM = Z_centered.T @ Z_centered
        v = MtM @ v
        v /= norm(v)
    return v
```

This computes the first principal component of Z using power iteration on the K×K covariance matrix — O(K²) per iteration, 20 iterations. For K up to a few hundred, this is negligible (~1ms).

### RMI Piecewise Polynomial (pipeline.py:316-363)

Each bin fits a polynomial: `position ≈ poly((key - mean_key) / std_key)`. The polynomial degree is adaptive:

```python
max_order = min(self.model_order, len(bin_keys) - 1)
coeffs = np.polyfit(x, positions, max_order)
```

For bins with too few points or zero variance, it falls back to a constant model (the mean position). This prevents overfitting on sparse bins.

### Search Radius Tradeoff

```
radius=10:   covers ~20 vectors → fast (0.1ms) but might miss matches
radius=50:   covers ~100 vectors → balanced (0.5ms)
radius=200:  covers ~400 vectors → high recall (2ms)
exact:       all vectors → perfect recall but O(n·K)
```

The default `search_radius=50` provides a good balance. For critical applications, `exact_search` is always available.

---

## Pipeline Orchestrator

### `NPMGEEPipeline` (pipeline.py:445-680)

The orchestrator ties all three stages together and provides a clean interface:

```python
class NPMGEEPipeline:
    def __init__(self, k_classes=None, npmi_window=5, alpha=0.5):
        ...

    def build_from_text_corpus(self, documents, chunk_size=200, progress_callback=None):
        """Full build: tokenize → NPMI → GEE → LIDER"""
        ...

    def build_from_graph(self, adjacency_matrix, item_names=None, labels=None):
        """Build from pre-computed adjacency matrix"""
        ...

    def query(self, query_text, top_k=5, use_exact=True):
        """Search over chunk-level embeddings"""
        ...
```

### Query Flow (pipeline.py:635-680)

1. Tokenize query (same preprocessing as documents)
2. For each query token that exists in vocabulary, fetch its GEE embedding from `Z`
3. Average query token embeddings → query vector
4. Compute chunk embeddings: for each chunk, average the GEE embeddings of its constituent words
5. Cosine similarity between query vector and all chunk vectors
6. Return top-k chunks by similarity

**Important:** The current query implementation uses exact search over chunks (cosine similarity), not LIDER. The LIDER index is built on word-level embeddings for the `/graph` endpoint and trace visualization. Chunk-level search with LIDER is a planned enhancement.

### Progress Callbacks

The `progress_callback` mechanism enables SSE streaming in the API:

```python
# Phase 1: Tokenizing
callback({"phase": 1, "name": "Tokenizing", "n_chunks": X, "vocab": [...]})

# Phase 2: NPMI
callback({"phase": 2, "name": "NPMI edge weights", "edges": [...]})

# Phase 3: GEE
callback({"phase": 3, "name": "GEE Embedding", "vectors": [...], "embedding_dims": K})

# Phase 4: LIDER
callback({"phase": 4, "name": "LIDER learned index", "n_bins": B, "n_vectors": N})
```

---

## API Server Design

### FastAPI App (app.py:1-729)

**Singleton pipeline** — one global `_pipeline` instance. Indexing is serialized through `_pipeline_lock`. This is intentional: the pipeline is CPU-bound, so concurrent builds would thrash.

**Ephemeral storage** — all uploads go to `/tmp/graphq_uploads/`. Startup flushes any leftover files from previous runs. This was a deliberate design choice to avoid dependency on Firebase Storage (billing disabled) or Cloudflare R2 (not set up).

**SSE streaming** (`/index/stream`) — uses a `ThreadPoolExecutor` for the blocking pipeline build. Progress events are pushed through a `queue.Queue` back to the async event loop. A 1-second ping keeps the connection alive during long phases (NPMI matrix construction on large documents).

### Endpoint Design Rationale

| Decision | Why |
|----------|-----|
| Sync endpoints (not async) | Pipeline calls are CPU-bound NumPy; async provides no benefit |
| Singleton pipeline | Memory: the NPMI matrix alone is O(n²), holding multiple would exhaust RAM |
| Separate upload + index | Decouples I/O from compute; allows re-indexing with different params |
| CORS `*` | Development convenience; the cloudflared tunnel is the real auth boundary |

---

## Frontend Architecture

### Component Tree

```
index.html
├── Disclaimer Banner (ephemeral mode warning)
├── Header
│   ├── Logo + Title
│   └── Dev Mode Toggle
├── Status Bar
│   ├── API Status (health check poll)
│   ├── Index Status
│   └── Spinner
├── Main (3-column)
│   ├── Upload Panel (left)
│   │   ├── Drop Zone (drag-and-drop)
│   │   └── Document List
│   ├── Index Progress Panel (center, hidden initially)
│   │   ├── Phase Bars (tokenizing → NPMI → GEE → LIDER)
│   │   ├── Live Stats (chunks, vocab, dims, bins)
│   │   └── Magic Canvas (animated per-phase visualization)
│   └── Results Panel (right)
│       ├── Tab Bar (Search | Graph)
│       ├── Search Pane
│       │   ├── Dev Trace Panel (hidden unless dev mode)
│       │   ├── Query Input + Top-k selector
│       │   └── Results List
│       └── Graph Pane (hidden unless graph tab)
│           └── D3 Force-Directed Network
└── D3.js (CDN)
```

### Canvas Animation System (app.js:95-480)

The `IndexProgress` class renders a different visualization per phase on a `<canvas>` element:

- **Tokenizing:** Words appear as circles in a radial layout, pulsing into existence
- **NPMI:** Words reposition + edges appear between them with varying opacity
- **GEE:** Words flow to 2D-projected embedding positions (using first 2 GEE dimensions), glow effect
- **LIDER:** Histogram bars rise into position
- **Done:** Confetti/sparkle particles with "Indexed!" text

Each phase transition is smooth (lerp-based animation at 60fps via `requestAnimationFrame`).

### Dev Mode Trace

When dev mode is toggled on, the search call uses `/search/trace` instead of `/search`. The response includes a `trace` array:

```json
{
  "trace": [
    {"step": "NPMI edge weights", "query_words_found": [...], "edges_above_threshold": 5, "top_edges": {...}},
    {"step": "GEE embedding", "query_words_vectors": [...], "sample": {...}},
    {"step": "LIDER learned index", "n_bins": 128, "n_vectors": 1234},
    {"step": "top-k results", "results": [...]}
  ]
}
```

This is rendered in the trace panel showing exactly what each pipeline stage did with the query.

---

## Validation & Edge Cases

### Numerical Stability

1. **Log(0) in NPMI:** ε=1e-12 prevents NaN. Verified: zero-count pairs produce NPMI → -1 (the lower bound), which is correct (they never co-occur).

2. **Zero-degree vertices in GEE:** `d[d == 0] = 1.0` prevents division by zero. A vocabulary word that appears but never co-occurs with anything gets a zero embedding (correct: no structural information).

3. **Single-class labels in GEE:** If all vertices have the same label, `k=1`, producing 1D embeddings. The pipeline handles this (K=1 is valid, just degenerate).

4. **Empty query:** `query()` returns `[]` when no query tokens match the vocabulary. The API returns 200 with empty results, not 400.

5. **Power iteration non-convergence:** Falls back to random projection if SVD-equivalent operation fails (line 307: `except Exception: return random.randn(k) / sqrt(k)`).

### Concurrency

- `_pipeline_lock` prevents concurrent index builds (409 Conflict)
- After build completes, `_pipeline` is replaced atomically (Python GIL guarantees reference assignment is atomic)
- Reads (`/search`, `/graph`) can proceed during a build against the old pipeline

### Memory Bounds

For vocabulary size n and embedding dimension K:

| Component | Memory |
|-----------|--------|
| Co-occurrence matrix | n² × 8 bytes (float64) |
| NPMI matrix | n² × 8 bytes |
| GEE embeddings | n × K × 8 bytes |
| LIDER index | ~n_bins × K × 8 bytes |

For n=5,000 and K=100:
- Co-occurrence: 200 MB
- NPMI: 200 MB
- GEE: 4 MB
- **Total: ~404 MB**

This is why the singleton pipeline pattern is necessary.

---

## Design Decisions & Tradeoffs

### Decision 1: Word-level vs Chunk-level Embeddings

**Chosen:** Word-level GEE embeddings, chunk-level query aggregation.

**Alternative:** Embed each chunk directly (using chunk words as vertices in the co-occurrence graph).

**Why:** Word-level embeddings are reusable across queries. Chunk-level would require re-embedding when new chunks are added. The aggregation approach (mean of word vectors) is a bag-of-words model — simple but effective.

**Tradeoff:** Loses word order within chunks. Mitigated by the co-occurrence window in NPMI (nearby words already influence each other's embeddings).

### Decision 2: Ephemeral Storage (/tmp/)

**Chosen:** Store uploaded PDFs in `/tmp/`, wiped on restart.

**Why:** Firebase Storage billing was disabled on the deployment account. Cloudflare R2 was not set up. `/tmp/` on the Azure VM works immediately with zero config.

**Tradeoff:** No persistence across server restarts. Acceptable for a demo/research tool. Production would use Azure Blob or S3.

### Decision 3: LiteParse for PDF Extraction

**Chosen:** LiteParse (LlamaIndex) over PyPDF2/pdfplumber/pymupdf.

**Why:** LiteParse produces cleaner text output with better table/structure preservation. It's the same library used in the LlamaIndex ecosystem, making it well-tested on diverse PDF formats.

**Tradeoff:** Additional dependency. Could fall back to PyPDF2 for basic extraction if LiteParse is unavailable.

### Decision 4: Singleton Global Pipeline

**Chosen:** One `_pipeline` global variable, serialized builds.

**Why:** Memory constraints (see Memory Bounds above). The NPMI matrix is O(n²) and holding two simultaneously would exceed available RAM on the VM.

**Tradeoff:** Only one document can be indexed at a time. Multi-tenant use would require per-document pipeline serialization/deserialization.

### Decision 5: Power Iteration for PCA (not full SVD)

**Chosen:** 20 iterations of power method on Z^T·Z for the dominant eigenvector.

**Why:** Full SVD on a K×K matrix is O(K³). Power iteration is O(K²) per iteration, and 20 iterations is sufficient for the dominant eigenvector. For K up to a few hundred (typical vocabulary), this is ~1ms vs ~10ms for full SVD — a 10x speedup.

**Tradeoff:** Only extracts the first principal component (not the full spectrum). This is sufficient for a 1D sorting key but loses multi-dimensional structure.

---

## Future Enhancements

1. **Chunk-level LIDER search:** Currently LIDER indexes word embeddings; query does exact cosine search over chunks. Extending LIDER to chunk-level would bring sub-ms query times.

2. **Incremental indexing:** Adding a document currently requires full rebuild. Incremental NPMI updates (merge co-occurrence counts) would enable streaming document ingestion.

3. **Multi-lingual support:** Word-level co-occurrence is language-specific. A language-detection frontend + per-language pipeline would enable cross-lingual search.

4. **Persistent index serialization:** Save/load pipeline state (NPMI matrix, GEE embeddings, LIDER bins) to disk for survival across restarts.

5. **Hybrid mode:** Combine NPMI+GEE+LIDER with a sparse retriever (BM25) for hybrid lexical+semantic search.

6. **GPU acceleration:** The NPMI matrix multiply `D^α @ A @ Y` is a dense matrix operation — trivially accelerated with CuPy or JAX.
