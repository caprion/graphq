#!/usr/bin/env python3
"""
GraphQ API Server — End-to-End Document Q&A Pipeline
=====================================================
FastAPI server wrapping: LiteParse + NPMI + GEE + LIDER

Ephemeral Mode: All data is stored in /tmp/ and wiped on server restart.
DO NOT upload personal or sensitive documents — they are not stored permanently
but are briefly held in server memory and disk for processing.

Endpoints:
  POST /documents/upload   — Upload PDF, persist to /tmp/graphq_uploads/ (ephemeral)
  POST /documents/{id}/index — Build NPMI→GEE→LIDER index for that document
  GET  /search             — Query the index (sub-ms)
  GET  /stats              — Pipeline statistics
  GET  /graph              — Full graph data for D3 visualization
  GET  /search/trace       — Search with step-by-step trace + metrics
  GET  /health             — Health check
"""

import os, sys, json, math, uuid, time, traceback, asyncio
import threading, queue
from pathlib import Path
from typing import Optional, List, AsyncGenerator
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import uvicorn

# ── Ensure our venv packages are available ────────────────────────────────────
_site_packages = os.environ.get("GRAPHQ_VENV_SITE_PACKAGES", "")
if _site_packages and _site_packages not in sys.path:
    sys.path.insert(0, _site_packages)

# ── Import pipeline components ─────────────────────────────────────────────────
_pipeline_dir = os.environ.get("GRAPHQ_PIPELINE_DIR", os.path.dirname(os.path.abspath(__file__)))
if _pipeline_dir not in sys.path:
    sys.path.insert(0, _pipeline_dir)
from pipeline import NPMGEEPipeline, compute_npmi, gee_embed, build_cooccurrence_matrix

# ── LiteParse ─────────────────────────────────────────────────────────────────
from liteparse import LiteParse

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(title="GraphQ API", version="1.1.0",
    description="NPMI+GEE+LIDER pipeline for sub-ms document search. EPHEMERAL — data wiped on restart.")

# ── CORS — allow all origins for development ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # permissive for local dev tool
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Global pipeline instance (built on /index call)
_pipeline: Optional[NPMGEEPipeline] = None
_pipeline_lock = False

# ── Ephemeral local file storage ─────────────────────────────────────────────
UPLOAD_DIR = Path("/tmp/graphq_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Flush any leftover files from previous runs on startup
_removed = 0
for f in UPLOAD_DIR.glob("*.pdf"):
    f.unlink()
    _removed += 1
print(f"[startup] Flushed {_removed} leftover PDF(s) from /tmp/graphq_uploads/")

# In-memory document registry (ephemeral — cleared on restart)
_documents: dict = {}

# =============================================================================
# Helpers
# =============================================================================

def parse_pdf_to_text(file_path: str, max_pages: int = 200) -> List[str]:
    """Extract text from PDF using LiteParse. Returns list of page texts."""
    parser = LiteParse(quiet=True, max_pages=max_pages)
    result = parser.parse(file_path)
    pages = []
    for page in result.pages:
        if page.text and page.text.strip():
            pages.append(page.text.strip())
    return pages


def load_corpus_from_dir(dir_path: str, extensions: list = None) -> List[str]:
    """Walk a directory, extract text from all PDFs."""
    if extensions is None:
        extensions = [".pdf"]
    texts = []
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if any(fname.lower().endswith(ext) for ext in extensions):
                fpath = os.path.join(root, fname)
                try:
                    pages = parse_pdf_to_text(fpath)
                    texts.append("\n".join(pages))
                except Exception as e:
                    print(f"  [skip] {fname}: {e}")
    return texts


def pipeline_stats(p: NPMGEEPipeline) -> dict:
    """Collect readable stats from a built pipeline."""
    n_chunks = len(p.document_chunks) if p.document_chunks else 0
    vocab_size = len(p.item_names) if p.item_names else 0
    embed_dim = p.Z.shape[1] if (hasattr(p, 'Z') and p.Z is not None) else 0
    n_bins = p.index.n_bins if (hasattr(p, 'index') and p.index is not None) else 0
    return {
        "n_chunks": n_chunks,
        "vocab_size": vocab_size,
        "embedding_dim": embed_dim,
        "lider_bins": n_bins,
        "npmi_window": p.npmi_window,
        "alpha": p.alpha,
    }


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def make_metrics(start_ns: int, token_count: int = 0, extra: dict = None) -> dict:
    """Build a metrics dict from a start time (nanoseconds from time.time_ns())."""
    elapsed_ms = (time.time_ns() - start_ns) / 1e6
    m = {
        "elapsed_ms": round(elapsed_ms, 3),
        "tokens_estimate": token_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        m.update(extra)
    return m

# =============================================================================
# Endpoints (all sync def — pipeline calls are CPU-bound, not async)
# =============================================================================

@app.get("/health")
def health():
    status = "ready" if _pipeline is not None else "not_indexed"
    return {"status": status, "pipeline": status}


# ── Legacy /parse endpoint (still used internally) ────────────────────────────
@app.post("/parse")
def parse_pdf(file: UploadFile = File(...)):
    t0 = time.time_ns()
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    temp_path = f"/tmp/{file.filename}"
    content = file.file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        pages = parse_pdf_to_text(temp_path)
        total_chars = sum(len(p) for p in pages)
        return JSONResponse({
            "filename": file.filename,
            "num_pages": len(pages),
            "total_chars": total_chars,
            "tokens_estimate": estimate_tokens("".join(pages)),
            "pages": [{"page": i+1, "chars": len(p), "preview": p[:200]}
                      for i, p in enumerate(pages)],
            "metrics": make_metrics(t0, estimate_tokens("".join(pages))),
        })
    finally:
        os.unlink(temp_path)


# ── Legacy /index endpoint ─────────────────────────────────────────────────────
@app.post("/index")
def build_index(
    directory: Optional[str] = Form(None),
    text_corpus: Optional[str] = Form(None),
    chunk_size: int = Form(200),
):
    global _pipeline, _pipeline_lock
    t0 = time.time_ns()

    if _pipeline_lock:
        raise HTTPException(409, "Index build already in progress")
    _pipeline_lock = True

    try:
        if directory:
            docs = load_corpus_from_dir(directory)
        elif text_corpus:
            docs = [d.strip() for d in text_corpus.split("\n\n") if d.strip()]
        else:
            raise HTTPException(400, "Provide either 'directory' or 'text_corpus'")

        if not docs:
            raise HTTPException(400, "No documents found")

        total_chars = sum(len(d) for d in docs)
        pipeline = NPMGEEPipeline(npmi_window=5, alpha=0.5)
        pipeline.build_from_text_corpus(docs, chunk_size=chunk_size)
        _pipeline = pipeline
        stats = pipeline_stats(pipeline)

        return JSONResponse({
            "status": "indexed",
            "stats": stats,
            "metrics": make_metrics(t0, estimate_tokens(text_corpus or ""), {
                "n_docs": len(docs),
                "total_chars": total_chars,
            }),
        })
    finally:
        _pipeline_lock = False


# ── SSE Streaming /index/stream endpoint ─────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=1)

def _run_pipeline_build(docs, chunk_size, callback_queue, done_event):
    """
    Run the blocking pipeline build in a thread pool, pushing phase events
    back to the async event loop via a queue.
    """
    def progress_callback(event_dict):
        callback_queue.put(event_dict)

    pipeline = NPMGEEPipeline(npmi_window=5, alpha=0.5)
    pipeline.build_from_text_corpus(docs, chunk_size=chunk_size, progress_callback=progress_callback)
    done_event.set()  # signal that build is complete

async def _stream_index_build(
    directory: Optional[str],
    text_corpus: Optional[str],
    chunk_size: int,
    doc_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE events while the pipeline builds in a thread.
    Optionally loads document text from uploaded-documents store by doc_id.
    """
    global _pipeline, _pipeline_lock

    if _pipeline_lock:
        yield _sse_event("error", {"message": "Index build already in progress"})
        return

    _pipeline_lock = True
    callback_queue = queue.Queue()
    done_event = threading.Event()
    docs = None

    try:
        if doc_id:
            # Load from uploaded documents registry
            if doc_id not in _documents:
                yield _sse_event("error", {"message": f"Document '{doc_id}' not found — upload it first"})
                return
            meta = _documents[doc_id]
            file_path = Path(meta["file_path"])
            if not file_path.exists():
                yield _sse_event("error", {"message": "PDF file missing on server (ephemeral)"})
                return
            pages = parse_pdf_to_text(str(file_path))
            docs = pages  # each page is a "chunk"
        elif directory:
            docs = load_corpus_from_dir(directory)
        elif text_corpus:
            docs = [d.strip() for d in text_corpus.split("\n\n") if d.strip()]
        else:
            yield _sse_event("error", {"message": "Provide either 'directory', 'text_corpus', or 'doc_id'"})
            return

        if not docs:
            yield _sse_event("error", {"message": "No documents found"})
            return

        total_chars = sum(len(d) for d in docs)

        # Kick off blocking work in thread pool
        loop = asyncio.get_running_loop()
        build_future = loop.run_in_executor(
            _executor,
            lambda: _run_pipeline_build(docs, chunk_size, callback_queue, done_event)
        )

        # Ping interval tracker
        ping_interval = 1.0  # seconds
        last_ping = loop.time()

        while True:
            # Wait for callback queue events with ping timeout
            try:
                event = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: callback_queue.get(block=True, timeout=ping_interval)),
                    timeout=ping_interval + 0.5
                )
                if isinstance(event, dict):
                    pass  # normal event
                else:
                    continue
            except asyncio.TimeoutError:
                # Send a ping comment every second while waiting
                yield ": ping\n\n"
                last_ping = loop.time()
                continue

            # Got a phase event from the thread
            phase_num = event.get("phase")
            if phase_num is not None:
                yield _sse_event("phase", event)
            else:
                yield _sse_event("phase", event)

            # Check if build is done
            if phase_num == 4:
                break

        # Wait for the build thread to finish and get the pipeline
        try:
            pipeline = await asyncio.wait_for(build_future, timeout=300)
        except asyncio.TimeoutError:
            yield _sse_event("error", {"message": "Build timed out after 5 minutes"})
            return

        _pipeline = pipeline
        stats = pipeline_stats(pipeline)
        metrics = make_metrics(0, estimate_tokens(text_corpus or ""), {
            "n_docs": len(docs),
            "total_chars": total_chars,
        })

        yield _sse_event("done", {
            "status": "indexed",
            "stats": stats,
            "metrics": metrics,
        })

    finally:
        _pipeline_lock = False

def _sse_event(event_type: str, data: dict) -> str:
    """Format a dict as a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

@app.post("/index/stream")
async def build_index_stream(
    request: Request,
    directory: Optional[str] = Form(None),
    text_corpus: Optional[str] = Form(None),
    chunk_size: int = Form(200),
    doc_id: Optional[str] = Form(None),
):
    """
    StreamingResponse SSE endpoint — streams progress events as the
    NPMI→GEE→LIDER index builds in the background.

    Yields Server-Sent Events:
      event: phase — progress for each indexing phase
      event: done  — final result when complete
      event: error — if something goes wrong

    Supports three modes:
      • doc_id       — loads previously-uploaded PDF from /tmp/graphq_uploads/
      • text_corpus  — directly passes text (small docs / testing)
      • directory   — loads .txt/.md files from a server directory
    """
    # Support GET query parameter for text_corpus (matches /index signature)
    if text_corpus is None:
        text_corpus = request.query_params.get("text_corpus")

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked",
        "Content-Type": "text/event-stream; charset=utf-8",
    }

    return StreamingResponse(
        _stream_index_build(directory, text_corpus, chunk_size, doc_id=doc_id),
        media_type="text/event-stream",
        headers=headers,
    )


@app.get("/search")
def search(query: str, top_k: int = 5):
    t0 = time.time_ns()
    if _pipeline is None:
        raise HTTPException(400, "Index not built. POST /index first.")

    results = _pipeline.query(query, top_k=top_k)
    return JSONResponse({
        "query": query,
        "top_k": top_k,
        "query_tokens_estimate": estimate_tokens(query),
        "results": [
            {
                "idx": int(idx),
                "score": float(score),
                "text": str(doc_text)[:500],
                "chunk_chars": len(str(doc_text)),
            }
            for idx, score, doc_text in results
        ],
        "metrics": make_metrics(t0, estimate_tokens(query)),
    })


@app.get("/search/trace")
def search_trace(query: str, top_k: int = 5):
    """
    Developer-mode search — returns full step-by-step trace of how the
    query travels through: tokenization → NPMI edges → GEE embedding → LIDER lookup.
    """
    t0 = time.time_ns()
    if _pipeline is None:
        raise HTTPException(400, "Index not built. POST /index first.")

    p = _pipeline

    # Step 1: Tokenize the query
    import re
    raw_tokens = re.findall(r"[a-zA-Z0-9_]+", query.lower())
    p = _pipeline
    # Build word_to_idx from the pipeline's _vocab list
    word_to_idx = {w: i for i, w in enumerate(p._vocab)}
    query_tokens = [t for t in raw_tokens if t in word_to_idx]

    # Step 2: Which NPMI edges did these tokens participate in?
    trace_steps = []

    if hasattr(p, '_npmi_matrix') and p._npmi_matrix is not None:
        vocab_in_query = [t for t in query_tokens if t in word_to_idx]
        vocab_indices = [word_to_idx[t] for t in vocab_in_query]

        # Get NPMI sub-matrix for query words
        npmi_sub = {}
        for i, ti in enumerate(vocab_indices):
            for j, tj in enumerate(vocab_indices):
                if i < j and p._npmi_matrix[ti, tj] > 0.1:
                    npmi_sub[f"{vocab_in_query[i]}_{vocab_in_query[j]}"] = float(p._npmi_matrix[ti, tj])

        trace_steps.append({
            "step": "NPMI edge weights",
            "description": "Co-occurrence strength between query words in the document",
            "query_words_found": vocab_in_query,
            "edges_above_threshold": len(npmi_sub),
            "top_edges": dict(sorted(npmi_sub.items(), key=lambda x: -x[1])[:10]),
        })

    # Step 3: GEE embedding lookup for query words
    gee_vectors = {}
    if hasattr(p, 'Z') and p.Z is not None:
        from numpy.linalg import norm as np_norm
        for token in query_tokens:
            if token in word_to_idx:
                idx = word_to_idx[token]
                vec = p.Z[idx]
                # cosine similarity to a few nearest vocab words
                norms = p.Z / (np_norm(p.Z, axis=1, keepdims=True) + 1e-10)
                qvec = vec / (np_norm(vec) + 1e-10)
                sims = (norms @ qvec).flatten()
                top_idx = sims.argsort()[-5:][::-1]
                gee_vectors[token] = {
                    "embedding_dim": int(p.Z.shape[1]),
                    "nearest_words": [(p.item_names[i], round(float(sims[i]), 4))
                                      for i in top_idx if i < len(p.item_names)],
                }

        trace_steps.append({
            "step": "GEE embedding",
            "description": "Graph Embedding (GEE) vectors for each query word + nearest neighbours",
            "query_words_vectors": list(gee_vectors.keys()),
            "sample": dict(list(gee_vectors.items())[:3]),
        })

    # Step 4: LIDER lookup
    if hasattr(p, 'index'):
        trace_steps.append({
            "step": "LIDER learned index",
            "description": f"Binned {p.Z.shape[0]} vocabulary words into {p.index.n_bins} bins for sub-ms search",
            "n_bins": p.index.n_bins,
            "n_vectors": int(p.Z.shape[0]),
        })

    # Step 5: Final results
    results = _pipeline.query(query, top_k=top_k)
    trace_steps.append({
        "step": "top-k results",
        "description": f"Returned top {len(results)} chunks by composite score",
        "results": [
            {"rank": i+1, "idx": int(idx), "score": round(float(score), 6), "preview": str(doc_text)[:150]}
            for i, (idx, score, doc_text) in enumerate(results)
        ],
    })

    return JSONResponse({
        "query": query,
        "query_tokens_estimate": estimate_tokens(query),
        "pipeline_version": "NPMI+GEE+LIDER v1.0",
        "trace": trace_steps,
        "metrics": make_metrics(t0, estimate_tokens(query), {"n_steps": len(trace_steps)}),
    })


@app.get("/graph")
def get_graph(limit: int = 100):
    """
    Return the full co-occurrence graph for D3.js visualization.
    Nodes = top-N vocabulary words with GEE embeddings.
    Edges = NPMI-weighted co-occurrence (thresholded).
    """
    if _pipeline is None:
        raise HTTPException(400, "Index not built. POST /index first.")

    p = _pipeline
    if not hasattr(p, '_npmi_matrix') or p._npmi_matrix is None:
        raise HTTPException(400, "Graph not available — rebuild index first")

    # Nodes: top words by frequency
    n = min(limit, len(p._vocab))
    vocab = p._vocab[:n]
    npmi = p._npmi_matrix[:n, :n]

    nodes = []
    for i, word in enumerate(vocab):
        vec = p.Z[i] if hasattr(p, 'Z') and p.Z is not None else None
        freq = int(p._cooc_matrix[i, i]) if hasattr(p, '_cooc_matrix') else 1
        nodes.append({
            "id": word,
            "idx": int(i),
            "token": word,
            "embedding": vec.tolist() if vec is not None else [],
            "freq": freq,
        })

    # Edges: NPMI > threshold
    threshold = 0.05
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(npmi[i, j])
            if w > threshold:
                edges.append({"source": vocab[i], "target": vocab[j], "weight": round(w, 4)})

    return JSONResponse({
        "nodes": nodes,
        "edges": edges,
        "stats": pipeline_stats(p),
        "vocab_size": len(p.item_names),
    })


@app.get("/stats")
def stats():
    if _pipeline is None:
        raise HTTPException(400, "Index not built. POST /index first.")
    return JSONResponse(pipeline_stats(_pipeline))


# =============================================================================
# Document Storage (Ephemeral — /tmp/ only, wiped on restart)
# =============================================================================

@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    """
    EPHEMERAL: PDF is saved to /tmp/graphq_uploads/{doc_id}.pdf.
    Data is NOT persisted — it will be deleted when the server restarts.
    DO NOT upload personal, sensitive, or private documents.

    Flow: upload → parse → store metadata (no auto-index)
    """
    t0 = time.time_ns()
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    doc_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{doc_id}.pdf"
    content = file.file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        pages = parse_pdf_to_text(str(file_path))
        num_pages = len(pages)
        total_chars = sum(len(p) for p in pages)
        preview = pages[0][:300] if pages else ""
    except Exception:
        num_pages = 0
        total_chars = len(content)
        preview = ""
        pages = []

    meta = {
        "doc_id": doc_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "num_pages": num_pages,
        "total_chars": total_chars,
        "tokens_estimate": estimate_tokens("".join(pages)),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "file_path": str(file_path),
        "status": "uploaded",
    }
    _documents[doc_id] = meta

    return JSONResponse({
        "doc_id": doc_id,
        "filename": file.filename,
        "num_pages": num_pages,
        "total_chars": total_chars,
        "tokens_estimate": estimate_tokens("".join(pages)),
        "preview": preview,
        "uploaded_at": meta["uploaded_at"],
        "status": "uploaded",
        "disclaimer": "⚠️ EPHEMERAL — Data is stored in server RAM and /tmp/ only. Wiped on server restart. Do NOT upload personal documents.",
        "next_step": f"POST /documents/{doc_id}/index to build the search index",
        "metrics": make_metrics(t0, estimate_tokens("".join(pages))),
    })


@app.get("/documents")
def list_documents():
    docs = sorted(_documents.values(), key=lambda d: d["uploaded_at"], reverse=True)
    return JSONResponse({
        "count": len(docs),
        "ephemeral": True,
        "disclaimer": "⚠️ All data wiped on server restart",
        "documents": [
            {
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "num_pages": d["num_pages"],
                "total_chars": d["total_chars"],
                "tokens_estimate": d.get("tokens_estimate", 0),
                "uploaded_at": d["uploaded_at"],
                "status": d["status"],
            }
            for d in docs
        ]
    })


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    if doc_id not in _documents:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    d = _documents[doc_id]
    d["disclaimer"] = "⚠️ EPHEMERAL — Data is stored in server RAM and /tmp/ only. Wiped on server restart."
    return JSONResponse(d)


@app.post("/documents/{doc_id}/index")
def index_document(doc_id: str, chunk_size: int = 200):
    global _pipeline, _pipeline_lock
    t0 = time.time_ns()

    if doc_id not in _documents:
        raise HTTPException(404, f"Document '{doc_id}' not found")

    meta = _documents[doc_id]
    file_path = meta["file_path"]

    pages = parse_pdf_to_text(file_path)
    full_text = "\n".join(pages)

    if _pipeline_lock:
        raise HTTPException(409, "Index build already in progress")
    _pipeline_lock = True

    try:
        pipeline = NPMGEEPipeline(npmi_window=5, alpha=0.5)
        pipeline.build_from_text_corpus([full_text], chunk_size=chunk_size)
        _pipeline = pipeline

        meta["status"] = "indexed"
        meta["indexed_at"] = datetime.now(timezone.utc).isoformat()

        stats = pipeline_stats(pipeline)

        return JSONResponse({
            "doc_id": doc_id,
            "status": "indexed",
            "stats": stats,
            "disclaimer": "⚠️ EPHEMERAL — Index held in server RAM. Wiped on restart.",
            "metrics": make_metrics(t0, estimate_tokens(full_text), {
                "total_chars": len(full_text),
                "chunk_size": chunk_size,
            }),
        })
    finally:
        _pipeline_lock = False


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if doc_id not in _documents:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    meta = _documents.pop(doc_id)
    file_path = Path(meta["file_path"])
    if file_path.exists():
        file_path.unlink()
    return JSONResponse({
        "deleted": doc_id,
        "filename": meta["filename"],
        "note": "File removed from /tmp/graphq_uploads/",
    })


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("GRAPHQ_PORT", 8766))
    print(f"GraphQ API starting on port {port}...")
    print(f"⚠️  EPHEMERAL MODE — /tmp/graphq_uploads/ flushed on every startup")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
