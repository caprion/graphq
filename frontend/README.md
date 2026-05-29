---
name: graphq-browser
description: End-to-end PDF Question-Answering app. Upload a PDF → LiteParse extracts text → NPMI+GEE+LIDER pipeline builds index → semantic search returns relevant chunks. Frontend on Cloudflare Pages, backend on Azure VM (cloudflared tunnel), files stored locally on the VM filesystem (Option C).
triggers: [graphq, pdf-qa, pdf-question-answer, liteparse, document-search, cloudflare-pages]
---

# GraphQ Browser — PDF Q&A Skill

## Live URLs

- **Frontend**: `https://915fefef.graphq.pages.dev` (Cloudflare Pages — latest deploy)
- **Backend**: `https://tumor-radiation-economics-integral.trycloudflare.com` (cloudflared tunnel → Azure VM :8766)

> ⚠️ The cloudflared **quick tunnel URL changes on every restart**. For production, set up a **named tunnel** with a permanent subdomain.

## What's New (v1.1)

- **🔬 Dev Mode**: Toggle in header — enables step-by-step `/search/trace` view showing NPMI edges, GEE vectors, LIDER bins, and final results
- **🕸️ Graph View**: D3.js force-directed network of word co-occurrences — zoom, drag, click nodes to trace search paths
- **⏱️ Metrics**: Every API response now includes `elapsed_ms`, `tokens_estimate`, and `timestamp`
- **⚠️ Ephemeral Disclaimer**: Banner on every page — data is stored in `/tmp/` only, wiped on restart

---

## Architecture (Option C — Azure VM filesystem)

```
User Browser
  └── https://541723cc.graphq.pages.dev  (Cloudflare Pages, static)
          │
          │  POST /documents/upload  (PDF file)
          ▼
    Azure VM :8766 (FastAPI)
    ├── LiteParse  ────────────────────► PDF → page texts
    │                                       ↓
    │   /tmp/graphq_uploads/{id}.pdf   (local file storage)
    │                                       ↓
    ├── NPMI + GEE + LIDER  ◄─────────── text_corpus
    │
    └── GET /search?query=...  ◄──────── sub-ms results
          │
          ▼
    Results returned to browser
```

### Why Option C?
- ✅ Firebase Storage **blocked** — billing account disabled on `hermes-fd22c`
- ✅ Cloudflare R2 **not set up** yet
- ✅ Azure VM filesystem **works immediately** — no extra config needed
- ⚠️ Files exist only on the VM — server restart clears `/tmp/`

---

## What Each Part Does

| Component | Technology | Role |
|-----------|-----------|------|
| Frontend | Vanilla JS, CSS | Upload, parse, index, query, display results |
| File storage | Firebase Storage | Persist uploaded PDFs, generate signed URLs |
| Metadata | Firestore | Document registry, index stats, status tracking |
| Text extraction | LiteParse | PDF/DOCX/HTML → structured text |
| Pipeline | NPMI + GEE + LIDER | Zero-chunking semantic search index |
| Backend API | FastAPI | HTTP endpoints for all pipeline operations |
| Tunnel | cloudflared | Exposes Azure VM :8766 to the public internet |

---

## Deployment Checklist

### 1. Start the Backend (Azure VM)

```bash
# Kill any existing server on port 8766
kill $(lsof -ti:8766) 2>/dev/null

# Navigate to pipeline directory
cd /home/sumit/.hermes/skills/npmi-gee-lider

# Start the FastAPI server (background)
GRAPHQ_PORT=8766 /home/sumit/.hermes/venvs/docgraph/bin/python app.py &
echo $! > /tmp/graphq_server.pid

# Verify
curl http://localhost:8766/health
# Expected: {"status":"ready","pipeline":"ready"}
```

### 2. Start cloudflared Tunnel (so the browser can reach the backend)

```bash
# Quick tunnel (URL changes on each run — NOT for production)
# Cloudflared binary at /tmp/cloudflared

/tmp/cloudflared tunnel --url http://localhost:8766 2>&1 &
# Wait ~10 seconds, then find the URL:
# grep "trycloudflare.com" ~/.hermes/profiles/arc/home/.config/.wrangler/logs/wrangler-*.log
```

**For a permanent URL** (recommended for production):
```bash
# Requires a Cloudflare Zero Trust account + named tunnel
/tmp/cloudflared tunnel --protocol quic --token <YOUR_TUNNEL_TOKEN>
```

### 3. Update the Frontend API URL

Edit `assets/app.js` line 17:
```javascript
const API_BASE = 'https://YOUR-PERMANENT-TUNNEL-URL.trycloudflare.com';
```

### 4. Deploy to Cloudflare Pages

```bash
SKILL_DIR=/home/sumit/.hermes/skills/graphq-browser

# Copy assets to public/
cp $SKILL_DIR/assets/index.html $SKILL_DIR/assets/style.css $SKILL_DIR/assets/app.js $SKILL_DIR/public/

# Deploy
CLOUDFLARE_API_TOKEN=YOUR_CLOUDFLARE_API_TOKEN \
CLOUDFLARE_ACCOUNT_ID=721040363c7e8a4b9bbd99ec75b2e729 \
/tmp/node_modules/.bin/wrangler pages deploy $SKILL_DIR/public \
  --project-name=graphq \
  --commit-message="GraphQ PDF Q&A — v1"
```

---

## API Reference

| Method | Endpoint | Body/Params | Response |
|--------|----------|-------------|----------|
| `GET` | `/health` | — | `{"status": "ready\|not_indexed", "pipeline": "..."}` |
| `POST` | `/parse` | `multipart/form-data` with `file` | `{"num_pages": N, "total_chars": N, "pages": [{"page": N, "preview": "..."}]}` |
| `POST` | `/index` | `text_corpus` (string) + `chunk_size` (int) | `{"status": "indexed", "stats": {...}}` |
| `GET` | `/search?query=&top_k=` | — | `{"query": "...", "results": [{"idx": N, "score": 0.85, "text": "..."}]}` |
| `GET` | `/stats` | — | `{"n_chunks": N, "vocab_size": N, ...}` |

### `/search` Response Example

```json
{
  "query": "multi-agent collaboration",
  "top_k": 5,
  "results": [
    {"idx": 634, "score": 0.8532, "text": "The multi-agent gap in Hermes today..."},
    {"idx": 12,  "score": 0.8209, "text": "The profile is the agent..."},
    {"idx": 554, "score": 0.7188, "text": "SQLite queue for inter-process..."}
  ]
}
```

---

## Firebase Integration

### Service Account
Path: `/home/sumit/.config/firebase/service-account.json`

### Firestore — `graphq_indexes` Collection

Each document represents one indexed PDF:

```json
{
  "doc_id": "uuid-here",
  "original_filename": "kanban-spec.pdf",
  "storage_path": "pdfs/uuid-here.pdf",
  "chunk_size": 200,
  "status": "ready",
  "n_chunks": 461,
  "vocab_size": 2005,
  "embedding_dim": 461,
  "lider_bins": 128,
  "uploaded_at": "2026-05-28T16:00:00Z",
  "updated_at": "2026-05-28T16:01:00Z",
  "last_queried_at": "2026-05-28T17:00:00Z"
}
```

### Using the Firebase Integration Module

```python
from scripts.firebase_integration import FirestoreIndex, FirebaseStorage

creds = '/home/sumit/.config/firebase/service-account.json'
project = 'hermes-fd22c'

# Firestore — register/list/update documents
firestore = FirestoreIndex(creds, project)
docs = firestore.list_documents()
print(f'Found {len(docs)} indexed documents')

# Register before indexing
firestore.register_document(
    doc_id='my-pdf-001',
    original_filename='annual-report-2024.pdf',
    storage_path='pdfs/my-pdf-001.pdf',
    chunk_size=200,
)

# Update after index is built
firestore.update_stats(
    doc_id='my-pdf-001',
    n_chunks=461,
    vocab_size=2005,
    embedding_dim=461,
    lider_bins=128,
)

# Firebase Storage — upload PDF
storage = FirebaseStorage(creds, project)
result = storage.upload_pdf('/path/to/file.pdf', 'pdfs/my-pdf-001.pdf')
print('Uploaded:', result['download_url'])
```

---

## File Structure

```
graphq-browser/
├── SKILL.md                        ← This file
├── assets/
│   ├── index.html                 ← Main SPA
│   ├── style.css                  ← Dark theme styles
│   └── app.js                     ← Frontend logic (API calls, UI state)
├── public/                        ← Built artifacts (gitignored)
│   ├── index.html
│   ├── style.css
│   └── app.js
└── scripts/
    ├── firebase_integration.py    ← Firebase Storage + Firestore client
    └── deploy.sh                  ← One-command deploy script
```

---

## LiteParse — Supported File Types

LiteParse 2.0.1 supports:
- **PDF** — best with text-layer PDFs; OCR warnings for image-only pages
- **DOCX** — full text extraction with structure
- **HTML** — clean markdown-like output
- **Plain text** — direct pass-through

For **images, URLs, web pages** → use **Jina Reader** (`r.jina.ai/http://URL`) instead. Jina Reader handles rendered web content and images; LiteParse handles files you host.

---

## Limitations

1. **Quick tunnel URL changes** on each `cloudflared` restart — must redeploy frontend to update `API_BASE`
2. **No user authentication** — app is open to all; add Firebase Auth for multi-user
3. **No incremental indexing** — `POST /index` rebuilds the entire index from scratch
4. **Azure VM network** — the server has no inbound HTTPS/API access from outside; cloudflared tunnel is the workaround
5. **No chunk citations** — results show `chunk_N` but not the page number from the original PDF

---

## Future Improvements

1. **Named Cloudflare Tunnel** — permanent URL, no redeploy needed on restart
2. **Firebase Auth** — per-user document collections and quotas
3. **Streaming index progress** — SSE or Firestore real-time updates as index builds
4. **Gemini summarization** — pipe top chunks to Gemini API for natural-language answers
5. **Multi-document search** — index across all PDFs, search with document filtering
6. **Cloud Run deployment** — containerize `app.py`, deploy as managed service (needs Cloud Run Admin role)
