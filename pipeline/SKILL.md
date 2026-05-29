---
name: npmi-gee-lider
description: |
  Zero-chunking document retrieval pipeline: NPMI edge weighting → Graph Encoder Embedding (GEE) 
  → LIDER learned index. Turns raw text/graph into sub-millisecond searchable vectors without 
  neural networks or spectral decomposition. Includes Jina Reader (URL→markdown) and multi-tier 
  search fallback (Tavily → Brave → DuckDuckGo).
  
trigger: |
  Use when user wants to:
  - Query documents without chunking (PDF, text, web pages)
  - Build fast vector search without training/embeddings
  - Extract content from URLs into a search pipeline
  - Do RAG alternative with community-structure-based retrieval
  
  NOT for: heavy neural approaches, when you have pre-trained embedding models available.
inputs:
  documents: list[str] — text documents or URLs
  query: str — search query
  mode: "text_corpus" | "from_graph" | "from_urls" — pipeline mode
  top_k: int — number of results (default 5)
outputs:
  ranked_results: list of (item, score) — retrieved items with distances
  pipeline: NPMGEEPipeline — built pipeline for reuse
  embeddings: numpy array — n×K GEE embedding matrix
setup: |
  # Install dependencies (in venv)
  python3 -m venv ~/.hermes/venvs/docgraph
  ~/.hermes/venvs/docgraph/bin/pip install numpy scikit-learn scipy pandas
  
  # Or run directly — script auto-creates venv if missing
  python3 pipeline.py
  
  # For LiteParse PDF parsing (optional):
  pip install liteparse  # or: npx @llamaindex/liteparse
  lit parse document.pdf --format text -o output.txt
  # Then feed output.txt as document to pipeline
features:
  NPMI:
    description: "Normalized Pointwise Mutual Information for edge weighting"
    formula: "NPMI(i,j) = ln(P(i,j)/(P(i)P(j))) / -ln(P(i,j))"
    range: "[-1, 1] — +1=perfect co-occur, 0=independence, -1=never co-occur"
    effect: "Suppresses high-frequency hub nodes, amplifies meaningful weak connections"
  GEE:
    description: "Graph Encoder Embedding — model-free, non-iterative"
    formula: "Z = D^α × A × Y (one-hot label propagation)"
    dims: "K dimensions where K = number of classes (document categories)"
    advantage: "No eigen-decomposition, no training, no gradient descent"
  LIDER:
    description: "Learned Index for Large-scale Dense Retrieval"
    components:
      - "SK-LSH: SortingKeys-Locality Sensitive Hashing for dimension reduction"
      - "RMI: Recursive Model Index — predicts where match lives, no ANN tree traversal"
    speed: "Sub-millisecond at billion scale on commodity hardware"
  Jina_Reader:
    description: "URL to markdown via r.jina.ai — no API key needed"
    usage: "fetch_url(url) → clean markdown text for pipeline input"
    headers: "X-Return-Format: markdown"
  Multi_Tier_Search:
    description: "Automatic fallback chain for search"
    chain: "Tavily → Brave Search → DuckDuckGo"
    vault_keys: "TAVILY_API_KEY, BRAVE_SEARCH_API_KEY (from claw-agent vault)"
env:
  VAULT_SECRET(secret_name): |
    Reads secret from Azure KeyVault `claw-agent` via managed identity.
    Uses VM MSI endpoint: http://169.254.169.254/metadata/identity/oauth2/token
    Returns secret value or None if inaccessible (read-only RBAC).
usage: |
  # 1. Text corpus mode
  from npmi_gee_lider import NPMGEEPipeline
  pipeline = NPMGEEPipeline()
  pipeline.build_from_text_corpus(["doc1 text...", "doc2 text..."])
  results = pipeline.query("search term", top_k=5)
  
  # 2. URL mode (Jina Reader)
  from jina_search import build_from_weburls
  pipeline = build_from_weburls(["https://github.com/run-llama/liteparse"])
  results = pipeline.query("PDF parsing")
  
  # 3. PDF mode (LiteParse → pipeline)
  # Run: lit parse document.pdf --format text -o doc.txt
  # Then feed doc.txt to pipeline above
  
  # 4. Search web
  from jina_search import search_web
  results = search_web("Graph Encoder Embedding", provider_priority=["tavily","brave","duckduckgo"])
  
  # 5. Direct CLI
  python3 npmi_gee_lider/pipeline.py  # runs demo with 5 synthetic documents
  python3 jina_search.py  # runs Jina + search demos
integration: |
  To add a new search provider (e.g., SerpAPI, Serper):
  1. Implement function: search_serpapi(query, api_key, max_results) → list[dict]
  2. Add to provider_priority in search_web(): ["tavily", "serpapi", "brave", "duckduckgo"]
  3. Get API key, store in vault with: TAVILY_API_KEY, BRAVE_SEARCH_API_KEY, SERPAPI_KEY
  
  To add Tavily:
  - Get key at https://app.tavily.com → API Keys
  - Store in vault (requires vault write access — currently read-only on this VM)
pitfalls:
  - NPMI on empty/sparse matrix → all -1 (anti-correlations everywhere)
  - GEE unsupervised needs ~n/k > 10 ratio for meaningful clusters  
  - LIDER bins < 16 → poor RMI predictions, increase n_bins
  - Jina Reader fails on some sites (Cloudflare block) → try Brave or direct HTML scrape
  - Tavily key at capacity → check usage at app.tavily.com, fallback to Brave
verification: |
  # Run the demo
  /home/sumit/.hermes/venvs/docgraph/bin/python /home/sumit/.hermes/skills/npmi-gee-lider/pipeline.py
  
  # Expected output:
  # - 120 words → 5D embeddings (one per document category)
  # - Query "neural networks deep learning" returns word_clusters etc.
  # - Query "graph community detection" returns word_applications, word_analysis
  # - Distances should be ascending (lower = better match)
  
  # Test Jina
  /home/sumit/.hermes/venvs/docgraph/bin/python /home/sumit/.hermes/skills/npmi-gee-lider/jina_search.py
  
  # Verify vault access (read existing secrets)
  python3 -c "from jina_search import get_vault_secret; print(get_vault_secret('TAVILY_API_KEY'))"
files:
  - pipeline.py: Core NPMI+GEE+LIDER implementation + NPMGEEPipeline class
  - jina_search.py: Jina Reader + multi-tier search + vault helpers
  - SKILL.md: This file
---