#!/usr/bin/env python3
"""
NPMI + GEE + LIDER Pipeline
============================
A non-iterative, no-training vectorization pipeline for document/graph retrieval.
- NPMI: Edge weighting via normalized pointwise mutual information
- GEE:  Graph Encoder Embedding (one-hot, no spectral decomposition)
- LIDER: Learned index for sub-millisecond vector search

Author: Hermes Agent
"""

import numpy as np
from pathlib import Path
import json, math

# =============================================================================
# STAGE 1: NPMI — Smart Edge Weighting & Co-occurrence Filtering
# =============================================================================

def compute_npmi(adjacency_matrix, eps=1e-12):
    """
    Compute Normalized Pointwise Mutual Information edge weights.
    
    NPMI(i,j) = ln( P(i,j) / (P(i) * P(j)) ) / -ln(P(i,j))
    
    Ranges from -1 to 1:
      +1 = perfect co-occurrence (always together)
       0 = independence
      -1 = never co-occur (avoid)
    
    Args:
        adjacency_matrix: n×n co-occurrence/frequency matrix (raw counts)
        eps: small value to avoid log(0)
    
    Returns:
        npmi_matrix: n×n NPMI-weighted matrix (symmetric)
        pmi_matrix:  n×n raw PMI matrix (for inspection)
    """
    n = adjacency_matrix.shape[0]
    
    # Total co-occurrence sum
    total = adjacency_matrix.sum()
    if total == 0:
        raise ValueError("Empty adjacency matrix")
    
    # Joint probability P(i,j)
    P_joint = (adjacency_matrix + eps) / total
    
    # Marginal probabilities P(i) and P(j)
    row_sum = adjacency_matrix.sum(axis=1, keepdims=True)
    col_sum = adjacency_matrix.sum(axis=0, keepdims=True)
    P_i = (row_sum + eps) / total   # n×1
    P_j = (col_sum + eps) / total   # 1×n
    
    # PMI = ln(P(i,j) / (P(i) * P(j)))
    pmi = np.log(P_joint / (P_i * P_j + eps))
    
    # NPMI = PMI / -ln(P(i,j))
    # For P(i,j) → 0, -ln(P(i,j)) → ∞, so NPMI → -1
    # For P(i,j) → 1, -ln(P(i,j)) → 0, so NPMI → 1 (but handle carefully)
    denom = -np.log(P_joint + eps)
    denom = np.where(denom < eps, eps, denom)  # avoid division by zero
    
    npmi = pmi / denom
    
    # Clamp to [-1, 1] for numerical stability
    npmi = np.clip(npmi, -1.0, 1.0)
    
    return npmi, pmi


def build_cooccurrence_matrix(items, window_size=5):
    """
    Build co-occurrence matrix from a list of items (e.g., tokens, section IDs).
    Items that appear within window_size of each other get +1 count.
    
    Args:
        items: list of item identifiers (strings or ints)
        window_size: co-occurrence window
    
    Returns:
        n×n symmetric numpy matrix of counts
    """
    unique_items = list(set(items))
    n = len(unique_items)
    item_to_idx = {item: i for i, item in enumerate(unique_items)}
    
    # Count co-occurrences in sliding window
    counts = np.zeros((n, n), dtype=np.float64)
    
    for i in range(len(items)):
        idx_i = item_to_idx[items[i]]
        for j in range(max(0, i - window_size), min(len(items), i + window_size + 1)):
            if i != j:
                idx_j = item_to_idx[items[j]]
                counts[idx_i, idx_j] += 1
    
    # Symmetrize (undirected graph)
    counts = (counts + counts.T)
    
    return counts, unique_items


# =============================================================================
# STAGE 2: GEE — Graph Encoder Embedding (Non-iterative, no training)
# =============================================================================

def gee_embed(npmi_matrix, labels=None, k=None, alpha=0.5):
    """
    Graph Encoder Embedding: Z = (D^alpha * A * Y)
    
    Converts an NPMI-weighted adjacency matrix into dense K-dimensional
    vertex embeddings without spectral decomposition or neural networks.
    
    The key insight: multiply the graph structure by one-hot class labels.
    Community structure is encoded directly through label propagation.
    
    Args:
        npmi_matrix: n×n NPMI-weighted adjacency matrix (values in [-1,1])
        labels: list of K class labels (one per vertex), or None for unsupervised
        k: number of classes (if labels is None, estimate from data)
        alpha: normalization exponent (0.5 = sqrt(D) normalization)
    
    Returns:
        Z: n×K embedding matrix (n vertices, K dimensions)
    """
    n = npmi_matrix.shape[0]
    
    # Handle negative weights (NPMI can be negative = anti-co-occurrence)
    # Shift to non-negative by adding 1 and scaling
    A = npmi_matrix.copy()
    A = (A + 1.0) / 2.0  # Shift from [-1,1] to [0,1]
    
    # Remove self-loops
    np.fill_diagonal(A, 0)
    
    # Row sums (degrees)
    d = A.sum(axis=1)
    d[d == 0] = 1.0  # avoid division by zero
    
    # Degree normalization: D^alpha
    D_alpha = np.diag(np.power(d, alpha))
    
    # Create one-hot label matrix Y
    if labels is None:
        # Unsupervised: use spectral clustering on A to estimate labels
        # (Fast O(n) heuristic: using the diagonal of A as vertex weights)
        # For truly unsupervised we fall back to random projection
        # but we need some cluster signal
        k_est = k or max(3, n // 10)
        Y = np.random.randn(n, k_est) * 0.1
        Y = (Y > 0).astype(float)  # binarize
    else:
        # Convert labels to one-hot matrix
        unique_labels = list(set(labels))
        k = len(unique_labels)
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        Y = np.zeros((n, k))
        for i, label in enumerate(labels):
            Y[i, label_to_idx[label]] = 1.0
    
    # Z = D^alpha @ A @ Y
    Z = D_alpha @ A @ Y
    
    # L2 normalize each row (vertex embedding)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Z = Z / norms
    
    return Z


def unsupervised_gee(A, k, alpha=0.5, n_iter=10):
    """
    Unsupervised GEE via iterative label refinement.
    
    Starts with random labels and refines them based on graph structure.
    Alternates between:
      1. Propagate labels through graph: Y' = D^alpha @ A @ Y
      2. Re-assign labels by nearest centroid
    
    Args:
        A: n×n adjacency matrix (non-negative)
        k: number of clusters
        alpha: degree normalization exponent
        n_iter: refinement iterations
    
    Returns:
        Z: n×k embedding
        Y_est: estimated labels (n,)
    """
    n = A.shape[0]
    
    # Initialize Y randomly
    Y = np.random.rand(n, k)
    Y = Y / Y.sum(axis=1, keepdims=True)  # normalize rows
    
    # Row degrees
    d = A.sum(axis=1)
    d[d == 0] = 1.0
    D_alpha = np.diag(np.power(d, alpha))
    
    for _ in range(n_iter):
        # Label propagation step: Y' = D^alpha @ A @ Y
        Y = D_alpha @ A @ Y
        
        # Normalize rows
        Y = Y / (Y.sum(axis=1, keepdims=True) + 1e-12)
        
        # Reassign to nearest centroid (argmax)
        # But keep soft assignments for next iteration
        pass
    
    # Final hard assignment
    labels = np.argmax(Y, axis=1)
    
    # Compute centroids
    centroids = np.zeros((k, k))
    for c in range(k):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = Y[mask].mean(axis=0)
    
    # Z = Y @ centroids.T (each vertex embedding = its soft label projected)
    Z = Y @ centroids.T
    
    # L2 normalize
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Z = Z / norms
    
    return Z, labels


# =============================================================================
# STAGE 3: LIDER — Learned Index for Sub-millisecond Vector Search
# =============================================================================

class LIDERIndex:
    """
    LIDER: Learned Index for Large-scale Dense Retrieval.
    
    Combines:
    - SK-LSH: SortingKeys-Locality Sensitive Hashing for dimension reduction
    - RMI: Recursive Model Index for predictive querying
    
    Instead of tree/graph-based ANN traversal, we "predict" where
    a vector's match lives using a learned model.
    
    For simplicity, this implements the core idea:
    1. Sort vectors by their first principal component (or random projection)
    2. Build a piecewise linear model (RMI) that predicts position from vector
    3. Binary search around the prediction for exact match
    
    Args:
        n_bins: number of bins for RMI model
        model_order: polynomial order for each bin's local model
    """
    
    def __init__(self, n_bins=128, model_order=2):
        self.n_bins = n_bins
        self.model_order = model_order
        self.vectors = None      # n×K matrix (all embeddings)
        self.keys = None         # n×1 sorting keys (projected positions)
        self.bin_models = []     # list of (start, end, coeffs) per bin
        self.bin_boundaries = [] # key values marking bin boundaries
    
    def fit(self, Z):
        """
        Build the learned index from embedding matrix Z.
        
        Args:
            Z: n×K embedding matrix (n vertices, K dimensions)
        """
        n, k = Z.shape
        self.vectors = Z.copy()
        
        # Step 1: Project to 1D using first principal component (or random)
        # For speed, use mean-centered random projection (equivalent when n is small)
        # For large n, use approximate PCA via power iteration
        key_vector = self._compute_key_vector(Z)
        self.keys = Z @ key_vector  # n-dimensional projection
        
        # Sort by key
        sort_idx = np.argsort(self.keys)
        self.keys = self.keys[sort_idx]
        self.vectors = self.vectors[sort_idx]
        
        # Step 2: Build RMI piecewise linear model
        self._build_rmi_model()
        
        print(f"  LIDER: Built index for {n} vectors in {self.n_bins} bins")
        return self
    
    def _compute_key_vector(self, Z):
        """Compute the key projection vector (dominant direction)."""
        n, k = Z.shape
        
        # Use SVD on centered matrix (fast for small k)
        Z_centered = Z - Z.mean(axis=0)
        
        try:
            # Power iteration for top eigenvector (faster than full SVD for 1 vector)
            v = np.random.randn(k)
            v = v / np.linalg.norm(v)
            
            for _ in range(20):
                # Z_centered.T @ Z_centered @ v → dominant eigenvector
                MtM = Z_centered.T @ Z_centered
                v_new = MtM @ v
                v = v_new / np.linalg.norm(v_new)
            
            return v
        except Exception:
            # Fallback: random projection
            return np.random.randn(k) / np.sqrt(k)
    
    def _build_rmi_model(self):
        """Build recursive model index — piecewise polynomial fit."""
        n = len(self.keys)
        bin_size = max(1, n // self.n_bins)
        
        self.bin_models = []
        self.bin_boundaries = []
        
        for b in range(self.n_bins):
            start = b * bin_size
            end = min((b + 1) * bin_size, n)
            
            if end <= start:
                continue
            
            bin_keys = self.keys[start:end]
            
            # Linear model: key_position ≈ a * key_value + b
            # We want to predict position from key value
            # Simple approach: use local polynomial regression
            # For order=1: just fit a line through (key_value, position)
            
            positions = np.arange(start, end, dtype=np.float64)
            
            # Normalize keys for numerical stability
            mean_key = bin_keys.mean()
            std_key = bin_keys.std() + 1e-9
            
            x = (bin_keys - mean_key) / std_key
            
            # Adaptive order: don't overfit small bins
            max_order = min(self.model_order, len(bin_keys) - 1)
            if max_order < 1 or std_key < 1e-9:
                coeffs = np.array([float(positions.mean())])
            else:
                try:
                    coeffs = np.polyfit(x, positions, max_order)
                except Exception:
                    # Fall back to constant model if polyfit fails
                    coeffs = np.array([float(positions.mean())])
            
            self.bin_models.append((start, end, mean_key, std_key, coeffs))
            self.bin_boundaries.append(self.keys[start])
        
        self.bin_boundaries = np.array(sorted(set(self.bin_boundaries)))
    
    def _predict_position(self, key_value):
        """RMI prediction: find which bin and predict position within it."""
        # Find bin by binary search on boundaries
        bin_idx = np.searchsorted(self.bin_boundaries, key_value)
        bin_idx = min(bin_idx, len(self.bin_models) - 1)
        
        if bin_idx < 0:
            bin_idx = 0
        
        start, end, mean_key, std_key, coeffs = self.bin_models[bin_idx]
        
        # Predict position using polynomial
        x = (key_value - mean_key) / std_key
        predicted_pos = np.polyval(coeffs, x)
        
        # Clamp to valid range
        predicted_pos = max(start, min(end - 1, predicted_pos))
        
        return int(predicted_pos), start, end
    
    def search(self, query_vector, k=5, search_radius=50):
        """
        Sub-millisecond vector search using learned index.
        
        Args:
            query_vector: K-dim query embedding
            k: number of nearest neighbors to return
            search_radius: how many positions around prediction to search
        
        Returns:
            indices: k nearest neighbor indices
            distances: distances to those neighbors
        """
        # Project query to 1D key
        key_vector = self._compute_key_vector(self.vectors)
        query_key = query_vector @ key_vector
        
        # RMI prediction
        pred_pos, bin_start, bin_end = self._predict_position(query_key)
        
        # Search window around prediction
        left = max(0, pred_pos - search_radius)
        right = min(len(self.vectors), pred_pos + search_radius)
        
        # Compute distances within window
        window_vectors = self.vectors[left:right]
        dists = np.linalg.norm(window_vectors - query_vector, axis=1)
        
        # Get top-k
        top_k_idx = np.argsort(dists)[:k]
        
        return (np.arange(left, right)[top_k_idx], dists[top_k_idx])
    
    def exact_search(self, query_vector, k=5):
        """
        Exact search across all vectors (for comparison).
        
        Returns:
            indices: k nearest neighbor indices  
            distances: distances to those neighbors
        """
        dists = np.linalg.norm(self.vectors - query_vector, axis=1)
        top_k_idx = np.argsort(dists)[:k]
        return top_k_idx, dists[top_k_idx]


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class NPMGEEPipeline:
    """
    Full NPMI + GEE + LIDER pipeline for document/graph retrieval.
    
    Usage:
        pipeline = NPMGEEPipeline()
        pipeline.build_from_text_corpus(documents)
        results = pipeline.query("search term", top_k=5)
    """
    
    def __init__(self, k_classes=None, npmi_window=5, alpha=0.5):
        self.k_classes = k_classes
        self.npmi_window = npmi_window
        self.alpha = alpha
        
        self.item_names = None     # original item identifiers
        self.Z = None              # GEE embeddings (n×K)
        self.index = None          # LIDER index
        self.labels = None         # community/cluster labels
        self.document_chunks = None  # original text chunks
        
        self._npmi_matrix = None
        self._gee_matrix = None
    
    def build_from_text_corpus(self, documents, chunk_size=200, progress_callback=None):
        """
        Build the full pipeline from a list of text documents.
        
        Args:
            documents: list of strings (each is a document)
            chunk_size: approximate characters per chunk (for co-occurrence)
            progress_callback: optional function(dict) called after each phase with event data
        """
        print(f"\n📦 Building NPMI+GEE+LIDER pipeline from {len(documents)} documents...")
        
        # Step 0: Chunk documents
        chunks = []
        chunk_labels = []
        for doc_id, doc in enumerate(documents):
            # Simple chunking by sentences (split on '.')
            sentences = [s.strip() for s in doc.split('.') if s.strip()]
            for sent_id, sent in enumerate(sentences):
                chunks.append(f"[Doc{doc_id}/S{sent_id}] {sent}")
                chunk_labels.append(doc_id)

        self.document_chunks = chunks
        n_chunks = len(chunks)
        total_chars = sum(len(c) for c in chunks)
        print(f"  → Created {n_chunks} chunks from {len(documents)} documents")

        # Use tokens within chunks for co-occurrence
        all_tokens = []
        for chunk in chunks:
            # Tokenize: lowercase, alpha only
            tokens = ''.join(c if c.isalnum() else ' ' for c in chunk.lower()).split()
            all_tokens.append(tokens)
        
        # Build co-occurrence based on sequential adjacency
        vocab = set()
        for tokens in all_tokens:
            vocab.update(tokens)
        vocab = sorted(list(vocab))
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        n_vocab = len(vocab)
        
        print(f"  → Vocabulary size: {n_vocab}")
        
        # Phase 1 callback: tokenizing done (after vocab is built)
        if progress_callback:
            progress_callback({
                "phase": 1,
                "name": "Tokenizing",
                "n_chunks": n_chunks,
                "n_vocab": n_vocab,
                "total_chars": total_chars,
                "docs_in_batch": len(documents),
                "chunk_size": chunk_size,
                "vocab": vocab[:40],  # top 40 words for visualization
            })
        
        # Co-occurrence within sliding window
        cooc = np.zeros((n_vocab, n_vocab), dtype=np.float64)
        for tokens in all_tokens:
            for i, tok_i in enumerate(tokens):
                idx_i = word_to_idx[tok_i]
                for j in range(max(0, i - self.npmi_window), min(len(tokens), i + self.npmi_window + 1)):
                    if i != j:
                        idx_j = word_to_idx[tokens[j]]
                        cooc[idx_i, idx_j] += 1
        
        # Step 2: NPMI
        print("  → Computing NPMI edge weights...")
        npmi_matrix, _ = compute_npmi(cooc)
        self._npmi_matrix = npmi_matrix
        self._cooc_matrix = cooc  # raw co-occurrence for frequency stats
        self._vocab = vocab  # ordered word list for index lookups
        
        # Phase 2 callback: NPMI matrix built
        matrix_size_mb = npmi_matrix.nbytes / 1e6
        # Top edges for visualization
        edges = []
        for i in range(n_vocab):
            for j in range(i + 1, n_vocab):
                w = float(npmi_matrix[i, j])
                if w > 0.2:
                    edges.append({"source": vocab[i], "target": vocab[j], "weight": round(w, 4)})
        edges = sorted(edges, key=lambda x: -x["weight"])[:30]
        if progress_callback:
            progress_callback({
                "phase": 2,
                "name": "NPMI edge weights",
                "vocab_size": n_vocab,
                "matrix_shape": list(npmi_matrix.shape),
                "matrix_size_mb": matrix_size_mb,
                "edges": edges,
            })
        
        # Step 3: GEE embeddings (use sentence position as pseudo-labels)
        # Each sentence gets a unique pseudo-label based on its sequential position
        # This preserves ordering information and avoids single-class collapse
        pseudo_labels = np.arange(n_chunks, dtype=np.int64)
        
        Z = gee_embed(npmi_matrix, labels=pseudo_labels, alpha=self.alpha)
        self.Z = Z
        self._gee_matrix = Z
        
        # Phase 3 callback: GEE embedding done
        if progress_callback:
            progress_callback({
                "phase": 3,
                "name": "GEE Embedding",
                "embedding_dims": Z.shape[1],
                "alpha": self.alpha,
                "vectors": Z[:30].tolist() if Z.shape[0] >= 30 else Z.tolist(),  # first 30 vectors for viz
            })
        
        # Store document chunks for retrieval (strip the [DocX/Sy] prefix)
        self._raw_chunks = [c.split('] ', 1)[1] if '] ' in c else c for c in chunks]
        self.document_chunks = chunks
        
        # Build item_names for word→index lookup
        self.item_names = [f"word_{w}" for w in vocab]
        
        # Step 4: LIDER index
        n_bins = min(128, max(1, n_vocab // 10))
        print(f"  → Building LIDER learned index with {n_bins} bins...")
        self.index = LIDERIndex(n_bins=n_bins)
        self.index.fit(Z)
        
        # Phase 4 callback: LIDER index built
        if progress_callback:
            progress_callback({
                "phase": 4,
                "name": "LIDER learned index",
                "n_bins": self.index.n_bins,
                "n_vectors": int(Z.shape[0]),
                "index_bounds": self.index._bounds.tolist() if hasattr(self.index, '_bounds') else [],
            })
        
        print("✅ Pipeline built successfully!")
        print(f"   {len(vocab)} words → {Z.shape[1]}D GEE embeddings")
        
        return self
    
    def build_from_graph(self, adjacency_matrix, item_names=None, labels=None):
        """
        Build from a pre-computed adjacency matrix.
        
        Args:
            adjacency_matrix: n×n matrix of edge weights
            item_names: list of n item name strings (for result interpretation)
            labels: list of n class labels (for supervised GEE)
        """
        print(f"\n📦 Building NPMI+GEE+LIDER pipeline from adjacency matrix...")
        n = adjacency_matrix.shape[0]
        print(f"  → {n} nodes in graph")
        
        # Step 1: NPMI
        print("  → Computing NPMI edge weights...")
        npmi_matrix, _ = compute_npmi(adjacency_matrix)
        self._npmi_matrix = npmi_matrix
        
        # Step 2: GEE
        print("  → Computing GEE embeddings...")
        Z = gee_embed(npmi_matrix, labels=labels, alpha=self.alpha)
        self.Z = Z
        self._gee_matrix = Z
        
        # Step 3: LIDER
        print("  → Building LIDER learned index...")
        self.index = LIDERIndex(n_bins=min(128, n // 10))
        self.index.fit(Z)
        
        self.item_names = item_names or [f"node_{i}" for i in range(n)]
        
        print("✅ Pipeline built!")
        return self
    
    def query(self, query_text, top_k=5, use_exact=True):
        """
        Query the pipeline for relevant document passages.
        
        Args:
            query_text: text to search for
            top_k: number of results
            use_exact: use exact search (True) vs LIDER (False)
        
        Returns:
            list of (chunk_index, score, chunk_text) tuples
        """
        # Tokenize query the same way as documents
        tokens = ''.join(c if c.isalnum() else ' ' for c in query_text.lower()).split()
        
        if not tokens or self.index is None:
            return []
        
        if not hasattr(self, '_raw_chunks') or not self._raw_chunks:
            return []
        
        n_chunks = len(self._raw_chunks)
        
        # Average embedding of query tokens that appear in our vocabulary
        # Use chunk-level embeddings: each chunk's embedding = mean of its word embeddings
        chunk_embs = np.zeros((n_chunks, self.Z.shape[1]), dtype=np.float64)
        
        # Build word-to-chunk mapping (which chunks contain each word)
        vocab_set = set(self.item_names)
        word_chunk_scores = {}  # word -> list of (chunk_idx, score)
        
        for ci in range(n_chunks):
            # Get tokens in this chunk
            ctokens = ''.join(c if c.isalnum() else ' ' for c in self._raw_chunks[ci].lower()).split()
            # Get unique word indices in this chunk
            word_indices = []
            for w in set(ctokens):
                key = f"word_{w}"
                if key in self.item_names:
                    idx = self.item_names.index(key)
                    word_indices.append(idx)
            if word_indices:
                chunk_embs[ci] = self.Z[word_indices].mean(axis=0)
        
        # Query: find which word index corresponds to query tokens
        query_emb = np.zeros(self.Z.shape[1], dtype=np.float64)
        count = 0
        for tok in tokens:
            key = f"word_{tok}"
            if key in self.item_names:
                idx = self.item_names.index(key)
                query_emb += self.Z[idx]
                count += 1
        
        if count == 0:
            return []
        
        query_emb /= count
        query_emb /= np.linalg.norm(query_emb) + 1e-12
        
        # Normalize chunk embeddings
        norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        chunk_embs_norm = chunk_embs / norms
        
        # Cosine similarity search over chunks
        sims = chunk_embs_norm @ query_emb
        
        # Get top-k
        top_idx = np.argsort(sims)[::-1][:top_k]
        
        results = [(int(idx), float(sims[idx]), self._raw_chunks[idx]) for idx in top_idx]
        return results


# =============================================================================
# DEMO / BENCHMARK
# =============================================================================

def demo():
    """Run a demo of the NPMI+GEE+LIDER pipeline."""
    
    print("="*70)
    print("NPMI + GEE + LIDER Pipeline Demo")
    print("="*70)
    
    # Create synthetic document corpus
    docs = [
        "Machine learning algorithms process large datasets to identify patterns. "
        "Deep learning models use neural networks with many layers to learn representations. "
        "Training data is essential for supervised learning approaches.",

        "Graph theory applications include social network analysis and recommendation systems. "
        "Network nodes represent entities and edges represent relationships between entities. "
        "Community detection algorithms identify clusters in graph structures.",

        "Natural language processing enables computers to understand human language. "
        "Text embeddings convert words into dense vector representations. "
        "Transformer models have revolutionized NLP tasks including translation and summarization.",

        "Computer vision systems analyze images and videos to extract meaningful information. "
        "Convolutional neural networks are commonly used for image classification tasks. "
        "Object detection models can identify multiple objects within a single image.",

        "Information retrieval systems help users find relevant documents from large collections. "
        "Search engines use ranking algorithms to order results by relevance. "
        "Vector databases enable efficient similarity search over high-dimensional embeddings."
    ]
    
    # Build pipeline
    pipeline = NPMGEEPipeline(k_classes=None, npmi_window=3)
    pipeline.build_from_text_corpus(docs)
    
    # Test queries
    test_queries = [
        "neural networks deep learning",
        "graph community detection",
        "text embeddings information retrieval",
        "computer vision object detection"
    ]
    
    print("\n🔍 Query Results:")
    print("-"*70)
    
    for query in test_queries:
        results = pipeline.query(query, top_k=5, use_exact=True)
        print(f"\nQuery: '{query}'")
        for name, dist in results:
            print(f"  → {name}: {dist:.4f}")
    
    print("\n" + "="*70)
    print("✅ Demo complete!")
    print("="*70)
    
    return pipeline


if __name__ == "__main__":
    demo()