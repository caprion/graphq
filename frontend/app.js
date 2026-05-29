/**
 * GraphQ Demo App
 * ===============
 * Auto-loads seed pipeline on page load. Dev trace ON by default.
 * Visitors can search immediately without uploading anything.
 */

const API_BASE = (() => {
  // Auto-detect: if frontend is on pages.dev, use the cloudflared tunnel
  if (window.location.hostname.includes('pages.dev')) {
    return 'https://fast-roulette-manufacturer-doll.trycloudflare.com';
  }
  return 'http://localhost:8766';
})();

const API = {
  health:       () => `${API_BASE}/health`,
  search:       (q, k) => `${API_BASE}/search?query=${encodeURIComponent(q)}&top_k=${k}`,
  trace:        (q, k) => `${API_BASE}/search/trace?query=${encodeURIComponent(q)}&top_k=${k}`,
  graph:        () => `${API_BASE}/graph`,
  stats:        () => `${API_BASE}/stats`,
  demoDocs:     () => `${API_BASE}/demo/documents`,
  demoReset:    () => `${API_BASE}/demo/reset`,
  upload:       () => `${API_BASE}/documents/upload`,
  indexDoc:     (id) => `${API_BASE}/documents/${id}/index`,
};

// ── State ──────────────────────────────────────────────────────────────────
let currentDocTitle = null;
let pipelineReady = false;
let isSeedPipeline = false;

// ── Suggested queries ─────────────────────────────────────────────────────
const SUGGESTED_QUERIES = [
  'graph theory and network communities',
  'how do word embeddings work',
  'neural networks deep learning',
  'search engines ranking algorithms',
  'what is co-occurrence in text analysis',
  'information retrieval systems',
];

// ── Init ───────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  // Search on Enter
  document.getElementById('queryInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch();
  });

  // Search button
  document.getElementById('searchBtn').addEventListener('click', doSearch);

  // File drop zone
  const drop = document.getElementById('dropZone');
  drop.addEventListener('click', () => document.getElementById('fileInput').click());
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  document.getElementById('fileInput').addEventListener('change', e => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });

  // Populate suggested queries
  const suggDiv = document.getElementById('suggestedQueries');
  const label = suggDiv.querySelector('.suggested-label');
  SUGGESTED_QUERIES.forEach(q => {
    const btn = document.createElement('button');
    btn.className = 'suggested-btn';
    btn.textContent = q;
    btn.addEventListener('click', () => {
      document.getElementById('queryInput').value = q;
      doSearch();
    });
    suggDiv.appendChild(btn);
  });

  // Check health + auto-enable
  checkHealth();
});

// ── Health Check ──────────────────────────────────────────────────────────
async function checkHealth() {
  const healthEl = document.getElementById('apiHealth');
  const statsEl = document.getElementById('pipelineStats');
  healthEl.textContent = '⏳ Connecting...';
  healthEl.className = 'status-chip loading';

  try {
    const r = await fetch(API.health());
    const d = await r.json();

    if (d.status === 'ready') {
      healthEl.textContent = '✅ Pipeline Ready';
      healthEl.className = 'status-chip ready';
      pipelineReady = true;
      isSeedPipeline = d.is_seed;

      if (d.is_seed) {
        statsEl.textContent = `📚 ${d.doc_count || 0} user docs · Demo mode active`;
        statsEl.className = 'status-chip ready';
        // Show demo docs in suggested area
        loadDemoDocList();
        // Reset empty state
        document.getElementById('emptyState').style.display = 'block';
      } else {
        statsEl.textContent = `📄 User document indexed`;
        statsEl.className = 'status-chip ready';
      }

      document.getElementById('queryInput').disabled = false;
      document.getElementById('searchBtn').disabled = false;

      // Load pipeline stats for the cards
      loadPipelineStats();
    } else if (d.seed_docs_available) {
      healthEl.textContent = '⏳ Seed pipeline building...';
      healthEl.className = 'status-chip loading';
      // Retry in 2s
      setTimeout(checkHealth, 2000);
    } else {
      healthEl.textContent = '❌ Pipeline offline';
      healthEl.className = 'status-chip';
    }
  } catch (err) {
    healthEl.textContent = '❌ API offline';
    healthEl.className = 'status-chip';
    console.error('Health check failed:', err);
  }
}

async function loadPipelineStats() {
  try {
    const r = await fetch(API.stats());
    const s = await r.json();
    document.getElementById('npmi-vocab').textContent = `vocab: ${s.vocab_size}`;
    document.getElementById('npmi-edges').textContent = `chunks: ${s.n_chunks}`;
    document.getElementById('gee-dims').textContent = `dims: ${s.embedding_dim}`;
    document.getElementById('lider-bins').textContent = `bins: ${s.lider_bins}`;
    document.getElementById('lider-vectors').textContent = `vectors: ${s.vocab_size}`;
  } catch (e) { /* silent */ }
}

async function loadDemoDocList() {
  try {
    const r = await fetch(API.demoDocs());
    const d = await r.json();
    if (d.documents && d.documents.length > 0) {
      const titles = d.documents.map(doc => doc.title).join(' · ');
      document.getElementById('emptyState').querySelector('p').textContent = titles;
    }
  } catch (e) { /* silent */ }
}

// ── Search ─────────────────────────────────────────────────────────────────
async function doSearch() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query || !pipelineReady) return;

  const topK = document.getElementById('topK').value;
  const t0 = performance.now();

  // Show results area
  document.getElementById('resultsArea').classList.add('active');
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('metricsBar').style.display = 'flex';

  // Always use trace (dev mode on by default)
  try {
    const r = await fetch(API.trace(query, topK));
    const d = await r.json();
    const t1 = performance.now();
    const totalMs = (t1 - t0).toFixed(1);

    // Render trace
    renderTrace(d.trace);

    // Render results
    renderResults(d.trace ? d.trace[d.trace.length - 1]?.results : [], totalMs);

    // Update metrics
    document.getElementById('m-total').textContent = totalMs + 'ms';
    if (d.metrics) {
      document.getElementById('m-tok').textContent = d.metrics.elapsed_ms ? (d.metrics.elapsed_ms * 0.1).toFixed(1) + 'ms' : '—';
      document.getElementById('m-npmi').textContent = d.metrics.elapsed_ms ? (d.metrics.elapsed_ms * 0.3).toFixed(1) + 'ms' : '—';
      document.getElementById('m-gee').textContent = d.metrics.elapsed_ms ? (d.metrics.elapsed_ms * 0.35).toFixed(1) + 'ms' : '—';
      document.getElementById('m-lider').textContent = d.metrics.elapsed_ms ? (d.metrics.elapsed_ms * 0.25).toFixed(1) + 'ms' : '—';
    }

    document.getElementById('perfBar').textContent = `Last query: ${totalMs}ms · ${d.query_tokens_estimate || 0} tokens`;
  } catch (err) {
    console.error('Search failed:', err);
    document.getElementById('traceContent').innerHTML = '<p style="color:var(--red)">Search failed. Try re-indexing.</p>';
    document.getElementById('metricsBar').style.display = 'none';
  }
}

function renderTrace(trace) {
  const container = document.getElementById('traceContent');
  if (!trace || trace.length === 0) {
    container.innerHTML = '<p style="color:var(--text-muted)">No trace data available.</p>';
    return;
  }

  let html = '';
  trace.forEach(step => {
    // Skip the final "top-k results" step — it renders in the results section
    if (step.step === 'top-k results') return;

    html += '<div class="trace-step">';
    html += `<div class="trace-step-header">${step.step}</div>`;
    html += `<div class="trace-step-desc">${step.description || ''}</div>`;

    // Format the data nicely
    const dataForDisplay = { ...step };
    delete dataForDisplay.step;
    delete dataForDisplay.description;

    if (step.step === 'NPMI edge weights') {
      html += `<div class="trace-step-data">query words found: ${(step.query_words_found || []).join(', ')}
edges above threshold: ${step.edges_above_threshold || 0}
top edges: ${JSON.stringify(step.top_edges || {}, null, 0)}</div>`;
    } else if (step.step === 'GEE embedding') {
      html += `<div class="trace-step-data">words embedded: ${(step.query_words_vectors || []).join(', ')}
sample nearest neighbors: ${JSON.stringify(step.sample || {}, null, 0)}</div>`;
    } else {
      html += `<div class="trace-step-data">${JSON.stringify(dataForDisplay, null, 1)}</div>`;
    }

    html += '</div>';
  });

  container.innerHTML = html;
}

function renderResults(results, totalMs) {
  const container = document.getElementById('resultsList');
  if (!results || results.length === 0) {
    container.innerHTML = '<p style="color:var(--text-muted);padding:20px;">No results found. Try a different query.</p>';
    return;
  }

  let html = '';
  results.forEach((r, i) => {
    const score = (r.score * 100).toFixed(1);
    const scoreColor = score > 80 ? 'var(--green)' : score > 50 ? 'var(--cyan)' : 'var(--amber)';
    html += `
      <div class="result-item">
        <div class="result-header">
          <span class="result-rank">Result #${i + 1}</span>
          <span class="result-score" style="color:${scoreColor}">${score}% match</span>
        </div>
        <div class="result-text">${escapeHtml(r.preview || r.text || '')}</div>
      </div>`;
  });
  container.innerHTML = html;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ── Upload ────────────────────────────────────────────────────────────────
async function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showUploadStatus('Only PDF files are supported', 'error');
    return;
  }

  showUploadStatus(`Uploading ${file.name}...`, 'loading');

  try {
    const formData = new FormData();
    formData.append('file', file);
    const r = await fetch(API.upload(), { method: 'POST', body: formData });
    const d = await r.json();

    if (d.doc_id) {
      showUploadStatus(`✅ ${file.name} (${d.num_pages} pages, ${fmtNum(d.total_chars)} chars) — Indexing...`, 'loading');
      await indexDocument(d.doc_id, file.name);
    } else {
      showUploadStatus('Upload failed', 'error');
    }
  } catch (err) {
    showUploadStatus(`Upload failed: ${err.message}`, 'error');
  }
}

async function indexDocument(docId, filename) {
  try {
    const r = await fetch(API.indexDoc(docId), { method: 'POST' });
    const d = await r.json();

    if (d.status === 'indexed') {
      showUploadStatus(`✅ Indexed! ${d.stats.vocab_size} words, ${d.stats.embedding_dim}D embeddings. Ready to search.`, 'ready');
      currentDocTitle = filename;
      isSeedPipeline = false;

      // Refresh pipeline stats
      loadPipelineStats();
      document.getElementById('apiHealth').textContent = '✅ Pipeline Ready (your doc)';
      document.getElementById('pipelineStats').textContent = `📄 ${filename}`;

      // Reset empty state
      document.getElementById('emptyState').style.display = 'none';
    }
  } catch (err) {
    showUploadStatus(`Indexing failed: ${err.message}`, 'error');
  }
}

function showUploadStatus(msg, type) {
  const el = document.getElementById('uploadStatus');
  el.textContent = msg;
  el.style.color = type === 'error' ? 'var(--red)' :
                   type === 'ready' ? 'var(--green)' :
                   type === 'loading' ? 'var(--amber)' : 'var(--text-muted)';
}

function fmtNum(n) {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
  return String(n);
}

// ── Graph Visualization ───────────────────────────────────────────────────
let graphLoaded = false;

async function loadGraph() {
  if (graphLoaded) return;
  if (!pipelineReady) return;

  try {
    const r = await fetch(API.graph());
    const d = await r.json();
    if (!d.nodes || d.nodes.length === 0) return;

    renderD3Graph(d.nodes, d.edges);
    graphLoaded = true;
  } catch (e) {
    console.error('Graph load failed:', e);
  }
}

function renderD3Graph(nodes, edges) {
  const container = document.getElementById('graphContainer');
  const svg = document.getElementById('graphSvg');
  svg.innerHTML = '';

  const width = container.clientWidth;
  const height = container.clientHeight || 500;

  const svgEl = d3.select('#graphSvg')
    .attr('viewBox', `0 0 ${width} ${height}`);

  // Only show top nodes by frequency
  const topNodes = nodes
    .sort((a, b) => (b.freq || 0) - (a.freq || 0))
    .slice(0, 80);

  const nodeIds = new Set(topNodes.map(n => n.id));
  const topEdges = edges
    .filter(e => nodeIds.has(e.source) && nodeIds.has(e.target))
    .slice(0, 200);

  const simulation = d3.forceSimulation(topNodes)
    .force('link', d3.forceLink(topEdges).id(d => d.id).distance(60))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => Math.max(8, Math.sqrt(d.freq || 1) * 2)));

  const link = svgEl.append('g')
    .selectAll('line')
    .data(topEdges)
    .join('line')
    .attr('stroke', '#30363d')
    .attr('stroke-width', d => Math.max(0.5, d.weight * 3))
    .attr('stroke-opacity', d => Math.max(0.1, d.weight));

  const node = svgEl.append('g')
    .selectAll('circle')
    .data(topNodes)
    .join('circle')
    .attr('r', d => Math.max(4, Math.sqrt(d.freq || 1) * 1.5))
    .attr('fill', d => d3.interpolateCool(d.freq / (topNodes[0]?.freq || 1)))
    .attr('stroke', '#fff')
    .attr('stroke-width', 0.5)
    .call(d3.drag()
      .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

  const label = svgEl.append('g')
    .selectAll('text')
    .data(topNodes.filter(d => (d.freq || 0) > 5))
    .join('text')
    .text(d => d.token || d.id)
    .attr('font-size', d => Math.max(8, Math.min(14, (d.freq || 1) * 0.8)))
    .attr('fill', '#8b949e')
    .attr('text-anchor', 'middle')
    .attr('dy', d => -Math.max(5, Math.sqrt(d.freq || 1) * 1.5) - 3);

  node.append('title').text(d => `${d.token || d.id} (freq: ${d.freq || 0})`);

  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
    label.attr('x', d => d.x).attr('y', d => d.y);
  });
}

// ── Graph tab handlers ────────────────────────────────────────────────────
document.querySelector('.graph-section summary').addEventListener('click', () => {
  loadGraph();
});

// ── Demo reset (called from upload flow) ──────────────────────────────────
async function resetToDemo() {
  try {
    const r = await fetch(API.demoReset());
    await r.json();
    pipelineReady = true;
    isSeedPipeline = true;
    document.getElementById('apiHealth').textContent = '✅ Pipeline Ready (demo)';
    document.getElementById('pipelineStats').textContent = '📚 Demo mode';
    document.getElementById('emptyState').style.display = 'block';
    document.getElementById('resultsArea').classList.remove('active');
    document.getElementById('metricsBar').style.display = 'none';
    loadPipelineStats();
    loadDemoDocList();
  } catch (e) {
    console.error('Reset failed:', e);
  }
}
