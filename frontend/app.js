/**
 * GraphQ Browser App — PDF Q&A + Developer Metrics + Graph Visualization
 * =======================================================================
 * Flow:
 *  1. Upload PDF  → POST /documents/upload
 *  2. Build Index → POST /documents/{id}/index
 *  3. Search      → GET  /search?query=...  (sub-ms)
 *  4. Dev Mode   → GET  /search/trace      (step-by-step trace)
 *  5. Graph View → GET  /graph             (D3 co-occurrence network)
 */

const API_BASE = 'https://fast-roulette-manufacturer-doll.trycloudflare.com';

const API = {
  health:       () => `${API_BASE}/health`,
  documents:    () => `${API_BASE}/documents`,
  upload:       () => `${API_BASE}/documents/upload`,
  indexDoc:     (id) => `${API_BASE}/documents/${id}/index`,
  search:       (q, k) => `${API_BASE}/search?query=${encodeURIComponent(q)}&top_k=${k}`,
  searchTrace:  (q, k) => `${API_BASE}/search/trace?query=${encodeURIComponent(q)}&top_k=${k}`,
  graph:        () => `${API_BASE}/graph`,
};

// ── State ──────────────────────────────────────────────────────────────────
let currentDocId = null;
let devMode = false;
let searchAnimation = null;

// ── Init ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  // Developer mode toggle
  document.getElementById('devToggle').addEventListener('change', e => {
    devMode = e.target.checked;
    document.getElementById('devPanel').style.display = devMode ? 'block' : 'none';
  });

  // Search on Enter
  document.getElementById('queryInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch();
  });

  // Graph tab
  document.getElementById('graphTab').addEventListener('click', () => {
    document.getElementById('searchTab').classList.remove('active');
    document.getElementById('graphTab').classList.add('active');
    document.getElementById('searchPane').style.display = 'none';
    document.getElementById('graphPane').style.display = 'block';
    loadGraph();
  });

  document.getElementById('searchTab').addEventListener('click', () => {
    document.getElementById('searchTab').classList.add('active');
    document.getElementById('graphTab').classList.remove('active');
    document.getElementById('searchPane').style.display = 'block';
    document.getElementById('graphPane').style.display = 'none';
  });

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

  checkHealth();
  loadDocList();
});

// ── Health + Doc List ───────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(API.health());
    const d = await r.json();
    document.getElementById('apiStatus').textContent = d.status === 'ready' ? '✅ Ready' : '⏳ ' + d.status;
    document.getElementById('indexStatus').textContent = d.pipeline === 'ready' ? '✅ Built' : '❌ Not built';
    if (d.status === 'ready') document.getElementById('queryInput').disabled = false;
  } catch {
    document.getElementById('apiStatus').textContent = '❌ Offline';
    document.getElementById('indexStatus').textContent = '❌ Offline';
  }
}

async function loadDocList() {
  try {
    const r = await fetch(API.documents());
    const d = await r.json();
    const list = document.getElementById('docList');
    list.innerHTML = '';
    if (d.documents && d.documents.length === 0) {
      list.innerHTML = '<p class="muted">No documents uploaded yet.</p>';
      return;
    }
    (d.documents || []).forEach(doc => {
      const div = document.createElement('div');
      div.className = 'doc-item';
      div.innerHTML = `
        <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
        <span class="doc-meta">${doc.num_pages || 0}p · ${fmtNum(doc.total_chars || 0)} chars · ${fmtNum(doc.tokens_estimate || 0)} tokens</span>
        <span class="doc-status ${doc.status}">${doc.status === 'indexed' ? '✅ Indexed' : '📤 Uploaded'}</span>
      `;
      list.appendChild(div);
    });
  } catch { /* ignore */ }
}

// ── Index Progress ──────────────────────────────────────────────────────────
class IndexProgress {
  constructor() {
    this.phases = ['idle', 'tokenizing', 'npmi', 'gee', 'lider', 'done'];
    this.currentPhase = 'idle';
    this.data = {};
    this.animFrame = null;
    this.canvas = document.getElementById('magic-canvas');
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.words = [];
    this.edges = [];
    this.bins = [];
    this.sparkles = [];
    this.t = 0;
    this._startLoop();
  }

  _startLoop() {
    if (!this.ctx) return;
    const loop = () => {
      this._draw();
      this.t += 0.016;
      this.animFrame = requestAnimationFrame(loop);
    };
    this.animFrame = requestAnimationFrame(loop);
  }

  reset() {
    this.currentPhase = 'idle';
    this.data = {};
    this.words = [];
    this.edges = [];
    this.bins = [];
    this.sparkles = [];
    this.t = 0;
    if (this.animFrame) cancelAnimationFrame(this.animFrame);

    // Reset all bars
    ['tokenizing', 'npmi', 'gee', 'lider'].forEach(p => {
      const bar = document.getElementById(`bar-${p}`);
      if (bar) {
        bar.style.width = '0%';
        bar.classList.remove('active', 'done');
      }
    });

    // Reset stats
    ['chunks', 'vocab', 'dims', 'bins'].forEach(s => {
      const el = document.getElementById(`stat-${s}`);
      if (el) el.textContent = '—';
    });

    // Clear canvas
    if (this.ctx) {
      this.ctx.clearRect(0, 0, 400, 300);
    }

    // Hide panel
    const panel = document.getElementById('indexProgressPanel');
    if (panel) panel.style.display = 'none';
  }

  setPhase(phase, data = {}) {
    this.currentPhase = phase;
    this.data = data;
    this.t = 0;

    // Update bars
    const phaseOrder = ['tokenizing', 'npmi', 'gee', 'lider'];
    const phaseIndex = phaseOrder.indexOf(phase);

    phaseOrder.forEach((p, i) => {
      const bar = document.getElementById(`bar-${p}`);
      if (!bar) return;
      bar.classList.remove('active', 'done');

      if (i < phaseIndex) {
        // Completed
        bar.style.width = '100%';
        bar.classList.add('done');
      } else if (i === phaseIndex) {
        // Active
        bar.style.width = '60%';
        bar.classList.add('active');
      } else {
        // Pending
        bar.style.width = '0%';
      }
    });

    // Done phase
    if (phase === 'done') {
      const bar = document.getElementById('bar-lider');
      if (bar) {
        bar.style.width = '100%';
        bar.classList.remove('active');
        bar.classList.add('done');
      }
    }

    // Update stats
    if (data.n_chunks !== undefined) {
      const el = document.getElementById('stat-chunks');
      if (el) el.textContent = data.n_chunks;
    }
    if (data.vocab_size !== undefined) {
      const el = document.getElementById('stat-vocab');
      if (el) el.textContent = data.vocab_size;
    }
    if (data.dims !== undefined) {
      const el = document.getElementById('stat-dims');
      if (el) el.textContent = data.dims;
    }
    if (data.bins !== undefined) {
      const el = document.getElementById('stat-bins');
      if (el) el.textContent = data.bins;
    }

    // Show panel
    const panel = document.getElementById('indexProgressPanel');
    if (panel) panel.style.display = 'flex';

    // Prepare canvas data per phase
    this._preparePhaseData(phase, data);
  }

  _preparePhaseData(phase, data) {
    if (phase === 'tokenizing' && data.vocab) {
      this.words = data.vocab.slice(0, 40).map((w, i) => {
        const angle = (i / 40) * Math.PI * 2;
        const r = 80 + Math.random() * 30;
        return {
          token: w,
          x: 200 + Math.cos(angle) * r,
          y: 150 + Math.sin(angle) * r,
          targetX: 200 + Math.cos(angle) * (60 + Math.random() * 80),
          targetY: 150 + Math.sin(angle) * (60 + Math.random() * 80),
          alpha: 0,
          targetAlpha: 0.6 + Math.random() * 0.4,
          size: 4 + Math.random() * 4
        };
      });
      this.edges = [];
      this.bins = [];
      this.sparkles = [];
    } else if (phase === 'npmi' && data.vocab) {
      // Keep words, add edges from NPMI data
      this.words.forEach(w => {
        w.targetX = w.x + (Math.random() - 0.5) * 40;
        w.targetY = w.y + (Math.random() - 0.5) * 40;
      });
      this.edges = [];
      if (data.edges) {
        // Build a quick word→index map
        const vIdx = {};
        if (this.words.length && this.words[0].token) {
          this.words.forEach((w, i) => { vIdx[w.token] = i; });
        }
        this.edges = data.edges.slice(0, 30).map(e => ({
          from: vIdx[e.source],
          to: vIdx[e.target],
          alpha: 0
        })).filter(e => e.from !== undefined && e.to !== undefined);
      } else {
        // Auto-generate some edges between nearby words
        for (let i = 0; i < this.words.length - 1; i++) {
          if (Math.random() > 0.5) {
            this.edges.push({
              from: this.words[i],
              to: this.words[i + 1],
              alpha: 0
            });
          }
        }
      }
      this.bins = [];
      this.sparkles = [];
    } else if (phase === 'gee' && data.vectors) {
      // Words drift to 2D GEE positions using first 2 dims
      this.words.forEach((w, i) => {
        if (data.vectors[i]) {
          w.targetX = 200 + data.vectors[i][0] * 60;
          w.targetY = 150 + data.vectors[i][1] * 60;
        }
      });
      this.edges = [];
      this.bins = [];
      this.sparkles = [];
    } else if (phase === 'lider' && data.bins) {
      this.bins = data.bins.map((h, i) => ({
        x: 40 + i * 40,
        height: 0,
        targetHeight: h * 200,
        color: `hsl(${(i / (data.bins.length || 10)) * 360}, 70%, 55%)`
      }));
      this.words = [];
      this.edges = [];
      this.sparkles = [];
    } else if (phase === 'done') {
      // Generate sparkles
      this.sparkles = Array.from({ length: 60 }, () => ({
        x: Math.random() * 400,
        y: Math.random() * 300,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        alpha: 1,
        size: 1 + Math.random() * 3,
        hue: Math.random() * 360
      }));
    }
  }

  _animate() {
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
    const loop = () => {
      this._draw();
      this.t += 0.016;
      this.animFrame = requestAnimationFrame(loop);
    };
    loop();
  }

  _draw() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const W = 400, H = 300;

    // Clear
    ctx.clearRect(0, 0, W, H);

    if (this.currentPhase === 'idle') return;

    if (this.currentPhase === 'tokenizing') {
      this._drawTokenizing(ctx, W, H);
    } else if (this.currentPhase === 'npmi') {
      this._drawNpmi(ctx, W, H);
    } else if (this.currentPhase === 'gee') {
      this._drawGee(ctx, W, H);
    } else if (this.currentPhase === 'lider') {
      this._drawLider(ctx, W, H);
    } else if (this.currentPhase === 'done') {
      this._drawDone(ctx, W, H);
    }
  }

  _drawTokenizing(ctx, W, H) {
    // Background
    ctx.fillStyle = 'rgba(13,17,23,0.9)';
    ctx.fillRect(0, 0, W, H);

    // Draw words appearing
    const revealCount = Math.min(this.words.length, Math.floor(this.t * 8));
    this.words.forEach((w, i) => {
      if (i < revealCount) {
        w.alpha = Math.min(w.targetAlpha, w.alpha + 0.05);
        w.x += (w.targetX - w.x) * 0.05;
        w.y += (w.targetY - w.y) * 0.05;
      } else {
        w.alpha = Math.max(0, w.alpha - 0.05);
      }

      if (w.alpha > 0) {
        ctx.globalAlpha = w.alpha;
        ctx.fillStyle = '#22d3ee';
        ctx.beginPath();
        ctx.arc(w.x, w.y, w.size, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#e6edf3';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(w.token.slice(0, 6), w.x, w.y - w.size - 3);
        ctx.globalAlpha = 1;
      }
    });

    // Center label
    if (this.t < 2) {
      ctx.fillStyle = '#8b949e';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Tokenizing words...', W / 2, H / 2 + 100);
    }
  }

  _drawNpmi(ctx, W, H) {
    ctx.fillStyle = 'rgba(13,17,23,0.9)';
    ctx.fillRect(0, 0, W, H);

    // Draw edges appearing
    const edgeReveal = Math.min(this.edges.length, Math.floor(this.t * 5));
    for (let i = 0; i < edgeReveal; i++) {
      const e = this.edges[i];
      e.alpha = Math.min(0.8, e.alpha + 0.05);
      ctx.globalAlpha = e.alpha;
      ctx.strokeStyle = '#60a5fa';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(e.from.x, e.from.y);
      ctx.lineTo(e.to.x, e.to.y);
      ctx.stroke();
    }

    // Draw words
    this.words.forEach(w => {
      w.x += (w.targetX - w.x) * 0.03;
      w.y += (w.targetY - w.y) * 0.03;
      ctx.globalAlpha = 0.7;
      ctx.fillStyle = '#6366f1';
      ctx.beginPath();
      ctx.arc(w.x, w.y, w.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#e6edf3';
      ctx.font = '8px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(w.token.slice(0, 5), w.x, w.y - w.size - 2);
      ctx.globalAlpha = 1;
    });

    ctx.fillStyle = '#8b949e';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Building NPMI matrix...', W / 2, H / 2 + 100);
  }

  _drawGee(ctx, W, H) {
    ctx.fillStyle = 'rgba(13,17,23,0.9)';
    ctx.fillRect(0, 0, W, H);

    // Words flowing to embedding positions
    this.words.forEach(w => {
      w.x += (w.targetX - w.x) * 0.04;
      w.y += (w.targetY - w.y) * 0.04;

      // Glow
      const gradient = ctx.createRadialGradient(w.x, w.y, 0, w.x, w.y, w.size * 3);
      gradient.addColorStop(0, 'rgba(34,211,238,0.3)');
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(w.x, w.y, w.size * 3, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#22d3ee';
      ctx.beginPath();
      ctx.arc(w.x, w.y, w.size, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#e6edf3';
      ctx.font = '8px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(w.token.slice(0, 5), w.x, w.y - w.size - 2);
    });

    ctx.fillStyle = '#8b949e';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('GEE embedding...', W / 2, H / 2 + 100);
  }

  _drawLider(ctx, W, H) {
    ctx.fillStyle = 'rgba(13,17,23,0.9)';
    ctx.fillRect(0, 0, W, H);

    // Bars rising
    this.bins.forEach(b => {
      b.height += (b.targetHeight - b.height) * 0.08;
      const barW = 30;
      const barH = Math.max(0, b.height);
      const x = b.x;
      const y = H - 40 - barH;

      const gradient = ctx.createLinearGradient(x, y, x, y + barH);
      gradient.addColorStop(0, b.color);
      gradient.addColorStop(1, 'rgba(0,0,0,0.3)');
      ctx.fillStyle = gradient;
      ctx.fillRect(x, y, barW, barH);

      // Bar outline
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, barW, barH);
    });

    ctx.fillStyle = '#8b949e';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('LIDER binning...', W / 2, H - 10);
  }

  _drawDone(ctx, W, H) {
    ctx.fillStyle = 'rgba(13,17,23,0.95)';
    ctx.fillRect(0, 0, W, H);

    // Sparkles/confetti
    this.sparkles.forEach(s => {
      s.x += s.vx;
      s.y += s.vy;
      s.vy += 0.02; // gravity
      s.alpha -= 0.008;

      if (s.alpha > 0) {
        ctx.globalAlpha = s.alpha;
        ctx.fillStyle = `hsl(${s.hue}, 90%, 65%)`;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
        ctx.fill();

        // Star shape
        ctx.strokeStyle = `hsl(${s.hue}, 90%, 80%)`;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(s.x - s.size * 2, s.y);
        ctx.lineTo(s.x + s.size * 2, s.y);
        ctx.moveTo(s.x, s.y - s.size * 2);
        ctx.lineTo(s.x, s.y + s.size * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });

    // "Ready!" text
    const textAlpha = Math.min(1, this.t * 0.5);
    ctx.globalAlpha = textAlpha;
    ctx.fillStyle = '#22c55e';
    ctx.font = 'bold 32px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('✅ Indexed!', W / 2, H / 2 - 20);

    ctx.fillStyle = '#8b949e';
    ctx.font = '13px monospace';
    ctx.fillText('Search is ready', W / 2, H / 2 + 15);
    ctx.globalAlpha = 1;
  }

  _stop() {
    if (this.animFrame) {
      cancelAnimationFrame(this.animFrame);
      this.animFrame = null;
    }
  }
}

const indexProgress = new IndexProgress();

// ── SSE Helper ──────────────────────────────────────────────────────────────
function connectSSE(sseUrl, onPhase, onDone) {
  // EventSource only supports GET — use it only if no POST body is needed.
  // For doc_id, the server reads it from Form fields, so we must use fetch.
  // Always use the fetch+ReadableStream fallback for reliability.
  fallbackSSE(sseUrl, onPhase, onDone);
}

async function fallbackSSE(sseUrl, onPhase, onDone) {
  // sseUrl may include ?doc_id=...  (query param for GET fallback)
  // But we need POST with Content-Type: application/x-www-form-urlencoded
  try {
    const r = await fetch(sseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({ doc_id: currentDocId }).toString(),
    });
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const d = JSON.parse(line.slice(6));
            if (d.status === 'indexed' || d.stats) {
              onDone(d);
              return;
            } else {
              onPhase(d);
            }
          } catch {}
        }
        if (line.startsWith('event: error')) {
          onPhase({ error: true, message: 'Build failed — check server logs' });
        }
      }
    }
  } catch (err) {
    console.error('SSE fallback error:', err);
    onPhase({ error: true, message: err.message });
  }
}

// ── File Upload ─────────────────────────────────────────────────────────────
async function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    alert('Only PDF files supported.');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    alert('Max file size: 20MB');
    return;
  }

  const fd = new FormData();
  fd.append('file', file);

  setStatus('Uploading + parsing...', 'loading');
  try {
    const r = await fetch(API.upload(), { method: 'POST', body: fd });
    const d = await r.json();

    if (!r.ok) throw new Error(d.detail || 'Upload failed');

    currentDocId = d.doc_id;
    showMetricsPanel([d.metrics]);

    const preview = document.getElementById('preview');
    preview.innerHTML = `
      <div class="result-card">
        <div class="result-meta">📄 ${d.filename} · ${d.num_pages} pages · ${fmtNum(d.total_chars)} chars · ~${fmtNum(d.tokens_estimate)} tokens</div>
        <div class="result-text">${escHtml(d.preview)}</div>
        <div class="disclaimer">⚠️ EPHEMERAL — Stored in server RAM /tmp/. Wiped on restart. Do NOT upload personal docs.</div>
        <button class="btn btn-primary" onclick="buildIndex()">⚡ Build NPMI+GEE+LIDER Index</button>
      </div>
    `;
    setStatus(`✅ Uploaded: ${d.filename}`, 'ok');
    loadDocList();
  } catch (err) {
    setStatus('❌ Upload failed: ' + err.message, 'error');
  }
}

async function buildIndex() {
  if (!currentDocId) return;
  const btn = document.querySelector('.btn-primary');
  if (!btn) return;

  // Disable button, show indexing state
  btn.disabled = true;
  btn.textContent = '⏳ Indexing...';

  // Reset and show progress panel
  indexProgress.reset();
  indexProgress.setPhase('tokenizing', {});

  setStatus('Building NPMI+GEE+LIDER index...', 'loading');

  const t0 = performance.now();

  // Connect to SSE stream
  const sseUrl = `${API_BASE}/index/stream`;

  connectSSE(
    sseUrl,
    (data) => {
      // Phase event — map integer phase numbers to string phase names
      const phaseMap = { 1: 'tokenizing', 2: 'npmi', 3: 'gee', 4: 'lider' };
      const phase = data.phase in phaseMap ? phaseMap[data.phase] : (data.name || String(data.phase));
      indexProgress.setPhase(phase, data);
      setStatus(`⏳ ${phase}...`, 'loading');
    },
    (data) => {
      // Done event
      const elapsed = (performance.now() - t0).toFixed(1);
      indexProgress.setPhase('done', data);

      showMetricsPanel([data.metrics, { elapsed_ms: parseFloat(elapsed), note: 'user-visible wall-clock' }]);

      document.getElementById('queryInput').disabled = false;
      document.getElementById('indexStatus').textContent = '✅ Built';
      setStatus(`✅ Indexed: ${data.stats.n_chunks} chunks, ${data.stats.vocab_size} words, ${elapsed}ms`, 'ok');

      btn.textContent = '✅ Indexed!';
      btn.classList.add('btn-success');
      checkHealth();
    }
  );
}

// ── Search ──────────────────────────────────────────────────────────────────
async function doSearch() {
  const q = document.getElementById('queryInput').value.trim();
  const k = parseInt(document.getElementById('topK').value) || 5;
  if (!q) return;

  if (devMode) {
    await doSearchTrace(q, k);
  } else {
    await doSearchSimple(q, k);
  }
}

async function doSearchSimple(q, k) {
  setStatus('Searching...', 'loading');
  clearResults();
  startSpinner();

  try {
    const r = await fetch(API.search(q, k));
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);

    stopSpinner();
    showMetricsPanel([d.metrics]);
    d.results.forEach((res, i) => addResult(res, i));
    setStatus(`✅ Found ${d.results.length} results in ${d.metrics.elapsed_ms}ms`, 'ok');
  } catch (err) {
    stopSpinner();
    setStatus('❌ Search failed: ' + err.message, 'error');
  }
}

async function doSearchTrace(q, k) {
  setStatus('Tracing search pipeline...', 'loading');
  clearResults();
  startSpinner();

  try {
    const r = await fetch(API.searchTrace(q, k));
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);

    stopSpinner();
    showMetricsPanel([d.metrics]);

    // Render trace steps as an expandable timeline
    const container = document.getElementById('results');
    const traceDiv = document.createElement('div');
    traceDiv.className = 'trace-container';

    let html = `<div class="trace-header">
      <span class="trace-label">🔬 DEVELOPER TRACE</span>
      <span class="trace-meta">${d.query_tokens_estimate} tokens · ${d.trace.length} steps · ${d.metrics.elapsed_ms}ms</span>
    </div>`;

    d.trace.forEach((step, i) => {
      html += `<div class="trace-step" onclick="toggleTrace(this)">
        <div class="trace-step-header">
          <span class="step-num">${i + 1}</span>
          <span class="step-name">${escHtml(step.step)}</span>
          <span class="step-desc">${escHtml(step.description || '')}</span>
        </div>
        <div class="trace-step-body" style="display:none">
          <pre>${escHtml(JSON.stringify(step, null, 2))}</pre>
        </div>
      </div>`;
    });

    traceDiv.innerHTML = html;
    container.appendChild(traceDiv);

    // Also show regular results
    if (d.trace[d.trace.length - 1].results) {
      const results = d.trace[d.trace.length - 1].results;
      results.forEach(res => {
        const div = document.createElement('div');
        div.className = 'result-card trace-result';
        div.innerHTML = `
          <div class="result-meta">#${res.rank} · chunk ${res.idx} · score: ${res.score}</div>
          <div class="result-text">${escHtml(res.preview)}...</div>
        `;
        container.appendChild(div);
      });
    }

    setStatus(`🔬 Traced ${d.trace.length} pipeline steps in ${d.metrics.elapsed_ms}ms`, 'ok');
  } catch (err) {
    stopSpinner();
    setStatus('❌ Trace failed: ' + err.message, 'error');
  }
}

function toggleTrace(el) {
  const body = el.querySelector('.trace-step-body');
  body.style.display = body.style.display === 'none' ? 'block' : 'none';
}

// ── Graph Visualization ──────────────────────────────────────────────────────
let graphSimulation = null;

async function loadGraph() {
  const container = document.getElementById('graphSvg');
  container.innerHTML = '<text x="50%" y="50%" text-anchor="middle" fill="#888">Loading graph...</text>';

  try {
    const r = await fetch(API.graph());
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);

    renderGraph(d.nodes, d.edges, d.stats);
  } catch (err) {
    container.innerHTML = `<text x="50%" y="50%" text-anchor="middle" fill="red">Failed: ${escHtml(err.message)}</text>`;
  }
}

function renderGraph(nodes, edges, stats) {
  const container = document.getElementById('graphSvg');
  container.innerHTML = '';

  const W = container.clientWidth || 800;
  const H = container.clientHeight || 500;

  const svg = d3.select(container)
    .append('svg')
    .attr('width', W)
    .attr('height', H)
    .attr('viewBox', `0 0 ${W} ${H}`);

  // Stats overlay
  svg.append('text')
    .attr('x', 10).attr('y', 20)
    .attr('fill', '#6b7280').attr('font-size', '11px')
    .text(`GraphQ Network: ${nodes.length} words · ${edges.length} edges · ${stats.n_chunks} chunks · ${stats.vocab_size}D GEE embedding`);

  // Zoom
  const g = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.2, 5]).on('zoom', e => g.attr('transform', e.transform)));

  // Edges
  const link = g.append('g')
    .selectAll('line')
    .data(edges)
    .enter().append('line')
    .attr('stroke', d => d3.interpolateRgb('#4b5563', '#60a5fa')(d.weight * 5))
    .attr('stroke-width', d => Math.max(0.5, d.weight * 3))
    .attr('stroke-opacity', 0.6);

  // Nodes
  const node = g.append('g')
    .selectAll('g')
    .data(nodes)
    .enter().append('g')
    .attr('class', 'graph-node')
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) graphSimulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => { if (!e.active) graphSimulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );

  // Node circles
  node.append('circle')
    .attr('r', d => Math.sqrt(d.freq + 1) * 2.5)
    .attr('fill', '#3b82f6')
    .attr('stroke', '#93c5fd').attr('stroke-width', 1.5)
    .attr('opacity', 0.85);

  // Labels
  node.append('text')
    .attr('dy', d => Math.sqrt(d.freq + 1) * 2.5 + 12)
    .attr('text-anchor', 'middle')
    .attr('fill', '#9ca3af').attr('font-size', '9px')
    .attr('font-family', 'monospace')
    .text(d => d.token.length > 10 ? d.token.slice(0, 8) + '..' : d.token);

  // Hover tooltip
  node.on('mouseover', function(e, d) {
    d3.select(this).select('circle').attr('fill', '#60a5fa').attr('r', Math.sqrt(d.freq + 1) * 3.5);
    d3.select(this).select('text').attr('fill', '#e5e7eb').attr('font-size', '11px');
    showTooltip(e, `<b>${d.token}</b><br>idx: ${d.idx}<br>freq: ${d.freq}`);
  }).on('mouseout', function(e, d) {
    d3.select(this).select('circle').attr('fill', '#3b82f6').attr('r', Math.sqrt(d.freq + 1) * 2.5);
    d3.select(this).select('text').attr('fill', '#9ca3af').attr('font-size', '9px');
    hideTooltip();
  });

  // Animated search overlay
  node.on('click', function(e, d) {
    const query = document.getElementById('queryInput').value.trim();
    if (query) animateSearchPath(query, d.token);
  });

  // Force simulation
  graphSimulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(edges).id(d => d.id || d.token).distance(60).strength(0.3))
    .force('charge', d3.forceManyBody().strength(-80))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collide', d3.forceCollide().radius(d => Math.sqrt(d.freq + 1) * 3 + 5))
    .on('tick', () => {
      link
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
}

// ── Search Animation ────────────────────────────────────────────────────────
async function animateSearchPath(query, targetToken) {
  const svgEl = document.querySelector('#graphSvg svg');
  if (!svgEl) return;
  const svg = d3.select(svgEl);
  const W = svgEl.clientWidth || 800;
  const H = svgEl.clientHeight || 500;

  // Cancel previous
  if (searchAnimation) searchAnimation.stop();

  // Tokenize query
  const queryTokens = query.toLowerCase().match(/[a-z0-9_]+/g) || [];
  const sim = graphSimulation;

  // Find matching nodes
  const matchingNodes = queryTokens
    .map(t => sim.nodes().find(n => n.token === t))
    .filter(Boolean);

  if (matchingNodes.length === 0) return;

  // Build path: query tokens → all nodes → target
  const pathNodes = [...matchingNodes];
  if (targetToken) {
    const target = sim.nodes().find(n => n.token === targetToken);
    if (target) pathNodes.push(target);
  }

  // Draw animated path
  const pathLine = svg.append('line')
    .attr('stroke', '#f59e0b').attr('stroke-width', 3)
    .attr('stroke-linecap', 'round').attr('opacity', 0);

  const particle = svg.append('circle')
    .attr('r', 6).attr('fill', '#fbbf24')
    .attr('stroke', '#fff').attr('stroke-width', 2)
    .attr('opacity', 0);

  // Highlight nodes in path
  matchingNodes.forEach(n => {
    d3.selectAll('.graph-node circle')
      .filter(d => d.token === n.token)
      .transition().duration(300)
      .attr('fill', '#f59e0b').attr('r', Math.sqrt(n.freq + 1) * 4);
  });

  // Animate particle along nodes
  if (pathNodes.length < 2) return;

  let step = 0;
  const totalSteps = pathNodes.length * 8;

  searchAnimation = d3.timer(() => {
    step++;
    const idx = Math.floor(step / 8);
    const t = (step % 8) / 8;
    const from = pathNodes[idx % pathNodes.length];
    const to = pathNodes[(idx + 1) % pathNodes.length];

    const x = from.x + (to.x - from.x) * t;
    const y = from.y + (to.y - from.y) * t;

    particle.attr('cx', x).attr('cy', y).attr('opacity', 1);

    if (step >= totalSteps) {
      searchAnimation.stop();
      particle.transition().duration(500).attr('opacity', 0);
      pathLine.transition().duration(500).attr('opacity', 0).remove();

      // Reset node colors after 1s
      setTimeout(() => {
        d3.selectAll('.graph-node circle')
          .transition().duration(300)
          .attr('fill', '#3b82f6')
          .attr('r', d => Math.sqrt(d.freq + 1) * 2.5);
      }, 1000);
    }
  });
}

// ── Tooltip ─────────────────────────────────────────────────────────────────
let tooltipEl = null;
function showTooltip(e, html) {
  if (!tooltipEl) {
    tooltipEl = document.createElement('div');
    tooltipEl.id = 'graphTooltip';
    document.body.appendChild(tooltipEl);
  }
  tooltipEl.innerHTML = html;
  tooltipEl.style.cssText = 'position:fixed;background:#1f2937;border:1px solid #374151;border-radius:6px;padding:8px 12px;color:#e5e7eb;font-size:12px;pointer-events:none;z-index:9999;max-width:200px;box-shadow:0 4px 12px rgba(0,0,0,0.4)';
  positionTooltip(e);
}
function hideTooltip() { if (tooltipEl) tooltipEl.style.display = 'none'; }
function positionTooltip(e) {
  if (!tooltipEl) return;
  tooltipEl.style.display = 'block';
  tooltipEl.style.left = (e.clientX + 15) + 'px';
  tooltipEl.style.top = (e.clientY - 10) + 'px';
}
document.addEventListener('mousemove', e => { if (tooltipEl && tooltipEl.style.display === 'block') positionTooltip(e); });

// ── UI Helpers ──────────────────────────────────────────────────────────────
function setStatus(msg, cls) {
  const el = document.getElementById('statusBar');
  el.textContent = msg;
  el.className = 'status-bar ' + cls;
}

function clearResults() {
  document.getElementById('results').innerHTML = '';
}

function startSpinner() {
  document.getElementById('spinner').style.display = 'inline-block';
}
function stopSpinner() {
  document.getElementById('spinner').style.display = 'none';
}

function addResult(res, i) {
  const container = document.getElementById('results');
  const div = document.createElement('div');
  div.className = 'result-card';
  div.innerHTML = `
    <div class="result-header">
      <span class="result-rank">#${i + 1}</span>
      <span class="result-score" title="Relevance score">${(res.score * 100).toFixed(1)}%</span>
      <span class="result-chunk">chunk ${res.idx}</span>
    </div>
    <div class="result-text">${escHtml(res.text || '')}</div>
    <div class="result-chars">${fmtNum(res.chunk_chars || 0)} chars</div>
  `;
  container.appendChild(div);
}

function showMetricsPanel(metricsList) {
  const panel = document.getElementById('metricsOutput');
  panel.innerHTML = metricsList.map(m => `<pre>${escHtml(JSON.stringify(m, null, 2))}</pre>`).join('');
}

function fmtNum(n) {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
  return String(n);
}

function escHtml(s) {
  if (!s) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
