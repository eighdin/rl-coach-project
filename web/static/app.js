const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const fileName    = document.getElementById('file-name');
const playerInput = document.getElementById('player-name');
const analyzeBtn  = document.getElementById('analyze-btn');
const status      = document.getElementById('status');
const statusMsg   = document.getElementById('status-msg');
const report      = document.getElementById('report');
const historyList = document.getElementById('history-list');

let selectedFile = null;

// ── File selection ────────────────────────────────────────────────────────────

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) selectFile(fileInput.files[0]);
});

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) selectFile(f);
});

function selectFile(f) {
  if (!f.name.endsWith('.replay')) {
    showError('Only .replay files are supported.');
    return;
  }
  selectedFile = f;
  fileName.textContent = f.name;
  analyzeBtn.disabled = false;
  hideStatus();
  report.classList.remove('visible');
}

// ── Analyze ───────────────────────────────────────────────────────────────────

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  showSpinner('Parsing replay and detecting events…');
  report.classList.remove('visible');

  const form = new FormData();
  form.append('file', selectedFile);
  form.append('player_name', playerInput.value.trim());

  try {
    // Give feedback that the AI step is running (no SSE, just update the message after a delay)
    setTimeout(() => {
      if (status.classList.contains('visible') && !status.classList.contains('error')) {
        statusMsg.textContent = 'Querying AI coach — this may take a minute…';
      }
    }, 3000);

    const res = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Unknown error');

    hideStatus();
    renderReport(data);
    loadHistory();
  } catch (err) {
    showError(err.message);
  } finally {
    analyzeBtn.disabled = false;
  }
});

// ── Render report ─────────────────────────────────────────────────────────────

function renderReport(data, meta = null) {
  report.classList.add('visible');

  const metaEl = document.getElementById('report-meta');
  metaEl.innerHTML = meta ? `
    <span><strong>${meta.player_name}</strong></span>
    <span>${meta.playlist_name}</span>
    <span class="badge ${meta.outcome}">${meta.outcome.toUpperCase()} ${meta.team_score}–${meta.opponent_score}</span>
    ${meta.played_at ? `<span>${formatDate(meta.played_at)}</span>` : ''}
  ` : '';

  const container = document.getElementById('coaching-points');
  container.innerHTML = '';

  document.getElementById('game-summary').textContent = data.game_summary || '';

  (data.coaching_points || []).forEach(pt => {
    const el = document.createElement('div');
    el.className = `coaching-point ${pt.impact}`;
    el.innerHTML = `
      <div class="point-header">
        <div class="rank">${pt.rank}</div>
        <div class="point-label">${pt.label}</div>
        <span class="impact-badge ${pt.impact}">${pt.impact}</span>
        <span class="category-tag">${pt.category.replace('_', ' ')}</span>
      </div>
      <div class="point-body">
        <div class="point-section">
          <label>Observation</label>
          <p>${pt.observation}</p>
        </div>
        <div class="point-section">
          <label>Why it matters</label>
          <p>${pt.why_it_matters}</p>
        </div>
        <div class="point-section fix-box">
          <label>Fix</label>
          <p>${pt.fix}</p>
        </div>
      </div>
    `;
    container.appendChild(el);
  });

  report.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── History ───────────────────────────────────────────────────────────────────

async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const items = await res.json();

    if (!items.length) {
      historyList.innerHTML = '<p class="empty-history">No replays analyzed yet.</p>';
      return;
    }

    historyList.innerHTML = '';
    items.forEach(item => {
      const el = document.createElement('div');
      el.className = 'history-item';
      el.dataset.hash = item.replay_hash;
      el.innerHTML = `
        <div class="player">${item.player_name}</div>
        <div class="meta">
          <span class="badge ${item.outcome}">${item.outcome.toUpperCase()} ${item.team_score}–${item.opponent_score}</span>
          <span>${item.playlist_name}</span>
          ${item.played_at ? `<span>${formatDate(item.played_at)}</span>` : ''}
        </div>
      `;
      el.addEventListener('click', () => loadCachedReport(item.replay_hash, item, el));

      const delBtn = document.createElement('button');
      delBtn.className = 'delete-btn';
      delBtn.title = 'Delete record';
      delBtn.innerHTML = '&times;';
      delBtn.addEventListener('click', async e => {
        e.stopPropagation();
        const wasActive = el.classList.contains('active');
        await fetch(`/api/session/${item.replay_hash}`, { method: 'DELETE' });
        if (wasActive) report.classList.remove('visible');
        loadHistory();
      });
      el.appendChild(delBtn);
      historyList.appendChild(el);
    });
  } catch (e) {
    console.error('Failed to load history:', e);
  }
}

async function loadCachedReport(hash, meta, el) {
  document.querySelectorAll('.history-item').forEach(i => i.classList.remove('active'));
  el.classList.add('active');

  showSpinner('Loading cached report…');
  report.classList.remove('visible');

  try {
    const res = await fetch(`/api/report/${hash}`);
    if (!res.ok) throw new Error('Report not found.');
    const data = await res.json();
    hideStatus();
    renderReport(data, meta);
  } catch (err) {
    showError(err.message);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function showSpinner(msg) {
  status.className = 'status visible';
  status.innerHTML = `<div class="spinner"></div><span id="status-msg">${msg}</span>`;
}

function showError(msg) {
  status.className = 'status visible error';
  status.innerHTML = `<span>⚠ ${msg}</span>`;
}

function hideStatus() {
  status.className = 'status';
}

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  } catch { return ''; }
}

// ── Init ──────────────────────────────────────────────────────────────────────

loadHistory();
