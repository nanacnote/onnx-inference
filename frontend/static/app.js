'use strict';

// ─── Tab switching ────────────────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    });
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));

    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');
    document.getElementById(btn.dataset.tab).classList.remove('hidden');
  });
});

// ─── Live range labels ────────────────────────────────────────────────────────

document.querySelectorAll('input[type="range"]').forEach(input => {
  const out = input.nextElementSibling;
  if (out && out.tagName === 'OUTPUT') {
    input.addEventListener('input', () => { out.value = input.value; });
  }
});

// ─── LLM ─────────────────────────────────────────────────────────────────────

document.getElementById('llm-form').addEventListener('submit', async e => {
  e.preventDefault();

  const prompt = document.getElementById('llm-prompt').value.trim();
  if (!prompt) return;

  const btn     = e.submitter;
  const result  = document.getElementById('llm-result');
  const output  = document.getElementById('llm-output');
  const errEl   = document.getElementById('llm-err');

  setLoading(btn, true);
  result.classList.remove('hidden');
  output.textContent = '';
  errEl.textContent  = '';

  try {
    const res  = await fetch('/api/generate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        prompt,
        max_new_tokens: parseInt(document.getElementById('llm-tokens').value, 10),
        temperature:    parseFloat(document.getElementById('llm-temp').value),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    output.textContent = data.text;
  } catch (err) {
    errEl.textContent = err.message;
  } finally {
    setLoading(btn, false);
  }
});

// ─── TTS ─────────────────────────────────────────────────────────────────────

document.getElementById('tts-form').addEventListener('submit', async e => {
  e.preventDefault();

  const text = document.getElementById('tts-text').value.trim();
  if (!text) return;

  const btn    = e.submitter;
  const result = document.getElementById('tts-result');
  const player = document.getElementById('tts-player');
  const errEl  = document.getElementById('tts-err');

  setLoading(btn, true);
  result.classList.remove('hidden');
  errEl.textContent = '';

  // Revoke the previous object URL to avoid memory leaks
  if (player.src) URL.revokeObjectURL(player.src);
  player.removeAttribute('src');

  try {
    const res = await fetch('/api/synthesize', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        text,
        speed: parseFloat(document.getElementById('tts-speed').value),
      }),
    });

    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.error || `HTTP ${res.status}`);
    }

    const blob = await res.blob();
    player.src = URL.createObjectURL(blob);
    player.play();
  } catch (err) {
    errEl.textContent = err.message;
  } finally {
    setLoading(btn, false);
  }
});

// ─── Embedding ────────────────────────────────────────────────────────────────

document.getElementById('embed-form').addEventListener('submit', async e => {
  e.preventDefault();

  const text = document.getElementById('embed-text').value.trim();
  if (!text) return;

  const btn    = e.submitter;
  const result = document.getElementById('embed-result');
  const errEl  = document.getElementById('embed-err');

  setLoading(btn, true);
  result.classList.remove('hidden');
  errEl.textContent = '';

  try {
    const res  = await fetch('/api/embed', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    renderVector(data.vector, data.dimensions);
  } catch (err) {
    errEl.textContent = err.message;
    document.getElementById('embed-meta').textContent  = '';
    document.getElementById('embed-bars').innerHTML    = '';
    document.getElementById('embed-full').textContent  = '';
    document.getElementById('embed-full').classList.add('hidden');
    document.getElementById('embed-toggle').textContent = 'Show full vector';
  } finally {
    setLoading(btn, false);
  }
});

function renderVector(vector, dimensions) {
  document.getElementById('embed-meta').textContent =
    `${dimensions} dimensions · unit-normalised float32`;

  const barsEl = document.getElementById('embed-bars');
  barsEl.innerHTML = '';

  const maxAbs = Math.max(...vector.map(Math.abs));

  vector.forEach(v => {
    const bar = document.createElement('div');
    bar.className       = 'vector-bar';
    bar.style.height    = `${(Math.abs(v) / maxAbs) * 100}%`;
    bar.style.background = v >= 0 ? '#0d47a1' : '#b71c1c';
    bar.title           = v.toFixed(6);
    barsEl.appendChild(bar);
  });

  const fullEl   = document.getElementById('embed-full');
  const toggleEl = document.getElementById('embed-toggle');

  fullEl.textContent = '';
  fullEl.classList.add('hidden');
  toggleEl.textContent = 'Show full vector';

  toggleEl.onclick = () => {
    if (fullEl.classList.contains('hidden')) {
      fullEl.textContent = '[' + vector.map(v => v.toFixed(6)).join(', ') + ']';
      fullEl.classList.remove('hidden');
      toggleEl.textContent = 'Hide full vector';
    } else {
      fullEl.classList.add('hidden');
      toggleEl.textContent = 'Show full vector';
    }
  };
}

// ─── OCR ─────────────────────────────────────────────────────────────────────

let ocrFile = null;

document.getElementById('ocr-input').addEventListener('change', e => {
  ocrFile = e.target.files[0] ?? null;
  document.getElementById('ocr-drop-label').textContent =
    ocrFile ? ocrFile.name : 'Click to select an image — JPEG, PNG, BMP, TIFF, WebP';
});

document.getElementById('ocr-form').addEventListener('submit', async e => {
  e.preventDefault();
  if (!ocrFile) return;

  const btn    = e.submitter;
  const result = document.getElementById('ocr-result');
  const errEl  = document.getElementById('ocr-err');

  setLoading(btn, true);
  result.classList.remove('hidden');
  errEl.textContent = '';

  const form = new FormData();
  form.append('image', ocrFile);

  try {
    const res  = await fetch('/api/ocr', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    renderOCR(ocrFile, data.results);
  } catch (err) {
    errEl.textContent = err.message;
    document.getElementById('ocr-frame').classList.add('hidden');
    document.getElementById('ocr-table').classList.add('hidden');
    document.getElementById('ocr-none').classList.add('hidden');
  } finally {
    setLoading(btn, false);
  }
});

function renderOCR(file, results) {
  const frame  = document.getElementById('ocr-frame');
  const img    = document.getElementById('ocr-img');
  const canvas = document.getElementById('ocr-canvas');
  const table  = document.getElementById('ocr-table');
  const none   = document.getElementById('ocr-none');

  frame.classList.remove('hidden');

  // Revoke previous object URL
  if (img.src) URL.revokeObjectURL(img.src);

  img.onload = () => {
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    results.forEach(r => {
      const [x1, y1, x2, y2, x3, y3, x4, y4] = r.box;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x3, y3);
      ctx.lineTo(x4, y4);
      ctx.closePath();
      ctx.strokeStyle = '#0d47a1';
      ctx.lineWidth   = Math.max(1, img.naturalWidth / 400);
      ctx.stroke();
      ctx.fillStyle = 'rgba(13, 71, 161, 0.08)';
      ctx.fill();
    });
  };

  img.src = URL.createObjectURL(file);

  if (results.length === 0) {
    table.classList.add('hidden');
    none.classList.remove('hidden');
    return;
  }

  none.classList.add('hidden');
  table.classList.remove('hidden');

  const tbody = table.querySelector('tbody');
  tbody.innerHTML = results.map((r, i) => `<tr>
    <td>${i + 1}</td>
    <td>${escHtml(r.text)}</td>
    <td>${(r.confidence * 100).toFixed(1)} %</td>
  </tr>`).join('');
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function setLoading(btn, on) {
  if (on) {
    btn.dataset.label = btn.textContent;
    btn.disabled      = true;
    btn.innerHTML     = '<span class="spinner"></span>Running\u2026';
  } else {
    btn.disabled  = false;
    btn.innerHTML = btn.dataset.label;
  }
}

function escHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
