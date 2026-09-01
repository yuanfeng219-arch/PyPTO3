/* IR Pipeline scrubber + inspector (stage 2).
   Data: window.PTO_IR_PIPELINE, generated from the real passes_dump. */
(function () {
  'use strict';

  const DATA = window.PTO_IR_PIPELINE;
  if (!DATA) return;

  const P = DATA.passes;
  const STRATA = DATA.strata;
  const LAST = P.length - 1;

  const FKEY = [
    { k: 'orch',   label: 'Orchestration', c: '#4E7C90' },
    { k: 'inline', label: 'Inline',        c: '#7E8A90' },
    { k: 'incore', label: 'InCore',        c: '#A2814F' },
    { k: 'aic',    label: 'AIC',           c: '#CE5622' },
    { k: 'aiv',    label: 'AIV',           c: '#E09258' },
    { k: 'group',  label: 'Group',         c: '#BC7440' },
    { k: 'spmd',   label: 'Spmd',          c: '#6E8288' }
  ];

  const MODES = {
    delta: { label: '改动量', segs: (p) => [{ v: p.d || 0, c: stratumVar(p.i) }] },
    ops:   { label: '算子词汇', segs: (p) => [
              { v: p.o.te, c: '#4E7C90', name: 'tensor.*' },
              { v: p.o.ti, c: '#C4612F', name: 'tile.*' }] },
    funcs: { label: '函数拓扑', segs: (p) => FKEY.map(k => ({ v: p.f[k.k], c: k.c, name: k.label })) },
    mem:   { label: '内存', segs: (p) => [
              { v: p.o.mr, c: '#6E8288', name: 'MemRef' },
              { v: p.o.al, c: '#CE5622', name: 'tile.alloc' }] }
  };

  const state = { mode: 'delta', sel: 0, onlyChanged: false, done: -1 };

  const $ = (s, r) => (r || document).querySelector(s);
  const esc = (s) => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const stratumOf = (i) => STRATA.find(s => i >= s.from && i <= s.to);
  const stratumVar = (i) => 'var(--irp-' + stratumOf(i).id.toLowerCase() + ')';
  const flat = (p) => p.i > 0 && !p.d;

  let els = null;

  /* ---------- markup ---------- */

  function mount() {
    const strip = $('#passStrip');
    if (!strip || !strip.parentNode) return false;

    const root = document.createElement('div');
    root.className = 'kf-irp';
    root.id = 'irPipeline';
    root.innerHTML =
      '<div class="kf-irp-bar">' +
        '<div class="kf-irp-seg" id="irpModes" role="group" aria-label="柱高编码">' +
          Object.keys(MODES).map(k =>
            '<button type="button" data-irp-mode="' + k + '" aria-pressed="' + (k === state.mode) + '">' +
            MODES[k].label + '</button>').join('') +
        '</div>' +
        '<button class="kf-irp-toggle" id="irpFilter" type="button" aria-pressed="false">' +
          '<i></i>只看有改动的 ' + P.filter(p => p.i > 0 && !flat(p)).length + ' 步</button>' +
        '<span class="kf-irp-spacer"></span>' +
        '<span class="kf-irp-pos"><b id="irpPos">00</b> / ' + String(LAST).padStart(2, '0') + '</span>' +
        '<div class="kf-irp-nav">' +
          '<button type="button" data-irp-step="-1" aria-label="上一个 pass">‹</button>' +
          '<button type="button" data-irp-step="1" aria-label="下一个 pass">›</button>' +
        '</div>' +
      '</div>' +
      '<div class="kf-irp-track">' +
        '<div class="kf-irp-strata" id="irpStrata"></div>' +
        '<div class="kf-irp-cols" id="irpCols"></div>' +
        '<div class="kf-irp-axis" id="irpAxis"></div>' +
      '</div>' +
      '<div class="kf-irp-body">' +
        '<aside class="kf-irp-info" id="irpInfo"></aside>' +
        '<section class="kf-irp-diff" id="irpDiff"></section>' +
      '</div>';

    strip.parentNode.insertBefore(root, strip.nextSibling);

    const tip = document.createElement('div');
    tip.className = 'kf-irp-tip';
    tip.id = 'irpTip';
    document.body.appendChild(tip);

    els = {
      root, tip,
      cols: $('#irpCols'), axis: $('#irpAxis'), strata: $('#irpStrata'),
      info: $('#irpInfo'), diff: $('#irpDiff'), pos: $('#irpPos'),
      modes: $('#irpModes'), filter: $('#irpFilter')
    };

    els.strata.innerHTML = STRATA.map(s => {
      const n = s.to - s.from + 1;
      return '<span style="--irp-c: var(--irp-' + s.id.toLowerCase() + '); flex: ' + n + ' 1 0">' +
             (n > 2 ? s.id + ' ' + s.name : s.id) + '</span>';
    }).join('');

    return true;
  }

  /* ---------- scrubber ---------- */

  function renderTrack() {
    const segsOf = MODES[state.mode].segs;
    const max = Math.max.apply(null, P.map(p => segsOf(p).reduce((a, b) => a + b.v, 0)).concat([1]));

    els.cols.innerHTML = P.map(p => {
      const segs = segsOf(p);
      const tot = segs.reduce((a, b) => a + b.v, 0);
      const h = Math.max((tot / max) * 100, 0.8);
      const isFlat = state.mode === 'delta' && flat(p);
      const stack = segs.filter(s => s.v > 0).map(s =>
        '<span class="kf-irp-bar-seg" style="--irp-c:' + s.c + '; height:' + (tot ? (s.v / tot) * 100 : 0) + '%"></span>'
      ).join('') || '<span class="kf-irp-bar-seg" style="--irp-c: var(--border-strong); height:100%"></span>';

      return '<button type="button" class="kf-irp-col' +
        (isFlat ? ' is-flat' : '') +
        (state.onlyChanged && flat(p) ? ' is-muted' : '') +
        (p.i <= state.done ? ' is-done' : '') +
        '" data-irp-i="' + p.i + '" aria-pressed="' + (p.i === state.sel) + '"' +
        ' aria-label="' + esc(p.i + ' ' + p.name) + '">' +
        '<span class="kf-irp-stack" style="height:' + h + '%">' + stack + '</span></button>';
    }).join('');

    els.axis.innerHTML = P.map(p =>
      '<span class="' + (p.i === state.sel ? 'is-on' : '') + '">' +
      String(p.i).padStart(2, '0') + '</span>').join('');

    els.pos.textContent = String(state.sel).padStart(2, '0');
  }

  /* ---------- inspector ---------- */

  function delta(cur, prev) {
    if (prev === null || prev === undefined) return '';
    const d = cur - prev;
    if (!d) return '<em class="flat">±0</em>';
    return '<em class="' + (d > 0 ? 'up' : 'down') + '">' + (d > 0 ? '+' : '') + d + '</em>';
  }

  function renderInfo() {
    const p = P[state.sel];
    const prev = state.sel > 0 ? P[state.sel - 1] : null;
    const s = stratumOf(p.i);
    const sv = stratumVar(p.i);
    const ftot = FKEY.reduce((a, k) => a + p.f[k.k], 0);
    const pftot = prev ? FKEY.reduce((a, k) => a + prev.f[k.k], 0) : null;

    const props =
      p.gain.map(x => '<span class="kf-irp-prop">' + x + '</span>').join('') +
      p.lose.map(x => '<span class="kf-irp-prop is-lost">' + x + '</span>').join('') ||
      '<span class="kf-irp-prop is-none">不声明 produced / invalidated</span>';

    const shown = FKEY.filter(k => p.f[k.k] > 0);
    const ribbon = shown.map(k =>
      '<span style="--irp-c:' + k.c + '; flex:' + p.f[k.k] + ' 1 0" title="' + k.label + ' × ' + p.f[k.k] + '"></span>').join('');
    const key = shown.map(k =>
      '<span style="--irp-c:' + k.c + '"><i></i>' + k.label + ' ' + p.f[k.k] + '</span>').join('');

    els.info.innerHTML =
      '<div class="kf-irp-head" style="--irp-c:' + sv + '">' +
        '<span class="kf-irp-idx">' + String(p.i).padStart(2, '0') + '</span>' +
        '<span class="kf-irp-title">' + esc(p.name) + '</span>' +
        '<span class="kf-irp-tag is-stratum" style="--irp-c:' + sv + '">' + s.id + ' ' + s.name + '</span>' +
        '<span class="kf-irp-tag">' + p.c + '</span>' +
      '</div>' +
      '<p class="kf-irp-desc">' + esc(p.desc) + '</p>' +
      '<dl class="kf-irp-metrics">' +
        '<div><dt>IR 行数</dt><dd>' + p.l + delta(p.l, prev && prev.l) + '</dd></div>' +
        '<div><dt>改动行</dt><dd>' + (p.d === null ? '—' : p.d) + '</dd></div>' +
        '<div><dt>函数</dt><dd>' + ftot + delta(ftot, pftot) + '</dd></div>' +
        '<div><dt>tensor 算子</dt><dd>' + p.o.te + delta(p.o.te, prev && prev.o.te) + '</dd></div>' +
        '<div><dt>tile 算子</dt><dd>' + p.o.ti + delta(p.o.ti, prev && prev.o.ti) + '</dd></div>' +
        '<div><dt>MemRef</dt><dd>' + p.o.mr + delta(p.o.mr, prev && prev.o.mr) + '</dd></div>' +
      '</dl>' +
      '<p class="kf-irp-sec">性质流转</p>' +
      '<div class="kf-irp-props">' + props + '</div>' +
      '<p class="kf-irp-sec">函数组成</p>' +
      '<div class="kf-irp-ribbon">' + ribbon + '</div>' +
      '<div class="kf-irp-key">' + key + '</div>';
  }

  function renderDiff() {
    const p = P[state.sel];
    const head =
      '<div class="kf-irp-head">' +
        '<span class="kf-irp-title">IR 变化 · ' + esc(p.file) + '</span>' +
        '<span class="kf-irp-tag">' + (p.hunks.length ? p.hunks.length + ' 个片段' : '无改动') + '</span>' +
      '</div>';

    if (!p.hunks.length) {
      els.diff.innerHTML = head + '<p class="kf-irp-empty">' +
        (p.i === 0 ? '基准快照,无前序 pass 可比。'
          : p.c === '?' ? '该 pass 在当前编译器源码中已不存在,本次运行也未改动 IR。'
          : '本次运行中该 pass 未改动 IR —— 前后快照逐字节相同。') + '</p>';
      return;
    }

    els.diff.innerHTML = head + '<div class="kf-irp-diff-body">' + p.hunks.map(h =>
      '<div class="kf-irp-hunk">' +
        '<div class="kf-irp-hunk-h">@ line ' + h.at + '</div>' +
        (h.b.length ? '<pre class="is-before">' + esc(h.b.join('\n')) + '</pre>' : '') +
        (h.a.length ? '<pre class="is-after">' + esc(h.a.join('\n')) + '</pre>' : '') +
      '</div>').join('') + '</div>';
  }

  function render() { renderTrack(); renderInfo(); renderDiff(); syncStageHead(); }

  function syncStageHead() {
    const p = P[state.sel];
    const nameEl = $('#activePassName');
    if (nameEl) nameEl.textContent = p.name;
  }

  function select(i) {
    state.sel = Math.max(0, Math.min(LAST, i));
    render();
  }

  /* ---------- events ---------- */

  function wire() {
    els.cols.addEventListener('click', e => {
      const b = e.target.closest('.kf-irp-col');
      if (b) select(+b.dataset.irpI);
    });

    els.cols.addEventListener('mousemove', e => {
      const b = e.target.closest('.kf-irp-col');
      if (!b) { els.tip.style.opacity = '0'; return; }
      const p = P[+b.dataset.irpI];
      const segs = MODES[state.mode].segs(p).filter(s => s.v > 0);
      const detail = state.mode === 'delta'
        ? (p.d === null ? '基准快照' : p.d + ' 行改动')
        : segs.map(s => (s.name || '') + ' ' + s.v).join(' · ');
      els.tip.innerHTML = '<b>' + String(p.i).padStart(2, '0') + ' ' + esc(p.name) + '</b><br>' + esc(detail);
      els.tip.style.opacity = '1';
      const r = els.tip.getBoundingClientRect();
      els.tip.style.left = Math.min(e.clientX + 12, window.innerWidth - r.width - 8) + 'px';
      els.tip.style.top = Math.max(e.clientY - r.height - 10, 8) + 'px';
    });
    els.cols.addEventListener('mouseleave', () => { els.tip.style.opacity = '0'; });

    els.modes.addEventListener('click', e => {
      const b = e.target.closest('button');
      if (!b) return;
      state.mode = b.dataset.irpMode;
      Array.prototype.forEach.call(els.modes.querySelectorAll('button'),
        x => x.setAttribute('aria-pressed', String(x === b)));
      renderTrack();
    });

    els.filter.addEventListener('click', () => {
      state.onlyChanged = !state.onlyChanged;
      els.filter.setAttribute('aria-pressed', String(state.onlyChanged));
      renderTrack();
    });

    els.root.addEventListener('click', e => {
      const b = e.target.closest('[data-irp-step]');
      if (!b) return;
      let i = state.sel + (+b.dataset.irpStep);
      if (state.onlyChanged) while (i > 0 && i < LAST && flat(P[i])) i += (+b.dataset.irpStep);
      select(i);
    });

    document.addEventListener('keydown', e => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      const stage = document.querySelector('[data-stage="2"]');
      if (!stage || !stage.classList.contains('is-active')) return;
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT')) return;
      e.preventDefault();
      select(state.sel + (e.key === 'ArrowRight' ? 1 : -1));
    });

    // Sweep the real pipeline alongside the existing compile run.
    const run = document.getElementById('runCompile');
    if (run) run.addEventListener('click', sweep);
  }

  /* ---------- compile sweep ---------- */

  let sweeping = false;
  async function sweep() {
    if (sweeping) return;
    sweeping = true;
    state.done = -1;

    // index 0 is the frontend baseline, not a pass
    const changed = P.filter(p => p.i > 0 && !flat(p)).length;
    for (let i = 0; i <= LAST; i += 1) {
      state.done = i;
      state.sel = i;
      renderTrack(); renderInfo(); renderDiff(); syncStageHead();
      const col = els.cols.querySelector('[data-irp-i="' + i + '"]');
      if (col) {
        col.classList.add('is-running');
        col.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      }
      await new Promise(r => setTimeout(r, flat(P[i]) ? 12 : 46));
      if (col) col.classList.remove('is-running');
    }

    const status = $('#compileStatus');
    if (status) {
      status.textContent = LAST + ' / ' + LAST + ' Pass 通过';
      status.className = 'kf-state-chip good';
    }
    const summary = $('#guardSummary');
    if (summary) summary.textContent = changed + ' 个 pass 改写了 IR · ' + (P.length - changed) + ' 个未命中';

    select(13);
    sweeping = false;
  }

  /* ---------- boot ---------- */

  function boot() {
    if (!mount()) return;
    wire();
    select(0);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
