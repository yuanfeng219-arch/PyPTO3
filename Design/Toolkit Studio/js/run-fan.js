/* Run composition fan — what this run is made of, one facet at a time.

   Centre    the run itself, its ring drawn from the real busy envelope of the
             73 core lanes across the 993 µs span
   Facets    the five countable dimensions of the run
   Leaves    the members of the selected facet, each with the one number that
             matters for it and a segment bar of the verdicts it actually has

   The segment bar is per-facet, not a fixed three: a kernel genuinely has three
   independent verdicts (memory / intent / diagnostics), a pass has three
   (function count / scope count / IR size), a core has one. Padding a facet out
   to three cells would be inventing a judgement it does not have.

   Data: js/ir-kernels-data.js + js/run-trace-data.js (both generated).
   Mounted by task-history.js as the 运行切片 tab. */
(function () {
  'use strict';

  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.prototype.slice.call((r || document).querySelectorAll(s));
  const esc = (s) => String(s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const kb = (b) => b >= 1048576 ? (b / 1048576).toFixed(1) + ' MB'
                  : b >= 1024 ? Math.round(b / 1024) + ' KB' : b + ' B';
  const num = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

  const SPACE_LABEL = { Vec: 'Vec (UB)', Mat: 'Mat (L1)', Acc: 'Acc (L0C)',
                        Left: 'Left (L0A)', Right: 'Right (L0B)' };
  const spLabel = (s) => SPACE_LABEL[s] || s;

  const st = { facet: 'fn', sel: null, host: null, opts: {} };

  const K = () => window.PTO_IR_KERNELS;
  const D = () => window.PTO_RUN_TRACE;

  const pct = (a, b) => (b ? (a / b) * 100 : 0);
  function worst(k) {
    const C = K();
    let w = -1, sp = null, u = 0;
    for (const s of C.spaces) {
      const p = pct(k.mem[s] || 0, C.limits[s]);
      if (p > w) { w = p; sp = s; u = k.mem[s] || 0; }
    }
    return { p: w, sp, u, lim: C.limits[sp] };
  }

  /* ---------- centre ring ----------
     64 buckets across the run; each bar is how busy the whole chip was in that
     slice. Read off dfx segments, so the shape is the run's own profile. */
  function envelope(n) {
    const T = D();
    if (!T) return null;
    const b = new Array(n).fill(0);
    T.segs.forEach(s => {
      const from = s[2], to = s[2] + s[3];
      const i0 = Math.max(0, Math.floor(from / T.span * n));
      const i1 = Math.min(n - 1, Math.floor(to / T.span * n));
      for (let i = i0; i <= i1; i++) {
        const lo = Math.max(from, i / n * T.span), hi = Math.min(to, (i + 1) / n * T.span);
        if (hi > lo) b[i] += hi - lo;
      }
    });
    const max = Math.max.apply(null, b) || 1;
    return b.map(v => v / max);
  }

  function ring() {
    const n = 72, e = envelope(n);
    if (!e) return '';
    const cx = 92, cy = 92, r0 = 46;
    const bars = e.map((v, i) => {
      const a = (i / n) * Math.PI * 2 - Math.PI / 2;
      const len = 6 + v * 26;
      const x1 = cx + Math.cos(a) * r0, y1 = cy + Math.sin(a) * r0;
      const x2 = cx + Math.cos(a) * (r0 + len), y2 = cy + Math.sin(a) * (r0 + len);
      return '<line x1="' + x1.toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + x2.toFixed(1) +
        '" y2="' + y2.toFixed(1) + '" opacity="' + (0.25 + v * 0.6).toFixed(2) + '"/>';
    }).join('');
    return '<svg class="kf-fan-ring" viewBox="0 0 184 184" aria-hidden="true">' +
      '<g class="kf-fan-ring-bars">' + bars + '</g>' +
      '<circle class="kf-fan-orb" cx="92" cy="92" r="26"/>' +
    '</svg>';
  }

  /* ---------- facets ---------- */
  function facets() {
    const C = K(), T = D();
    const out = [
      { k: 'fn', n: C.kernels.length, label: '生成函数', sub: 'passes_dump 末态' },
      { k: 'pass', n: C.passNames.length, label: '编译 Pass', sub: '逐 pass 快照' }
    ];
    if (T) out.push({ k: 'core', n: T.counts.lanes, label: '参与核心', sub: 'dfx 泳道' });
    out.push({ k: 'space', n: C.spaces.length, label: '内存空间', sub: '片上 buffer' });
    if (T) out.push({ k: 'tensor', n: T.counts.tensors, label: '张量', sub: 'deps.json' });
    return out;
  }

  /* ---------- leaves ----------
     Each returns { id, name, tag, chip, chipU, segs:[tone], note, title }.
     `tone` is one of ok / warn / bad / dim. */
  const band = (p) => p > 100 ? 'bad' : p >= 80 ? 'warn' : p >= 40 ? 'ok' : 'dim';

  function leavesFn() {
    return K().kernels.map(k => {
      const w = worst(k);
      const mem = !w.u ? 'dim' : w.u > w.lim ? 'bad' : w.u === w.lim ? 'warn' : band(w.p);
      const intent = !k.intent.declared.length ? 'dim' : k.intent.demoted > 0 ? 'warn' : 'ok';
      const diag = k.diags.length ? 'warn' : 'ok';
      return {
        id: k.name, name: k.name, tag: k.type,
        chip: w.u ? Math.round(w.p) : '—', chipU: w.u ? '%' : '',
        segs: [mem, intent, diag],
        title: k.name + ' · ' + k.type +
          (w.u ? '\n' + spLabel(w.sp) + ' ' + kb(w.u) + ' / ' + kb(w.lim) + '（' + Math.round(w.p) + '%）'
               : '\n无片上分配') +
          '\n声明流水 ' + k.intent.declared.length + ' · 被降级 ' + k.intent.demoted +
          '\n诊断 ' + k.diags.length + ' 条' +
          '\n点击在 IR Pass 快照中展开'
      };
    }).sort((a, b) => (b.chip === '—' ? -1 : b.chip) - (a.chip === '—' ? -1 : a.chip));
  }

  function leavesPass() {
    const C = K();
    return C.passNames.map((n, i) => {
      const l = C.lineage[i], p = i ? C.lineage[i - 1] : null;
      const d = (a, b) => a === b ? 'dim' : a > b ? 'ok' : 'warn';
      return {
        id: n, name: n, tag: String(i).padStart(2, '0'),
        chip: l ? l.n : '—', chipU: ' 函数',
        segs: l && p ? [d(l.n, p.n), d(l.at, p.at), d(l.lines, p.lines)] : ['dim', 'dim', 'dim'],
        note: l ? l.at + ' 作用域 · ' + num(l.lines) + ' 行 IR' : '',
        title: n + '\n函数 ' + (l ? l.n : '—') + ' · 作用域 ' + (l ? l.at : '—') +
          ' · IR ' + (l ? num(l.lines) : '—') + ' 行' +
          (l && l.born.length ? '\n新增 ' + l.born.join('、') : '') +
          (l && l.gone.length ? '\n消失 ' + l.gone.join('、') : '')
      };
    });
  }

  function leavesCore() {
    const T = D();
    const busy = {};
    T.segs.forEach(s => { busy[s[0]] = (busy[s[0]] || 0) + s[3]; });
    return T.lanes.map((l, i) => {
      const p = pct(busy[i] || 0, T.span);
      return {
        id: l.n, name: l.n, tag: l.k,
        chip: Math.round(p), chipU: '%',
        segs: [band(p)],
        title: l.n + ' · ' + l.k + '\n忙碌 ' + (busy[i] || 0).toFixed(1) + ' µs / ' +
          T.span.toFixed(1) + ' µs（' + Math.round(p) + '%）'
      };
    }).sort((a, b) => b.chip - a.chip);
  }

  function leavesSpace() {
    const C = K();
    return C.spaces.map(s => {
      const on = C.kernels.filter(k => (k.mem[s] || 0) > 0);
      const peak = on.reduce((a, k) => Math.max(a, pct(k.mem[s], C.limits[s])), 0);
      const full = on.filter(k => k.mem[s] === C.limits[s]).length;
      return {
        id: s, name: spLabel(s), tag: on.length + ' kernel',
        chip: Math.round(peak), chipU: '%',
        segs: [band(peak)],
        note: '上限 ' + kb(C.limits[s]) + (full ? ' · ' + full + ' 个填满' : ''),
        title: spLabel(s) + '\n' + on.length + ' 个 kernel · 上限 ' + kb(C.limits[s]) +
          '\n峰值 ' + Math.round(peak) + '%' + (full ? '\n' + full + ' 个正好填满（零余量）' : '')
      };
    }).sort((a, b) => b.chip - a.chip);
  }

  // widths the dump actually emits; anything else falls back to 2 B/element
  const DTYPE = { FLOAT32: ['FP32', 4], BFLOAT16: ['BF16', 2], FLOAT16: ['FP16', 2],
                  INT32: ['INT32', 4], INT8: ['INT8', 1] };

  function leavesTensor() {
    return D().tensors.map((t, i) => {
      const d = DTYPE[t.d] || [t.d, 2];
      const b = t.n * d[1];
      const mb = b >= 1048576;
      return {
        id: 't' + i, name: num(t.n) + ' 元素', tag: d[0], bytes: b,
        chip: mb ? (b / 1048576).toFixed(b >= 10485760 ? 0 : 1) : Math.round(b / 1024),
        chipU: mb ? ' MB' : ' KB',
        segs: [],
        note: '@' + t.a.slice(-8),
        title: t.d + ' · ' + num(t.n) + ' 元素 · ' + kb(b) + '\n地址 ' + t.a
      };
    }).sort((a, b) => b.bytes - a.bytes);
  }

  const LEAVES = { fn: leavesFn, pass: leavesPass, core: leavesCore,
                   space: leavesSpace, tensor: leavesTensor };

  const CAPTION = {
    fn: ['每个函数一条线', '数字是它最紧的那块片上空间用了多少 · 三段依次为 内存 / 意图兑现 / 诊断'],
    pass: ['每个 pass 一条线', '数字是该 pass 之后还剩多少函数 · 三段依次为 函数数 / 作用域数 / IR 行数 相对上一个 pass'],
    core: ['每条泳道一条线', '数字是这颗核在 993 µs 里忙碌的比例'],
    space: ['每块片上空间一条线', '数字是这块空间上的峰值水位'],
    tensor: ['每个张量一条线', '数字是它的逻辑字节数']
  };

  /* ---------- render ---------- */
  function render() {
    if (!st.host || !K()) return;
    const F = facets();
    const rows = LEAVES[st.facet]();
    const cap = CAPTION[st.facet];
    const C = K();
    const m = (C.source || '').match(/^_jit_(.+)_(\d{8})_(\d{6})$/);
    const diags = C.kernels.reduce((a, k) => a + k.diags.length, 0);

    st.host.innerHTML =
      '<div class="kf-fan-h">运行切片<small>中心是这次运行，选一个切面看它由什么组成 · ' +
        esc(cap[1]) + '</small></div>' +
      '<div class="kf-fan">' +
        '<div class="kf-fan-centre">' +
          ring() +
          '<b>' + esc(m ? m[1] : C.source) + '</b>' +
          '<code>' + esc(m ? m[2] + '_' + m[3] : '') + '</code>' +
          '<span class="kf-fan-chip">诊断 ' + diags + 'P</span>' +
        '</div>' +
        '<div class="kf-fan-facets" role="tablist">' + F.map(f =>
          '<button type="button" role="tab" data-fan-facet="' + f.k + '"' +
            ' aria-selected="' + (f.k === st.facet) + '">' +
            '<b>' + num(f.n) + '</b><span>' + f.label + '</span><small>' + f.sub + '</small>' +
          '</button>').join('') + '</div>' +
        '<div class="kf-fan-leaves"><div class="kf-fan-scroll" id="fanScroll">' + rows.map(r =>
          '<button type="button" class="kf-fan-leaf' + (r.id === st.sel ? ' is-sel' : '') + '"' +
            ' data-fan-leaf="' + esc(r.id) + '" title="' + esc(r.title) + '">' +
            '<b>' + esc(r.name) + '</b>' +
            '<span class="kf-fan-leaf-chip">' + r.chip + '<i>' + esc(r.chipU) + '</i></span>' +
            '<em>' + esc(r.tag) + '</em>' +
            (r.segs.length
              ? '<span class="kf-fan-segs">' + r.segs.map(s =>
                  '<i class="is-' + s + '"></i>').join('') + '</span>'
              : '<span class="kf-fan-segs"></span>') +
            (r.note ? '<small>' + esc(r.note) + '</small>' : '') +
          '</button>').join('') + '</div></div>' +
        '<svg class="kf-fan-edges" id="fanEdges" aria-hidden="true"></svg>' +
      '</div>';

    draw();
  }

  /* Edges are measured, not laid out: the leaf column scrolls and the facet
     column is centred, so the only reliable endpoints are real bounding boxes. */
  function draw() {
    const wrap = $('.kf-fan', st.host), svg = $('#fanEdges', st.host);
    if (!wrap || !svg) return;
    const from = $('[data-fan-facet][aria-selected="true"]', st.host);
    const box = wrap.getBoundingClientRect();
    svg.setAttribute('viewBox', '0 0 ' + box.width + ' ' + box.height);
    if (!from) { svg.innerHTML = ''; return; }
    const f = from.getBoundingClientRect();
    const x1 = f.right - box.left, y1 = f.top + f.height / 2 - box.top;
    const clip = $('.kf-fan-leaves', st.host).getBoundingClientRect();

    svg.innerHTML = $$('[data-fan-leaf]', st.host).map(el => {
      const r = el.getBoundingClientRect();
      const cy = r.top + r.height / 2;
      if (cy < clip.top - 4 || cy > clip.bottom + 4) return '';   // scrolled out
      const x2 = r.left - box.left, y2 = cy - box.top;
      const dx = Math.max(40, (x2 - x1) * 0.55);
      const tone = (el.querySelector('.kf-fan-segs i.is-bad') && 'bad') ||
                   (el.querySelector('.kf-fan-segs i.is-warn') && 'warn') || 'ok';
      return '<path class="is-' + tone + (el.classList.contains('is-sel') ? ' is-sel' : '') + '" d="M' +
        x1.toFixed(1) + ' ' + y1.toFixed(1) + ' C' + (x1 + dx).toFixed(1) + ' ' + y1.toFixed(1) +
        ', ' + (x2 - dx).toFixed(1) + ' ' + y2.toFixed(1) + ', ' +
        x2.toFixed(1) + ' ' + y2.toFixed(1) + '"/>';
    }).join('');
  }

  function wire() {
    st.host.addEventListener('click', e => {
      const f = e.target.closest('[data-fan-facet]');
      if (f) { st.facet = f.dataset.fanFacet; st.sel = null; render(); return; }
      const l = e.target.closest('[data-fan-leaf]');
      if (!l) return;
      st.sel = l.dataset.fanLeaf;
      render();
      // the compile guard lives on another tab, so the host decides how to get
      // there rather than this module reaching across for it
      if (st.facet === 'fn' && st.opts.onKernel) st.opts.onKernel(st.sel);
    });
    // the scroll container is replaced on every render, so delegate from the
    // host rather than rebinding; capture, because scroll does not bubble
    st.host.addEventListener('scroll', draw, { capture: true, passive: true });
    if (typeof ResizeObserver === 'function') {
      if (ro) ro.disconnect();
      ro = new ResizeObserver(draw);
      ro.observe(st.host);
    }
  }
  let ro = null;

  window.PTO_FAN = {
    /* The IR tab rebuilds its panel, so mount() is called with a fresh host
       each time it opens. State (facet, selection) is module-level and rides
       across, which is what makes the tab feel like it was left where it was. */
    mount(host, opts) {
      if (!host || !K()) return false;
      const first = st.host !== host;
      st.host = host;
      st.opts = opts || {};
      render();
      if (first) wire();
      return true;
    },
    redraw: draw
  };
})();
