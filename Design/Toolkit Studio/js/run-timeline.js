/* Execution timeline for one run — the "what actually ran" view.

   Two tracks share one time axis:
     逻辑任务  428 tasks from deps.json, grouped onto the 38 kernels they lower to
     核心泳道  73 AIC / AIV cores, from the Chrome-Trace swimlane
   Selecting a block opens an inspector with that task's real record: IO tensors
   (dtype / shape / strides), dependency edges, and the files it produced.

   Data: js/run-trace-data.js (generated). Times are microseconds from run start.
   Mounted by task-history.js into the run detail page. */
(function () {
  'use strict';

  const $ = (s, r) => (r || document).querySelector(s);
  const esc = (s) => String(s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const st = { sel: null, tab: 'sum', track: 'both', host: null,
               scroll: { tasks: 0, cores: 0 }, reveal: false };

  const us = (v) => v >= 1000 ? (v / 1000).toFixed(2) + ' ms' : v.toFixed(2) + ' µs';
  const kbytes = (b) => b >= 1048576 ? (b / 1048576).toFixed(1) + ' MB'
                      : b >= 1024 ? Math.round(b / 1024) + ' KB' : b + ' B';

  function data() { return window.PTO_RUN_TRACE || null; }

  /* ---------- axis ---------- */
  function axis(D) {
    const step = 100, out = [];
    for (let t = 0; t <= D.span; t += step) {
      out.push('<span style="left:' + (t / D.span * 100) + '%">' + t + '</span>');
    }
    return '<div class="kf-tl-axis"><div class="kf-tl-lbl">时间 µs</div>' +
      '<div class="kf-tl-ax">' + out.join('') + '</div></div>';
  }

  /* ---------- track 1: logical tasks, grouped by kernel ---------- */
  function taskTrack(D) {
    // one row per kernel that actually executed, earliest first
    const rows = D.kroll.slice().sort((a, b) => a.s - b.s);
    return rows.map(k => {
      const name = D.kernels[k.k].name;
      const be = D.kernels[k.k].be || '—';
      // a task can lower to more than one kernel, so it appears under each of
      // them, positioned by that kernel's own span rather than the task's
      const mine = [];
      D.tasks.forEach((t, i) => t.ks.forEach(x => { if (x.k === k.k) mine.push([t, i, x]); }));
      const blocks = mine.map(([t, i, x]) => {
        const w = Math.max(0.16, (x.e - x.s) / D.span * 100);
        const on = st.sel === i ? ' is-sel' : '';
        const title = name + ' · ' + t.r + 't' + t.ti +
          '\n' + us(x.s) + ' → ' + us(x.e) + '（wall ' + us(x.e - x.s) + '）' +
          '\n' + x.c + ' 核 · 累计忙碌 ' + us(x.b) +
          (t.ks.length > 1 ? '\n同一任务还下沉为 ' + t.ks.filter(y => y.k !== k.k).map(y => y.kn).join('、') : '');
        return '<i class="kf-tl-b' + on + '" data-tl-task="' + i + '"' +
          ' style="left:' + (x.s / D.span * 100) + '%;width:' + w + '%"' +
          ' title="' + esc(title) + '"></i>';
      }).join('');
      const hit = mine.some(m => m[1] === st.sel);
      return '<div class="kf-tl-row is-' + be.toLowerCase() + (hit ? ' is-hit' : '') +
        '" data-tl-kernel="' + k.k + '">' +
        '<div class="kf-tl-lbl" title="' + esc(name + ' · ' + be + ' · ' + k.n + ' 次调用 · 累计忙碌 ' + us(k.busy) + ' · 最多 ' + k.maxc + ' 核并发') + '"><code>' + esc(name) + '</code>' +
          '<em>' + esc(be) + '</em><b>' + k.n + '</b></div>' +
        '<div class="kf-tl-lane">' + blocks + '</div>' +
      '</div>';
    }).join('');
  }

  /* ---------- track 2: physical cores ---------- */
  function laneTrack(D) {
    const byLane = [];
    D.lanes.forEach(() => byLane.push([]));
    D.segs.forEach(s => { if (byLane[s[0]]) byLane[s[0]].push(s); });
    return D.lanes.map((l, li) => {
      const blocks = byLane[li].map(s => {
        const t = D.tasks[s[1]];
        const w = Math.max(0.16, s[3] / D.span * 100);
        const on = st.sel === s[1] ? ' is-sel' : '';
        return '<i class="kf-tl-b' + on + '" data-tl-task="' + s[1] + '"' +
          ' style="left:' + (s[2] / D.span * 100) + '%;width:' + w + '%"' +
          ' title="' + esc(l.n + ' · ' + (t ? t.kn : '') + '\n' + us(s[2]) + ' + ' + us(s[3])) + '"></i>';
      }).join('');
      const busy = byLane[li].reduce((a, s) => a + s[3], 0);
      const hit = st.sel != null && D.tasks[st.sel] && D.tasks[st.sel].l.indexOf(li) >= 0;
      return '<div class="kf-tl-row is-core is-' + l.k.toLowerCase() + (hit ? ' is-hit' : '') + '">' +
        '<div class="kf-tl-lbl" title="' + esc(l.n + ' · 忙碌 ' + us(busy) + ' / ' + us(D.span) + ' · ' + byLane[li].length + ' 段') + '"><code>' + esc(l.n) + '</code>' +
          '<b>' + Math.round(busy / D.span * 100) + '%</b></div>' +
        '<div class="kf-tl-lane">' + blocks + '</div>' +
      '</div>';
    }).join('');
  }

  /* ---------- inspector ---------- */
  const TABS = [['sum', '概览'], ['io', '输入输出'], ['dep', '依赖'], ['art', '产物']];

  function inspector(D) {
    if (st.sel == null) {
      return '<div class="kf-tl-insp is-empty">' +
        '<p>点击任一色块，查看该次任务的完整记录。</p>' +
        '<dl><dt>逻辑任务</dt><dd>' + D.counts.tasks + '</dd>' +
          '<dt>下沉 kernel</dt><dd>' + D.counts.kernels + '</dd>' +
          '<dt>参与核心</dt><dd>' + D.counts.lanes + '</dd>' +
          '<dt>依赖边</dt><dd>' + D.counts.edges + '</dd></dl>' +
      '</div>';
    }
    const t = D.tasks[st.sel];
    const K = D.kernels[t.k] || { name: t.kn, be: '', files: [] };
    const tabs = TABS.map(x =>
      '<button type="button" class="' + (st.tab === x[0] ? 'is-on' : '') + '" data-tl-tab="' + x[0] + '">' +
      x[1] + '</button>').join('');

    let body = '';
    if (st.tab === 'sum') {
      const rows = [
        ['任务 ID', t.id],
        ['轮次 / 序号', (t.r || '—') + ' · t' + (t.ti < 0 ? '?' : t.ti)],
        ['下沉 kernel', t.ks.map(x => x.kn + (D.kernels[x.k] && D.kernels[x.k].be ? '（' + D.kernels[x.k].be + '）' : '')).join(' + ')],
        ['调度域', t.sc === 'manual' ? 'manual（源码显式）' : t.sc === 'auto' ? 'auto（编译器推导）' : '—'],
        ['起 / 止', us(t.s) + ' → ' + us(t.e)],
        ['墙上时间', us(t.e - t.s)],
        ['占用核心', t.c + ' 个'],
        ['累计忙碌', us(t.b) + (t.c > 1 ? '（' + t.c + ' 核之和）' : '')],
        ['核心', t.l.map(i => D.lanes[i] ? D.lanes[i].n : i).join(' · ')]
      ];
      // the mixed-kernel case: one logical task, two lowered kernels on two
      // core kinds — worth spelling out rather than hiding behind a "+"
      const split = t.ks.length > 1
        ? '<div class="kf-tl-split"><h5>双核拆分 · ExpandMixedKernel</h5>' +
          t.ks.map(x => '<div><code>' + esc(x.kn) + '</code>' +
            '<em>' + (D.kernels[x.k] ? D.kernels[x.k].be : '') + '</em>' +
            '<small>' + x.c + ' 核 · ' + us(x.b) + '</small></div>').join('') +
          '<p>同一个逻辑任务在两类核上同时执行，产物也是两个独立 kernel。</p></div>'
        : '';
      body = '<dl class="kf-tl-kv">' + rows.map(r =>
        '<dt>' + esc(r[0]) + '</dt><dd>' + esc(String(r[1])) + '</dd>').join('') + '</dl>' + split;
    } else if (st.tab === 'io') {
      body = t.io && t.io.length
        ? '<table class="kf-tl-tbl"><thead><tr><th>#</th><th>方向</th><th>dtype</th><th>shape</th><th>strides</th><th>buffer</th></tr></thead><tbody>' +
          t.io.map(a => {
            const tn = D.tensors[a.x];
            return '<tr><td>' + a.i + '</td>' +
              '<td class="is-' + a.t + '">' + a.t + '</td>' +
              '<td>' + esc(a.d) + '</td>' +
              '<td>[' + a.sh.join(', ') + ']</td>' +
              '<td>[' + a.st.join(', ') + ']</td>' +
              '<td>' + (tn ? tn.n.toLocaleString() + ' elem' : '—') + '</td></tr>';
          }).join('') + '</tbody></table>'
        : '<p class="kf-tl-none">这条记录来自运行时轨迹，deps.json 中没有对应的参数表。</p>';
    } else if (st.tab === 'dep') {
      const pre = D.edges.filter(e => e[1] === st.sel);
      const suc = D.edges.filter(e => e[0] === st.sel);
      const list = (arr, pick) => arr.length
        ? '<div class="kf-tl-deps">' + arr.slice(0, 24).map(e => {
            const j = pick(e), o = D.tasks[j];
            return '<button type="button" data-tl-task="' + j + '">' +
              '<code>' + esc(o ? o.kn : '?') + '</code>' +
              '<em>' + (e[2] ? '数据' : '显式') + '</em>' +
              '<small>' + (o ? us(o.s) : '') + '</small></button>';
          }).join('') + (arr.length > 24 ? '<span class="kf-tl-more">…另有 ' + (arr.length - 24) + ' 条</span>' : '') + '</div>'
        : '<p class="kf-tl-none">无</p>';
      body = '<div class="kf-tl-dsec"><h5>前驱 · ' + pre.length + '</h5>' + list(pre, e => e[0]) + '</div>' +
             '<div class="kf-tl-dsec"><h5>后继 · ' + suc.length + '</h5>' + list(suc, e => e[1]) + '</div>' +
             '<p class="kf-tl-note">「数据」= 由张量产出关系推出；「显式」= 源码里写死的顺序。</p>';
    } else {
      const groups = t.ks.map(x => [x.kn, (D.kernels[x.k] || {}).files || []])
        .filter(g => g[1].length);
      body = groups.length
        ? groups.map(([kn, files]) =>
            (t.ks.length > 1 ? '<h5 class="kf-tl-fh">' + esc(kn) + '</h5>' : '') +
            '<div class="kf-tl-files">' + files.map(f =>
              '<div><code>' + esc(f[0]) + '</code><small>' + kbytes(f[1]) + '</small></div>'
            ).join('') + '</div>').join('')
        : '<p class="kf-tl-none">该 kernel 没有独立落盘的产物。</p>';
    }

    return '<div class="kf-tl-insp">' +
      '<div class="kf-tl-ihead">' +
        '<div><span class="kf-eyebrow">' + esc(t.r || 'task') + ' · t' + (t.ti < 0 ? '?' : t.ti) + '</span>' +
          '<h4>' + esc(t.ks.length > 1 ? t.ks.map(x => x.kn).join(' + ') : K.name) + '</h4></div>' +
        '<button type="button" class="kf-tl-x" data-tl-close aria-label="关闭">×</button>' +
      '</div>' +
      '<div class="kf-tl-tabs">' + tabs + '</div>' +
      '<div class="kf-tl-ibody">' + body + '</div>' +
    '</div>';
  }

  /* Both tracks are short scroll boxes over long lists (39 kernel rows, 73
     lanes), and every selection re-renders them wholesale — so scroll offsets
     have to be carried across the re-render by hand, and the rows that the
     selection actually hit have to be brought into view. */
  function saveScroll() {
    if (!st.host) return;
    const a = $('.kf-tl-tasks', st.host), b = $('.kf-tl-cores', st.host);
    if (a) st.scroll.tasks = a.scrollTop;
    if (b) st.scroll.cores = b.scrollTop;
  }

  function restoreScroll() {
    const a = $('.kf-tl-tasks', st.host), b = $('.kf-tl-cores', st.host);
    if (a) { a.scrollTop = st.scroll.tasks; a.addEventListener('scroll', saveScroll, { passive: true }); }
    if (b) { b.scrollTop = st.scroll.cores; b.addEventListener('scroll', saveScroll, { passive: true }); }
  }

  // centre a hit row inside its own scroll box; leave it alone if already shown
  function revealIn(box, el) {
    if (!box || !el) return;
    const row = el.closest('.kf-tl-row') || el;
    const b = box.getBoundingClientRect(), r = row.getBoundingClientRect();
    if (r.top >= b.top && r.bottom <= b.bottom) return;
    box.scrollTop += (r.top - b.top) - (b.height - r.height) / 2;
    saveScroll();
  }

  function revealSelection() {
    if (st.sel == null || !st.host) return;
    ['.kf-tl-tasks', '.kf-tl-cores'].forEach(sel => {
      const box = $(sel, st.host);
      revealIn(box, box && $('.kf-tl-b.is-sel', box));
    });
  }

  /* ---------- mount ---------- */
  function mount(host) {
    const D = data();
    if (!host) return;
    if (st.host === host) saveScroll();
    st.host = host;
    if (!D) { host.innerHTML = ''; return; }

    const wide = D.kroll[0];
    const wideT = D.tasks.reduce((a, t, i) =>
      (t.e - t.s) > (a ? D.tasks[a].e - D.tasks[a].s : -1) ? i : a, null);
    const W = D.tasks[wideT];

    host.innerHTML =
      '<section class="kf-rd-sec kf-tl">' +
        '<div class="kf-rd-h">执行时间线' +
          '<small>' + D.counts.tasks + ' 个逻辑任务落到 ' + D.counts.kernels + ' 个 kernel · ' +
          D.counts.lanes + ' 个核心 · 全程 ' + us(D.span) +
          ' · 来自 dfx_outputs/</small></div>' +
        '<div class="kf-tl-hint">' +
          '<span><i class="is-aic"></i>AIC（Cube）</span>' +
          '<span><i class="is-aiv"></i>AIV（Vector）</span>' +
          '<span class="kf-tl-fact">最宽任务 <code>' + esc(W ? W.kn : '') + '</code> ' +
            (W ? us(W.e - W.s) : '') + ' 墙上时间 / ' + (W ? W.c : 0) + ' 核并发 / 累计忙碌 ' +
            (W ? us(W.b) : '') + '</span>' +
        '</div>' +
        '<div class="kf-tl-grid">' +
          '<div class="kf-tl-chart">' +
            axis(D) +
            '<div class="kf-tl-thead">逻辑任务 · 按 kernel 归组 <b>' + D.counts.tasks + '</b></div>' +
            '<div class="kf-tl-scroll kf-tl-tasks">' + taskTrack(D) + '</div>' +
            '<div class="kf-tl-thead">核心泳道 · AIC / AIV <b>' + D.counts.lanes + '</b></div>' +
            '<div class="kf-tl-scroll kf-tl-cores">' + laneTrack(D) + '</div>' +
          '</div>' +
          inspector(D) +
        '</div>' +
      '</section>';

    restoreScroll();
    if (st.reveal) { st.reveal = false; revealSelection(); }
  }

  function rerender() { if (st.host) mount(st.host); }

  document.addEventListener('click', (e) => {
    if (!st.host || !st.host.contains(e.target)) return;
    if (e.target.closest('[data-tl-close]')) { st.sel = null; rerender(); return; }
    const tab = e.target.closest('[data-tl-tab]');
    if (tab) { st.tab = tab.dataset.tlTab; rerender(); return; }
    const b = e.target.closest('[data-tl-task]');
    if (b) { st.sel = Number(b.dataset.tlTask); st.reveal = true; rerender(); return; }
    const k = e.target.closest('[data-tl-kernel]');
    if (k) {                                   // row label: jump to its first task
      const kid = Number(k.dataset.tlKernel);
      const D = data();
      const i = D.tasks.findIndex(t => t.k === kid);
      if (i >= 0) { st.sel = i; st.reveal = true; rerender(); }
    }
  });

  window.PTO_TIMELINE = { mount, reset() { st.sel = null; st.tab = 'sum'; } };
})();
