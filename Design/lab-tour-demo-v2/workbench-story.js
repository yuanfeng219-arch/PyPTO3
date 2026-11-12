/* 「工作怎么分」图形化视图（只接 qk_pv；其他 Kernel 仍用 lens.js 原图）。
 * 顶部一条流程条代替文字说明：5 个图标节点，每个一个图形 + 一个数字 + 两三个字，点节点即切换：
 *   ① 列活 1280 → ② 排队 992｜288 → ③ 发给 24 核 → ④ 54 轮 → ⑤ 分得好不好（最慢核 / 平均）
 * 下方同一批方块（一格 = 一份活）在五种排法之间移动；画布上只用短标签。
 * ⑤ 回答“计算逻辑写得好不好”：每个核一条长条，长度 = 这个核要做的工作量；所有核都要等最慢的那个，
 *   所以右端齐 = 分得好，参差 = 有核在空等（琥珀斜纹）。工作量可按“份数”或按“要搬的 KV 行数”看，
 *   排法可在“现在（真活在前）”与“对照：不排队直接发”之间切，方块会飞到对照方案里各自的核上。
 *   KV 行数是估算口径（Synthetic 输入，未上板），需 profiling 确认。
 * 数据取 LensData.buildQkScenario（与 lens.js 同一份）。lens.js 仍是选中 / 筛选 / 悬停的唯一来源：
 * 它的原图隐藏着照常工作，这里按 data-key 同步 is-sel / is-mate / is-out；方块沿用 .lane-cell[data-key][data-lane]。 */
(function () {
  'use strict';
  const D = window.LensData;
  if (!D || !D.buildQkScenario) return;
  const $ = function (id) { return document.getElementById(id); };
  const host = $('kernelLanes'), work = $('kernelWork'), bar = $('kernelBar'), tabs = $('kernelTabs'), meta = $('kernelMeta');
  if (!host || !bar || !tabs || !meta) return;
  const NS = 'http://www.w3.org/2000/svg';
  const F = D.FACTS, N = F.NUM_QK_CORES, SB = F.SPARSE_BLOCKS, PER_REQ = F.S;
  const pad2 = function (n) { return (n < 10 ? '0' : '') + n; };
  const esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
  const fmt = function (n) { return n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(Math.round(n)); };

  // ── 状态 ──
  let S = null, facts = null, rankShown = -1, step = null, evalMode = false, weight = 'rows', plan = 'now';
  let root = null, svg = null, cellLayer = null, annLayer = null, cells = [], W = 0, H = 0, timer = 0, pendingResize = false, lastStep = null, lastKey = '';

  const onTab = function (t) { const b = tabs.querySelector('[data-tab="' + t + '"]'); return !!b && b.getAttribute('aria-selected') === 'true'; };
  function curStep() {
    if (document.body.dataset.level !== '2' || !/^qk_pv\b/.test(meta.textContent)) return null;
    if (onTab('pipeline')) return evalMode ? 'eval' : 'rounds';
    if (!onTab('assignment')) return null;
    evalMode = false;
    const b = bar.querySelector('[data-l2="phase"][aria-pressed="true"]');
    return b ? b.dataset.v : 'assignment';
  }
  function curRank() { const s = bar.querySelector('[data-l2="rank"]'); return s ? Number(s.value) || 0 : 0; }

  // ── 数据与“好不好”的度量 ──
  // 两种排法：now = 现在（真活在前、空活在后，再按队序轮流发）；flat = 对照（不排队，按产生顺序直接轮流发）
  // 两种工作量口径：count = 每份真活算 1（Cube 按整块算，份数决定算力）；rows = 每份真活算它要搬的有效 KV 行数（估算）
  const costOf = function (it, w) { return !it.valid ? 0 : w === 'count' ? 1 : it.kv; };
  function assign(sc, p) {
    const lanes = []; for (let l = 0; l < N; l++) lanes.push([]);
    (p === 'now' ? sc.queue : sc.items).forEach(function (it, i) { lanes[i % N].push(it); });
    return lanes;
  }
  function measure(lanes, w) {
    const load = lanes.map(function (ls) { return ls.reduce(function (s, it) { return s + costOf(it, w); }, 0); });
    const max = Math.max.apply(null, load), avg = load.reduce(function (a, b) { return a + b; }, 0) / load.length;
    return { load: load, max: max, avg: avg, ratio: max / avg, wait: 1 - avg / max, slow: load.indexOf(max) };
  }
  function computeFacts(sc) {
    const total = sc.items.length, valid = sc.validCount, per = sc.lanes.map(function (l) { return l.items.length; });
    const fullReal = Math.floor(valid / N), mixed = valid % N ? 1 : 0;
    const A = { now: assign(sc, 'now'), flat: assign(sc, 'flat') }, M = {};
    ['now', 'flat'].forEach(function (p) { M[p] = { count: measure(A[p], 'count'), rows: measure(A[p], 'rows') }; });
    return { total: total, valid: valid, empty: total - valid, rounds: sc.iters, tokens: total / SB, reqs: total / SB / PER_REQ,
      per: per, fullReal: fullReal, mixed: mixed, emptyRounds: sc.iters - fullReal - mixed, A: A, M: M };
  }
  const verdict = function (r) { return r <= 1.03 ? 'is-good' : r <= 1.06 ? 'is-ok' : 'is-bad'; };

  // ── 流程条：图形 + 数字 + 两三个字 ──
  const G = {
    generation: '<svg viewBox="0 0 40 28" aria-hidden="true">' + (function () { let s = ''; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) s += '<rect x="' + (4 + c * 6.6) + '" y="' + (3 + r * 6) + '" width="5.4" height="4.8" rx="1" class="' + (c === 0 ? 'g-win' : (r + c) % 4 === 3 ? 'g-idle' : 'g-cmp') + '"/>'; return s; })() + '</svg>',
    ordering: '<svg viewBox="0 0 40 28" aria-hidden="true"><rect x="2" y="10" width="27" height="8" rx="2" class="g-cmp"/><rect x="2" y="10" width="7" height="8" rx="2" class="g-win"/><rect x="30" y="10" width="8" height="8" rx="2" class="g-idle"/><path d="M2 23 H38" class="g-line"/><path d="M34 20.5 L38 23 L34 25.5" class="g-line"/></svg>',
    assignment: '<svg viewBox="0 0 40 28" aria-hidden="true"><rect x="2" y="10" width="7" height="8" rx="1.5" class="g-cmp"/>' + [3, 9, 15, 21].map(function (y, i) { return '<path d="M10 14 L28 ' + (y + 2) + '" class="g-line"/><rect x="29" y="' + y + '" width="9" height="4" rx="1" class="' + (i % 2 ? 'g-cmp' : 'g-win') + '"/>'; }).join('') + '</svg>',
    rounds: '<svg viewBox="0 0 40 28" aria-hidden="true">' + (function () { let s = ''; for (let r = 0; r < 4; r++) for (let c = 0; c < 6; c++) s += '<rect x="' + (2 + c * 6.2) + '" y="' + (3 + r * 6) + '" width="5" height="4.6" rx="1" class="' + (c >= 4 ? 'g-idle' : (r + c) % 3 ? 'g-cmp' : 'g-win') + '"/>'; return s; })() + '<path d="M26.6 1 V27" class="g-cut"/></svg>',
    eval: '<svg viewBox="0 0 40 28" aria-hidden="true">' + [[3, 30], [9, 24], [15, 33], [21, 27]].map(function (b) { return '<rect x="2" y="' + b[0] + '" width="' + b[1] + '" height="4" rx="1" class="g-cmp"/>' + (b[1] < 33 ? '<rect x="' + (2 + b[1]) + '" y="' + b[0] + '" width="' + (33 - b[1]) + '" height="4" class="g-wait"/>' : ''); }).join('') + '<path d="M35.5 1 V27" class="g-finish"/></svg>'
  };
  const FLOW = ['generation', 'ordering', 'assignment', 'rounds', 'eval'];
  function flowNode(k) {
    const f = facts, m = f.M[plan][weight];
    const v = {
      generation: ['<b>' + f.total + '</b>', '份活'],
      ordering: ['<b>' + f.valid + '</b><i>｜</i><b class="is-dim">' + f.empty + '</b>', '真活在前'],
      assignment: ['<b>' + N + '</b>', '个核轮流领'],
      rounds: ['<b>' + f.rounds + '</b>', '轮'],
      eval: ['<b class="' + verdict(m.ratio) + '">' + m.ratio.toFixed(2) + '×</b>', '好不好 · 空等 ' + Math.round(m.wait * 100) + '%']
    }[k];
    return '<button type="button" class="wb-flow-n' + (k === step ? ' is-on' : '') + '" data-wb-flow="' + k + '" aria-pressed="' + (k === step) + '">'
      + '<span class="wb-flow-g">' + G[k] + '</span><span class="wb-flow-v">' + v[0] + '</span><span class="wb-flow-k">' + v[1] + '</span></button>';
  }
  function paintFlow() {
    root.querySelector('.wb-flow').innerHTML = FLOW.map(function (k, i) { return (i ? '<span class="wb-flow-arrow" aria-hidden="true"></span>' : '') + flowNode(k); }).join('');
    const tools = root.querySelector('.wb-eval-tools');
    tools.hidden = step !== 'eval';
    if (step === 'eval') {
      const seg = function (name, opts, cur) { return '<span class="wb-seg" role="group" aria-label="' + name + '">' + opts.map(function (o) { return '<button type="button" data-wb-' + name + '="' + o[0] + '" aria-pressed="' + (o[0] === cur) + '">' + o[1] + '</button>'; }).join('') + '</span>'; };
      tools.innerHTML = seg('weight', [['rows', 'KV 行数'], ['count', '份数']], weight) + seg('plan', [['now', '现在'], ['flat', '不排队']], plan);
    }
  }

  // ── 骨架 ──
  function build() {
    root = document.createElement('div');
    root.className = 'wb-story';
    root.innerHTML = '<div class="wb-story-head"><nav class="wb-flow" aria-label="工作怎么分：五步"></nav><span class="wb-eval-tools" hidden></span></div>'
      + '<div class="wb-story-canvas"></div>'
      + '<div class="wb-story-key"></div>';
    svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('class', 'wb-story-svg');
    svg.setAttribute('role', 'group');
    svg.innerHTML = '<defs>'
      + '<pattern id="wbIdle" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="4" height="4" fill="#1d1d1d"/><rect width="1.2" height="4" fill="#5a5a5a"/></pattern>'
      + '<pattern id="wbWait" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="rgb(251 197 104 / 10%)"/><rect width="1.4" height="5" fill="rgb(251 197 104 / 55%)"/></pattern>'
      + '</defs><g class="wb-under"></g><g class="wb-cells"></g><g class="wb-ann"></g>';
    root.querySelector('.wb-story-canvas').appendChild(svg);
    cellLayer = svg.querySelector('.wb-cells');
    annLayer = svg.querySelector('.wb-ann');
    root.addEventListener('click', onClick);
    if (window.ResizeObserver) new ResizeObserver(function () { schedule(true); }).observe(root.querySelector('.wb-story-canvas'));
  }
  function loadData(rank) {
    S = D.buildQkScenario(rank); facts = computeFacts(S); rankShown = rank;
    cellLayer.innerHTML = '';
    cells = S.items.map(function (it) {
      const r = document.createElementNS(NS, 'rect');
      r.setAttribute('class', 'lane-cell wb-cell ' + (!it.valid ? 'wb-k-idle' : it.sb === 0 ? 'k-win' : 'k-cmp'));
      r.dataset.key = it.id; r.dataset.lane = it.lane;
      const t = document.createElementNS(NS, 'title');
      t.textContent = 'Token ' + it.token + ' · KV 块 ' + it.sb + (it.valid ? ' · 有效 KV ' + it.kv + ' 行' : ' · 空活（' + it.reason + '）') + ' · 队列第 ' + it.q + ' 位 · 核 ' + it.lane + ' 第 ' + (it.iter + 1) + ' 轮';
      r.appendChild(t);
      cellLayer.appendChild(r);
      return r;
    });
    svg.classList.add('is-still');
  }

  // ── 图例：只放色块与两三个字 ──
  function paintKey() {
    const base = '<span><i class="k-win"></i>滑窗块</span><span><i class="k-cmp"></i>压缩块</span><span><i class="k-idle"></i>空活</span>';
    const extra = step === 'eval'
      ? '<span><i class="k-wait"></i>空等最慢的核</span><span><i class="k-finish"></i>全部做完</span><span class="k-note">' + (weight === 'rows' ? '长度 = 要搬的有效 KV 行数（估算）' : '长度 = 真活份数') + '</span>'
      : '<span class="k-note">一格 = 一份活</span>';
    root.querySelector('.wb-story-key').innerHTML = base + extra + '<span class="k-syn" title="有效 / 空块来自 Synthetic 输入；分配规则取自源码；未上板">Synthetic · 未上板</span>';
  }

  // ── 五种排法 ──
  function layoutGeneration() {
    const reqs = facts.reqs, cg = 8, lh = 14;
    let best = null;
    [[reqs, 1], [reqs / 2, 2], [reqs / 4, 4], [reqs / 8, 8]].forEach(function (o) {
      const cr = o[0], rr = o[1];
      const t = Math.min((W - 30 - (cr - 1) * cg) / (cr * SB), (H - rr * lh - (rr - 1) * cg) / (rr * PER_REQ), 18);
      if (!best || t > best.t) best = { cr: cr, rr: rr, t: t };
    });
    const t = Math.max(3, Math.floor(best.t)), cw = SB * t, ch = PER_REQ * t;
    const totalW = best.cr * cw + (best.cr - 1) * cg, x0 = Math.max(30, (W - totalW) / 2);
    const pos = S.items.map(function (it) {
      const ci = it.req % best.cr, ri = Math.floor(it.req / best.cr);
      return { x: x0 + ci * (cw + cg) + it.sb * t, y: ri * (ch + lh + cg) + lh + (it.token % PER_REQ) * t, w: t - 1, h: t - 1 };
    });
    let ann = '';
    for (let r = 0; r < reqs; r++) {
      const ci = r % best.cr, ri = Math.floor(r / best.cr);
      ann += '<text class="wb-ann-k" x="' + (x0 + ci * (cw + cg)) + '" y="' + (ri * (ch + lh + cg) + lh - 4) + '">' + (cw >= 40 ? '请求 ' + r : r) + '</text>';
    }
    // 第一张卡旁画坐标：↓ token、→ KV 块
    ann += '<path class="wb-ann-axis-l" d="M' + (x0 - 8) + ' ' + lh + ' V ' + (lh + ch) + '" marker-end="url(#wbAx)"/>'
      + '<text class="wb-ann-k is-strong" x="' + (x0 - 12) + '" y="' + (lh + ch / 2 + 4) + '" text-anchor="end">token</text>';
    const usedH = best.rr * (ch + lh) + (best.rr - 1) * cg;
    return { pos: pos, ann: ann, used: usedH, order: function (it) { return it.id; } };
  }
  function layoutOrdering() {
    const lh = 18, gapRow = 22;
    let best = null;
    for (let c = 24; c <= 160; c += 4) {
      const vr = Math.ceil(facts.valid / c), er = Math.ceil(facts.empty / c);
      const t = Math.min(W / c, (H - lh - gapRow) / (vr + er), 18);
      if (!best || t > best.t) best = { c: c, t: t, vr: vr };
    }
    const t = Math.max(3, Math.floor(best.t)), c = best.c, x0 = Math.max(0, (W - c * t) / 2), emptyTop = lh + best.vr * t + gapRow;
    const pos = S.items.map(function (it) {
      if (it.valid) return { x: x0 + (it.q % c) * t, y: lh + Math.floor(it.q / c) * t, w: t - 1, h: t - 1 };
      const k = it.q - facts.valid;
      return { x: x0 + (k % c) * t, y: emptyTop + Math.floor(k / c) * t, w: t - 1, h: t - 1 };
    });
    const ann = '<text class="wb-ann-k is-strong" x="' + x0 + '" y="' + (lh - 5) + '">▶ 队头 · 真活 ' + facts.valid + '</text>'
      + '<text class="wb-ann-k is-strong" x="' + x0 + '" y="' + (emptyTop - 7) + '">队尾 · 空活 ' + facts.empty + '</text>';
    return { pos: pos, ann: ann, used: emptyTop + Math.ceil(facts.empty / c) * t, order: function (it) { return it.q; } };
  }
  function matrix(rightW) {
    const lw = 48, top = 22, bottomAxis = 14, cols = facts.rounds, rows = N;
    const tw = Math.max(3, Math.min((W - lw - rightW) / cols, 24)), th = Math.max(3, Math.min((H - top - bottomAxis) / rows, 22));
    const usedW = lw + cols * tw + rightW, x0 = Math.max(0, (W - usedW) / 2);
    return { lw: lw, top: top, tw: tw, th: th, x0: x0, cx: x0 + lw, used: top + rows * th + bottomAxis };
  }
  function laneLabels(m, mark) {
    const every = m.th >= 9 ? 1 : m.th >= 5 ? 2 : 4;
    let s = '';
    for (let l = 0; l < N; l++) {
      if (l % every && l !== mark) continue;
      s += '<text class="lane-label wb-lane-k' + (l === mark ? ' is-mark' : '') + '" data-lane="' + l + '" x="' + (m.x0 + m.lw - 8) + '" y="' + (m.top + l * m.th + m.th / 2 + 3.5) + '" text-anchor="end">核 ' + pad2(l) + '</text>';
    }
    return s;
  }
  function layoutAssignment() {
    const m = matrix(0);
    const pos = S.items.map(function (it) { return { x: m.cx + it.iter * m.tw, y: m.top + it.lane * m.th, w: m.tw - 1.5, h: m.th - 1 }; });
    let ann = laneLabels(m);
    for (let r = 0; r < 2; r++) {
      ann += '<rect class="wb-ann-col" x="' + (m.cx + r * m.tw - 1.5) + '" y="' + (m.top - 3) + '" width="' + (m.tw + 1.5) + '" height="' + (N * m.th + 4) + '" rx="3"/>';
    }
    ann += '<text class="wb-ann-k is-strong" x="' + m.cx + '" y="' + (m.top - 8) + '">第 1 轮 · 第 2 轮 → …</text>';
    return { pos: pos, ann: ann, used: m.used, order: function (it) { return it.q; } };
  }
  function layoutRounds() {
    const m = matrix(40);
    const pos = S.items.map(function (it) { return { x: m.cx + it.iter * m.tw, y: m.top + it.lane * m.th, w: m.tw - 1.5, h: m.th - 1 }; });
    const bx = m.cx + facts.fullReal * m.tw, ex = m.cx + (facts.fullReal + facts.mixed) * m.tw, endX = m.cx + facts.rounds * m.tw, bottom = m.top + N * m.th;
    let ann = '<rect class="wb-ann-zone" x="' + (ex - 1) + '" y="' + (m.top - 3) + '" width="' + (endX - ex + 2) + '" height="' + (N * m.th + 4) + '" rx="3"/>'
      + '<line class="wb-ann-cut" x1="' + (bx - .75) + '" y1="' + (m.top - 16) + '" x2="' + (bx - .75) + '" y2="' + (bottom + 2) + '"/>'
      + '<text class="wb-ann-k is-strong" x="' + (bx - 6) + '" y="' + (m.top - 8) + '" text-anchor="end">← 真活 ' + facts.fullReal + ' 轮</text>'
      + '<text class="wb-ann-k is-muted" x="' + (ex + 4) + '" y="' + (m.top - 8) + '">空活 ' + facts.emptyRounds + ' 轮 →</text>';
    ann += laneLabels(m);
    const every = m.th >= 9 ? 1 : m.th >= 5 ? 2 : 4;
    for (let l = 0; l < N; l += every) ann += '<text class="wb-ann-k" x="' + (endX + 8) + '" y="' + (m.top + l * m.th + m.th / 2 + 3.5) + '">' + facts.per[l] + '</text>';
    if (H - bottom >= 12) [1, 10, 20, 30, 40, 50].filter(function (r) { return r <= facts.rounds; }).forEach(function (r) {
      ann += '<text class="wb-ann-axis" x="' + (m.cx + (r - .5) * m.tw) + '" y="' + (bottom + 11) + '" text-anchor="middle">' + r + '</text>';
    });
    return { pos: pos, ann: ann, used: m.used, order: function () { return 0; } };
  }
  // ⑤ 分得好不好：每个核一条长条，长度 = 工作量；终点线 = 最慢的核；琥珀斜纹 = 其他核在空等
  function layoutEval() {
    const lanes = facts.A[plan], m0 = facts.M[plan][weight];
    const m = matrix(96), x0 = m.cx, span = (W - m.lw - 96 - m.x0) - 8, k = span / m0.max, bottom = m.top + N * m.th;
    const pos = new Array(S.items.length), byId = {};
    S.items.forEach(function (it, i) { byId[it.id] = i; });
    lanes.forEach(function (ls, l) {
      let x = x0;
      ls.forEach(function (it) {
        const w = costOf(it, weight) * k;
        pos[byId[it.id]] = { x: x, y: m.top + l * m.th, w: Math.max(0, w - (w > 3 ? .8 : 0)), h: m.th - 1, hide: !w };
        x += w;
      });
    });
    const fx = x0 + m0.max * k, ax = x0 + m0.avg * k;
    let under = '';
    m0.load.forEach(function (v, l) {
      const ex = x0 + v * k;
      if (fx - ex > .5) under += '<rect class="wb-wait" x="' + ex.toFixed(1) + '" y="' + (m.top + l * m.th) + '" width="' + (fx - ex).toFixed(1) + '" height="' + (m.th - 1) + '"/>';
    });
    let ann = laneLabels(m, m0.slow)
      + '<line class="wb-ann-finish" x1="' + fx + '" y1="' + (m.top - 14) + '" x2="' + fx + '" y2="' + (bottom + 2) + '"/>'
      + '<line class="wb-ann-avg" x1="' + ax + '" y1="' + (m.top - 4) + '" x2="' + ax + '" y2="' + (bottom + 2) + '"/>'
      + '<text class="wb-ann-k is-strong" x="' + (fx + 4) + '" y="' + (m.top - 6) + '">全部做完</text>'
      + '<text class="wb-ann-k is-muted" x="' + (ax - 4) + '" y="' + (m.top - 6) + '" text-anchor="end">平均</text>';
    // 右侧：每个核的工作量；最慢的核标出来
    const every = m.th >= 9 ? 1 : m.th >= 5 ? 2 : 4, unit = weight === 'rows' ? ' 行' : ' 份';
    m0.load.forEach(function (v, l) {
      if (l % every && l !== m0.slow) return;
      ann += '<text class="wb-ann-k' + (l === m0.slow ? ' is-slow' : '') + '" x="' + (fx + 8) + '" y="' + (m.top + l * m.th + m.th / 2 + 3.5) + '">' + (l === m0.slow ? '最慢 ' : '') + fmt(v) + unit + '</text>';
    });
    // 结论徽标：最慢 / 平均、空等占比
    const bx = x0 + 6, by = bottom + 8;
    if (H - bottom >= 26) ann += '<g class="wb-verdict-g ' + verdict(m0.ratio) + '"><text x="' + bx + '" y="' + (by + 12) + '"><tspan class="v-big">' + m0.ratio.toFixed(2) + '×</tspan><tspan dx="6">最慢 / 平均</tspan><tspan dx="16" class="v-big">' + Math.round(m0.wait * 100) + '%</tspan><tspan dx="6">算力在空等</tspan></text></g>';
    return { pos: pos, ann: ann, under: under, used: bottom + (H - bottom >= 26 ? 26 : 0), order: function (it) { return it.lane; }, spread: 500 };
  }
  const LAYOUT = { generation: layoutGeneration, ordering: layoutOrdering, assignment: layoutAssignment, rounds: layoutRounds, eval: layoutEval };

  // ── 画 ──
  function paint(resized) {
    const box = root.querySelector('.wb-story-canvas');
    W = Math.floor(box.clientWidth); H = Math.floor(box.clientHeight);
    if (!W || !H) return;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    paintFlow(); paintKey();
    const L = LAYOUT[step]();
    const dy = Math.max(0, Math.floor((H - (L.used || H)) / 2));
    const key = step + ':' + weight + ':' + plan;
    const animate = !resized && lastKey && lastKey !== key && !svg.classList.contains('is-still');
    svg.classList.toggle('is-still', !animate);
    const total = facts.total, spread = L.spread || (step === 'assignment' ? 1100 : step === 'ordering' ? 700 : 400);
    S.items.forEach(function (it, i) {
      const p = L.pos[i], el = cells[i];
      el.setAttribute('width', Math.max(p.hide ? 0 : 1, p.w).toFixed(2));
      el.setAttribute('height', Math.max(1, p.h).toFixed(2));
      el.classList.toggle('wb-hide', !!p.hide);
      el.style.transitionDelay = animate ? Math.round(L.order(it) / (step === 'eval' ? N : total) * spread) + 'ms' : '0ms';
      el.style.transform = 'translate(' + p.x.toFixed(2) + 'px,' + (p.y + dy).toFixed(2) + 'px)';
    });
    svg.querySelector('.wb-under').innerHTML = '<g transform="translate(0 ' + dy + ')">' + (L.under || '') + '</g>';
    annLayer.innerHTML = '<defs><marker id="wbAx" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#c6c7cb"/></marker></defs><g transform="translate(0 ' + dy + ')">' + L.ann + '</g>';
    annLayer.classList.remove('is-on');
    setTimeout(function () { annLayer.classList.add('is-on'); }, 20);
    lastStep = step; lastKey = key;
    svg.setAttribute('aria-label', { generation: '一共 ' + facts.total + ' 份活', ordering: '真活在前，空活在后', assignment: '轮流发给 ' + N + ' 个核', rounds: facts.rounds + ' 轮', eval: '最慢核是平均的 ' + facts.M[plan][weight].ratio.toFixed(2) + ' 倍' }[step]);
    syncState();
    if (!animate) setTimeout(function () { svg.classList.remove('is-still'); }, 40);
  }

  // 选中 / 筛选 / 同伴 / Valid KV Rows 明暗：从 lens.js 的原图逐格抄过来
  function syncState() {
    if (!cells.length) return;
    const src = {};
    host.querySelectorAll(':scope > svg .lane-cell[data-key]').forEach(function (el) { src[el.dataset.key] = el; });
    let selLane = null;
    const lab = host.querySelector(':scope > svg .asg-label.is-lane.is-sel, :scope > svg .lane-row.is-sel .lane-cell');
    if (lab) selLane = Number(lab.dataset.lane);
    cells.forEach(function (el) {
      const s = src[el.dataset.key];
      ['is-sel', 'is-mate', 'is-out'].forEach(function (c) { el.classList.toggle(c, !!s && s.classList.contains(c)); });
      el.style.fillOpacity = step === 'eval' ? '' : s && s.style.fillOpacity ? s.style.fillOpacity : '';
    });
    svg.querySelectorAll('.wb-lane-k').forEach(function (t) { t.classList.toggle('is-sel', Number(t.dataset.lane) === selLane); });
  }

  // ── 流程条与评估开关的点击 ──
  function go(k) {
    const pipe = onTab('pipeline');
    if (k === 'rounds' || k === 'eval') {
      evalMode = k === 'eval';
      if (!pipe) { const r = tabs.querySelector('[data-tab="pipeline"]'); if (r && !r.disabled) r.click(); }
      schedule(false);
      return;
    }
    evalMode = false;
    // 在 Lane 轮次里用 workbench-polish.js 的代理按钮回到 Assignment 各阶段；否则直接点 lens.js 的阶段按钮
    const b = pipe ? bar.querySelector('[data-wb-phase="' + k + '"]') : bar.querySelector('[data-l2="phase"][data-v="' + k + '"]');
    if (b) b.click();
  }
  function onClick(e) {
    const n = e.target.closest('[data-wb-flow]');
    if (n) { go(n.dataset.wbFlow); return; }
    const w = e.target.closest('[data-wb-weight]');
    if (w) { weight = w.dataset.wbWeight; schedule(false); return; }
    const p = e.target.closest('[data-wb-plan]');
    if (p) { plan = p.dataset.wbPlan; schedule(false); }
  }

  // ── 何时显示、何时重画 ──
  function schedule(resized) {
    // setTimeout 而不是 requestAnimationFrame：页面在后台（或宿主面板隐藏）时 rAF 不触发，视图会挂不上
    clearTimeout(timer);
    if (resized) pendingResize = true;
    timer = setTimeout(function () { const r = pendingResize; pendingResize = false; update(r); }, 16);
  }
  function update(resized) {
    const next = curStep();
    work.classList.toggle('wb-story-on', !!next);
    host.classList.toggle('wb-story-on', !!next);
    if (!next) { if (root && root.parentNode) root.remove(); lastKey = ''; return; }
    if (!root) build();
    if (root.parentNode !== host) host.appendChild(root);
    const rank = curRank();
    if (rank !== rankShown) { loadData(rank); lastKey = ''; }
    const key = next + ':' + weight + ':' + plan;
    if (key !== lastKey || resized) { step = next; paint(resized); }
    else syncState();
  }
  new MutationObserver(function () { schedule(false); }).observe(bar, { childList: true, subtree: true, attributes: true, attributeFilter: ['aria-pressed'] });
  bar.addEventListener('change', function () { schedule(false); });
  new MutationObserver(function () { schedule(false); }).observe(tabs, { attributes: true, subtree: true, attributeFilter: ['aria-selected'] });
  new MutationObserver(function () { schedule(false); }).observe(meta, { childList: true, characterData: true, subtree: true });
  new MutationObserver(function () { schedule(false); }).observe(document.body, { attributes: true, attributeFilter: ['data-level'] });
  // lens.js 重画原图（换阶段、选中、筛选）或整块替换 #kernelLanes 时，重新挂上并同步状态
  new MutationObserver(function (list) {
    if (list.every(function (m) { return root && (m.target === root || root.contains(m.target)); })) return;
    schedule(false);
  }).observe(host, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style'] });
  schedule(false);

  // 演示深链：#L2:generation / #L2:ordering 停在第 ① / ② 步；#L2:eval 停在第 ⑤ 步
  const deep = /^#L2:(generation|ordering|eval)$/.exec(location.hash);
  if (deep) setTimeout(function () {
    const lv = document.querySelector('#lensLevels [data-level="2"]'); if (lv) lv.click();
    if (deep[1] === 'eval') { go('eval'); return; }
    const t = tabs.querySelector('[data-tab="assignment"]'); if (t) t.click();
    const b = bar.querySelector('[data-l2="phase"][data-v="' + deep[1] + '"]'); if (b) b.click();
  }, 0);
})();
