/* =============================================================
 * Tuning Atlas
 *
 * 结构照搬 The Visual Agency《Codex Atlanticus》：
 *   第 1 层  两条可刷选的总览带（一条按序，一条按时间）
 *   第 2 层  排列 / 过滤开关
 *   第 3 层  主体 —— 一个执行块一格，颜色 = 分类
 *   第 4 层  一个大号数字 —— 当前选中了多少格
 * 四层互相过滤，页面上除两个浮层外不放正文。
 *
 * 数据直接复用 operator-tuning-console 生成的 data.js，
 * 不复制一份：那边 build-data.cjs 重跑后这边跟着变。
 * ============================================================= */
(function () {
  'use strict';

  var RUNS = window.TUNING_RUNS;
  if (!RUNS) return;

  /* 颜色给「单块耗时」而不是 scope：本 run 有 62 个 scope，取前五个
   * 只盖住 18.6% 的格子，矩阵会有 81% 是无信息的灰。按耗时分档是
   * 完全划分，而且五档正好对上参考图的五个分类。
   * 色值与 atlas.css 的 --at-c1..c5 一致（取样自参考图原图）。 */
  var BANDS = [
    { lo: 0,   label: '< 5 us',    color: '#3197cd' },
    { lo: 5,   label: '5 – 20',    color: '#4ec58c' },
    { lo: 20,  label: '20 – 100',  color: '#ffbc52' },
    { lo: 100, label: '100 – 400', color: '#f8957c' },
    { lo: 400, label: '≥ 400 us',  color: '#e0685c' },
  ];
  var NB = BANDS.length;
  var NSC = 5;          /* scope 筛选行取前五 + 其余 */
  var DIM = 0.14;
  var ACCENT = '#ffbc52';

  function bandOf(d) {
    for (var i = NB - 1; i > 0; i--) if (d >= BANDS[i].lo) return i;
    return 0;
  }

  /* ------------------------------------------------------------- 工具 */
  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function el(tag, cls, txt) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt != null) n.textContent = txt;
    return n;
  }
  function group(n) { return Math.round(n).toLocaleString('en-US'); }
  function pct(a, b) { return b ? (a / b * 100).toFixed(1) : '0.0'; }

  /* canvas 按设备像素比放大，否则 1px 的格子边在高分屏上糊成两像素 */
  function hidpi(cv, W, H) {
    var dpr = window.devicePixelRatio || 1;
    cv.width = Math.max(1, Math.round(W * dpr));
    cv.height = Math.max(1, Math.round(H * dpr));
    var g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return g;
  }

  /* ----------------------------------------------------------- 运行目录 */
  var LIST = [];
  Object.keys(RUNS).forEach(function (ck) {
    var c = RUNS[ck];
    Object.keys(c.ranks).forEach(function (rk) {
      LIST.push({ ck: ck, rk: rk, label: c.case.label + ' · ' + rk });
    });
  });

  var S = {
    ci: 0,
    order: 'time',     /* time | core */
    pathOnly: false,
    seq: null,         /* [i0, i1]，下发序下标区间 */
    win: null,         /* [t0, t1]，墙钟窗口 us */
    bands: {}, bandsOn: false,
    scopes: {}, scopesOn: false,
    overlay: null,
  };

  /* --------------------------------------------------------- 视图装配 */
  var CACHE = {};
  function view() {
    if (CACHE[S.ci]) return CACHE[S.ci];
    var ent = LIST[S.ci];
    var c = RUNS[ent.ck];
    var r = c.ranks[ent.rk];
    var sw = r.swimlane;

    /* 任务 → scope 名。tasks[].scope 在 outline 之前的任务上是 "auto"，
     * 权威映射在 scopes[].tags 上，所以反查而不是直接读字段。 */
    var tagScope = {};
    r.scopes.forEach(function (s) {
      s.tags.forEach(function (t) { tagScope[t] = s.name; });
    });
    /* scopes 已按 Σ core-time 降序，前五个各占一行，其余合并 */
    var topSc = r.scopes.slice(0, NSC);
    var scIdx = {};
    topSc.forEach(function (s, i) { scIdx[s.name] = i; });

    /* 白边 = 所属任务在依赖关键路径上（cpath.cpm，不是按耗时排出来的） */
    var onCpm = {};
    r.cpath.cpm.tags.forEach(function (t) { onCpm[t] = 1; });

    var blocks = [];
    sw.blocks.forEach(function (slices, lane) {
      slices.forEach(function (b) {
        var t = r.tasks[b[2]];
        if (!t) return;
        var sn = tagScope[t.tag] || t.rawName;
        /* tag 是 rNtM：N = 下发环，M = 环内的下发序号。这是任务被排进
         * 队列的次序，和它实际跑起来的次序不是一回事（本 dump 里有
         * 几十处倒置），所以它能当第二条独立的轴用。 */
        var m = /^r(\d+)t(\d+)$/.exec(t.tag);
        blocks.push({
          st: b[0], dur: b[1], lane: lane,
          tag: t.tag, callable: t.callable || t.rawName,
          ring: m ? +m[1] : 0, ti: m ? +m[2] : 0,
          scope: sn,
          si: scIdx[sn] == null ? NSC : scIdx[sn],
          band: bandOf(b[1]),
          crit: onCpm[t.tag] ? 1 : 0,
          seq: 0,
        });
      });
    });

    var idx = blocks.map(function (_, i) { return i; });
    var byProg = idx.slice().sort(function (a, b) {
      var p = blocks[a], q = blocks[b];
      return p.ring - q.ring || p.ti - q.ti || p.lane - q.lane;
    });
    var byTime = idx.slice().sort(function (a, b) {
      return blocks[a].st - blocks[b].st || blocks[a].lane - blocks[b].lane;
    });
    var byCore = idx.slice().sort(function (a, b) {
      return blocks[a].lane - blocks[b].lane || blocks[a].st - blocks[b].st;
    });
    /* 左边那条带刷的是下发序，不是时间；两条带因此不是同一个轴 */
    byProg.forEach(function (bi, p) { blocks[bi].seq = p; });

    var legend = BANDS.map(function (b, i) {
      return { i: i, name: b.label, color: b.color, sum: 0, n: 0 };
    });
    var scopes = topSc.map(function (s, i) {
      return { i: i, name: s.name, sum: 0, n: 0 };
    });
    scopes.push({
      i: NSC,
      name: '其余 ' + Math.max(0, r.scopes.length - NSC) + ' 个',
      sum: 0, n: 0,
    });

    var totalDur = 0, critN = 0;
    blocks.forEach(function (b) {
      legend[b.band].sum += b.dur; legend[b.band].n += 1;
      scopes[b.si].sum += b.dur; scopes[b.si].n += 1;
      totalDur += b.dur; critN += b.crit;
    });

    /* 下发序和时间序差多少：按任务数，不按块数（同一任务的块是一起下发的） */
    var prog = r.tasks.map(function (t, i) {
      var mm = /^r(\d+)t(\d+)$/.exec(t.tag);
      return { i: i, ring: mm ? +mm[1] : 0, ti: mm ? +mm[2] : 0 };
    }).sort(function (a, b) { return a.ring - b.ring || a.ti - b.ti; });
    var inversions = 0;
    for (var q = 1; q < prog.length; q++) if (prog[q].i < prog[q - 1].i) inversions++;

    var V = {
      ent: ent, caseObj: c, rank: r,
      blocks: blocks, byProg: byProg, byTime: byTime, byCore: byCore,
      legend: legend, scopes: scopes,
      span: sw.spanUs, lanes: sw.laneNames.length,
      totalDur: totalDur, critN: critN, inversions: inversions,
      cols: {},   /* 总览带的逐像素列，按宽度缓存 */
      layout: null, order: null,
    };
    CACHE[S.ci] = V;
    return V;
  }

  /* --------------------------------------------------------- 过滤判定 */
  function pass(b) {
    if (S.seq && (b.seq < S.seq[0] || b.seq > S.seq[1])) return false;
    /* 时间窗按「有重叠」判定，一个块跨在窗口边界上也算进来 */
    if (S.win && (b.st + b.dur <= S.win[0] || b.st >= S.win[1])) return false;
    if (S.bandsOn && !S.bands[b.band]) return false;
    if (S.scopesOn && !S.scopes[b.si]) return false;
    if (S.pathOnly && !b.crit) return false;
    return true;
  }
  function anyOn(m) { return Object.keys(m).some(function (k) { return m[k]; }); }
  function syncSets() {
    S.bandsOn = anyOn(S.bands);
    S.scopesOn = anyOn(S.scopes);
  }
  function anyFilter() {
    return !!(S.seq || S.win || S.bandsOn || S.scopesOn || S.pathOnly);
  }

  /* --------------------------------------------------- 总览带的列数据 */
  function seqCols(V, W) {
    var key = 'seq:' + W;
    if (V.cols[key]) return V.cols[key];
    var n = V.byProg.length, out = new Float32Array(W);
    for (var p = 0; p < n; p++) {
      var i = Math.min(W - 1, Math.floor(p / n * W));
      var d = V.blocks[V.byProg[p]].dur;
      if (d > out[i]) out[i] = d;       /* 列内取最大，尖峰不会被均值抹掉 */
    }
    /* 块时长跨三个数量级，线性画出来是一条平线加几根针；开方压一下
     * 才看得出形状。纵轴因此不是等比的，只用来找位置，不用来读数。 */
    for (var k = 0; k < W; k++) out[k] = Math.sqrt(out[k]);
    V.cols[key] = out;
    return out;
  }
  function timeCols(V, W) {
    var key = 'time:' + W;
    if (V.cols[key]) return V.cols[key];
    /* 并发 = 同时忙着的核数（一个块占一个核），用差分数组累加 */
    var delta = new Float32Array(W + 2), span = V.span || 1;
    V.blocks.forEach(function (b) {
      var a = Math.floor(b.st / span * W);
      var z = Math.ceil((b.st + b.dur) / span * W);
      if (a < 0) a = 0;
      if (z > W) z = W;
      if (z <= a) z = a + 1;
      delta[a] += 1; delta[z] -= 1;
    });
    var out = new Float32Array(W), acc = 0;
    for (var i = 0; i < W; i++) { acc += delta[i]; out[i] = acc; }
    V.cols[key] = out;
    return out;
  }

  /* ------------------------------------------------------- 总览带绘制 */
  function drawStrip(which) {
    var cv = $('[data-canvas="' + which + '"]');
    if (!cv) return;
    var W = Math.max(1, Math.round(cv.clientWidth));
    var H = Math.max(1, Math.round(cv.clientHeight));
    var g = hidpi(cv, W, H);
    var V = view();
    var data = which === 'seq' ? seqCols(V, W) : timeCols(V, W);
    var max = 0;
    for (var i = 0; i < W; i++) if (data[i] > max) max = data[i];
    if (!max) max = 1;

    var rng = which === 'seq' ? S.seq : S.win;
    var dom = which === 'seq' ? [0, V.byProg.length - 1] : [0, V.span];
    var x0 = 0, x1 = 0;
    if (rng) {
      var d = (dom[1] - dom[0]) || 1;
      x0 = (rng[0] - dom[0]) / d * W;
      x1 = (rng[1] - dom[0]) / d * W;
    }

    function bars(alpha, color) {
      g.globalAlpha = alpha;
      g.fillStyle = color;
      for (var k = 0; k < W; k++) {
        var h = Math.round(data[k] / max * (H - 3));
        if (h > 0) g.fillRect(k, H - h, 1, h);
      }
      g.globalAlpha = 1;
    }

    g.clearRect(0, 0, W, H);
    if (rng) {
      bars(0.16, '#ffffff');
      g.save();
      g.beginPath();
      g.rect(x0, 0, Math.max(1, x1 - x0), H);
      g.clip();
      bars(0.95, ACCENT);
      g.restore();
      g.fillStyle = ACCENT;
      g.fillRect(Math.round(x0), 0, 1, H);
      g.fillRect(Math.round(x1) - 1, 0, 1, H);
    } else {
      bars(0.62, '#ffffff');
    }
    /* 基线：没有它，零并发的区段读不出是"空"还是"没画" */
    g.fillStyle = 'rgba(255,255,255,0.14)';
    g.fillRect(0, H - 1, W, 1);
  }

  /* ----------------------------------------------------------- 矩阵布局 */
  function layout(W, H, n) {
    var AR = 1.5, GX = 0.26, GY = 0.2, best = null;
    for (var cw = 1.5; cw <= 20; cw += 0.25) {
      var ch = cw * AR, px = cw * (1 + GX), py = ch * (1 + GY);
      var cols = Math.max(1, Math.floor(W / px));
      var rows = Math.ceil(n / cols);
      if (rows * py <= H) best = { cw: cw, ch: ch, px: px, py: py, cols: cols, rows: rows };
    }
    if (!best) {
      /* 格子再小也塞不下时不缩到看不见，宁可让最后几行被裁掉 */
      var cw2 = 1.5, ch2 = cw2 * AR, px2 = cw2 * (1 + GX), py2 = ch2 * (1 + GY);
      var cols2 = Math.max(1, Math.floor(W / px2));
      best = { cw: cw2, ch: ch2, px: px2, py: py2, cols: cols2, rows: Math.ceil(n / cols2) };
    }
    return best;
  }

  function drawMatrix() {
    var wrap = $('.matrix-wrap'), cv = $('[data-canvas="matrix"]');
    if (!wrap || !cv) return;
    var W = Math.max(1, Math.round(wrap.clientWidth));
    var H = Math.max(1, Math.round(wrap.clientHeight));
    var g = hidpi(cv, W, H);
    var V = view();
    var order = S.order === 'core' ? V.byCore : S.order === 'prog' ? V.byProg : V.byTime;
    var L = layout(W, H, order.length);
    V.layout = L;
    V.order = order;

    g.clearRect(0, 0, W, H);
    for (var p = 0; p < order.length; p++) {
      var b = V.blocks[order[p]];
      var col = p % L.cols, row = (p - col) / L.cols;
      var x = col * L.px, y = row * L.py;
      var ok = pass(b);
      g.globalAlpha = ok ? 1 : DIM;
      g.fillStyle = BANDS[b.band].color;
      g.fillRect(x, y, L.cw, L.ch);
      if (ok && b.crit && L.cw >= 3) {
        g.globalAlpha = 1;
        g.strokeStyle = 'rgba(255,255,255,0.95)';
        g.lineWidth = 1;
        g.strokeRect(x + 0.5, y + 0.5, L.cw - 1, L.ch - 1);
      }
    }
    g.globalAlpha = 1;
  }

  function hitMatrix(mx, my) {
    var V = view(), L = V.layout;
    if (!L || !V.order) return null;
    var col = Math.floor(mx / L.px), row = Math.floor(my / L.py);
    if (col < 0 || col >= L.cols || row < 0) return null;
    if (mx - col * L.px > L.cw || my - row * L.py > L.ch) return null;
    var p = row * L.cols + col;
    if (p < 0 || p >= V.order.length) return null;
    return V.blocks[V.order[p]];
  }

  /* --------------------------------------------------------- 右栏图例 */
  function drawLegend() {
    var V = view();

    var host = $('[data-bind="legend"]');
    host.textContent = '';
    V.legend.forEach(function (lg) {
      var b = el('button', 'lg' + (S.bandsOn && !S.bands[lg.i] ? ' off' : ''));
      b.type = 'button';
      b.style.background = lg.color;
      b.appendChild(el('span', 'n', lg.name));
      b.appendChild(el('span', 'c', group(lg.sum)));
      b.title = lg.name + '：' + group(lg.n) + ' 块（占格子 ' + pct(lg.n, V.blocks.length)
        + '%）· Σ ' + group(lg.sum) + ' us（占 core-time ' + pct(lg.sum, V.totalDur) + '%）';
      b.addEventListener('click', function () {
        S.bands[lg.i] = !S.bands[lg.i];
        syncSets();
        paint();
      });
      host.appendChild(b);
    });

    var sh = $('[data-bind="scopes"]');
    sh.textContent = '';
    V.scopes.forEach(function (sc) {
      var b = el('button', 'sc' + (S.scopes[sc.i] ? ' on' : ''));
      b.type = 'button';
      b.appendChild(el('span', 'n', sc.name));
      b.appendChild(el('span', 'c', group(sc.sum)));
      b.title = sc.name + '：' + group(sc.n) + ' 块 · Σ ' + group(sc.sum)
        + ' us（占 core-time ' + pct(sc.sum, V.totalDur) + '%）';
      b.addEventListener('click', function () {
        S.scopes[sc.i] = !S.scopes[sc.i];
        syncSets();
        paint();
      });
      sh.appendChild(b);
    });

    var pn = $('[data-bind="pathnote"]');
    pn.textContent = '';
    pn.appendChild(el('i'));
    pn.appendChild(el('span', null, '白边 = 依赖关键路径'));
    pn.appendChild(el('b', null, group(V.critN) + ' 块'));
    pn.title = '本路 ' + V.rank.cpath.cpm.tags.length + ' 个任务在依赖关键路径上，'
      + '共 ' + group(V.critN) + ' 个块（占 ' + pct(V.critN, V.blocks.length) + '%）。';
  }

  /* ------------------------------------------------------------ 读数 */
  function drawReadout() {
    var V = view();
    var n = 0, sum = 0, lanes = {}, tmin = Infinity, tmax = -Infinity, crit = 0;
    V.blocks.forEach(function (b) {
      if (!pass(b)) return;
      n += 1; sum += b.dur; lanes[b.lane] = 1; crit += b.crit;
      if (b.st < tmin) tmin = b.st;
      if (b.st + b.dur > tmax) tmax = b.st + b.dur;
    });
    $('[data-bind="bigNum"]').textContent = group(n);

    var sub = $('[data-bind="bigSub"]');
    sub.textContent = '';
    if (!n) {
      sub.appendChild(el('span', null, '当前条件下没有块 —— 收窄的是刷选区间，还是图例？'));
    } else {
      [
        ['Σ core-time', group(sum) + ' us'],
        ['占全部', pct(sum, V.totalDur) + '%'],
        ['落在', Object.keys(lanes).length + ' / ' + V.lanes + ' 核'],
        ['跨', group(tmax - tmin) + ' us'],
        ['关键路径上', group(crit) + ' 块'],
      ].forEach(function (p, i) {
        if (i) sub.appendChild(el('span', null, '  ·  '));
        var one = el('span', 'p');
        one.appendChild(el('span', null, p[0] + ' '));
        one.appendChild(el('b', null, p[1]));
        sub.appendChild(one);
      });
    }

    $('[data-bind="seqRange"]').textContent = S.seq
      ? '第 ' + group(S.seq[0] + 1) + ' – ' + group(S.seq[1] + 1) + ' 块'
      : '全部 ' + group(V.byProg.length) + ' 块';
    $('[data-bind="timeRange"]').textContent = S.win
      ? group(S.win[0]) + ' – ' + group(S.win[1]) + ' us'
      : '0 – ' + group(V.span) + ' us';
    $('[data-act="reset"]').disabled = !anyFilter();
  }

  /* --------------------------------------------------------- 顶部控件 */
  function drawChrome() {
    var V = view();
    var runSeg = $('#runSeg');
    if (!runSeg.childNodes.length) {
      LIST.forEach(function (ent, i) {
        var b = el('button', null, ent.label);
        b.type = 'button';
        b.addEventListener('click', function () {
          if (S.ci === i) return;
          S.ci = i;
          clearFilters();
          paint();
        });
        runSeg.appendChild(b);
      });
    }
    $$('button', runSeg).forEach(function (b, i) { b.classList.toggle('on', i === S.ci); });

    seg('scope-mode', [
      { k: false, label: '全部块' },
      { k: true, label: '仅关键路径' },
    ], S.pathOnly, function (k) { S.pathOnly = k; paint(); });

    seg('order', [
      { k: 'prog', label: '下发序' },
      { k: 'time', label: '时间序' },
      { k: 'core', label: '核序' },
    ], S.order, function (k) { S.order = k; paint(); });

    $('[data-bind="footL"]').textContent = '数据源 ' + V.caseObj.source;
    $('[data-bind="footC"]').textContent =
      '本页只画采到的执行块；没被 chip swimlane 记下的时间不出现在这张图上';
  }

  function seg(name, opts, cur, onPick) {
    var host = $('[data-seg="' + name + '"]');
    if (host.dataset.built !== '1') {
      opts.forEach(function (o) {
        var b = el('button', null, o.label);
        b.type = 'button';
        b.addEventListener('click', function () { onPick(o.k); });
        host.appendChild(b);
      });
      host.dataset.built = '1';
    }
    $$('button', host).forEach(function (b, i) { b.classList.toggle('on', opts[i].k === cur); });
  }

  /* ------------------------------------------------------------- 刷选 */
  function bindBrush(which) {
    var cv = $('[data-canvas="' + which + '"]');
    var dragging = false, ax = 0;

    function domainAt(clientX) {
      var r = cv.getBoundingClientRect();
      var f = Math.min(1, Math.max(0, (clientX - r.left) / (r.width || 1)));
      var V = view();
      return which === 'seq' ? f * (V.byProg.length - 1) : f * V.span;
    }
    function apply(a, b) {
      var lo = Math.min(a, b), hi = Math.max(a, b);
      if (which === 'seq') S.seq = [Math.round(lo), Math.round(hi)];
      else S.win = [Math.round(lo * 100) / 100, Math.round(hi * 100) / 100];
    }

    cv.addEventListener('pointerdown', function (e) {
      dragging = true;
      ax = domainAt(e.clientX);
      cv.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    cv.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      apply(ax, domainAt(e.clientX));
      paint();
    });
    function end(e) {
      if (!dragging) return;
      dragging = false;
      var bx = domainAt(e.clientX);
      var full = which === 'seq' ? view().byProg.length : view().span;
      /* 点一下（没拖出宽度）= 取消这一条的刷选，和参考图一致 */
      if (Math.abs(bx - ax) / (full || 1) < 0.004) {
        if (which === 'seq') S.seq = null; else S.win = null;
      } else {
        apply(ax, bx);
      }
      paint();
    }
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);
  }

  /* ------------------------------------------------------------- 悬浮 */
  function bindHover() {
    var cv = $('[data-canvas="matrix"]');
    var tip = $('[data-bind="tip"]');
    cv.addEventListener('mousemove', function (e) {
      var r = cv.getBoundingClientRect();
      var b = hitMatrix(e.clientX - r.left, e.clientY - r.top);
      if (!b) { tip.hidden = true; return; }
      tip.textContent = '';
      tip.appendChild(el('div', 't', b.scope + (b.crit ? '  ◂ 关键路径' : '')));
      /* 纯 Cube / Vec 的 scope 不改名，callable 和 scope 同名，重复一遍没意义 */
      tip.appendChild(el('div', 's',
        b.dur.toFixed(2) + ' us  ·  ' + view().rank.swimlane.laneNames[b.lane]
        + '  ·  t=' + b.st.toFixed(1)
        + (b.callable && b.callable !== b.scope ? '  ·  ' + b.callable : '')));
      tip.hidden = false;
      var x = Math.min(window.innerWidth - tip.offsetWidth - 12, e.clientX + 14);
      var y = e.clientY - tip.offsetHeight - 12;
      if (y < 8) y = e.clientY + 18;
      tip.style.left = Math.max(8, x) + 'px';
      tip.style.top = y + 'px';
    });
    cv.addEventListener('mouseleave', function () { tip.hidden = true; });
    /* 点格子 = 按它的 scope 筛选，等价于点右栏那一行 */
    cv.addEventListener('click', function (e) {
      var r = cv.getBoundingClientRect();
      var b = hitMatrix(e.clientX - r.left, e.clientY - r.top);
      if (!b) return;
      S.scopes[b.si] = !S.scopes[b.si];
      syncSets();
      paint();
    });
  }

  /* ------------------------------------------------------------- 浮层 */
  function sheetRead() {
    var V = view();
    var hot = V.legend[NB - 1];
    var h = [];
    h.push('<h2>怎么读</h2>');
    h.push('<p>这张图把一次真实上板执行的每一个 AICore 执行块画成一格，本路共 <code>'
      + group(V.blocks.length) + '</code> 格。四层逻辑，从上往下：</p>');
    h.push('<div class="lay">'
      + '<span class="i">1</span><span class="d"><b>两条总览带，两个不同的轴。</b>'
      + '左边是<b>下发序</b> —— 任务被排进队列的次序（tag 里的 <code>rNtM</code>），'
      + '柱高是这一块自己的耗时；右边是<b>墙钟</b> —— 它实际跑起来的时刻，'
      + '柱高是那一刻同时忙着的核数。两者不是同一个顺序：本路有 <code>' + V.inversions
      + '</code> 处任务跑在了比它先下发的任务前面，所以两条带各刷各的，叠加起来才有意义。'
      + '在带上拖动选一段，点一下取消该条。'
      + '（左边纵轴开过方，否则跨三个数量级的时长画出来是一条平线加几根针。）</span>'
      + '<span class="i">2</span><span class="d"><b>排列开关。</b>'
      + '网格可以按下发序、时间序或核序铺开 —— 前两个正对着上面那两条带，'
      + '换一个排法就能看见调度把哪些块挪了位置；按核序排时同一个核的块连在一起，'
      + '颜色的断层就是这个核负载的变化。</span>'
      + '<span class="i">3</span><span class="d"><b>右栏两组筛选。</b>'
      + '上面一组是<b>单块耗时的五档</b>，它决定格子的颜色；下面一组是 <b>scope</b>，'
      + '只筛不上色。两组都能多选，点矩阵里的格子等同于点它所属的 scope 那一行。</span>'
      + '<span class="i">4</span><span class="d"><b>数字。</b>'
      + '页面上唯一的大号文字，是当前选中的块数；下面一行给 Σ core-time、占比、铺在几个核上、跨多长时间。</span>'
      + '</div>');
    h.push('<h3>为什么颜色给耗时，不给 scope</h3>');
    h.push('<p>本路有 <code>' + V.rank.scopes.length + '</code> 个 scope，'
      + '按 Σ core-time 取前五个只盖住不到两成的格子，矩阵会有八成是无信息的灰。'
      + '按耗时分档是完全划分，而且第一眼就能看见这件事：最粗的一档 <code>'
      + BANDS[NB - 1].label + '</code> 只有 <b>' + group(hot.n) + '</b> 格（占 '
      + pct(hot.n, V.blocks.length) + '%），却吃掉 <b>' + pct(hot.sum, V.totalDur)
      + '%</b> 的 core-time。格子多不等于花时间多。</p>');
    h.push('<h3>白边是什么</h3>');
    h.push('<p>格子带白边 = 它所属的任务在<b>依赖关键路径</b>上。'
      + '这条路径是在 happens-before 过滤后的依赖图上取最长链算出来的，'
      + '不是按耗时排序挑出来的 —— 本路 <code>' + V.rank.cpath.cpm.tags.length
      + '</code> 个任务在上面，共 <code>' + group(V.critN) + '</code> 个块。</p>');
    h.push('<p class="warn">Σ core-time 大不等于拖慢墙钟：一个 scope 可能摊在 '
      + V.lanes + ' 个核上一次跑完。要判断值不值得动，看它在不在路径上，'
      + '而不是看它的格子多不多、颜色深不深。</p>');
    return h.join('');
  }

  function sheetAbout() {
    var V = view(), ci = V.caseObj.case;
    var h = [];
    h.push('<h2>关于这份 dump</h2>');
    h.push('<p>数据来自 <code>' + V.caseObj.source + '</code>，由 '
      + '<code>Design/operator-tuning-console/build-data.cjs</code> 从 chip swimlane 记录生成。'
      + '本页直接引用那份 <code>data.js</code>，不另存一份副本。</p>');
    h.push('<h3>这一路的规模</h3>');
    h.push('<p><code>' + ci.model + '</code> · ' + ci.backend + ' · ' + ci.numCores + ' 核（AIC '
      + ci.aicCount + ' / AIV ' + ci.aivCount + '）· 时钟 ' + (ci.clockHz / 1e6) + ' MHz<br>'
      + '本路 <code>' + V.rank.tasks.length + '</code> 个任务、<code>' + V.rank.scopes.length
      + '</code> 个 scope、<code>' + group(V.blocks.length) + '</code> 个执行块，'
      + '墙钟 <code>' + group(V.span) + ' us</code>，Σ core-time <code>'
      + group(V.totalDur) + ' us</code>。</p>');
    h.push('<h3>这张图能回答什么</h3>');
    h.push('<p>块有多大、什么时候跑、属于哪个 scope、铺在多少个核上、哪些落在关键路径上，'
      + '以及任意一段时间窗或一段发射区间里这几件事分别是什么样。</p>');
    h.push('<h3>不能单独回答什么</h3>');
    h.push('<p>单个 kernel 内部为什么慢 —— 那要看核内流水的 MTE / Vector / Cube 占用；'
      + '以及 Host 侧有没有贡献 —— 这份 dump 只有 STRACE span，没有 BenchmarkStats，'
      + '给不出 rounds / warmup 的统计量。这两件事在完整版控制台里分别是 L1 页和 E2E 页。</p>');
    h.push('<p class="warn">一次采集 = 一个样本。这里所有数字都是这一次运行的观测值，'
      + '不是多次运行的统计量，不能当作性能结论直接对外报。</p>');
    return h.join('');
  }

  function drawOverlay() {
    var ov = $('[data-bind="overlay"]');
    if (!S.overlay) { ov.hidden = true; return; }
    $('[data-bind="overlayBody"]').innerHTML =
      S.overlay === 'read' ? sheetRead() : sheetAbout();
    ov.hidden = false;
  }

  /* ------------------------------------------------------------- 绘制 */
  function paint() {
    drawChrome();
    drawStrip('seq');
    drawStrip('time');
    drawMatrix();
    drawLegend();
    drawReadout();
    drawOverlay();
  }

  function clearFilters() {
    S.seq = null; S.win = null;
    S.bands = {}; S.scopes = {};
    S.pathOnly = false;
    syncSets();
  }

  /* ------------------------------------------------------------- 绑定 */
  function boot() {
    syncSets();

    $$('[data-overlay]').forEach(function (b) {
      b.addEventListener('click', function () { S.overlay = b.dataset.overlay; drawOverlay(); });
    });
    $('[data-act="close-overlay"]').addEventListener('click', function () {
      S.overlay = null; drawOverlay();
    });
    $('[data-bind="overlay"]').addEventListener('click', function (e) {
      if (e.target === e.currentTarget) { S.overlay = null; drawOverlay(); }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && S.overlay) { S.overlay = null; drawOverlay(); }
    });

    $('[data-act="reset"]').addEventListener('click', function () {
      clearFilters();
      paint();
    });

    bindBrush('seq');
    bindBrush('time');
    bindHover();

    var t = null;
    window.addEventListener('resize', function () {
      clearTimeout(t);
      t = setTimeout(function () {
        /* 列缓存是按像素宽度算的，宽度变了就作废 */
        Object.keys(CACHE).forEach(function (k) { CACHE[k].cols = {}; });
        paint();
      }, 120);
    });

    requestAnimationFrame(paint);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
