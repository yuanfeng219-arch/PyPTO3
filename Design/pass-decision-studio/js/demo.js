/* =============================================================
 * Pass Decision Studio · Pass 编译决策可视化 — 交互
 * ============================================================= */
(function () {
  'use strict';

  var D = window.PASS_DECISION_DATA;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var esc = function (s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  };

  var LAYER_TONE = { tiling: 'var(--tone-tiling)', memory: 'var(--tone-memory)', deps: 'var(--tone-deps)', sync: 'var(--tone-sync)' };
  var LAYER_NAME = {};
  D.layers.forEach(function (l) { LAYER_NAME[l.id] = l.name; });

  var STAGES = [
    { id: 'spine',    label: '决策脊 · 全景' },
    { id: 'timeline', label: '对象时间线' },
    { id: 'memmap',   label: 'Memory Map' },
    { id: 'swimlane', label: '同步泳道对比' },
    { id: 'diff',     label: '决策 diff' },
    { id: 'inline',   label: '源码内联标注' }
  ];

  var state = {
    step: 0,
    stage: 'spine',
    focus: null,
    mode: 'decisions',
    hiddenLayers: {},
    railAll: true,
    openPass: null,
    ackL3: false,
    showOverride: false,
    playing: false,
    timer: null
  };

  function decision(id) {
    for (var i = 0; i < D.decisions.length; i++) if (D.decisions[i].id === id) return D.decisions[i];
    return null;
  }
  function decisionsOfPass(idx) {
    return D.decisions.filter(function (d) { return d.pass === idx; });
  }
  function passByIdx(idx) {
    for (var i = 0; i < D.passes.length; i++) if (D.passes[i].idx === idx) return D.passes[i];
    return null;
  }
  function levelRank(l) { return { L0: 0, L1: 1, L2: 2, L3: 3 }[l] || 0; }

  function setActiveElement(el, scrollContainer) {
    if (!el) return;
    el.focus({ preventScroll: true });
    if (scrollContainer) {
      var r = el.getBoundingClientRect();
      var c = scrollContainer.getBoundingClientRect();
      if (r.top < c.top + 8) scrollContainer.scrollTop += r.top - c.top - 8;
      else if (r.bottom > c.bottom - 8) scrollContainer.scrollTop += r.bottom - c.bottom + 8;
    }
  }

  /* =========================================================
   * 左栏 · 决策脊（纵向）
   * ======================================================= */
  function renderLegend() {
    var el = $('#spine-legend');
    var dec = D.passes.filter(function (p) { return p.decisions > 0; }).length;
    el.innerHTML =
      '<div class="rail-modes" role="group" aria-label="Pass 列表范围">' +
      '<button type="button" class="rail-mode' + (state.railAll ? ' is-on' : '') + '" data-all="1">全部 ' + D.passes.length + '</button>' +
      '<button type="button" class="rail-mode' + (state.railAll ? '' : ' is-on') + '" data-all="0">仅决策点 ' + dec + '</button>' +
      '</div>' +
      D.layers.map(function (l) {
        var off = state.hiddenLayers[l.id] ? ' is-off' : '';
        return '<button type="button" class="spine-legend__item' + off + '" data-layer="' + l.id + '" ' +
               'style="--tone:' + LAYER_TONE[l.id] + '" title="显示 / 隐藏该层的决策">' +
               '<i class="spine-legend__dot"></i>' + esc(l.short) + '</button>';
      }).join('');

    $$('.rail-mode', el).forEach(function (b) {
      b.addEventListener('click', function () {
        state.railAll = b.dataset.all === '1';
        renderLegend(); renderRail();
      });
    });
    $$('.spine-legend__item', el).forEach(function (b) {
      b.addEventListener('click', function () {
        var id = b.dataset.layer;
        state.hiddenLayers[id] = !state.hiddenLayers[id];
        renderLegend(); renderRail(); renderStage();
      });
    });
  }

  function renderRail() {
    var rail = $('#spine-rail');
    var html = '';
    var shownPasses = 0, shownDec = 0;

    D.phases.forEach(function (ph) {
      var passes = D.passes.filter(function (p) {
        if (p.phase !== ph.id) return false;
        if (!state.railAll && p.decisions === 0) return false;
        if (p.layer && state.hiddenLayers[p.layer]) return false;
        return true;
      });
      if (!passes.length) return;

      var phDec = passes.reduce(function (a, p) { return a + p.decisions; }, 0);
      shownPasses += passes.length; shownDec += phDec;

      html += '<div class="rail-group">';
      html += '<div class="rail-group__head"><span class="rail-group__name">' + esc(ph.name) + '</span>' +
              '<span class="rail-group__range">' + esc(ph.range) + '</span></div>';

      passes.forEach(function (p) {
        var ds = decisionsOfPass(p.idx);
        var isDec = p.decisions > 0;
        var tone = p.layer ? LAYER_TONE[p.layer] : 'var(--border-default)';
        var focused = state.focus && decision(state.focus) && decision(state.focus).pass === p.idx;
        var open = state.openPass === p.idx;
        var dots = ds.length
          ? ds.map(function (d) { return '<i class="spine-dot ' + d.level.toLowerCase() + '"></i>'; }).join('')
          : (isDec ? '<i class="spine-dot l1"></i>' : '');

        html += '<div class="rail-pass' + (focused ? ' is-active' : '') + (open ? ' is-open' : '') +
                (isDec ? ' is-decision' : ' is-mechanical') + '" data-pass="' + p.idx + '" ' +
                'style="--tone:' + tone + '" tabindex="0" role="button" aria-expanded="' + (open ? 'true' : 'false') + '">' +
                '<span class="rail-pass__idx">' + (p.foreign ? '··' : (p.idx < 10 ? '0' + p.idx : p.idx)) + '</span>' +
                '<span class="rail-pass__name">' + esc(p.name.replace('PTOAS · ', '')) + '</span>' +
                '<span class="rail-pass__dots">' + dots +
                (isDec ? '<b class="rail-pass__n">' + p.decisions + '</b>' : '<span class="rail-pass__mech">机械</span>') +
                '</span></div>';

        if (open) {
          html += '<div class="rail-desc">' +
                  '<p class="rail-desc__why">' + esc(p.why) + '</p>' +
                  '<p class="rail-desc__note' + (isDec ? ' is-dec' : '') + '">' + esc(p.note) + '</p>' +
                  (ds.length ? '<div class="rail-desc__ds">' + ds.map(function (d) {
                    return '<button type="button" class="rail-desc__d" data-goto="' + d.id + '">' +
                           '<b class="dc__lvl ' + d.level + '">' + d.level + '</b>' + esc(d.title) + '</button>';
                  }).join('') + '</div>' : '') +
                  '</div>';
        }
      });
      html += '</div>';
    });

    rail.innerHTML = html;

    $$('.rail-pass', rail).forEach(function (row) {
      var go = function () {
        var idx = Number(row.dataset.pass);
        state.openPass = state.openPass === idx ? null : idx;
        renderRail();
      };
      row.addEventListener('click', go);
      row.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
    });
    $$('.rail-desc__d', rail).forEach(function (b) {
      b.addEventListener('click', function (e) { e.stopPropagation(); focusDecision(b.dataset.goto); });
    });

    var openRow = $('.rail-pass.is-open', rail);
    if (openRow) openRow.scrollIntoView({ block: 'nearest' });

    $('#spine-meta').textContent = shownPasses + ' Pass · ' + shownDec + ' 决策';
    $('#spine-foot').innerHTML = state.railAll
      ? '<b style="color:var(--foreground)">51 个 Pass</b>（pypto 47 + PTOAS 4）里只有 <b style="color:var(--warning)">' +
        D.passes.filter(function (p) { return p.decisions > 0; }).length +
        ' 个在做决定</b>，其余是机械改写：输入定则输出唯一。点任一行看它做什么。'
      : '只显示做决定的 Pass。机械改写属于 IR diff 的范畴，不属于决策——把 51 个平铺给用户是错的。';
  }

  /* =========================================================
   * 中栏 · stage tabs
   * ======================================================= */
  function renderStageTabs() {
    var el = $('#stage-tabs');
    el.innerHTML = STAGES.map(function (s) {
      return '<button type="button" class="stage-tab' + (state.stage === s.id ? ' is-on' : '') +
             '" data-stage="' + s.id + '" role="tab">' + esc(s.label) + '</button>';
    }).join('');
    $$('.stage-tab', el).forEach(function (b) {
      b.addEventListener('click', function () {
        state.stage = b.dataset.stage;
        if (state.stage === 'inline') setMode('source'); else setMode('decisions');
        renderStageTabs(); renderStage(); renderStageHead();
      });
    });
  }

  function renderStageHead() {
    var s = STAGES.filter(function (x) { return x.id === state.stage; })[0];
    $('#stage-title').textContent = s ? s.label : '';
    var f = state.focus ? decision(state.focus) : null;
    $('#stage-meta').textContent = f ? (f.id + ' · ' + LAYER_NAME[f.layer]) : (D.session.total + ' 个决策');
    $('#status-count').textContent = D.session.total + ' 决策 · ' + D.session.review + ' 需复核 · ' + D.session.override + ' 推翻显式意图';
    $('#status-focus').textContent = f ? ('聚焦 ' + f.id + '（' + f.level + '）') : '未选中决策';
  }

  /* =========================================================
   * 中栏 · stage 内容
   * ======================================================= */
  function stepLead() {
    var st = D.steps[state.step];
    if (!st) return '';
    return '<div class="stage-lead"><p class="stage-lead__t">' + esc(st.title) + '</p>' +
           '<p class="stage-lead__b">' + esc(st.body) + '</p></div>';
  }

  function renderStage() {
    var host = $('#stage');
    var body = '';
    if (state.stage === 'spine')    body = viewSpine();
    if (state.stage === 'timeline') body = viewTimeline();
    if (state.stage === 'memmap')   body = viewMemMap();
    if (state.stage === 'swimlane') body = viewSwimlane();
    if (state.stage === 'diff')     body = viewDiff();
    if (state.stage === 'inline')   body = '<div class="stage-lead"><p class="stage-lead__t">已切到「源码 / IR」视图</p>' +
      '<p class="stage-lead__b">决策标注挂在源码改动行旁边，点击行即可跳到对应决策卡。</p></div>';
    host.innerHTML = stepLead() + body;
    host.scrollTop = 0;
    bindStage(host);
  }

  function bindStage(host) {
    $$('[data-goto]', host).forEach(function (el) {
      el.addEventListener('click', function () { focusDecision(el.dataset.goto); });
    });
    $$('[data-stage-go]', host).forEach(function (el) {
      el.addEventListener('click', function () {
        state.stage = el.dataset.stageGo;
        renderStageTabs(); renderStage(); renderStageHead();
      });
    });
  }

  /* ---- 决策脊全景（横向 SVG） ---- */
  function viewSpine() {
    // 左侧为泳道标签保留独立列，右侧保留阶段与跨层关系的呼吸空间。
    // 图宽增加后，密集决策点不再依赖整体放大来勉强区分。
    var W = 760, laneH = 58, padL = 124, padR = 40, padT = 68;
    var layers = D.layers.filter(function (l) { return !state.hiddenLayers[l.id]; });
    var H = padT + layers.length * laneH + 36;

    var PHASE_SHORT = { front: '前端', scope: '作用域', tile: 'Tile', split: '拆分',
      mem: '内存', dep: '依赖', dist: '分布', tail: '收尾', ptoas: 'PTOAS' };

    // 全景只画决策点：机械改写 Pass 不占横轴位置
    var passes = D.passes.filter(function (p) { return p.decisions > 0 && p.layer && !state.hiddenLayers[p.layer]; });
    // 横轴用序数而非 Pass 编号：保执行序，避免 46 → 91 之间出现大片空白
    var ord = {};
    passes.forEach(function (p, i) { ord[p.idx] = i; });
    var span = Math.max(1, passes.length - 1);
    var plotW = W - padL - padR;
    // 节点中心不贴卡片边缘：为圆点、焦点环和编号留出完整安全区。
    var nodeInset = 22;
    var nodeW = plotW - nodeInset * 2;
    var x = function (idx) { return padL + nodeInset + (ord[idx] || 0) / span * nodeW; };
    var x0 = function (ordinal) { return padL + nodeInset + ordinal / span * nodeW; };
    var laneY = function (id) {
      var i = layers.map(function (l) { return l.id; }).indexOf(id);
      return i < 0 ? null : padT + i * laneH + laneH / 2;
    };

    var s = '<svg class="spinepan" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="决策脊全景">';
    s += '<defs>';
    s += '<marker id="arc-arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">' +
         '<path d="M0 0 L8 4 L0 8 z" fill="var(--warning)"/></marker>';
    layers.forEach(function (l) {
      s += '<linearGradient id="lane-grad-' + l.id + '" x1="0" y1="0" x2="0" y2="1">' +
           '<stop offset="0" stop-color="' + LAYER_TONE[l.id] + '" stop-opacity="0.12"/>' +
           '<stop offset="1" stop-color="' + LAYER_TONE[l.id] + '" stop-opacity="0.035"/></linearGradient>';
    });
    s += '</defs>';

    // 顶部向导：意图 → 代码 + 执行序刻度
    s += '<text class="spinepan__axis-label" x="' + padL + '" y="16">表达意图</text>';
    s += '<text class="spinepan__axis-label" x="' + (W - padR) + '" y="16" text-anchor="end">生成代码</text>';
    s += '<line class="sv-axis" x1="' + padL + '" y1="22" x2="' + (W - padR) + '" y2="22"/>';
    s += '<line class="spinepan__origin" x1="' + padL + '" y1="16" x2="' + padL + '" y2="28"/>';
    s += '<line class="spinepan__origin" x1="' + (W - padR) + '" y1="16" x2="' + (W - padR) + '" y2="28"/>';

    // 阶段分隔带：把决策点按阶段着色分区（横向）
    D.phases.forEach(function (ph) {
      var inPh = passes.filter(function (p) { return p.phase === ph.id; });
      if (!inPh.length) return;
      var idxs = inPh.map(function (p) { return ord[p.idx]; }).filter(function (v) { return v != null; });
      var lo = Math.min.apply(null, idxs), hi = Math.max.apply(null, idxs);
      var mid = (x0(lo) + x0(hi)) / 2;
      var rectY = padT - 24;
      s += '<line class="spinepan__phase-tick" x1="' + mid + '" y1="' + rectY + '" x2="' + mid + '" y2="' + (rectY + 3) + '"/>';
      s += '<text class="spinepan__phase-label" x="' + mid + '" y="' + (rectY - 3) + '" text-anchor="middle">' + (PHASE_SHORT[ph.id] || ph.id) + '</text>';
    });

    // 泳道：每层一条，tone 渐变底
    layers.forEach(function (l, i) {
      var y = padT + i * laneH;
      var labelY = y + laneH / 2;
      s += '<rect class="spinepan__lane" x="' + padL + '" y="' + (y + 4) + '" width="' + plotW + '" height="' + (laneH - 8) + '" rx="10" fill="url(#lane-grad-' + l.id + ')"/>';
      // 泳道说明放到绘图区外，避免与第一个决策点共用起始坐标。
      s += '<text class="sv-lane" x="' + (padL - 18) + '" y="' + (labelY - 3) + '" text-anchor="end" style="fill:' + LAYER_TONE[l.id] + '">' + esc(l.short) + '</text>';
      s += '<text class="spinepan__lane-sub" x="' + (padL - 18) + '" y="' + (labelY + 11) + '" text-anchor="end">' + esc(l.name) + '</text>';
    });

    // 跨层影响弧线：33 MemoryReuse → PTOAS InsertSync
    var y1 = laneY('memory'), y2 = laneY('sync');
    if (y1 != null && y2 != null && ord[33] != null && ord[91] != null) {
      var x1 = x(33), x2 = x(91);
      var top = Math.min(y1, y2) - 29;
      var yStart = y1 - 13, yEnd = y2 - 13;
      s += '<path class="spinepan__arc" marker-end="url(#arc-arrow)" d="M' + x1 + ' ' + yStart +
           ' C ' + x1 + ' ' + top + ', ' + x2 + ' ' + top + ', ' + x2 + ' ' + yEnd + '"/>';
      // 弧线两端落点画成实心圆点，起点/终点一眼可辨
      s += '<circle class="spinepan__arc-cap" cx="' + x1 + '" cy="' + yStart + '" r="2.5"/>';
      s += '<circle class="spinepan__arc-cap" cx="' + x2 + '" cy="' + yEnd + '" r="2.5"/>';
      s += '<text class="spinepan__arc-label" x="' + (x2 - 8) + '" y="' + (top - 6) + '" text-anchor="end">复用决策 → 下游同步决策</text>';
    }

    // 节点
    passes.forEach(function (p) {
      var cy = laneY(p.layer);
      if (cy == null) return;
      var cx2 = x(p.idx);
      var ds = decisionsOfPass(p.idx);
      var top2 = ds.length ? ds.slice().sort(function (a, b) { return levelRank(b.level) - levelRank(a.level); })[0] : null;
      var r = 4 + Math.min(4.5, p.decisions * 0.45);
      var tone = LAYER_TONE[p.layer];
      var gid = top2 ? ' data-goto="' + top2.id + '"' : '';

      s += '<g class="spinepan__node"' + gid + '>';
      if (top2 && top2.level === 'L3') {
        s += '<polygon points="' + cx2 + ',' + (cy - r - 4) + ' ' + (cx2 + r + 4) + ',' + (cy + r + 1) + ' ' + (cx2 - r - 4) + ',' + (cy + r + 1) + '" fill="var(--danger)"/>';
      } else {
        var fill = !top2 ? 'transparent' : (top2.level === 'L2' ? tone : 'color-mix(in srgb,' + tone + ' 30%, transparent)');
        s += '<circle cx="' + cx2 + '" cy="' + cy + '" r="' + r + '" fill="' + fill + '" stroke="' + tone + '" stroke-width="1.4"/>';
      }
      if (state.focus && top2 && top2.id === state.focus) {
        s += '<circle class="spinepan__focus" cx="' + cx2 + '" cy="' + cy + '" r="' + (r + 5) + '"/>';
      }
      s += '<text class="sv-label" x="' + cx2 + '" y="' + (cy + r + 10) + '" text-anchor="middle">' +
           (p.foreign ? ('AS' + p.idx) : p.idx) + '</text>';
      s += '<title>' + esc(p.name) + ' · ' + p.decisions + ' 个决策</title></g>';
    });

    s += '</svg>';

    // 图例：等级语义 + 层色，作为 SVG 的注脚而非散落各处
    var legend = '<div class="spinepan__legend">' +
      '<span class="spinepan__legend-item"><i class="spinepan__legend-sw" data-shape="tri" style="--tone:var(--danger)"></i>L3 必须回答</span>' +
      '<span class="spinepan__legend-item"><i class="spinepan__legend-sw" data-shape="dot" style="--tone:var(--warning)"></i>L2 需复核</span>' +
      '<span class="spinepan__legend-item"><i class="spinepan__legend-sw" data-shape="dot" data-l1="1" style="--tone:var(--tone-tiling)"></i>L1 记录</span>' +
      '<span class="spinepan__legend-item"><i class="spinepan__legend-sw" data-shape="dash" style="--tone:var(--warning)"></i>跨层影响</span>' +
      '</div>';

    var l3 = D.decisions.filter(function (d) { return d.level === 'L3'; });
    var l2 = D.decisions.filter(function (d) { return d.level === 'L2'; });
    var list = l3.concat(l2).map(function (d) {
      return '<div class="dd-row k-' + (d.level === 'L3' ? 'removed' : 'changed') + '" data-goto="' + d.id + '" style="cursor:pointer">' +
        '<span class="dd-kind">' + d.level + ' · ' + esc(LAYER_NAME[d.layer]) + '</span>' +
        '<span class="dd-text">' + esc(d.title) +
        '<span class="dd-note">' + esc(d.cost.metric) + ' · ' + esc(d.pass === 90 ? 'PTOAS InsertSync' : ('Pass ' + d.pass + ' ' + passByIdx(d.pass).name)) + '</span></span></div>';
    }).join('');

    return '<div class="viz"><div class="viz__head"><span class="viz__t">决策脊 · 按四层与执行序</span>' +
      '<span class="viz__m">点圆点跳到决策卡 · 弧线 = 跨层影响</span></div><div class="viz__scroll">' + s + '</div>' +
      legend + '</div>' +
      '<div class="viz"><div class="viz__head"><span class="viz__t">需要你看的 ' + (l3.length + l2.length) + ' 条</span>' +
      '<span class="viz__m">L0 决策不出现在这里</span></div>' + list + '</div>';
  }

  /* ---- 对象时间线 ---- */
  function viewTimeline() {
    var t = D.timeline;
    var rows = t.nodes.map(function (n) {
      var isD = !!n.decision;
      var pl = n.pass === 99 ? 'codegen' : (n.pass >= 90 ? 'PTOAS' : String(n.pass));
      return '<div class="tl__row' + (isD ? ' is-decision' : '') + '"' + (isD ? ' data-goto="' + n.decision + '"' : '') + '>' +
        '<div class="tl__pass">' + esc(pl) + '<b>' + esc(n.name.replace('PTOAS · ', '')) + '</b></div>' +
        '<div class="tl__spine"><i class="tl__mark ' + n.touch + '"></i></div>' +
        '<div><div class="tl__state">' + esc(n.state) +
        (isD ? '<span class="tl__chip">' + esc(n.decision) + '</span>' : '') + '</div>' +
        (n.warn ? '<div class="tl__warn">⚠ ' + esc(n.warn) + '</div>' : '') + '</div></div>';
    }).join('');

    return '<div class="viz"><div class="viz__head"><span class="viz__t">' + esc(t.object) + '</span>' +
      '<span class="viz__m">' + esc(t.kind) + ' · 点带 ⚠ 的节点看决策卡</span></div>' +
      '<div class="tl">' + rows + '</div></div>' +
      '<div class="viz"><div class="viz__head"><span class="viz__t">对照：issue #1475 的作者当年怎么走完这条线</span></div>' +
      '<div class="dd-row k-same"><span class="dd-kind">1–2</span><span class="dd-text">从 swimlane 气泡起疑，打开 28_after_InitMemRef.py 看到 4 个 Mat alloc</span></div>' +
      '<div class="dd-row k-same"><span class="dd-kind">3</span><span class="dd-text">打开 29_after_MemoryReuse.py，发现只剩 2 个——但不知道是哪两个被合了、为什么可以合</span></div>' +
      '<div class="dd-row k-same"><span class="dd-kind">4</span><span class="dd-text">打开 fa_qks_aic.cpp，数到第 90 / 114 行的 set_flag / wait_flag</span></div>' +
      '<div class="dd-row k-changed"><span class="dd-kind">5 · 人脑</span><span class="dd-text">关联出结论：这条 WAR 边只因为 buffer 被复用才存在<span class="dd-note">ir_trace 能帮到第 2–3 步，帮不到第 5 步——而第 5 步才是结论</span></span></div></div>';
  }

  /* ---- Memory Map ---- */
  function mmPanel(side, data, focusBase) {
    var W = 380, H = 250, padL = 8, padT = 22, padB = 26;
    var cap = D.memmap.capacity;
    var lr = D.memmap.lineRange;
    var ax = function (addr) { return padL + addr / cap * (W - padL - 10); };
    var aw = function (size) { return size / cap * (W - padL - 10); };
    var ay = function (line) { return padT + (line - lr[0]) / (lr[1] - lr[0]) * (H - padT - padB); };

    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(data.label) + '">';
    s += '<defs><marker id="mm-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">' +
         '<path d="M0 0 L8 4 L0 8 z" fill="var(--warning)"/></marker></defs>';

    // 轴
    s += '<text class="sv-label" x="' + padL + '" y="12">0</text>';
    s += '<text class="sv-label" x="' + (W - 10) + '" y="12" text-anchor="end">128 KB</text>';
    s += '<line class="sv-axis" x1="' + padL + '" y1="16" x2="' + (W - 10) + '" y2="16"/>';
    [0.25, 0.5, 0.75].forEach(function (f) {
      var xx = padL + f * (W - padL - 10);
      s += '<line class="sv-grid" x1="' + xx + '" y1="16" x2="' + xx + '" y2="' + (H - padB) + '"/>';
    });

    var pos = {};
    var labelText = function (b, cls) {
      // 标签统一放矩形上方，不塞进窄框里，避免文字溢出/重叠
      var ly = Math.max(13, ay(b.from) - 5);
      return '<text class="' + cls + '" x="' + ax(b.addr) + '" y="' + ly + '">' + esc(b.name) + '</text>';
    };
    data.boxes.forEach(function (b) {
      var bx = ax(b.addr), bw = Math.max(6, aw(b.size));
      var by = ay(b.from), bh = Math.max(8, ay(b.to) - ay(b.from));
      pos[b.name] = { x: bx + bw / 2, y: by, y2: by + bh, w: bw };
      var cls = 'mm-box role-' + b.role + (b.reused ? ' is-reused' : '') + (focusBase && b.base === focusBase ? ' is-focus' : '');
      s += '<g class="' + cls + '" data-goto="D-33-04">' +
           '<rect x="' + bx + '" y="' + by + '" width="' + bw + '" height="' + bh + '" rx="3"/>' +
           labelText(b, 'mm-box__label') +
           '<title>' + esc(b.base + ' · ' + b.name + ' · ' + b.size + ' B · 行 ' + b.from + '–' + b.to) + '</title></g>';
    });

    (data.blocked || []).forEach(function (b) {
      var bx = ax(b.addr), bw = Math.max(6, aw(b.size));
      var by = ay(b.from), bh = Math.max(8, ay(b.to) - ay(b.from));
      s += '<g class="mm-blocked"><rect x="' + bx + '" y="' + by + '" width="' + bw + '" height="' + bh + '" rx="3"/>' +
           labelText(b, 'mm-box__label mm-box__label--blocked') +
           '<title>' + esc(b.name + '：' + b.reason) + '</title></g>';
    });

    (data.edges || []).forEach(function (e, i) {
      var a = pos[e.from], b = pos[e.to];
      if (!a || !b) return;
      var mx = a.x + (i === 0 ? 0 : 6);
      s += '<path class="mm-edge" d="M' + mx + ' ' + a.y2 + ' C ' + (mx + 26) + ' ' + (a.y2 + 12) + ', ' + (mx + 26) + ' ' + (b.y - 12) + ', ' + mx + ' ' + (b.y - 3) + '"/>';
      if (i === 0) s += '<text class="mm-edge-label" x="' + (mx + 30) + '" y="' + ((a.y2 + b.y) / 2) + '">' + esc(e.pipe) + '</text>';
    });

    var hw = side === 'after' ? ax(98304) : ax(131072);
    s += '<line class="mm-hw" x1="' + hw + '" y1="16" x2="' + hw + '" y2="' + (H - padB) + '"/>';
    s += '<text class="sv-label" x="' + (hw - 4) + '" y="' + (H - 14) + '" text-anchor="end">high-water</text>';
    s += '</svg>';

    return '<div class="mm-panel"><div class="mm-panel__t">' + esc(data.label) + '</div>' + s + '</div>';
  }

  function viewMemMap() {
    var m = D.memmap;
    var body = '<div class="viz"><div class="viz__head"><span class="viz__t">' + esc(m.space) +
      ' · 地址 × 生命周期</span><span class="viz__m">左右地址轴对齐 · 橙线 = 复用引入的依赖边</span></div>' +
      '<div class="mm-grid">' + mmPanel('before', m.before, null) + mmPanel('after', m.after, 'mem_mat_12') + '</div>' +
      '<div class="mm-legend">' +
      '<span class="mm-legend__i"><i class="mm-legend__sw" style="background:color-mix(in srgb,var(--primary) 26%,var(--surface-1));border-color:var(--primary)"></i>K tile</span>' +
      '<span class="mm-legend__i"><i class="mm-legend__sw" style="background:color-mix(in srgb,var(--success) 22%,var(--surface-1));border-color:var(--success)"></i>Q tile</span>' +
      '<span class="mm-legend__i"><i class="mm-legend__sw" style="border-color:var(--warning);border-style:dashed"></i>复用分段（第二个住户）</span>' +
      '<span class="mm-legend__i"><i class="mm-legend__sw" style="border-color:var(--foreground-muted);border-style:dashed;background:transparent"></i>被 not_inplace_safe 挡住的复用</span>' +
      '<span class="mm-legend__i"><svg width="20" height="8"><line x1="0" y1="4" x2="18" y2="4" stroke="var(--warning)" stroke-width="1.6"/></svg>复用引入的 WAR 边</span>' +
      '</div></div>';

    body += '<div class="viz"><div class="viz__head"><span class="viz__t">现有 memory_map.html 有什么、缺什么</span></div>' +
      '<div class="dd-row k-same"><span class="dd-kind">已有</span><span class="dd-text">地址 × 生命周期矩形、分空间 panel、high-water、容量 pill、红框标不同 base 的重叠</span></div>' +
      '<div class="dd-row k-changed"><span class="dd-kind">新增 1</span><span class="dd-text">复用分段：一个物理 buffer 上先后住过谁，画在同一根竖条上</span></div>' +
      '<div class="dd-row k-changed"><span class="dd-kind">新增 2</span><span class="dd-text">复用引入的依赖边（橙线，标 pipe 对）<span class="dd-note">这是 #1475 缺的东西：复用发生了看得见，复用的代价看不见</span></span></div>' +
      '<div class="dd-row k-changed"><span class="dd-kind">新增 3</span><span class="dd-text">被挡住的复用（灰虚线）——「我为什么没帮你省这块」同样要可见<span class="dd-note">数据来自 ForbidAliasCollector 的 forbidden-input set</span></span></div></div>';
    return body;
  }

  /* ---- 同步泳道 ---- */
  function swPanel(data, better) {
    var W = 820, padL = 66, padT = 20, rowH = 30;
    var H = padT + D.swimlane.pipes.length * rowH + 24;
    var sx = function (t) { return padL + t / D.swimlane.span * (W - padL - 16); };
    var rowY = function (pipe) { return padT + D.swimlane.pipes.indexOf(pipe) * rowH; };

    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(data.label) + '">';

    // 顶部时间刻度尺（只标两端，避免喧宾夺主）
    s += '<line class="sv-axis" x1="' + padL + '" y1="14" x2="' + (W - 16) + '" y2="14"/>';
    s += '<text class="sv-label" x="' + padL + '" y="10">0</text>';
    s += '<text class="sv-label" x="' + (W - 16) + '" y="10" text-anchor="end">' + D.swimlane.span + '</text>';
    [0.25, 0.5, 0.75].forEach(function (f) {
      var xx = padL + f * (W - padL - 16);
      s += '<line class="sv-grid" x1="' + xx + '" y1="12" x2="' + xx + '" y2="16"/>';
    });

    D.swimlane.pipes.forEach(function (p) {
      var y = rowY(p);
      s += '<rect class="sw-lane" x="' + padL + '" y="' + (y + 2) + '" width="' + (W - padL - 16) + '" height="' + (rowH - 4) + '" rx="4"/>';
      s += '<text class="sv-name" x="' + (padL - 10) + '" y="' + (y + rowH / 2 + 3) + '" text-anchor="end">' + esc(p) + '</text>';
      s += '<line class="sv-grid" x1="' + padL + '" y1="' + (y + rowH - 4) + '" x2="' + (W - 16) + '" y2="' + (y + rowH - 4) + '"/>';
    });

    (data.gaps || []).forEach(function (g) {
      var y = rowY(g.pipe);
      s += '<g class="sw-gap"><rect x="' + sx(g.t) + '" y="' + (y + 3) + '" width="' + (sx(g.t + g.w) - sx(g.t)) + '" height="' + (rowH - 11) + '" rx="3"/>' +
           '<text x="' + (sx(g.t) + 6) + '" y="' + (y + rowH / 2 + 2) + '">' + esc(g.label) + '</text></g>';
    });

    data.ops.forEach(function (o) {
      var y = rowY(o.pipe);
      var x1 = sx(o.t), w = sx(o.t + o.w) - sx(o.t);
      s += '<g class="sw-op tone-' + o.tone + (o.blocked ? ' is-blocked' : '') + '">' +
           '<rect x="' + x1 + '" y="' + (y + 3) + '" width="' + w + '" height="' + (rowH - 11) + '"/>' +
           '<text x="' + (x1 + 5) + '" y="' + (y + rowH / 2 + 2) + '">' + esc(o.name) + '</text>' +
           (o.line ? '<title>fa_qks_aic.cpp:' + o.line + '</title>' : '') + '</g>';
    });

    (data.flags || []).forEach(function (f) {
      var y1 = rowY(f.from) + rowH / 2, y2 = rowY(f.to) + rowH / 2, xx = sx(f.t);
      s += '<g data-goto="' + f.decision + '" style="cursor:pointer">' +
           '<path class="sw-flag" d="M' + xx + ' ' + y1 + ' L ' + xx + ' ' + y2 + '"/>' +
           '<circle cx="' + xx + '" cy="' + y2 + '" r="3" fill="var(--danger)"/>' +
           '<text class="sw-flag-label" x="' + (xx + 6) + '" y="' + (y2 - 6) + '">' + esc(f.label) + '</text></g>';
    });
    s += '</svg>';

    return '<div class="sw-panel"><div class="viz__head"><span class="viz__t" style="font:var(--type-body-sm)">' + esc(data.label) +
      '</span><span class="viz__m"><span class="sw-total' + (better ? ' is-better' : '') + '">' + esc(data.total) + '</span></span></div>' +
      '<div class="viz__scroll">' + s + '</div>' +
      (data.note ? '<div class="dd-note" style="margin-top:6px">' + esc(data.note) + '</div>' : '') + '</div>';
  }

  function viewSwimlane() {
    var sl = D.swimlane;
    return '<div class="viz"><div class="viz__head"><span class="viz__t">双泳道对比</span>' +
      '<span class="viz__m">默认视图，不是高级功能</span></div>' +
      swPanel(sl.current, false) + swPanel(sl.counterfactual, true) + '</div>' +
      '<div class="viz"><div class="viz__head"><span class="viz__t">为什么对比必须是默认态</span></div>' +
      '<div class="dd-row k-changed"><span class="dd-kind">出处</span><span class="dd-text">PTOAS#226 里，issue 作者自己贴了 msprof op simulator 生成的 auto-sync（上）vs manual-sync（下）双时间线截图；另一位工程师从图里读出「L0 循环已经很优，manual 在 L1 循环上气泡更小，多半是手工预取」。<span class="dd-note">那段对话就是这个视图的需求文档</span></span></div>' +
      '<div class="dd-row k-same"><span class="dd-kind">confidence</span><span class="dd-text">每条 flag 要编码把握度：可证明安全（细线）/ 保守插入（正常）/ UNKNOWN 回退（加粗 + 问号）。UNKNOWN 标出「编译器自己也没把握」的位置，正是 PTOAS#646「删一条多余 barrier 把 FA 吞吐提到 ~165 TFLOP/s」那类机会所在。</span></div></div>';
  }

  /* ---- 决策 diff ---- */
  function viewDiff() {
    var d = D.diff;
    var rows = d.rows.map(function (r) {
      var kindText = { removed: '− 消失', changed: '~ 变化', same: '= 不变' }[r.kind];
      return '<div class="dd-row k-' + r.kind + '"' + (decision(r.id) ? ' data-goto="' + r.id + '" style="cursor:pointer"' : '') + '>' +
        '<span class="dd-kind">' + kindText + '</span>' +
        '<span class="dd-text">' + esc(r.text) + '<span class="dd-note">' + esc(r.id) + ' · ' + esc(r.note) + '</span></span></div>';
    }).join('');

    return '<div class="viz"><div class="viz__head"><span class="viz__t">决策集合做差</span>' +
      '<span class="viz__m">同一交互服务跨版本排查与 Baseline vs Candidate</span></div>' +
      '<div class="dd-head">' +
      '<div class="dd-side"><div class="dd-side__t">' + esc(d.left.label) + '</div><div class="dd-side__m">' + esc(d.left.sha) + '</div></div>' +
      '<div class="dd-side"><div class="dd-side__t">' + esc(d.right.label) + '</div><div class="dd-side__m">' + esc(d.right.sha) + '</div></div>' +
      '</div>' + rows +
      '<div class="dd-outcome">' +
      '<div><span class="dd-outcome__k">改动前</span><span class="dd-outcome__v">' + esc(d.outcome.before) + '</span></div>' +
      '<div><span class="dd-outcome__k">改动后</span><span class="dd-outcome__v is-good">' + esc(d.outcome.after) + '</span></div>' +
      '<div><span class="dd-outcome__k">变化</span><span class="dd-outcome__v is-good">' + esc(d.outcome.delta) + '</span></div>' +
      '<div><span class="dd-outcome__k">正确性</span><span class="dd-outcome__v">' + esc(d.outcome.correctness) + '</span></div>' +
      '</div></div>' +
      '<div class="viz"><div class="viz__head"><span class="viz__t">对标 PTOAS#1111</span></div>' +
      '<div class="dd-row k-changed"><span class="dd-kind">今天</span><span class="dd-text">v0.48 → v0.54 结果回归，算术 opcode 一条没变。排查方式：逐版本 bisect + 读汇编。</span></div>' +
      '<div class="dd-row k-removed"><span class="dd-kind">有决策 diff</span><span class="dd-text">一屏看到：auto-sync flag 对在 v0.49 消失、dcci scope 在 v0.50 从 OUT 变 ALL。</span></div></div>';
  }

  /* =========================================================
   * 右栏 · 决策卡
   * ======================================================= */
  function renderCard() {
    var host = $('#card-body');
    var d = state.focus ? decision(state.focus) : null;
    if (!d) {
      $('#card-meta').textContent = '未选中';
      host.innerHTML = '<div class="card-empty">在左侧决策脊或中间图上点一个决策点。<br><br>' +
        'L0 决策不会出现在界面上——它们只进 decisions.json。</div>';
      return;
    }
    $('#card-meta').textContent = d.issue || '—';
    var p = passByIdx(d.pass);
    var h = '';

    h += '<div class="dc__top" style="--tone:' + LAYER_TONE[d.layer] + '">' +
         '<span class="dc__lvl ' + d.level + '">' + d.level + '</span>' +
         '<span class="dc__id">' + esc(d.id) + '</span>' +
         '<span class="dc__layer"><i></i>' + esc(LAYER_NAME[d.layer]) + '</span></div>';
    h += '<h3 class="dc__title">' + esc(d.title) + '</h3>';
    h += '<p class="dc__actor">actor · ' + esc(p ? p.name : ('pass ' + d.pass)) + '　span · ' + esc(d.span.file.split('/').pop()) + ':' + d.span.line + '</p>';

    h += '<div class="dc__f"><span class="dc__k">SUBJECT 作用对象</span><span class="dc__v dc__sub">' +
         d.subject.map(function (s) { return '<code>' + esc(s) + '</code>'; }).join('') + '</span></div>';

    h += '<div class="dc__f"><span class="dc__k">TRIGGER 为什么触发</span><span class="dc__v"><strong>' +
         esc(d.trigger) + '</strong><br>' + esc(d.triggerText) + '</span></div>';

    h += '<div class="dc__f"><span class="dc__k">PROTECTS 它保护了什么</span><span class="dc__v">' + esc(d.protects) + '</span></div>';

    h += '<div class="dc__f"><span class="dc__k">DECLINED 它放弃了什么</span><span class="dc__v">' + esc(d.declined) + '</span></div>';

    h += '<div class="dc__f"><span class="dc__k">COST 代价</span><div class="dc__cost"><span class="dc__v">' +
         esc(d.cost.text) + '</span><span class="dc__cost-m">' + esc(d.cost.metric) + '</span></div></div>';

    h += '<div class="dc__f"><span class="dc__k">CONFIDENCE 把握度</span><span class="dc__v dc__conf">' +
         '<i class="dc__conf-dot ' + d.confidence + '"></i>' + esc(d.confidence) + '</span>' +
         (d.confidenceNote ? '<span class="dc__conf-note">' + esc(d.confidenceNote) + '</span>' : '') + '</div>';

    if (d.upstream) {
      h += '<div class="dc__f"><span class="dc__k">UPSTREAM 上游决策</span>' +
           '<button type="button" class="dc__link" data-goto="' + d.upstream + '">↑ ' + esc(d.upstream) + ' · 真正该被质疑的是它</button></div>';
    }

    h += '<div class="dc__f"><span class="dc__k">OVERRIDE 出口</span>' +
         '<pre class="dc__code">' + esc(d.overrideCode) + '</pre>' +
         '<button type="button" class="btn btn-sm dc__copy" data-copy="' + d.id + '">复制 override 代码</button>' +
         '<span class="dc__conf-note">按钮不叫「应用」——override 必须落到源码里，才能被 code review、进 git、进复现包。</span></div>';

    h += '<div class="dc__f"><span class="dc__k">EVIDENCE 证据</span>' +
         d.evidence.map(function (e) {
           return '<span class="dc__ev"><span class="dc__ev-p">' + esc(e.path) + ':' + e.line + '</span>' +
                  '<span class="dc__ev-n">' + esc(e.note) + '</span></span>';
         }).join('') + '</div>';

    host.innerHTML = h;
    host.scrollTop = 0;

    $$('[data-goto]', host).forEach(function (b) { b.addEventListener('click', function () { focusDecision(b.dataset.goto); }); });
    var copyBtn = $('[data-copy]', host);
    if (copyBtn) copyBtn.addEventListener('click', function () { copyText(d.overrideCode); });
  }

  function copyText(txt) {
    var done = function () { toast('override 代码已复制 — 粘贴到源码里，让它进 code review'); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).then(done, fallback);
    } else fallback();
    function fallback() {
      var ta = document.createElement('textarea');
      ta.value = txt; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); done(); } catch (e) { toast('复制失败，请手动选择'); }
      document.body.removeChild(ta);
    }
  }

  /* =========================================================
   * L3 告警条
   * ======================================================= */
  function renderAlert() {
    var slot = $('#alert-slot');
    var l3 = D.decisions.filter(function (d) { return d.level === 'L3'; })[0];
    if (!l3 || state.ackL3 || state.mode !== 'decisions') { slot.hidden = true; return; }
    slot.hidden = false;
    slot.innerHTML = '<div class="alert"><span class="alert__tag">L3</span>' +
      '<span class="alert__text">你写的 <b>pl.manual_scope()</b> 没有被执行——编译器最终发出的是 <b>PTO2_SCOPE(AUTO)</b>，' +
      l3.subject.length + ' 个 scope 受影响。这是唯一必须回答的一条。</span>' +
      '<span class="alert__act"><button type="button" class="btn btn-sm btn-solid" id="l3-view">查看决策</button>' +
      '<button type="button" class="btn btn-sm" id="l3-ack">我知道了</button></span></div>';
    $('#l3-view').addEventListener('click', function () { focusDecision(l3.id); });
    $('#l3-ack').addEventListener('click', function () {
      state.ackL3 = true; renderAlert();
      toast('已 ack — 真实产品里这一步会被记录：接受 / 局部否决 / 改代码，用于校准 L3 判据是否太松');
    });
  }

  /* =========================================================
   * 源码 / IR 视图
   * ======================================================= */
  function renderSource() {
    var tree = $('#file-tree');
    tree.innerHTML = D.tree.map(function (r) {
      return '<div class="tree-row' + (r.hot ? ' is-hot' : '') + '" style="padding-left:' + (10 + r.d * 13) + 'px">' +
        '<span>' + esc(r.t) + '</span>' +
        (r.kind !== 'dir' ? '<span class="tree-row__k">' + esc(r.kind) + '</span>' : '') +
        (r.badge ? '<span class="tree-row__badge">' + esc(r.badge) + '</span>' : '') + '</div>';
    }).join('');

    var prev = null;
    var code = D.source.lines.map(function (l) {
      var gap = (prev != null && l.n - prev > 1) ? '<div class="src-gapline">⋯ 省略 ' + (l.n - prev - 1) + ' 行 ⋯</div>' : '';
      prev = l.n;
      var d = l.dec ? decision(l.dec) : null;
      var cls = 'src-line' + (d ? ' has-dec' : '') + (d && d.id === state.focus ? ' is-focus' : '');
      return gap + '<div class="' + cls + '"' + (d ? ' data-goto="' + d.id + '"' : '') + '>' +
        '<span class="src-line__n">' + l.n + '</span>' +
        '<span class="src-line__t">' + esc(l.t) +
        (d ? '<span class="src-dec">ⓘ ' + esc(d.id) + ' · ' + esc(shortKind(d)) + '</span>' : '') + '</span></div>';
    }).join('');
    $('#src-editor').innerHTML = '<div class="src-code">' + code + '</div>';
    $('#src-title').textContent = D.source.file.split('/').pop();
    $('#src-meta').textContent = D.source.file;

    var withDec = D.source.lines.filter(function (l) { return l.dec; });
    $('#inline-body').innerHTML =
      '<div class="dd-note" style="margin-bottom:10px">ir_trace 回答「改了哪些行」；这一层回答「为什么这么改」。两者是两层，不互相替代。</div>' +
      withDec.map(function (l) {
        var d = decision(l.dec);
        return '<div class="dd-row k-' + (d.level === 'L3' ? 'removed' : (d.level === 'L2' ? 'changed' : 'same')) + '" data-goto="' + d.id + '" style="cursor:pointer">' +
          '<span class="dd-kind">行 ' + l.n + ' · ' + d.level + '</span>' +
          '<span class="dd-text">' + esc(d.title) + '<span class="dd-note">' + esc(d.cost.metric) + '</span></span></div>';
      }).join('');

    $$('[data-goto]', $('#src-editor')).concat($$('[data-goto]', $('#inline-body'))).forEach(function (el) {
      el.addEventListener('click', function () { focusDecision(el.dataset.goto); });
    });
  }

  function shortKind(d) {
    return { buffer_coalesce: '复用', scope_demotion: '降级', pipeline_depth_shed: '流水降深', tiling_declined: '放弃切分', tiling_applied: '切分', sync_insert: '插同步' }[d.kind] || d.kind;
  }

  /* =========================================================
   * 模式 / 聚焦 / 回放
   * ======================================================= */
  function setMode(m) {
    if (state.mode === m) return;
    state.mode = m;
    $('#decisions-view').hidden = m !== 'decisions';
    $('#source-view').hidden = m !== 'source';
    $$('.ide-mode-tab').forEach(function (b) {
      var on = b.dataset.mode === m;
      b.classList.toggle('is-selected', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    renderAlert();
    if (m === 'source') renderSource();
  }

  function focusDecision(id) {
    if (!decision(id)) return;
    state.focus = id;
    var d = decision(id);
    state.openPass = d.pass;
    if (state.mode === 'source') renderSource();
    renderRail(); renderCard(); renderStageHead();
    if (state.mode === 'decisions') renderStage();

    var rail = $('#spine-rail');
    var railEl = $('.rail-pass.is-open', rail) || $('.rail-pass.is-active', rail);
    setActiveElement(railEl, rail);
    var card = $('#card-body');
    var cardTop = $('.dc__top', card);
    setActiveElement(cardTop, card);
  }

  function renderStepper() {
    var el = $('#stepper');
    el.innerHTML = D.steps.map(function (s, i) {
      var cls = i === state.step ? ' is-on' : (i < state.step ? ' is-done' : '');
      return (i ? '<span class="step-sep">›</span>' : '') +
        '<button type="button" class="step-chip' + cls + '" data-i="' + i + '" role="tab">' + esc(s.phase) + '</button>';
    }).join('');
    $$('.step-chip', el).forEach(function (b) {
      b.addEventListener('click', function () { goStep(Number(b.dataset.i)); });
    });
    $('#counter').textContent = (state.step + 1) + ' / ' + D.steps.length;
    var st = D.steps[state.step];
    $('#readout').textContent = st ? st.readout : '';
    var f = st && st.focus ? decision(st.focus) : null;
    $('#transport-level').textContent = f ? (f.level + ' · ' + LAYER_NAME[f.layer]) : '总览';
  }

  function goStep(i) {
    if (i < 0 || i >= D.steps.length) { stopPlay(); return; }
    state.step = i;
    var st = D.steps[i];
    state.stage = st.stage;
    state.showOverride = !!st.showOverride;
    setMode(st.mode || 'decisions');
    if (st.focus) { state.focus = st.focus; state.openPass = decision(st.focus).pass; }
    else if (i === 0) { state.focus = null; state.openPass = null; state.ackL3 = false; }
    renderAll();
    if (st.showOverride) {
      setTimeout(function () {
        var c = $('.dc__copy');
        if (c) { c.focus({ preventScroll: true }); c.scrollIntoView({ block: 'center', behavior: 'smooth' }); }
      }, 90);
    }
  }

  function setPlayIcon(playing) {
    var playBtn = $('#step-play');
    var play = playBtn.querySelector('.ui-icon--play');
    var pause = playBtn.querySelector('.ui-icon--pause');
    if (play) play.hidden = playing;
    if (pause) pause.hidden = !playing;
    playBtn.setAttribute('aria-label', playing ? '暂停' : '自动播放');
    playBtn.setAttribute('title', playing ? '暂停（空格）' : '自动播放（空格）');
  }
  function stopPlay() {
    state.playing = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    setPlayIcon(false);
  }
  function togglePlay() {
    if (state.playing) { stopPlay(); return; }
    state.playing = true;
    setPlayIcon(true);
    state.timer = setInterval(function () {
      if (state.step >= D.steps.length - 1) { stopPlay(); return; }
      goStep(state.step + 1);
    }, 6200);
  }

  var toastTimer = null;
  function toast(msg) {
    var t = $('#toast');
    t.textContent = msg;
    t.classList.add('is-on');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.classList.remove('is-on'); }, 3200);
  }

  /* =========================================================
   * 面板开关 / 分栏
   * ======================================================= */
  var panels = { left: true, right: true };
  function applyPanels() {
    $$('.demo-view .ide-layout').forEach(function (layout) {
      var ps = Array.prototype.slice.call(layout.children).filter(function (el) { return el.classList.contains('ide-pane'); });
      if (ps.length < 3) return;
      setPane(ps[0], panels.left);
      setPane(ps[ps.length - 1], panels.right);
    });
    [['#wc-left', 'left'], ['#wc-right', 'right']].forEach(function (p) {
      var b = $(p[0]);
      b.classList.toggle('is-on', panels[p[1]]);
      b.setAttribute('aria-pressed', panels[p[1]] ? 'true' : 'false');
    });
  }
  function setPane(pane, open) {
    pane.hidden = !open;
    [pane.nextElementSibling, pane.previousElementSibling].forEach(function (g) {
      if (g && g.classList.contains('pto-workbench-shell__split-gutter')) g.hidden = !open;
    });
  }
  function initSplit() {
    if (!window.PtoWorkbenchShell || !window.PtoWorkbenchShell.initResizablePanes) return;
    $$('.ide-layout').forEach(function (layout, i) {
      var ps = Array.prototype.slice.call(layout.children).filter(function (el) { return el.classList.contains('ide-pane'); });
      if (ps.length < 2) return;
      window.PtoWorkbenchShell.initResizablePanes({
        root: layout, panes: ps, direction: 'horizontal',
        sizes: [21, 52, 27], minSize: [190, 400, 300], gutterSize: 10,
        storageKey: 'pass-decision-split-' + i, gutterLabel: '调整相邻面板宽度'
      });
    });
  }

  /* =========================================================
   * 渲染入口
   * ======================================================= */
  function renderAll() {
    renderLegend(); renderRail();
    renderStageTabs(); renderStageHead(); renderStage();
    renderCard(); renderAlert(); renderStepper();
    if (state.mode === 'source') renderSource();
  }

  function init() {
    $('#session-id').textContent = D.session.id;
    $('#session-title').textContent = D.session.title;
    $('#env-chip').textContent = D.session.env;

    $$('.ide-mode-tab').forEach(function (b) {
      b.addEventListener('click', function () {
        setMode(b.dataset.mode);
        if (b.dataset.mode === 'source') { state.stage = 'inline'; }
        else if (state.stage === 'inline') { state.stage = 'spine'; }
        renderStageTabs(); renderStage(); renderStageHead();
      });
    });

    $('#step-prev').addEventListener('click', function () { stopPlay(); goStep(state.step - 1); });
    $('#step-next').addEventListener('click', function () { stopPlay(); goStep(state.step + 1); });
    $('#step-play').addEventListener('click', togglePlay);
    $('#reset-btn').addEventListener('click', function () {
      stopPlay(); state.ackL3 = false; state.hiddenLayers = {}; state.openPass = null; goStep(0);
      toast('已回到第一步');
    });
    $('#wc-left').addEventListener('click', function () { panels.left = !panels.left; applyPanels(); });
    $('#wc-right').addEventListener('click', function () { panels.right = !panels.right; applyPanels(); });

    document.addEventListener('keydown', function (e) {
      if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
      // 交互元素已聚焦（例如决策脊行、决策卡、stage 里的可点节点）时不抢方向键/空格
      if (document.activeElement && document.activeElement.closest &&
          document.activeElement.closest('button,[role="tab"],[role="button"],[tabindex]')) {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft' || e.key === ' ') return;
      }
      if (e.key === 'ArrowRight') { stopPlay(); goStep(state.step + 1); e.preventDefault(); }
      if (e.key === 'ArrowLeft')  { stopPlay(); goStep(state.step - 1); e.preventDefault(); }
      if (e.key === ' ') { togglePlay(); e.preventDefault(); }
    });

    initSplit();
    applyPanels();
    goStep(0);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
