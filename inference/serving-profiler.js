/**
 * Observe profiler bridge
 *
 * 将 Toolkit Studio 的推理 profiling 数据与组件迁移到 Serving Observe。
 * 数据仍由 inference-profile-data.js 提供；这里负责目标页的浅色容器、
 * Service / Scheduler / Hardware Counter 摘要，以及算子分析页内交互。
 */
(function registerServingProfiler() {
  'use strict';

  var ctx = null;
  var GROUP_COLORS = {
    mlp: '#7f6bb3',
    attn: '#b06c3a',
    proj: '#356fae',
    norm: '#2f7a5a',
    boundary: '#8b929a',
    idle: '#a34d4d'
  };

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (c) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c];
    });
  }

  function fmt(value, digits) {
    return value == null ? '—' : Number(value).toFixed(digits == null ? 2 : digits);
  }

  function int(value) {
    return value == null ? '—' : Number(value).toLocaleString('en-US');
  }

  function profile() {
    return window.PtoInferenceProfile && window.PtoInferenceProfile.get
      ? window.PtoInferenceProfile.get()
      : null;
  }

  function currentState() {
    return ctx && ctx.state && ctx.state.profiler ? ctx.state.profiler : {};
  }

  function rerender() {
    if (ctx && ctx.render) ctx.render();
  }

  function metric(label, value, detail, tone) {
    return '<article class="so-profile-kpi"><span>' + esc(label) + '</span><strong>' + value + '</strong><small class="' + (tone || '') + '">' + esc(detail) + '</small></article>';
  }

  function renderLayerSummary(p) {
    var s = p.summary;
    var q = p.serving.queue;
    var kv = p.memory.kv;
    return '<div class="so-section-title"><h2>上层运行与硬件计数</h2><p>Service / Scheduler / Hardware runtime counter</p></div>'
      + '<div class="so-profile-layer-grid">'
      + '<section class="so-card so-profile-layer-card"><div class="so-card-head"><div><h3>Service</h3><div class="so-finding-summary">请求入口与 decode 服务状态</div></div><span class="so-status success">Completed</span></div><div class="so-card-body"><dl class="so-kv"><dt>吞吐</dt><dd>' + int(s.tps) + ' tok/s</dd><dt>TPOT p50 / p99</dt><dd>' + fmt(s.tpot.p50, 1) + ' / ' + fmt(s.tpot.p99, 1) + ' ms</dd><dt>TTFT p50</dt><dd>' + int(s.ttft) + ' ms</dd><dt>窗口</dt><dd>' + int(p.meta.steps) + ' steps · ' + fmt(p.meta.duration ? parseFloat(p.meta.duration) : 7.8, 1) + ' s</dd></dl></div></section>'
      + '<section class="so-card so-profile-layer-card"><div class="so-card-head"><div><h3>Scheduler</h3><div class="so-finding-summary">连续批处理与槽位复用</div></div><span class="so-pill accent">运行稳定</span></div><div class="so-card-body"><dl class="so-kv"><dt>运行中 / 等待中</dt><dd>' + q.running + ' / ' + q.waiting + '</dd><dt>等待 p50 / p99</dt><dd>' + q.waitP50 + ' / ' + q.waitP99 + ' ms</dd><dt>抢占 / 重计算</dt><dd>' + q.preempt + ' / ' + q.recompute + '</dd><dt>Chunked Prefill</dt><dd>' + q.chunkedPrefill + ' 个窗口</dd></dl></div></section>'
      + '<section class="so-card so-profile-layer-card"><div class="so-card-head"><div><h3>Hardware Runtime Counter</h3><div class="so-finding-summary">PMU 与带宽采集状态</div></div><span class="so-status success">已采集</span></div><div class="so-card-body"><dl class="so-kv"><dt>达成带宽</dt><dd>' + fmt(s.traffic.total / s.tpot.p50, 2) + ' TB/s · ' + fmt(s.traffic.total / s.tpot.p50 / p.meta.peakBw * 100, 1) + '% 峰值</dd><dt>MTE2 / Vector / Cube</dt><dd>' + fmt(s.sol[2].pct, 1) + '% / ' + fmt(s.sol[1].pct, 1) + '% / ' + fmt(s.sol[0].pct, 1) + '%</dd><dt>HBM 流量 / step</dt><dd>' + fmt(s.traffic.total, 2) + ' GB</dd><dt>KV Cache</dt><dd>' + fmt(kv.utilization, 1) + '% · 命中 ' + fmt(kv.hitRate, 1) + '%</dd></dl></div></section>'
      + '</div>';
  }

  function renderKpis(p) {
    var s = p.summary;
    return '<div class="so-profile-kpis">'
      + metric('TPOT · p50', fmt(s.tpot.p50, 1) + '<i> ms</i>', 'p90 ' + fmt(s.tpot.p90, 1) + ' · p99 ' + fmt(s.tpot.p99, 1) + ' ms')
      + metric('吞吐', int(s.tps) + '<i> tok/s</i>', 'batch ' + p.meta.batch + ' · 平均 ' + fmt(s.batchAvg, 1))
      + metric('达成带宽', fmt(s.traffic.total / s.tpot.p50, 2) + '<i> TB/s</i>', fmt(s.traffic.total / s.tpot.p50 / p.meta.peakBw * 100, 1) + '% 峰值', 'warn')
      + metric('KV Cache', fmt(s.kvUsed, 2) + '<i> GB</i>', int(p.memory.kv.pagesUsed) + ' / ' + int(p.memory.kv.pagesTotal) + ' 页 · 碎片 ' + fmt(p.memory.kv.fragmentation, 1) + '%')
      + metric('执行效率', fmt(s.efficiency, 1) + '<i> %</i>', '理论下界 ' + fmt(s.lowerBoundMs, 2) + ' ms')
      + '</div>';
  }

  function renderSol(p) {
    var rows = p.summary.sol.map(function (unit) {
      var bottleneck = unit.pct === Math.max.apply(null, p.summary.sol.map(function (item) { return item.pct; }));
      return '<div class="so-profile-solrow ' + (bottleneck ? 'is-bottleneck' : '') + '"><span>' + esc(unit.label) + '</span><div class="so-profile-soltrack"><span data-unit="' + esc(unit.id) + '" style="width:' + unit.pct + '%"></span></div><b>' + fmt(unit.pct, 1) + '%</b><small>' + esc(unit.detail) + '</small></div>';
    }).join('');
    return '<section class="so-card so-profile-card"><div class="so-card-head"><div><h3>Hardware unit utilization</h3><div class="so-finding-summary">每个 decode step 的时间去向</div></div><span class="so-pill warm">Memory-bound</span></div><div class="so-card-body"><div class="so-profile-sol">' + rows + '</div><div class="so-profile-verdict"><strong>内存搬运是当前长 pole</strong><p>每 step 从 HBM 读取 <code>' + fmt(p.summary.traffic.total, 2) + ' GB</code>，当前达成带宽为 <code>' + fmt(p.summary.traffic.total / p.summary.tpot.p50, 2) + ' TB/s</code>；Cube 仅占 <code>' + fmt(p.summary.sol[0].pct, 1) + '%</code>，优先检查 MTE2 重叠与权重复用。</p></div></div></section>';
  }

  function renderMix(p) {
    var total = p.summary.tpot.p50;
    return '<section class="so-card so-profile-card"><div class="so-card-head"><div><h3>耗时构成</h3><div class="so-finding-summary">按 Scope 进入算子分析</div></div><span class="so-pill">合计 ' + fmt(total, 1) + ' ms</span></div><div class="so-card-body"><div class="so-profile-mix">' + p.groups.map(function (g) {
      return '<button type="button" class="so-profile-mixrow" data-prof-mix="' + esc(g.id) + '"><i style="background:' + (GROUP_COLORS[g.id] || '#8b929a') + '"></i><b>' + esc(g.label) + '</b><strong>' + fmt(g.share, 1) + '%</strong><small>' + fmt(g.ms, 3) + ' ms · ' + esc(g.detail) + '</small></button>';
    }).join('') + '</div></div></section>';
  }

  function renderOverview(p) {
    return '<div class="so-profile-pane so-profile-overview">'
      + renderLayerSummary(p)
      + '<div class="so-section-title"><h2>全链路性能</h2><p>Service 指标与 Hardware counter 的同一份采集上下文</p></div>'
      + renderKpis(p)
      + renderSol(p)
      + '<div class="so-profile-grid2">' + renderMix(p) + renderBatchHistogram(p) + '</div>'
      + '</div>';
  }

  function renderBatchHistogram(p) {
    var values = (p.serving && p.serving.batchOverTime) || [];
    var max = Math.max.apply(null, values.concat([1]));
    var bars = values.slice(0, 32).map(function (value, index) {
      return '<i title="step ' + (index + 1) + ' · batch ' + value + '" style="height:' + Math.max(12, value / max * 100) + '%"></i>';
    }).join('');
    return '<section class="so-card so-profile-card"><div class="so-card-head"><div><h3>Batch 随时间</h3><div class="so-finding-summary">连续批处理窗口的实时 batch</div></div><span class="so-pill">平均 ' + fmt(p.serving && p.serving.batchAvg, 1) + '</span></div><div class="so-card-body"><div class="so-profile-hist">' + bars + '</div><div class="so-profile-hist-meta"><span>step 1</span><b>当前 batch ' + p.meta.batch + '</b><span>step ' + p.meta.steps + '</span></div></div></section>';
  }

  function visibleOps(p) {
    var s = currentState();
    var query = String(s.query || '').trim().toLowerCase();
    var list = p.ops.filter(function (op) {
      return (!s.groupFilter || op.group === s.groupFilter) && (!query || op.name.toLowerCase().indexOf(query) >= 0 || op.scope.toLowerCase().indexOf(query) >= 0);
    });
    var key = s.sortKey || 'totalMs';
    var dir = s.sortDir === 'asc' ? 1 : -1;
    return list.slice().sort(function (a, b) {
      var av = key === 'mte2' ? a.units.mte2 : key === 'efficiency' ? a.efficiency : a[key];
      var bv = key === 'mte2' ? b.units.mte2 : key === 'efficiency' ? b.efficiency : b[key];
      if (av == null) return 1;
      if (bv == null) return -1;
      return (typeof av === 'string' ? av.localeCompare(bv) : av - bv) * dir;
    });
  }

  function opRow(op, maxShare) {
    var s = currentState();
    var selected = op.id === s.selectedOp ? ' is-selected' : '';
    return '<tr class="so-prof-op-row' + selected + '" data-prof-op="' + esc(op.id) + '"><td><strong>' + esc(op.name) + '</strong><small>' + esc(op.scope) + '</small></td><td>' + int(op.calls) + '</td><td><b>' + fmt(op.totalMs, 3) + '</b> ms</td><td><span class="so-prof-share"><i style="width:' + (op.share / maxShare * 100) + '%;background:' + (GROUP_COLORS[op.group] || '#356fae') + '"></i>' + fmt(op.share, 1) + '%</span></td><td>' + (op.perLayerUs == null ? '—' : fmt(op.perLayerUs, 1) + ' μs') + '</td><td>' + fmt(op.units.mte2, 1) + '%</td><td>' + (op.achievedBw ? fmt(op.achievedBw, 2) + ' TB/s' : '—') + '</td><td><span class="so-prof-bound ' + esc(op.bound) + '">' + esc(op.boundLabel) + '</span></td><td>' + (op.efficiency == null ? '—' : op.efficiency + '%') + '</td></tr>';
  }

  function renderOpDetail(p) {
    var s = currentState();
    var op = p.ops.find(function (item) { return item.id === s.selectedOp; }) || p.ops[0];
    if (!op) return '';
    var units = ['cube', 'vector', 'mte2', 'mte3', 'sync'].filter(function (key) { return op.units[key] > 0; });
    var stack = units.map(function (key) { return '<i class="' + key + '" style="width:' + op.units[key] + '%"></i>'; }).join('');
    var spark = op.perLayer ? '<div class="so-prof-spark">' + op.perLayer.map(function (value, index) { return '<i style="height:' + (value / Math.max.apply(null, op.perLayer) * 100) + '%" title="L' + index + ' · ' + fmt(value, 2) + ' μs"></i>'; }).join('') + '</div>' : '<div class="so-empty">该任务没有逐层分布。</div>';
    return '<section class="so-card so-profile-card so-prof-detail"><div class="so-card-head"><div><h3>' + esc(op.name) + '</h3><div class="so-finding-summary">' + esc(op.scope) + ' · ' + esc(op.boundLabel) + ' bound</div></div><span class="so-pill accent">' + (op.efficiency == null ? '—' : op.efficiency + '% efficiency') + '</span></div><div class="so-card-body"><p class="so-prof-note">' + esc(op.note || '当前算子已关联硬件计数与逐层执行分布。') + '</p><div class="so-prof-detail-grid"><section><h4>Hardware counter</h4><div class="so-prof-stack">' + stack + '</div><div class="so-prof-legend"><span>MTE2 ' + fmt(op.units.mte2, 1) + '%</span><span>Vector ' + fmt(op.units.vector, 1) + '%</span><span>Cube ' + fmt(op.units.cube, 1) + '%</span></div></section><section><h4>Per-layer latency</h4>' + spark + '</section><section><h4>Runtime facts</h4><dl class="so-kv"><dt>总耗时</dt><dd>' + fmt(op.totalMs, 3) + ' ms</dd><dt>调用次数</dt><dd>' + int(op.calls) + '</dd><dt>达成带宽</dt><dd>' + (op.achievedBw ? fmt(op.achievedBw, 2) + ' TB/s' : '—') + '</dd><dt>Arithmetic intensity</dt><dd>' + fmt(op.ai, 1) + ' FLOP/B</dd></dl></section></div></div></section>';
  }

  function renderOps(p) {
    var s = currentState();
    var list = visibleOps(p);
    var maxShare = Math.max.apply(null, p.ops.map(function (op) { return op.share; }));
    var columns = [['name', '任务'], ['calls', '调用'], ['totalMs', '总耗时'], ['share', '占比'], ['perLayerUs', '每层'], ['mte2', 'MTE2'], ['achievedBw', '带宽'], ['bound', 'Bound'], ['efficiency', '效率']];
    var heads = columns.map(function (item) { var sorted = s.sortKey === item[0] ? ' is-sorted ' + s.sortDir : ''; return '<th class="' + sorted + '" data-prof-sort="' + item[0] + '">' + item[1] + '</th>'; }).join('');
    var group = s.groupFilter ? p.groups.find(function (item) { return item.id === s.groupFilter; }) : null;
    return '<div class="so-profile-pane so-profile-ops"><div class="so-prof-toolbar"><div class="so-prof-segment">' + [['flat', '按任务'], ['scope', '按 Scope'], ['bound', '按硬件单元']].map(function (item) { return '<button type="button" class="' + ((s.groupBy || 'flat') === item[0] ? 'is-active' : '') + '" data-prof-groupby="' + item[0] + '">' + item[1] + '</button>'; }).join('') + '</div><input class="so-input so-prof-search" type="search" data-prof-search placeholder="筛选任务或 Scope…" value="' + esc(s.query || '') + '">' + (group ? '<button type="button" class="so-pill accent so-prof-filter" data-prof-clear-filter>' + esc(group.label) + ' ×</button>' : '') + '<span class="so-prof-toolbar-spacer"></span><span class="so-finding-summary">' + list.length + ' / ' + p.ops.length + ' 项 · 合计 ' + fmt(list.reduce(function (sum, op) { return sum + op.totalMs; }, 0), 3) + ' ms</span></div><div class="so-table-wrap so-prof-table-wrap"><table class="so-table so-prof-table"><thead><tr>' + heads + '</tr></thead><tbody>' + list.map(function (op) { return opRow(op, maxShare); }).join('') + '</tbody></table></div>' + renderOpDetail(p) + '</div>';
  }

  function render(tab) {
    var p = profile();
    if (!p) return '<div class="so-empty">推理性能数据尚未加载。</div>';
    if (tab === 'ops') return renderOps(p);
    if (tab === 'memory') return '<div class="so-profile-pane so-profile-source">' + (window.PtoInferenceMemory ? window.PtoInferenceMemory.render(p) : '<div class="so-empty">访存数据模块尚未加载。</div>') + '</div>';
    if (tab === 'serving') return '<div class="so-profile-pane so-profile-source">' + (window.PtoInferenceServing ? window.PtoInferenceServing.render(p) : '<div class="so-empty">批处理数据模块尚未加载。</div>') + '</div>';
    return renderOverview(p);
  }

  function bind(nextCtx) {
    if (ctx) return;
    ctx = nextCtx;
    document.addEventListener('click', function (event) {
      var target = event.target;
      if (!target.closest('.so-profile-pane')) return;
      var sort = target.closest('[data-prof-sort]');
      if (sort) {
        var s = currentState();
        var key = sort.getAttribute('data-prof-sort');
        if (s.sortKey === key) s.sortDir = s.sortDir === 'desc' ? 'asc' : 'desc';
        else { s.sortKey = key; s.sortDir = key === 'name' || key === 'bound' ? 'asc' : 'desc'; }
        rerender();
        return;
      }
      var groupBy = target.closest('[data-prof-groupby]');
      if (groupBy) { currentState().groupBy = groupBy.getAttribute('data-prof-groupby'); rerender(); return; }
      var row = target.closest('[data-prof-op]');
      if (row) { currentState().selectedOp = row.getAttribute('data-prof-op'); rerender(); return; }
      var mix = target.closest('[data-prof-mix]');
      if (mix) { currentState().groupFilter = mix.getAttribute('data-prof-mix'); currentState().query = ''; ctx.state.observeTab = 'ops'; rerender(); return; }
      if (target.closest('[data-prof-clear-filter]')) { currentState().groupFilter = null; rerender(); }
    });
    document.addEventListener('input', function (event) {
      if (!event.target.matches('[data-prof-search]')) return;
      var caret = event.target.selectionStart;
      currentState().query = event.target.value;
      rerender();
      var next = document.querySelector('[data-prof-search]');
      if (next) { next.focus(); next.setSelectionRange(caret, caret); }
    });
  }

  window.PtoServingProfiler = { bind: bind, render: render };
})();
