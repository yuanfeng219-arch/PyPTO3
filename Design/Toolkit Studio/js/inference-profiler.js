/**
 * 推理性能分析 · 右侧大抽屉
 *
 * 六个页签：总览 / 时间线 / 算子分析 / 访存与缓存 / 批处理与调度 / 诊断建议
 * 本文件负责抽屉外壳、Run context bar、总览与算子分析；其余页签渲染在各自模块，
 * 本文件只持有 state 并转发交互。
 *
 * 与结构图的契约：op.id === baseGraph.nodes[].id（qwen3-model-viz.js）
 */
(function registerInferenceProfiler() {
  'use strict';

  const TABS = [
    { id: 'overview', label: '总览' },
    { id: 'timeline', label: '时间线', singleOnly: true },
    { id: 'ops', label: '算子分析' },
    { id: 'parallel', label: '并行与通信', distOnly: true },
    { id: 'memory', label: '访存与缓存' },
    { id: 'serving', label: '批处理与调度' },
    { id: 'advisor', label: '诊断建议' },
  ];

  /**
   * 「并行与通信」只对多机多卡采集出现；
   * 「时间线」按单卡 40 层串行建模，多机多卡的调度看「并行与通信」的流水线甘特图。
   */
  const visibleTabs = (p) => TABS.filter((t) => (!t.distOnly || !!p.dist) && (!t.singleOnly || !p.dist));

  const GROUP_COLOR = {
    mlp: 'var(--primary)',
    attn: 'var(--warning)',
    proj: 'var(--tone-blue-strong, #4a90d9)',
    norm: 'var(--tone-green-strong, #4caf7d)',
    boundary: 'color-mix(in srgb, var(--foreground) 42%, transparent)',
    comm: 'var(--tone-blue-strong, #4a90d9)',
    idle: 'color-mix(in srgb, var(--danger) 55%, transparent)',
  };

  const state = {
    open: false,
    runId: 'run-0803-a',
    runMenu: false,
    tab: 'overview',
    selectedOp: 'fa-fused',
    sortKey: 'totalMs',
    sortDir: 'desc',
    groupBy: 'flat',
    groupFilter: null,
    query: '',
    lastFocus: null,
    tl: { mode: 'layer', layer: 12, zoom: 1, gapThreshold: 1, criticalOnly: false, selected: null },
  };

  let root = null;

  const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const profile = () => window.PtoInferenceProfile?.get(state.runId);
  const fmt = (n, d = 2) => (n === null || n === undefined ? '—' : Number(n).toFixed(d));
  const int = (n) => (n === null || n === undefined ? '—' : Number(n).toLocaleString('en-US'));

  function deltaChip(value, invert) {
    if (value === null || value === undefined) return '<span class="kf-prof-delta flat">—</span>';
    if (Math.abs(value) < 0.05) return '<span class="kf-prof-delta flat">持平</span>';
    const worse = invert ? value < 0 : value > 0;
    return `<span class="kf-prof-delta ${worse ? 'up' : 'down'}">${value > 0 ? '▲' : '▼'} ${fmt(Math.abs(value), 1)}%</span>`;
  }

  const effClass = (v) => (v === null ? '' : v >= 85 ? 'good' : v >= 70 ? 'mid' : 'bad');

  /** 算子是否在结构图里有同名节点（__idle__ 之类的聚合项没有） */
  const hasGraphNode = (op) => !!window.PtoQwen3ModelViz?.graph?.nodes?.some((n) => n.id === op.id);

  /* ---------------- 总览 ---------------- */

  function renderKpis(p) {
    const s = p.summary;
    const cards = [
      ['TPOT · p50', `${fmt(s.tpot.p50, 1)}<i> ms</i>`, `p90 ${fmt(s.tpot.p90, 1)} · p99 ${fmt(s.tpot.p99, 1)}`, deltaChip(s.tpotDelta, false)],
      ['吞吐', `${int(s.tps)}<i> tok/s</i>`, `batch ${p.meta.batch} · 平均 ${fmt(s.batchAvg, 1)}`, deltaChip(s.tpsDelta, true)],
      ['TTFT · p50', `${int(s.ttft)}<i> ms</i>`, `prefill 平均 ${int(p.meta.seqAvg)} token`, deltaChip(s.ttftDelta, false)],
      ['达成带宽', `${fmt(s.traffic.total / s.tpot.p50, 2)}<i> TB/s</i>`, `${fmt(s.traffic.total, 2)} GB / step`, `<span class="kf-prof-delta flat">${fmt(s.traffic.total / s.tpot.p50 / p.meta.peakBw * 100, 1)}% 峰值</span>`],
      ['KV Cache', `${fmt(s.kvUsed, 2)}<i> / ${fmt(s.kvPool, 2)} GB</i>`, `${int(p.memory.kv.pagesUsed)} / ${int(p.memory.kv.pagesTotal)} 页 · 碎片 ${fmt(p.memory.kv.fragmentation, 2)}%`, `<span class="kf-prof-delta flat">${fmt(s.kvPct, 1)}%</span>`],
      ['执行效率', `${fmt(s.efficiency, 1)}<i> %</i>`, `理论下界 ${fmt(s.lowerBoundMs, 2)} ms`, `<span class="kf-prof-delta up">可回收 ${fmt(s.tpot.p50 - s.lowerBoundMs, 2)} ms</span>`],
    ];
    return `<div class="kf-prof-kpis">${cards.map(([label, value, sub, delta]) => `
      <article class="kf-prof-kpi"><span>${esc(label)}</span><b>${value}</b><small>${delta}<u style="text-decoration:none">${esc(sub)}</u></small></article>`).join('')}</div>`;
  }

  function renderSol(p) {
    const s = p.summary;
    const rows = s.sol.map((unit) => {
      const bottleneck = unit.pct === Math.max(...s.sol.map((u) => u.pct));
      return `<div class="kf-prof-solrow${bottleneck ? ' is-bottleneck' : ''}" data-unit="${unit.id}">
        <span>${esc(unit.label)}</span>
        <div class="kf-prof-soltrack"><div class="kf-prof-solfill" style="width:${unit.pct}%"></div></div>
        <div class="kf-prof-solval">${fmt(unit.pct, 1)}%</div>
        <div class="kf-prof-soldetail">${esc(unit.detail)}</div>
      </div>`;
    }).join('');
    return `<section class="kf-prof-card">
      <header><h3>Speed of Light · 单元占空比</h3><span>占 TPOT ${fmt(s.tpot.p50, 1)} ms 的时间比例</span></header>
      <div class="kf-prof-card__body">
        <div class="kf-prof-sol">${rows}</div>
        <div class="kf-prof-verdict">
          <i>◆</i>
          <b>Memory-bound · MTE2 权重搬运主导</b>
          <p>每 step 需从 HBM 读入 <code>${fmt(s.traffic.total, 2)} GB</code>（权重 ${fmt(s.traffic.weights, 2)} + KV ${fmt(s.traffic.kv, 2)} + 激活 ${fmt(s.traffic.act, 2)}）。
          按 ${fmt(p.meta.peakBw, 1)} TB/s 峰值算，理论下界 <code>${fmt(s.lowerBoundMs, 2)} ms</code>，当前 <code>${fmt(s.tpot.p50, 1)} ms</code>，效率 <code>${fmt(s.efficiency, 1)}%</code>。
          Cube 仅 ${fmt(s.sol[0].pct, 1)}%（${fmt(s.flops.achieved, 1)} / ${int(p.meta.peakFlops)} TFLOPS）——算力不是瓶颈，缺口在搬运效率与重叠度。</p>
        </div>
      </div>
    </section>`;
  }

  function renderRoofline(p) {
    const s = p.summary;
    const W = 560; const H = 340;
    const padL = 48; const padR = 16; const padT = 18; const padB = 36;
    const plotW = W - padL - padR; const plotH = H - padT - padB;
    const xMin = 0.2; const xMax = 1000; const yMin = 0.1; const yMax = 1000;
    const lx = (v) => padL + (Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin)) * plotW;
    const ly = (v) => padT + plotH - (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)) * plotH;
    const roofAt = (ai) => Math.min(p.meta.peakFlops, p.meta.peakBw * ai);

    const xTicks = [0.2, 1, 10, 100, 1000];
    const yTicks = [0.1, 1, 10, 100, 1000];
    const grid = [
      ...xTicks.map((t) => `<line class="grid" x1="${lx(t)}" y1="${padT}" x2="${lx(t)}" y2="${padT + plotH}"/><text class="tick" x="${lx(t)}" y="${padT + plotH + 14}" text-anchor="middle">${t}</text>`),
      ...yTicks.map((t) => `<line class="grid" x1="${padL}" y1="${ly(t)}" x2="${padL + plotW}" y2="${ly(t)}"/><text class="tick" x="${padL - 6}" y="${ly(t) + 3}" text-anchor="end">${t}</text>`),
    ].join('');

    const ridge = s.flops.ridge;
    const roof = `<polyline class="roof" points="${lx(xMin)},${ly(roofAt(xMin))} ${lx(ridge)},${ly(p.meta.peakFlops)} ${lx(xMax)},${ly(p.meta.peakFlops)}"/>
      <line class="ridge" x1="${lx(ridge)}" y1="${ly(p.meta.peakFlops)}" x2="${lx(ridge)}" y2="${padT + plotH}"/>
      <text class="rooflabel" x="${lx(1.6)}" y="${ly(roofAt(1.6)) - 7}" transform="rotate(-31 ${lx(1.6)} ${ly(roofAt(1.6)) - 7})">HBM ${fmt(p.meta.peakBw, 1)} TB/s</text>
      <text class="rooflabel" x="${lx(xMax) - 4}" y="${ly(p.meta.peakFlops) - 7}" text-anchor="end">Cube ${int(p.meta.peakFlops)} TFLOPS</text>
      <text class="tick" x="${lx(ridge)}" y="${padT + plotH - 6}" text-anchor="middle">脊点 ${fmt(ridge, 0)}</text>`;

    // AI≈16 的 GEMM 群（投影 + MLP + LM Head）挤在一起，合并成一个标注
    const gemm = p.ops.filter((o) => Math.abs(o.ai - 16) < 0.01);
    const singles = p.ops.filter((o) => o.ai !== null && o.ai > 0 && Math.abs(o.ai - 16) >= 0.01 && o.totalMs >= 0.2);

    const gemmDots = gemm.map((o) => `<circle cx="${lx(o.ai)}" cy="${ly(o.achievedTflops)}" r="4"/>`).join('');
    const gemmGroup = `<g class="pt mem" data-op="gate-up-proj"><title>投影 / MLP GEMM · 达成 70–76% 带宽屋顶</title>${gemmDots}
      <text x="${lx(16) + 10}" y="${ly(43.5) - 6}">GEMM ×${gemm.length} · 70–76% 屋顶</text></g>`;

    const singleDots = singles.map((o) => {
      const roofY = roofAt(o.ai);
      const ratio = o.achievedTflops / roofY;
      const cls = ratio > 0.65 ? 'mem' : ratio > 0.4 ? 'vec' : 'weak';
      const label = o.id === 'fa-fused'
        ? `<text x="${lx(o.ai) - 8}" y="${ly(o.achievedTflops) + 3}" text-anchor="end">fa_fused · ${fmt(ratio * 100, 0)}% 屋顶</text>
           <line class="ridge" x1="${lx(o.ai)}" y1="${ly(o.achievedTflops)}" x2="${lx(o.ai)}" y2="${ly(roofY)}"/>`
        : o.id === 'online-softmax' ? `<text x="${lx(o.ai) + 8}" y="${ly(o.achievedTflops) + 3}">online_softmax</text>`
          : o.id === 'silu' ? `<text x="${lx(o.ai) + 8}" y="${ly(o.achievedTflops) + 3}">silu</text>` : '';
      return `<g class="pt ${cls}" data-op="${o.id}"><title>${esc(o.name)} · AI ${fmt(o.ai, 1)} · ${fmt(o.achievedTflops, 2)} TFLOPS</title>
        <circle cx="${lx(o.ai)}" cy="${ly(o.achievedTflops)}" r="${o.id === 'fa-fused' ? 6 : 4}"/>${label}</g>`;
    }).join('');

    const work = `<g class="work"><title>整体工作点</title>
      <circle cx="${lx(s.flops.ai)}" cy="${ly(s.flops.achieved)}" r="6.5"/>
      <text x="${lx(s.flops.ai)}" y="${ly(s.flops.achieved) + 20}" text-anchor="middle">整体 AI ${fmt(s.flops.ai, 1)}</text></g>`;

    return `<section class="kf-prof-card">
      <header><h3>Roofline</h3><span>batch ${p.meta.batch} · 双对数 · 点击散点定位算子</span></header>
      <div class="kf-prof-card__body">
        <svg class="kf-prof-roofline" viewBox="0 0 ${W} ${H}" role="img" aria-label="Roofline 分析图">
          ${grid}
          <line class="axis" x1="${padL}" y1="${padT + plotH}" x2="${padL + plotW}" y2="${padT + plotH}"/>
          <line class="axis" x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}"/>
          ${roof}${gemmGroup}${singleDots}${work}
          <text class="axislabel" x="${padL + plotW / 2}" y="${H - 4}" text-anchor="middle">算术强度 (FLOP / Byte)</text>
          <text class="axislabel" x="12" y="${padT + plotH / 2}" text-anchor="middle" transform="rotate(-90 12 ${padT + plotH / 2})">达成算力 (TFLOPS)</text>
        </svg>
        <div class="kf-prof-rooflegend">
          <span><i style="background:var(--warning)"></i>贴合内存屋顶 (&gt;65%)</span>
          <span><i style="background:var(--tone-green-strong,#4caf7d)"></i>部分达成 (40–65%)</span>
          <span><i style="background:var(--danger)"></i>明显偏离 (&lt;40%)</span>
          <span><i style="background:var(--primary)"></i>整体工作点</span>
        </div>
      </div>
    </section>`;
  }

  function renderMix(p) {
    const R = 52; const C = 2 * Math.PI * R;
    let offset = 0;
    const arcs = p.groups.map((g) => {
      const len = C * (g.share / 100);
      const seg = `<circle r="${R}" cx="68" cy="68" stroke="${GROUP_COLOR[g.id]}" stroke-dasharray="${len.toFixed(2)} ${(C - len).toFixed(2)}" stroke-dashoffset="${(-offset).toFixed(2)}"/>`;
      offset += len;
      return seg;
    }).join('');
    const rows = p.groups.map((g) => `
      <button class="kf-prof-mixrow" type="button" data-mix="${g.id}">
        <i style="background:${GROUP_COLOR[g.id]}"></i>
        <b>${esc(g.label)}</b><u>${fmt(g.ms, 3)} ms</u><em>${fmt(g.share, 1)}%</em>
        <small>${esc(g.detail)}</small>
      </button>`).join('');
    return `<section class="kf-prof-card">
      <header><h3>耗时构成</h3><span>合计 ${fmt(p.summary.tpot.p50, 1)} ms · 点击筛选算子表</span></header>
      <div class="kf-prof-card__body"><div class="kf-prof-mix">
        <svg class="kf-prof-donut" viewBox="0 0 136 136" width="136" height="136" role="img" aria-label="耗时构成环形图">${arcs}</svg>
        <div class="kf-prof-mixlist">${rows}</div>
      </div></div>
    </section>`;
  }

  function renderHistogram(p) {
    const max = Math.max(...p.itlBins.map((b) => b[1]));
    const bars = p.itlBins.map(([ms, count]) => {
      const cls = ms === 15.2 ? ' is-p50' : ms >= 17.2 ? ' is-tail' : '';
      return `<div class="kf-prof-histbar${cls}" title="${ms} ms · ${count} steps">
        <span>${count}</span><i style="height:${Math.max(2, count / max * 100)}%"></i><span>${ms}</span></div>`;
    }).join('');
    const s = p.summary;
    return `<section class="kf-prof-card">
      <header><h3>Token 间隔分布</h3><span>${int(p.meta.steps)} steps · 单位 ms</span></header>
      <div class="kf-prof-card__body">
        <div class="kf-prof-hist">${bars}</div>
        <div class="kf-prof-histaxis">
          <span>p50 <b>${fmt(s.tpot.p50, 1)} ms</b></span>
          <span>p90 <b>${fmt(s.tpot.p90, 1)} ms</b></span>
          <span>p99 <b>${fmt(s.tpot.p99, 1)} ms</b></span>
          <span>抖动 p99/p50 <b>${fmt(s.tpot.p99 / s.tpot.p50, 2)}×</b></span>
          <span>抢占 <b>${s.preempt}</b></span>
        </div>
      </div>
    </section>`;
  }

  function renderOverview(p) {
    return renderKpis(p) + renderSol(p)
      + `<div class="kf-prof-grid2">${renderRoofline(p)}${renderMix(p)}</div>`
      + renderHistogram(p);
  }

  /* ---------------- 算子分析 ---------------- */

  const COLUMNS = [
    ['name', '任务', 'text'],
    ['calls', '调用', 'num'],
    ['totalMs', '总耗时', 'num'],
    ['share', '占比', 'num'],
    ['perLayerUs', '每层', 'num'],
    ['cube', 'Cube', 'num'],
    ['vector', 'Vector', 'num'],
    ['mte2', 'MTE2', 'num'],
    ['achievedBw', '达成带宽', 'num'],
    ['bound', 'Bound', 'text'],
    ['efficiency', '效率', 'num'],
  ];

  function sortValue(op, key) {
    if (['cube', 'vector', 'mte2'].includes(key)) return op.units[key];
    return op[key];
  }

  /** 分组用粗粒度 Scope（Scope 3 · MLP / Scope 3 · Norm 归为同一个 Scope 3），与结构图的阶段概念对齐 */
  const coarseScope = (op) => op.scope.split(' · ')[0];
  const SCOPE_ORDER = ['输入边界', 'Scope 1', 'Scope 2', 'Scope 3', '输出边界', '未归属'];
  const BOUND_ORDER = ['MTE2', 'Vector', 'Cube', 'Idle'];

  function visibleOps(p) {
    const q = state.query.trim().toLowerCase();
    let list = p.ops.filter((o) => (!state.groupFilter || o.group === state.groupFilter)
      && (!q || o.name.toLowerCase().includes(q) || o.scope.toLowerCase().includes(q)));
    const dir = state.sortDir === 'asc' ? 1 : -1;
    list = list.slice().sort((a, b) => {
      const av = sortValue(a, state.sortKey);
      const bv = sortValue(b, state.sortKey);
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      if (typeof av === 'string') return av.localeCompare(bv) * dir;
      return (av - bv) * dir;
    });
    return list;
  }

  function opRow(op, maxShare) {
    const selected = op.id === state.selectedOp ? ' is-selected' : '';
    const idle = op.group === 'idle' ? ' is-idle' : '';
    return `<tr class="${selected}${idle}" data-op-row="${op.id}">
      <td>${esc(op.name)}<small>${esc(op.scope)}</small></td>
      <td>${int(op.calls)}</td>
      <td><b>${fmt(op.totalMs, 3)}</b> ms</td>
      <td><span class="kf-prof-share"><span class="kf-prof-sharetrack"><span class="kf-prof-sharefill" style="width:${op.share / maxShare * 100}%;background:${GROUP_COLOR[op.group]}"></span></span>${fmt(op.share, 1)}%</span></td>
      <td>${op.perLayerUs === null ? '—' : `${fmt(op.perLayerUs, 1)} μs`}</td>
      <td>${op.units.cube ? `${fmt(op.units.cube, 1)}%` : '—'}</td>
      <td>${op.units.vector ? `${fmt(op.units.vector, 1)}%` : '—'}</td>
      <td>${op.units.mte2 ? `${fmt(op.units.mte2, 1)}%` : '—'}</td>
      <td>${op.achievedBw ? `${fmt(op.achievedBw, 2)} TB/s` : '—'}</td>
      <td><span class="kf-prof-tag ${op.bound}">${esc(op.boundLabel)}</span></td>
      <td class="kf-prof-eff ${effClass(op.efficiency)}">${op.efficiency === null ? '—' : `${op.efficiency}%`}</td>
    </tr>`;
  }

  function renderOpTable(p) {
    const list = visibleOps(p);
    const maxShare = Math.max(...p.ops.map((o) => o.share));
    let body = '';
    if (state.groupBy === 'flat') {
      body = list.map((op) => opRow(op, maxShare)).join('');
    } else {
      const byScope = state.groupBy === 'scope';
      const keyOf = byScope ? coarseScope : (op) => op.boundLabel;
      const order = byScope ? SCOPE_ORDER : BOUND_ORDER;
      const buckets = new Map();
      list.forEach((op) => {
        const k = keyOf(op);
        if (!buckets.has(k)) buckets.set(k, []);
        buckets.get(k).push(op);
      });
      const ordered = [...buckets.entries()].sort((a, b) => {
        const ia = order.indexOf(a[0]); const ib = order.indexOf(b[0]);
        return (ia < 0 ? order.length : ia) - (ib < 0 ? order.length : ib);
      });
      body = ordered.map(([label, ops]) => {
        const sum = ops.reduce((a, o) => a + o.totalMs, 0);
        const share = ops.reduce((a, o) => a + o.share, 0);
        return `<tr class="kf-prof-grouphead"><td colspan="11">${esc(label)} · <b>${fmt(sum, 3)} ms</b> · ${fmt(share, 1)}% · ${ops.length} 项</td></tr>`
          + ops.map((op) => opRow(op, maxShare)).join('');
      }).join('');
    }

    const head = COLUMNS.map(([key, label]) => {
      const sorted = state.sortKey === key ? ` is-sorted ${state.sortDir}` : '';
      return `<th class="${sorted.trim()}" data-sort="${key}">${esc(label)}</th>`;
    }).join('');

    const filtered = state.groupFilter ? p.groups.find((g) => g.id === state.groupFilter) : null;
    const chip = filtered
      ? `<button class="kf-prof-chip" type="button" data-clear-filter title="清除筛选"><i style="background:${GROUP_COLOR[filtered.id]}"></i>${esc(filtered.label)}<span>✕</span></button>`
      : '';

    return `<div class="kf-prof-optools">
        <div class="kf-prof-seg" role="group" aria-label="分组维度">
          ${[['flat', '按任务'], ['scope', '按 Scope'], ['bound', '按硬件单元']].map(([id, label]) => `<button type="button" class="${state.groupBy === id ? 'is-active' : ''}" data-groupby="${id}">${label}</button>`).join('')}
        </div>
        <input class="kf-prof-search" type="search" id="profOpSearch" placeholder="筛选任务或 Scope…" value="${esc(state.query)}" />
        ${chip}
        <span class="kf-prof-spacer" style="flex:1"></span>
        <span style="color:var(--foreground-muted);font-size:var(--kf-type-2xs)">${list.length} / ${p.ops.length} 项 · 合计 ${fmt(list.reduce((a, o) => a + o.totalMs, 0), 3)} ms</span>
        <button class="kf-prof-btn" type="button" data-prof-export>导出 CSV</button>
      </div>
      <div class="kf-prof-tablewrap"><table class="kf-prof-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function renderOpDetail(p) {
    const op = p.ops.find((o) => o.id === state.selectedOp);
    if (!op) return '';
    const u = op.units;
    const stack = ['cube', 'vector', 'mte2', 'mte3', 'sync']
      .filter((k) => u[k] > 0)
      .map((k) => `<i class="${k}" style="width:${u[k]}%" title="${k} ${fmt(u[k], 1)}%"></i>`).join('');
    const legend = ['cube', 'vector', 'mte2', 'mte3', 'sync']
      .filter((k) => u[k] > 0)
      .map((k) => `<span><i class="${k}" style="background:${{ cube: 'color-mix(in srgb,var(--tone-blue-strong,#4a90d9) 80%,transparent)', vector: 'color-mix(in srgb,var(--tone-green-strong,#4caf7d) 80%,transparent)', mte2: 'var(--warning)', mte3: 'color-mix(in srgb,var(--foreground) 32%,transparent)', sync: 'color-mix(in srgb,var(--danger) 55%,transparent)' }[k]}"></i>${k.toUpperCase()} ${fmt(u[k], 1)}%</span>`).join('');

    let spark = '<p style="margin:0;color:var(--foreground-muted);font-size:var(--kf-type-2xs)">该任务在整个 step 中只执行一次，无逐层分布。</p>';
    if (op.perLayer) {
      const max = Math.max(...op.perLayer);
      const min = Math.min(...op.perLayer);
      spark = `<div class="kf-prof-spark">${op.perLayer.map((v, i) => `<i class="${v === max ? 'is-peak' : ''}" style="height:${v / max * 100}%" title="L${i} · ${fmt(v, 2)} μs"></i>`).join('')}</div>
        <div class="kf-prof-sparkaxis"><span>L0</span><span>min ${fmt(min, 1)} · max ${fmt(max, 1)} μs · 极差 ${fmt((max - min) / min * 100, 1)}%</span><span>L39</span></div>`;
    }

    const roofRatio = op.ai ? op.achievedTflops / Math.min(p.meta.peakFlops, p.meta.peakBw * op.ai) * 100 : null;
    const cmp = op.static.length ? `<table class="kf-prof-cmp">
      <thead><tr><th>项</th><th>编译期静态</th><th>实测</th><th>偏差</th></tr></thead>
      <tbody>${op.static.map(([k, s, m, d, tone]) => `<tr><td>${esc(k)}</td><td>${esc(s)}</td><td>${esc(m)}</td><td class="delta ${tone}">${esc(d)}</td></tr>`).join('')}</tbody>
    </table>` : '<p style="margin:0;color:var(--foreground-muted);font-size:var(--kf-type-2xs)">该项未归属到具体任务，无静态对照。</p>';

    return `<section class="kf-prof-card kf-prof-detail">
      <div class="kf-prof-detail__head">
        <h3>${esc(op.name)}</h3>
        <span class="kf-prof-tag ${op.bound}">${esc(op.boundLabel)}-bound</span>
        <span style="color:var(--foreground-muted);font:var(--kf-type-2xs)/1 var(--font-mono)">${esc(op.scope)} · ${esc(op.source)}</span>
        <span class="kf-prof-spacer"></span>
        <span class="kf-prof-eff ${effClass(op.efficiency)}" style="font:700 var(--kf-type-sm)/1 var(--font-mono)">${op.efficiency === null ? '—' : `效率 ${op.efficiency}%`}</span>
      </div>
      <p class="kf-prof-note">${esc(op.note)}</p>
      <div class="kf-prof-detail__grid">
        <section>
          <h4>流水线拆解</h4>
          <div class="kf-prof-stack">${stack}</div>
          <div class="kf-prof-stacklegend">${legend}</div>
        </section>
        <section>
          <h4>逐层分布 · 40 层</h4>
          ${spark}
        </section>
        <section>
          <h4>搬运与并行</h4>
          <dl class="kf-prof-kv">
            <div><dt>MTE2 读入</dt><dd>${op.bytesIn >= 0.01 ? `${fmt(op.bytesIn, 3)} GB` : `${fmt(op.bytesIn * 1024, 2)} MB`}</dd></div>
            <div><dt>MTE3 写出</dt><dd>${op.bytesOut >= 0.01 ? `${fmt(op.bytesOut, 3)} GB` : `${fmt(op.bytesOut * 1024, 2)} MB`}</dd></div>
            <div><dt>片上复用率</dt><dd>${op.reuse === null ? '—' : `${fmt(op.reuse, 1)}×`}</dd></div>
            <div><dt>work items</dt><dd>${int(op.calls)}</dd></div>
            <div><dt>并行核数</dt><dd>${op.cores === null ? '—' : `${op.cores} · grid-stride`}</dd></div>
            <div><dt>负载不均衡 CV</dt><dd>${op.imbalance === null ? '—' : fmt(op.imbalance, 2)}</dd></div>
          </dl>
        </section>
        <section>
          <h4>Roofline 位置</h4>
          <dl class="kf-prof-kv">
            <div><dt>算术强度</dt><dd>${op.ai === null ? '—' : `${fmt(op.ai, 2)} FLOP/B`}</dd></div>
            <div><dt>达成算力</dt><dd>${fmt(op.achievedTflops, 2)} TFLOPS</dd></div>
            <div><dt>达成带宽</dt><dd>${fmt(op.achievedBw, 2)} TB/s</dd></div>
            <div><dt>占带宽峰值</dt><dd>${op.achievedBw ? `${fmt(op.achievedBw / p.meta.peakBw * 100, 1)}%` : '—'}</dd></div>
            <div><dt>距屋顶</dt><dd class="${roofRatio && roofRatio < 60 ? 'kf-prof-eff bad' : ''}">${roofRatio === null ? '—' : `${fmt(roofRatio, 1)}%`}</dd></div>
            <div><dt>总 FLOPs</dt><dd>${fmt(op.gflop, 2)} GFLOP</dd></div>
          </dl>
        </section>
        <section style="grid-column:1/-1">
          <h4>静态推断 vs 实测</h4>
          ${cmp}
        </section>
      </div>
      <div class="kf-prof-actions">
        ${hasGraphNode(op) ? `<button class="kf-prof-btn" type="button" data-goto-graph="${op.id}">↗ 在结构图中定位</button>`
        : '<button class="kf-prof-btn" type="button" disabled title="该项未归属到具体任务，结构图中没有对应节点">↗ 在结构图中定位</button>'}
        ${op.source === '—' ? '' : `<button class="kf-prof-btn" type="button" data-goto-source="${esc(op.source)}">↗ 打开 ${esc(op.source)}</button>`}
        <button class="kf-prof-btn" type="button" disabled title="时间线为 P3 计划内容">↗ 在时间线中查看</button>
        <button class="kf-prof-btn" type="button" disabled title="基线对比为 P5 计划内容">与基线 diff</button>
      </div>
    </section>`;
  }

  function renderOps(p) {
    return renderOpTable(p) + renderOpDetail(p);
  }

  /* ---------------- 外壳 ---------------- */

  function shell(p) {
    const m = p.meta;
    const tabs = visibleTabs(p).map((t) => `<button type="button" class="${state.tab === t.id ? 'is-active' : ''}" data-prof-tab="${t.id}" role="tab" aria-selected="${state.tab === t.id}">${esc(t.label)}</button>`).join('');
    return `<button class="kf-prof-scrim" type="button" data-prof-close aria-label="关闭推理性能分析"></button>
      <aside class="kf-prof-drawer" role="dialog" aria-modal="true" aria-label="推理性能分析">
        <header class="kf-prof-head">
          <div>
            <div class="kf-prof-crumb"><button type="button" data-prof-close>模型结构</button><span>/</span><button type="button" data-prof-close>${esc(m.model)}</button><span>/</span><b>推理性能分析</b></div>
            <h2>${esc(p.title)}</h2>
            <p>${esc(p.token)}</p>
          </div>
          <div class="kf-prof-headactions">
            <button class="kf-prof-btn" type="button" disabled title="基线对比为 P5 计划内容">对比基线</button>
            <button class="kf-prof-btn" type="button" data-prof-export>导出</button>
            <button class="kf-prof-close" type="button" data-prof-close aria-label="关闭">✕</button>
          </div>
        </header>
        <div class="kf-prof-context">
          <span class="kf-prof-runwrap">
            <button class="kf-prof-runpill" type="button" data-run-toggle aria-haspopup="listbox" aria-expanded="${state.runMenu}">${esc(p.id)} <span style="opacity:.6">▾</span></button>
            <div class="kf-prof-runmenu" role="listbox" ${state.runMenu ? '' : 'hidden'}>
              ${(window.PtoInferenceProfile.list() || []).map((r) => `<button type="button" role="option" aria-selected="${r.id === state.runId}" class="${r.id === state.runId ? 'is-active' : ''}" data-run-pick="${esc(r.id)}">
                <b>${esc(r.id)}</b><em>${fmt(r.tpot, 2)} ms · ${int(r.tps)} tok/s</em>
                <small>${esc(r.device)} · batch ${r.batch}${r.distributed ? ' · 多机多卡' : ' · 单卡'}</small>
              </button>`).join('')}
            </div>
          </span>
          <span class="kf-prof-ctxitem"><b>${esc(m.model)}</b> · BS ${m.batch} · ${esc(m.dtype)} · ${m.layers} 层</span>
          <span class="kf-prof-ctxsep"></span>
          <span class="kf-prof-ctxitem"><b>${esc(m.device)}</b> · ${m.hbm} GB / ${fmt(m.peakBw, 1)} TB/s</span>
          <span class="kf-prof-ctxsep"></span>
          <span class="kf-prof-ctxitem"><i></i>${esc(m.env)} 一致</span>
          <span class="kf-prof-ctxsep"></span>
          <span class="kf-prof-ctxitem">${int(m.steps)} steps · 平均 seq ${int(m.seqAvg)} · 采集 ${esc(m.capturedAt)}</span>
          <span class="kf-prof-spacer"></span>
          <span class="kf-prof-ctxitem">${esc(m.collector)}</span>
        </div>
        <nav class="kf-prof-tabs" role="tablist">${tabs}</nav>
        <div class="kf-prof-body" id="profBody"></div>
      </aside>`;
  }

  function renderBody() {
    const p = profile();
    const body = root?.querySelector('#profBody');
    if (!p || !body) return;
    const tab = visibleTabs(p).find((t) => t.id === state.tab) || TABS[0];
    if (tab.id === 'overview') body.innerHTML = renderOverview(p);
    else if (tab.id === 'parallel') body.innerHTML = window.PtoInferenceParallel.render(p);
    else if (tab.id === 'ops') body.innerHTML = renderOps(p);
    else if (tab.id === 'timeline') body.innerHTML = window.PtoInferenceTimeline.render(p, state.tl);
    else if (tab.id === 'memory') body.innerHTML = window.PtoInferenceMemory.render(p);
    else if (tab.id === 'serving') body.innerHTML = window.PtoInferenceServing.render(p);
    else if (tab.id === 'advisor') body.innerHTML = window.PtoInferenceAdvisor.render(p);
    body.scrollTop = 0;
  }

  function renderAll() {
    const p = profile();
    if (!p || !root) return;
    root.innerHTML = shell(p);
    renderBody();
  }

  function exportCsv() {
    const p = profile();
    const header = ['task', 'scope', 'calls', 'total_ms', 'share_pct', 'per_layer_us', 'cube_pct', 'vector_pct', 'mte2_pct', 'mte3_pct', 'sync_pct', 'achieved_bw_tbs', 'bound', 'efficiency_pct'];
    const lines = [header.join(',')].concat(p.ops.map((o) => [
      `"${o.name}"`, `"${o.scope}"`, o.calls ?? '', o.totalMs, o.share, o.perLayerUs ?? '',
      o.units.cube, o.units.vector, o.units.mte2, o.units.mte3, o.units.sync,
      o.achievedBw, o.boundLabel, o.efficiency ?? '',
    ].join(',')));
    const blob = new Blob([`﻿${lines.join('\n')}`], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${p.id}-ops.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function open() {
    const host = document.getElementById('modelArchitectureView');
    if (!host) return;
    if (!root) {
      root = document.createElement('div');
      root.className = 'kf-prof-root';
      root.id = 'inferenceProfiler';
      host.appendChild(root);
      root.addEventListener('click', onClick);
      root.addEventListener('input', onInput);
      root.addEventListener('change', onChange);
    }
    state.lastFocus = document.activeElement;
    state.open = true;
    root.hidden = false;
    renderAll();
    root.querySelector('.kf-prof-close')?.focus();
  }

  function close() {
    if (!root) return;
    state.open = false;
    root.hidden = true;
    if (state.lastFocus?.isConnected) state.lastFocus.focus();
  }

  function selectOp(id) {
    state.selectedOp = id;
    if (state.tab !== 'ops') {
      state.tab = 'ops';
      renderAll();
      return;
    }
    renderBody();
  }

  function onClick(event) {
    const t = event.target;
    if (t.closest('[data-prof-close]')) { close(); return; }

    if (t.closest('[data-run-toggle]')) { state.runMenu = !state.runMenu; renderAll(); return; }

    const pick = t.closest('[data-run-pick]');
    if (pick) {
      const next = pick.dataset.runPick;
      state.runMenu = false;
      if (next !== state.runId) {
        state.runId = next;
        const p = profile();
        // 切换采集后旧的选中项可能不存在：算子、层号、页签都要回到安全值
        if (!p.ops.some((o) => o.id === state.selectedOp)) state.selectedOp = p.ops[0].id;
        if (!visibleTabs(p).some((x) => x.id === state.tab)) state.tab = 'overview';
        state.groupFilter = null;
        state.query = '';
        state.tl = { ...state.tl, layer: Math.min(state.tl.layer, p.meta.layers - 1), selected: null, zoom: 1 };
      }
      renderAll();
      return;
    }
    if (state.runMenu && !t.closest('.kf-prof-runmenu')) { state.runMenu = false; renderAll(); return; }

    const tab = t.closest('[data-prof-tab]');
    if (tab) { state.tab = tab.dataset.profTab; renderAll(); return; }

    if (t.closest('[data-prof-export]')) { exportCsv(); return; }

    const sort = t.closest('[data-sort]');
    if (sort) {
      const key = sort.dataset.sort;
      if (state.sortKey === key) state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc';
      else { state.sortKey = key; state.sortDir = key === 'name' || key === 'bound' ? 'asc' : 'desc'; }
      renderBody();
      return;
    }

    const groupBy = t.closest('[data-groupby]');
    if (groupBy) { state.groupBy = groupBy.dataset.groupby; renderBody(); return; }

    const row = t.closest('[data-op-row]');
    if (row) { selectOp(row.dataset.opRow); return; }

    const point = t.closest('[data-op]');
    if (point) { selectOp(point.dataset.op); return; }

    if (t.closest('[data-clear-filter]')) { state.groupFilter = null; renderBody(); return; }

    /* ---- 时间线 ---- */
    const tlMode = t.closest('[data-tl-mode]');
    if (tlMode) { state.tl.mode = tlMode.dataset.tlMode; state.tl.zoom = 1; state.tl.selected = null; renderBody(); return; }

    const tlLayer = t.closest('[data-tl-layer]');
    if (tlLayer) { state.tl.layer = Number(tlLayer.dataset.tlLayer); state.tl.mode = 'layer'; state.tl.selected = null; renderBody(); return; }

    const tlZoom = t.closest('[data-tl-zoom]');
    if (tlZoom) {
      const dir = tlZoom.dataset.tlZoom;
      if (dir === 'fit') state.tl.zoom = 1;
      else state.tl.zoom = Math.max(1, Math.min(24, dir === 'in' ? state.tl.zoom * 2 : state.tl.zoom / 2));
      renderBody();
      return;
    }

    if (t.closest('[data-tl-critical]')) { state.tl.criticalOnly = !state.tl.criticalOnly; renderBody(); return; }

    const tlSpan = t.closest('[data-tl-span]');
    if (tlSpan) {
      state.tl.selected = state.tl.selected === tlSpan.dataset.tlSpan ? null : tlSpan.dataset.tlSpan;
      renderBody();
      return;
    }

    const tlGoto = t.closest('[data-tl-goto-op]');
    if (tlGoto) { selectOp(tlGoto.dataset.tlGotoOp); return; }

    const advGoto = t.closest('[data-advisor-op]');
    if (advGoto) { selectOp(advGoto.dataset.advisorOp); return; }

    const mix = t.closest('[data-mix]');
    if (mix) {
      state.query = '';
      state.groupFilter = mix.dataset.mix;
      state.tab = 'ops';
      const first = profile().ops
        .filter((o) => o.group === state.groupFilter)
        .sort((a, b) => b.totalMs - a.totalMs)[0];
      if (first) state.selectedOp = first.id;
      renderAll();
      return;
    }

    const graph = t.closest('[data-goto-graph]');
    if (graph) {
      const id = graph.dataset.gotoGraph;
      close();
      window.PtoQwen3ModelViz?.focusNode?.(id);
      return;
    }

    const source = t.closest('[data-goto-source]');
    if (source) {
      close();
      // 结构图视图里资源管理器是隐藏的，得先切回去再打开文件
      document.getElementById('activityExplorer')?.click();
      document.querySelector('[data-file="decode_layer.py"]')?.click();
    }
  }

  function onChange(event) {
    const t = event.target;
    if (t.id === 'tlLayerPick') { state.tl.layer = Number(t.value); state.tl.selected = null; renderBody(); return; }
    if (t.matches('[data-tl-threshold]')) { state.tl.gapThreshold = Number(t.value); renderBody(); }
  }

  function onInput(event) {
    if (event.target.id !== 'profOpSearch') return;
    state.query = event.target.value;
    const caret = event.target.selectionStart;
    renderBody();
    const next = root.querySelector('#profOpSearch');
    if (next) { next.focus(); next.setSelectionRange(caret, caret); }
  }

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && state.open) { event.stopPropagation(); close(); }
  });

  document.addEventListener('click', (event) => {
    if (event.target.closest('[data-open-profiler]')) {
      event.preventDefault();
      event.stopPropagation();
      open();
      return;
    }
    // 抽屉挂在结构图视图内，切走时一并收起，避免留下半开状态
    const rail = event.target.closest('[data-activity-view]');
    if (rail && rail.dataset.activityView !== 'model' && state.open) close();
  }, true);

  window.PtoInferenceProfiler = { open, close, selectOp, isOpen: () => state.open };
})();
