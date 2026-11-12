/* 算子开发页（workbench.html）视觉评审后的运行时调整，只在本 Demo 生效，../csa-operator-lens/ 的脚本不改。
 * 都是读页面已经写好的 DOM 再挪位置或换说法，不另算数据：
 *   1 状态栏去掉与画布提示重复的操作说明
 *   2 硅片落点：资源读数并入“视图”行；顶部加默认结论条（取状态栏告警 + 说明里的结论句）
 *   3 详情窗默认停靠在右侧、画布让位；拖动标题栏后变回自由浮窗 */
(function () {
  'use strict';
  const $ = function (id) { return document.getElementById(id); };
  const status = $('hpStatus');

  // ── 1 状态栏：操作提示画布左下角已经有，这里只留位置与对象 ──
  // 同时把 lens.js 写的“Pipeline”换成“Lane 轮次”：它和计算图里的 PL.PIPELINE（头块软件流水）不是一回事
  const DUP = /\s*·\s*双击模块展开，双击 Kernel 进入 L2\s*$/;
  function trimStatus() {
    const t = status.textContent, next = t.replace(DUP, '').replace(/·\s*Pipeline$/, '· Lane 轮次');
    if (next !== t) status.textContent = next;
  }
  new MutationObserver(trimStatus).observe(status, { childList: true, characterData: true, subtree: true });
  trimStatus();

  // ── 2 硅片落点 ──
  const head = document.querySelector('#viewKernel .lens-bottom-head');
  const chipHead = document.querySelector('#hpLensPane .hp-chip > .hp-pane-head');
  if (head && chipHead) {
    const extra = document.createElement('span');
    extra.className = 'wb-chip-extra';
    const stats = $('hpChipStats'), info = chipHead.querySelector('.pane-info');
    if (stats) extra.appendChild(stats);
    if (info) extra.appendChild(info);
    head.appendChild(extra);
  }
  const pane = $('hpLensPane'), caption = $('hpCaption');
  const verdict = document.createElement('div');
  verdict.className = 'wb-verdict'; verdict.hidden = true; verdict.setAttribute('role', 'note');
  if (pane) pane.insertBefore(verdict, pane.querySelector('.hp-chip'));
  const esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
  const ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4M12 17h.01"/></svg>';
  // 说明栏默认是“悬停…三处一起亮起”的操作提示 + 结论句；悬停时换成区域读数（提示标签变成“硅片 → 逻辑 …”）
  let sentence = '', shown = '';
  const atRest = function () { const k = caption.querySelector('.cap-k'); return !caption.textContent.trim() || !!(k && /^悬停/.test(k.textContent.trim())); };
  function paintCaption() { if (caption) caption.classList.toggle('wb-rest', atRest()); }
  function paintVerdict() {
    if (!pane || pane.hidden) return;
    paintCaption();
    const b = caption && atRest() ? caption.querySelector('b') : null;
    if (!sentence && b) sentence = b.textContent.trim();
    if (!status.classList.contains('is-warn')) return; // 悬停换成别的读数时保留上一条结论
    // 状态栏告警形如“! AIV1 空转（valid_shape=[0, 0]）· AIV0 Vec … · stage=2（2 槽）”：取第一段作结论；读数已在“视图”行，不重复
    const first = status.textContent.replace(/^!\s*/, '').split(/\s*·\s*/)[0];
    const m = /^(.*?)（(.*)）$/.exec(first);
    const key = m ? m[1] : first, why = sentence || (m ? m[2] : '');
    const html = '<span class="wb-verdict-k">' + ICON + esc(key) + '</span>' +
      '<span class="wb-verdict-t">' + esc(why).replace(/valid_shape=\[0, 0\]/g, '<code>valid_shape=[0, 0]</code>') + '</span>';
    if (html !== shown) { verdict.innerHTML = html; shown = html; }
    verdict.hidden = false;
  }
  if (caption) new MutationObserver(paintCaption).observe(caption, { childList: true, characterData: true, subtree: true });
  new MutationObserver(paintVerdict).observe(status, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['class'] });
  if (pane) new MutationObserver(paintVerdict).observe(pane, { attributes: true, attributeFilter: ['hidden'] });
  paintVerdict();

  // ── 4 Kernel 计算图：控制结构按类型画出不同的“形” ──
  // lens-tile.js 每次重画都整块替换 #hpLogic 的 SVG；这里在画完后补装饰，不改它的几何与交互（.lg-hit / .lg-frame 原样保留）。
  // 结构族统一用源码栏 PTO 框架标签的蓝灰（--fw），类型靠形状区分：
  //   SPMD 并行派发  = 实线框 + 背后两层错位“卡片叠”（24 份并行副本）+ 竖条组图标
  //   FOR 顺序循环   = 左侧虚线导轨（同源码缩进导轨）+ 右侧回到开头的回环箭头 + 循环图标
  //   PIPELINE 流水  = 实线框 + 表头色带 + 两条错位横条图标（前后两块重叠在飞）
  //   IF 分支        = 虚线框 + 菱形图标，与“① 拼 KV 块”并列，表示二选一
  // 计算阶段 ①–④ 是实心卡片、正文字标题，和等宽大写的结构关键字分开。
  const NS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs, parent, before) {
    const e = document.createElementNS(NS, tag);
    Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    parent.insertBefore(e, before || null);
    return e;
  }
  function shiftTexts(g, y, dx) {
    [].forEach.call(g.children, function (t) {
      if (t.tagName === 'text' && Math.abs(Number(t.getAttribute('y')) - y) < 1) t.setAttribute('x', Number(t.getAttribute('x')) + dx);
    });
  }
  function decorateLogic() {
    const svg = document.querySelector('#hpLogic > svg');
    if (!svg || svg.dataset.wb) return;
    svg.dataset.wb = '1';
    svg.classList.add('wb-logic');
    const defs = svg.querySelector('defs') || svgEl('defs', {}, svg, svg.firstChild);
    const mk = svgEl('marker', { id: 'wbLoopArrow', viewBox: '0 0 8 8', refX: '4', refY: '4', markerWidth: '7', markerHeight: '7', orient: 'auto' }, defs);
    svgEl('path', { d: 'M0 0 L8 4 L0 8 z', class: 'wb-loop-head' }, mk);
    const group = function (id) { return svg.querySelector('.lg-hit[data-node="' + id + '"]'); };
    // Q / KV 张量格写死了透镜原版的粉彩色（薄荷绿、淡紫），换成 Demo 色板的角色色：Q 来自 GM = 绿，KV 在 L1 喂 Cube = 紫
    const RECOLOR = { '#72d7af': 'var(--lime)', '#bca0f3': 'var(--lav)' };
    svg.querySelectorAll('[data-tensor] [fill]').forEach(function (e) {
      const to = RECOLOR[(e.getAttribute('fill') || '').toLowerCase()];
      if (to) e.style.fill = to;
    });

    const spmd = group('f-spmd');
    if (spmd) {
      const f = spmd.querySelector('.lg-frame');
      ['6', '3'].forEach(function (d) {
        svgEl('rect', { x: 12 + Number(d), y: 8 + Number(d), width: 696, height: 322, class: 'wb-deck' }, spmd, f);
      });
      shiftTexts(spmd, 25, 16);
      const ic = svgEl('g', { class: 'wb-ic', transform: 'translate(24 16)' }, spmd);
      [0, 3, 6, 9].forEach(function (x) { svgEl('rect', { x: x, y: 0, width: 1.8, height: 10, rx: .6 }, ic); });
    }
    const iter = group('f-iter');
    if (iter) {
      svgEl('path', { d: 'M25.5 41 V 317', class: 'wb-rail' }, iter);
      svgEl('path', { d: 'M691 314 V 48 Q 691 42 685 42 H 676', class: 'wb-loop', 'marker-end': 'url(#wbLoopArrow)' }, iter);
      svgEl('text', { x: 684, y: 55, 'text-anchor': 'end', class: 'sv-label wb-loop-k' }, iter).textContent = '逐个取下一项';
      shiftTexts(iter, 53, 16);
      const ic = svgEl('g', { class: 'wb-ic', transform: 'translate(36 44)' }, iter);
      svgEl('path', { d: 'M9 5 A4 4 0 1 1 7.6 1.9', class: 'wb-ic-line' }, ic);
      svgEl('path', { d: 'M5.6 0.4 L8.4 1.6 L7.2 4.4', class: 'wb-ic-line' }, ic);
    }
    const pipe = group('f-pipe');
    if (pipe) {
      const f = pipe.querySelector('.lg-frame');
      svgEl('path', { d: 'M36 154 V 132 Q 36 128 40 128 H 680 Q 684 128 684 132 V 154 Z', class: 'wb-band' }, pipe, f.nextSibling);
      shiftTexts(pipe, 145, 16);
      const ic = svgEl('g', { class: 'wb-ic', transform: 'translate(48 136)' }, pipe);
      svgEl('rect', { x: 0, y: 1, width: 7, height: 2.6, rx: 1 }, ic);
      svgEl('rect', { x: 4, y: 6.4, width: 7, height: 2.6, rx: 1 }, ic);
    }
    // “块无效时”原本是一行孤立的说明字：画成 IF 分支框，和 ① 拼 KV 块同高并列
    const lone = [].filter.call(svg.children, function (t) { return t.tagName === 'text' && t.textContent === '块无效时'; })[0];
    if (lone) {
      const br = svgEl('g', { class: 'wb-branch', 'pointer-events': 'none' }, svg, lone);
      svgEl('rect', { x: 534, y: 62, width: 152, height: 56, rx: 7, class: 'wb-branch-box' }, br);
      svgEl('path', { d: 'M544 68 L548.5 72.5 L544 77 L539.5 72.5 Z', class: 'wb-ic-line' }, br);
      svgEl('text', { x: 553, y: 76, class: 'sv-title wb-kw' }, br).textContent = 'IF';
      svgEl('text', { x: 570, y: 76, class: 'sv-label' }, br).textContent = '块无效 · 只写初值';
      lone.remove();
    }
    addFlowContext(svg);
  }

  // ── 4b Kernel 计算图带上模块内的上下游 ──
  // 取 L1 整层图里同一个模块的子图（csa-graph-model.js 的 LENS_CSA_GRAPH，与 L1 展开后看到的是同一份）：
  // 本 Kernel 之前的节点竖排在左、之后的竖排在右，中间就是原来的 qk_pv 循环体（SPMD 框）。
  // 只往 viewBox 两侧加列，不动原图几何；左右列的 Kernel 单击走 lens.js 的 data-lens-go 进入该 Kernel 的 L2。
  const FLOW_KERNEL = 'qk_pv';
  const FLOW_ICON = { synchronization: 'cache', planning: 'topk', position: 'rope', matmul: 'matmul', attention: 'attention', reduction: 'reduce', communication: 'exchange' };
  const FLOW = { colW: 150, gapL: 84, gapR: 108, nodeH: 28, dataH: 30, step: 42, frameL: 12, frameR: 708, mid: 169, top: 8, h: 338 };
  function flowContext() {
    const G = window.LENS_CSA_GRAPH;
    if (!G) return null;
    const owner = G.nodes.filter(function (n) { return n.composite && n.composite.subgraph.nodes.some(function (x) { return x.id === FLOW_KERNEL; }); })[0];
    if (!owner) return null;
    const sub = owner.composite.subgraph, at = sub.nodes.map(function (n) { return n.id; }).indexOf(FLOW_KERNEL);
    return { owner: owner, edges: sub.edges, up: sub.nodes.slice(0, at), down: sub.nodes.slice(at + 1) };
  }
  function flowNode(n, x, y) {
    const K = window.LensData && window.LensData.KERNELS[n.id];
    const tip = esc(n.name + '：' + (n.description || '')) + (K ? esc('（单击进入这个 Kernel）') : '');
    let body;
    if (n.type === 'data') {
      const d = n.data || {}, shape = d.shape && d.shape.allocated ? '[' + d.shape.allocated.join(', ') + ']' : '';
      body = window.PathGridGlyphs.tensor(n.id, n.name, [shape, d.dtype].filter(Boolean).join(' · '), x, y, FLOW.colW, FLOW.dataH, '#80b7f7');
    } else {
      const icon = (window.PtoOpIcons || []).filter(function (i) { return i.id === FLOW_ICON[(n.operation || {}).category]; })[0];
      body = window.VisualSystem.node('op', n.id, n.name, x, y, { width: FLOW.colW, height: FLOW.nodeH, icon: icon && icon.path, fontSize: 10.5 });
    }
    const go = K ? ' data-lens-go="2" data-kernel="' + esc(n.id) + '" tabindex="0" role="link" aria-label="' + esc('进入 ' + n.name) + '"' : '';
    return '<g class="wb-flow-node" data-flow="' + esc(n.id) + '"' + go + '><title>' + tip + '</title>' + body + '</g>';
  }
  // 一列：节点竖排、整列垂直居中；相邻节点之间按子图里的边画竖线（虚线 = TaskId / WAR 依赖）
  function flowColumn(list, x, edges) {
    const hOf = function (n) { return n.type === 'data' ? FLOW.dataH : FLOW.nodeH; };
    const total = list.reduce(function (s, n) { return s + hOf(n); }, 0) + FLOW.step * (list.length - 1);
    let y = FLOW.mid - total / 2, out = '';
    const box = {};
    list.forEach(function (n) { box[n.id] = { y: y, h: hOf(n) }; out += flowNode(n, x, y); y += hOf(n) + FLOW.step; });
    const cx = x + FLOW.colW / 2;
    for (let i = 1; i < list.length; i++) {
      const a = list[i - 1], b = list[i], e = edgeOf(edges, a.id, b.id);
      const y1 = box[a.id].y + box[a.id].h, y2 = box[b.id].y;
      out = flowEdge('M' + cx + ' ' + (y1 + 2) + ' V ' + (y2 - 2), e) + flowLabel(e, a, b, cx + 7, (y1 + y2) / 2 + 3, 'start') + out;
    }
    return { svg: out, box: box };
  }
  function edgeOf(edges, s, t) { return edges.filter(function (e) { return e.s === s && e.t === t; })[0] || null; }
  function flowEdge(d, e) {
    return '<path d="' + d + '" class="wb-flow-edge' + (e && e.type === 'control' ? ' is-control' : '') + '" marker-end="url(#wbFlowArrow)"/>';
  }
  // 边名与两端节点同名（如 qk_plan → qk_order 的“qk_order”）就不重复写
  function flowLabel(e, a, b, x, y, anchor) {
    if (!e || !e.label || e.label === a.name || e.label === b.name) return '';
    return '<text x="' + x + '" y="' + y + '" class="wb-flow-k"' + (anchor ? ' text-anchor="' + anchor + '"' : '') + '>' + esc(e.label.replace(/^返回\s*/, '')) + '</text>';
  }
  function addFlowContext(svg) {
    const c = flowContext();
    if (!c || (!c.up.length && !c.down.length)) return;
    const self = { id: FLOW_KERNEL, name: FLOW_KERNEL };
    const leftX = FLOW.frameL - FLOW.gapL - FLOW.colW, rightX = FLOW.frameR + FLOW.gapR;
    let s = '<marker id="wbFlowArrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" class="wb-flow-head"/></marker>';
    const defs = svg.querySelector('defs');
    if (defs) { defs.insertAdjacentHTML('beforeend', s); s = ''; }
    s += '<g class="wb-flow" aria-label="' + esc(c.owner.name + ' 内的上下游') + '">';
    let minX = 0, maxX = 720;
    if (c.up.length) {
      const col = flowColumn(c.up, leftX, c.edges), last = c.up[c.up.length - 1], b = col.box[last.id], y = b.y + b.h / 2, e = edgeOf(c.edges, last.id, FLOW_KERNEL);
      s += '<text x="' + leftX + '" y="25" class="wb-flow-cap">上游 · ' + esc(c.owner.name) + '</text>' + col.svg
        + flowEdge('M' + (leftX + FLOW.colW + 2) + ' ' + y + ' H ' + (FLOW.frameL - 2), e)
        + flowLabel(e, last, self, (leftX + FLOW.colW + FLOW.frameL) / 2, y - 6, 'middle');
      minX = leftX - 10;
    }
    if (c.down.length) {
      const col = flowColumn(c.down, rightX, c.edges), first = c.down[0], b = col.box[first.id], y = b.y + b.h / 2, e = edgeOf(c.edges, FLOW_KERNEL, first.id);
      s += '<text x="' + rightX + '" y="25" class="wb-flow-cap">下游 · ' + esc(c.owner.name) + '</text>' + col.svg
        + flowEdge('M' + (FLOW.frameR + 2) + ' ' + y + ' H ' + (rightX - 2), e)
        + flowLabel(e, self, first, (FLOW.frameR + rightX) / 2, y - 6, 'middle');
      maxX = rightX + FLOW.colW + 10;
    }
    svg.insertAdjacentHTML('beforeend', s + '</g>');
    svg.setAttribute('viewBox', minX + ' 0 ' + (maxX - minX) + ' ' + FLOW.h);
    svg.setAttribute('aria-label', (svg.getAttribute('aria-label') || '') + '，含 ' + c.owner.name + ' 内的上下游');
  }
  // 上下游 Kernel 可用键盘进入：Enter / 空格等同单击
  document.addEventListener('keydown', function (e) {
    const t = e.target.closest && e.target.closest('#hpLogic .wb-flow-node[data-lens-go]');
    if (t && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); t.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, detail: 1 })); }
  });
  const logic = $('hpLogic');
  if (logic) { new MutationObserver(decorateLogic).observe(logic, { childList: true }); decorateLogic(); }

  // ── 5 L2 下方按“问题”分组：工作怎么分（4 步）/ 放在哪 ──
  // lens.js 仍按 state.tab = assignment | pipeline | chip 工作；这里只换入口：
  //   “工作怎么分”页签在 assignment 与 pipeline 下都算选中；步骤条第 4 步「Lane 轮次」去点隐藏的 pipeline 页签，
  //   在 Lane 轮次里点 ①–③，先回 assignment 再点 lens.js 自己的阶段按钮。
  const tabs = $('kernelTabs'), bar = $('kernelBar');
  const tabBtn = function (t) { return tabs && tabs.querySelector('[data-tab="' + t + '"]'); };
  const onTab = function (t) { const b = tabBtn(t); return !!b && b.getAttribute('aria-selected') === 'true'; };
  let phaseCache = [['generation', 'Generation', '源码先产生哪些工作'], ['ordering', 'Ordering', 'qk_plan 怎样排队'], ['assignment', 'Assignment', 'SPMD Lane 怎样领取']];
  const ROUNDS_TIP = '每条 Lane 一轮轮领到哪些 Work Item；列是领取顺序，不是时间';
  // 步骤名换成大白话，原术语缩小跟在后面（给懂行的人对照源码）
  const PLAIN = { generation: '列出所有活', ordering: '排队', assignment: '分给各个核', rounds: '按轮干活' };
  const TERM = { generation: 'Generation', ordering: 'Ordering', assignment: 'Assignment', rounds: 'Lane 轮次' };
  function label(b, k) { if (PLAIN[k] && !b.querySelector('small')) b.innerHTML = esc(PLAIN[k]) + '<small>' + esc(TERM[k]) + '</small>'; }
  function roundsBtn(pressed) {
    const b = document.createElement('button');
    b.type = 'button'; b.dataset.wbStep = 'rounds'; b.title = ROUNDS_TIP;
    b.setAttribute('aria-pressed', String(pressed));
    label(b, 'rounds');
    return b;
  }
  function paintSteps() {
    if (!bar) return;
    const seg = bar.querySelector('.lens-seg[aria-label="Assignment 阶段"]');
    if (seg) {
      // 记下这个 Kernel 实际有哪几个阶段，Lane 轮次里画代理步骤时照抄
      phaseCache = [].map.call(seg.querySelectorAll('[data-l2="phase"]'), function (b) { return [b.dataset.v, TERM[b.dataset.v] || b.textContent, b.title]; });
      if (!seg.querySelector('[data-wb-step]')) {
        seg.classList.add('wb-steps'); seg.setAttribute('aria-label', '工作怎么分 · 步骤');
        seg.querySelectorAll('[data-l2="phase"]').forEach(function (b) { label(b, b.dataset.v); });
        seg.appendChild(roundsBtn(false));
      }
      return;
    }
    if (onTab('pipeline') && bar.firstChild && !bar.querySelector('.wb-steps')) {
      const proxy = document.createElement('div');
      proxy.className = 'lens-seg wb-steps'; proxy.setAttribute('role', 'group'); proxy.setAttribute('aria-label', '工作怎么分 · 步骤');
      phaseCache.forEach(function (p) {
        const b = document.createElement('button');
        b.type = 'button'; b.dataset.wbPhase = p[0]; b.title = p[2]; b.textContent = p[1]; b.setAttribute('aria-pressed', 'false');
        label(b, p[0]);
        proxy.appendChild(b);
      });
      proxy.appendChild(roundsBtn(true));
      const head = bar.querySelector('.lens-bar-k');
      if (head) head.hidden = true; // “256 Queries × 5 …”与底部图例重复
      bar.insertBefore(proxy, bar.firstChild);
    }
  }
  if (bar) new MutationObserver(paintSteps).observe(bar, { childList: true });
  if (tabs) new MutationObserver(paintSteps).observe(tabs, { attributes: true, subtree: true, attributeFilter: ['aria-selected'] });
  paintSteps();
  let passThrough = false;
  window.addEventListener('click', function (e) {
    if (passThrough) return;
    if (e.target.closest('[data-wb-step="rounds"]')) { e.stopPropagation(); const p = tabBtn('pipeline'); if (p && !p.disabled && !onTab('pipeline')) p.click(); return; }
    const ph = e.target.closest('[data-wb-phase]');
    if (ph) {
      e.stopPropagation();
      passThrough = true;
      try {
        tabBtn('assignment').click(); // lens.js 同步重画步骤条
        const real = bar.querySelector('[data-l2="phase"][data-v="' + ph.dataset.wbPhase + '"]');
        if (real) real.click();
      } finally { passThrough = false; }
      return;
    }
    // 已在 Lane 轮次时再点“工作怎么分”：留在原地，不退回 Assignment
    if (e.target.closest('#kernelTabs [data-tab="assignment"]') && onTab('pipeline')) e.stopPropagation();
  }, true);
  // pl.at Kernel 的说明卡里也有“Pipeline”字样
  const lanes = $('kernelLanes');
  if (lanes) new MutationObserver(function () {
    lanes.querySelectorAll('.lens-card p').forEach(function (p) { if (/Assignment \/ Pipeline/.test(p.innerHTML)) p.innerHTML = p.innerHTML.replace('Assignment / Pipeline', 'Lane 分工与轮次'); });
  }).observe(lanes, { childList: true });

  // ── 6 计算图标出“下方展开的是哪几层” ──
  // 工作怎么分 → 外两层 SPMD × 24 · FOR qk_it；放在哪 → 里层阶段 ①–④。高亮写在 CSS（#viewKernel 是否 .is-chip），这里补一句说明
  const kernelView = $('viewKernel'), topBar = document.querySelector('#viewKernel .lens-l2-top');
  const mapNote = document.createElement('span');
  mapNote.className = 'wb-map-note';
  if (topBar) topBar.appendChild(mapNote);
  function paintMap() {
    const chip = kernelView.classList.contains('is-chip'), shown = logic && !logic.hidden;
    mapNote.hidden = !shown;
    mapNote.innerHTML = chip ? '<i aria-hidden="true">↓</i>下方展开<b>里层</b>：阶段 ①–④ 的数据落在哪'
      : '<i aria-hidden="true">↓</i>下方展开<b>外两层</b>：SPMD × 24 · FOR qk_it';
  }
  if (kernelView) new MutationObserver(paintMap).observe(kernelView, { attributes: true, attributeFilter: ['class'] });
  if (logic) new MutationObserver(paintMap).observe(logic, { attributes: true, attributeFilter: ['hidden'] });
  if (kernelView) paintMap();

  // ── 7 模块 Kernel 链：从画布顶部的二选一切换，挪到顶栏路径旁（L1 模块与 L2 Kernel 之间） ──
  // 弹出一条迷你链：按源码顺序列出本模块派发的 Kernel，实线箭头是源码写出的 TaskId 依赖；
  // 点 Kernel 走 lens.js 已有的 data-lens-go 跳转；“在画布上看完整依赖链”去点隐藏的 #l2TopSwitch。
  const LD = window.LensData, crumb = $('lensCrumb'), topLeft = document.querySelector('.hp-top-left');
  const chainBtn = document.createElement('button');
  chainBtn.type = 'button'; chainBtn.className = 'wb-chain-btn'; chainBtn.hidden = true;
  chainBtn.setAttribute('aria-haspopup', 'dialog'); chainBtn.setAttribute('aria-expanded', 'false');
  const pop = document.createElement('div');
  pop.className = 'wb-chain-pop'; pop.hidden = true; pop.setAttribute('role', 'dialog'); pop.setAttribute('aria-label', '本模块的 Kernel 链');
  if (topLeft && crumb) { crumb.insertAdjacentElement('afterend', chainBtn); document.body.appendChild(pop); }
  function current() {
    if (!LD || document.body.dataset.level !== '2') return null;
    const last = [].slice.call(crumb.querySelectorAll('button')).pop();
    if (!last) return null;
    const parts = last.textContent.split(' / '), mod = LD.MODULES.filter(function (m) { return m.label === parts[0]; })[0];
    if (!mod || !LD.CHAINS[mod.id]) return null;
    const id = LD.CHAINS[mod.id].order.filter(function (k) { return LD.KERNELS[k] && LD.KERNELS[k].name === parts[1]; })[0];
    return { mod: mod, chain: LD.CHAINS[mod.id], id: id };
  }
  const dispatchOf = function (K) { return K.dispatch === 'spmd' ? 'SPMD × ' + (K.lanes || '?') : 'pl.at'; };
  function paintChain() {
    const c = current();
    chainBtn.hidden = !c;
    if (!c) { closeChain(); return; }
    const n = c.chain.order.length;
    chainBtn.innerHTML = 'Kernel 链 <b>' + n + '</b><i aria-hidden="true">▾</i>';
    chainBtn.title = c.mod.label + ' 派发的 ' + n + ' 个 Kernel';
    const edge = function (a, b) { return (c.chain.edges || []).filter(function (e) { return e[0] === a && e[1] === b; })[0]; };
    let row = '';
    c.chain.order.forEach(function (k, i) {
      const K = LD.KERNELS[k];
      if (i) {
        const e = edge(c.chain.order[i - 1], k);
        row += e ? '<span class="wb-chain-dep" title="源码写出的 TaskId 依赖">' + esc(e[2]) + '<i aria-hidden="true">→</i></span>' : '<span class="wb-chain-gap" aria-hidden="true"></span>';
      }
      row += '<button type="button" class="wb-chain-k" data-lens-go="2" data-kernel="' + k + '"' + (k === c.id ? ' aria-current="true"' : '') + ' title="' + esc(K.what || '') + '">'
        + '<b>' + esc(K.name) + '</b><small>' + esc(dispatchOf(K)) + (K.deep ? ' · 有计算图' : '') + '</small></button>';
    });
    const full = !$('chainFrame').hidden, K = LD.KERNELS[c.id];
    pop.innerHTML = '<div class="wb-chain-head"><b>' + esc(c.mod.label) + '</b> 派发的 Kernel · 按源码顺序<span>一次 pl.spmd / pl.at 就是一个</span></div>'
      + '<div class="wb-chain-row">' + row + '</div>'
      + '<div class="wb-chain-foot">' + (K && K.deep
        ? '<button type="button" class="wb-chain-toggle" data-wb-chain="' + (full ? 'logic' : 'chain') + '">' + (full ? '回到 ' + esc(K.name) + ' 计算图' : '在画布上看完整依赖链') + '</button>'
        : '<span>当前 Kernel 没有 Tile 计算图，画布上方显示的就是完整依赖链</span>')
      + '<span class="wb-chain-key"><i class="is-dep"></i>TaskId 依赖</span></div>';
  }
  function openChain() {
    paintChain();
    const r = chainBtn.getBoundingClientRect();
    pop.style.top = (r.bottom + 8) + 'px';
    pop.hidden = false; chainBtn.setAttribute('aria-expanded', 'true');
    // 宽度随链长；放得下就左对齐按钮，否则往左让，不出右边界
    pop.style.left = Math.max(12, Math.min(r.left, window.innerWidth - pop.offsetWidth - 12)) + 'px';
  }
  function closeChain() { pop.hidden = true; chainBtn.setAttribute('aria-expanded', 'false'); }
  chainBtn.addEventListener('click', function () { if (pop.hidden) openChain(); else closeChain(); });
  pop.addEventListener('click', function (e) {
    const t = e.target.closest('[data-wb-chain]');
    if (t) { const b = document.querySelector('#l2TopSwitch [data-l2top="' + t.dataset.wbChain + '"]'); if (b) b.click(); closeChain(); return; }
    if (e.target.closest('[data-lens-go]')) {
      // 跳转交给 lens.js 的全局点击；之后若选中的 Kernel 有计算图，画布回到计算图（不停在上次展开的完整链）
      setTimeout(function () {
        closeChain();
        const b = document.querySelector('#l2TopSwitch [data-l2top="logic"]');
        if (b && !b.disabled && !$('chainFrame').hidden) b.click();
      }, 0);
    }
  });
  document.addEventListener('pointerdown', function (e) { if (!pop.hidden && !pop.contains(e.target) && !chainBtn.contains(e.target)) closeChain(); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !pop.hidden) { closeChain(); chainBtn.focus(); } });
  if (crumb) new MutationObserver(function () { paintChain(); }).observe(crumb, { childList: true, subtree: true, characterData: true });
  new MutationObserver(function () { paintChain(); }).observe(document.body, { attributes: true, attributeFilter: ['data-level'] });
  if ($('chainFrame')) new MutationObserver(function () { if (!pop.hidden) paintChain(); }).observe($('chainFrame'), { attributes: true, attributeFilter: ['hidden'] });
  paintChain();

  // ── 3 详情窗停靠 ──
  const detail = $('lensDetail');
  if (detail) {
    let docked = true, down = null;
    const sync = function () {
      if (detail.hidden) docked = true; // 关掉后再打开，回到停靠位
      document.body.classList.toggle('wb-detail-docked', !detail.hidden && docked);
    };
    new MutationObserver(sync).observe(detail, { attributes: true, attributeFilter: ['hidden'] });
    detail.querySelectorAll('.lens-drag-handle').forEach(function (h) {
      h.addEventListener('pointerdown', function (e) { if (!e.target.closest('button')) down = [e.clientX, e.clientY]; });
      h.addEventListener('pointermove', function (e) {
        if (!down || !docked || !e.buttons) return;
        if (Math.abs(e.clientX - down[0]) + Math.abs(e.clientY - down[1]) > 3) { docked = false; sync(); }
      });
      h.addEventListener('pointerup', function () { down = null; });
    });
    sync();
  }
})();
