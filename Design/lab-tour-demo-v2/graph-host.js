/* 图宿主：用 Explorer-UI 的 PtoVisualization2Graph.mount 渲染语义图，并与外层页面互通。L1 与 L2 共用这一页：
 *   ?mode 缺省 → L1 整层图（LENS_CSA_GRAPH）；?mode=chain → L2 某个模块的 Kernel 链（外层发 chain 消息后再挂载）。
 * 两个方向都走 postMessage：用 file:// 直接打开时浏览器把 iframe 当成不同源，不能直接调用对方的函数。
 * 外层 → 本页：{ lens: 'graph', type: 'highlight' | 'clear' | 'select' | 'fit' | 'chain', ids, id, module }
 * 本页 → 外层：{ lens: 'l1' | 'l2', type: 'ready' | 'hover' | 'select' | 'drill', id }（L2 的 id 已换成 Kernel id） */
(function () {
  'use strict';
  const MODE = /[?&]mode=chain/.test(location.search) ? 'chain' : 'l1';
  const CG = window.LensChainGraph;
  let model = MODE === 'l1' ? window.LENS_CSA_GRAPH : null;
  const host = document.getElementById('viz2-graph-inter');
  const toOuter = function (id) { return MODE === 'chain' && id && CG ? (CG.kernelOf(id) || id) : id; };
  const toInner = function (id) { return MODE === 'chain' && id && CG && window.LensData.KERNELS[id] ? CG.nodeOf(id) : id; };
  function send(type, id, point) {
    if (window.parent && window.parent !== window) window.parent.postMessage({ lens: MODE === 'chain' ? 'l2' : 'l1', type: type, id: id == null ? null : toOuter(id), point: point }, '*');
  }
  if (MODE === 'chain') {
    const hint = document.querySelector('.lens-dock__hint');
    if (hint) hint.textContent = '单击选中 Kernel · 双击 qk_pv 进入 L3 · 实线 = 张量读写，虚线 = TaskId 依赖';
    const legend = document.getElementById('lensLegend');
    if (legend) legend.hidden = true; // Kernel 链只有 Kernel 一种图元，图例写在提示里
  }

  const scene = function () { return host.querySelector('[data-viz2-graph-scene]'); };
  const labels = function () { return host.querySelector('.viz2-graph-labels'); };
  const nodeEl = function (id) { return [].filter.call(host.querySelectorAll('[data-pypa-node]'), function (el) { return el.dataset.pypaNode === id; })[0]; };
  let view = { camera: null, expanded: [], nodeId: null, inspectorOpen: false, overlays: {} };
  let silent = false, ctl = null, hot = [];
  let focusId = null, userMoved = false; // L1 取景焦点；用户动过镜头后不再自动取景（见 frameFocus）

  function onView(patch) {
    view = Object.assign({}, view, patch, { inspectorOpen: false }); // 详情放在外层右栏，不用图内抽屉
    if (Object.prototype.hasOwnProperty.call(patch, 'nodeId') && patch.nodeId && !silent) {
      if (MODE === 'l1') focusId = patch.nodeId; // 画布随后变宽变窄（如详情窗停靠）时，按用户点的节点重新取景
      send('select', patch.nodeId);
    }
    ctl.update(null, view, { active: true });
    paintHot();
    const out = document.getElementById('lensScale');
    if (out && view.camera) out.textContent = Math.round(view.camera.scale * 100) + '%';
  }
  // 挂载（或换一个模型重新挂载）：同一个 host，旧实例先 destroy，事件随之解绑
  function mountModel(next) {
    if (ctl) ctl.destroy();
    model = next; hot = [];
    window.PTO_PYPA_SEMANTIC_GRAPH = model; // compute-graph-scene 取 overlay 时的默认模型
    view = { camera: null, expanded: [], nodeId: null, inspectorOpen: false, overlays: {} };
    ctl = window.PtoVisualization2Graph.mount(host, { onSelect: function () {}, onView: onView, execution: { blocks: 0, taskCount: 0 } });
    ctl.update(null, view, { active: true });
  }
  if (model) mountModel(model);

  // 首屏：按宽度适配、顶端对齐，节点名可读；“适应”再看全图
  // 阶段框（.kf-pypa-graph__group）的标题会伸出框外，取景要把它算进去，否则窄画布上右侧标题被裁
  const GROUP_SEL = '.kf-pypa-graph__group', LABEL_OVERHANG = 100, LABEL_TOP = 34, DOCK = 52, TOP_PAD = 20;
  const box = function (el) { return { x: parseFloat(el.style.left) || 0, y: parseFloat(el.style.top) || 0, w: parseFloat(el.style.width) || el.offsetWidth, h: parseFloat(el.style.height) || el.offsetHeight }; };
  function extent() {
    let w = 0, h = 0, hasGroup = false;
    host.querySelectorAll('[data-pypa-node], [data-pypa-group], ' + GROUP_SEL).forEach(function (el) {
      const b = box(el);
      if (el.matches(GROUP_SEL)) hasGroup = true;
      w = Math.max(w, b.x + b.w);
      h = Math.max(h, b.y + b.h);
    });
    return { w: w + (hasGroup ? LABEL_OVERHANG : 0), h: h };
  }
  // L1：整层图是一条竖着的长流水（约 1014 × 2616）。宽度适配，上限 L1_MAX、下限 L1_MIN：
  // 节点名在 1440 宽下约 10px，比原来的 62% 上限更好读；首屏随后由 frameFocus 对准当前阶段。
  // L2：整条链放得下（不小于 45%）就整体适配，否则按宽度适配——与 ../csa-operator-lens/ 原版一致。
  const L1_MAX = .8, L1_MIN = .62;
  function frameWidth() {
    const vp = host.querySelector('.viz2-graph-viewport'), ex = extent();
    if (!vp || !vp.clientWidth || !vp.clientHeight || !ex.w) return false;
    const byW = (vp.clientWidth - 48) / ex.w, byH = (vp.clientHeight - 56) / ex.h;
    let scale = Math.max(.3, Math.min(L1_MAX, byW));
    if (MODE === 'chain') scale = Math.min(byW, byH) >= .45 ? Math.min(.85, byW, byH) : Math.max(.35, Math.min(.85, byW));
    onView({ camera: { scale: scale, x: (vp.clientWidth - ex.w * scale) / 2, y: MODE === 'chain' && Math.min(byW, byH) >= .45 ? (vp.clientHeight - ex.h * scale) / 2 : TOP_PAD } });
    return true;
  }
  // L1 首屏对准当前节点所在阶段：优先“上一阶段整框 + 当前节点”，读得出来龙去脉；
  // 放不下（缩放会低于 L1_MIN）就从当前阶段的标题开始往下排。阶段标题始终完整露出，不再半截裁在上沿。
  function frameFocus(id) {
    const el = nodeEl(id), vp = host.querySelector('.viz2-graph-viewport');
    if (MODE !== 'l1' || !el || !vp || !vp.clientWidth || !vp.clientHeight) return false;
    const n = box(el), groups = [].map.call(host.querySelectorAll(GROUP_SEL), box).sort(function (a, b) { return a.y - b.y; });
    const gi = groups.findIndex(function (g) { return n.x >= g.x && n.x <= g.x + g.w && n.y >= g.y && n.y <= g.y + g.h; });
    if (gi < 0) return false;
    const W = vp.clientWidth - 48, H = vp.clientHeight - DOCK - TOP_PAD, g = groups[gi], p = groups[gi - 1];
    let top, bottom, left, right;
    if (p) {
      top = p.y - LABEL_TOP; bottom = n.y + n.h + 24;
      left = Math.min(p.x, g.x); right = Math.max(p.x + p.w, g.x + g.w) + LABEL_OVERHANG;
    }
    let scale = p ? Math.min(L1_MAX, W / (right - left), H / (bottom - top)) : 0;
    if (scale < L1_MIN) {
      top = g.y - LABEL_TOP; left = g.x; right = g.x + g.w + LABEL_OVERHANG;
      scale = Math.max(L1_MIN, Math.min(L1_MAX, W / (right - left)));
    }
    const fitsX = (right - left) * scale <= W;
    const x = fitsX ? (vp.clientWidth - (right + left) * scale) / 2 : vp.clientWidth / 2 - (n.x + n.w / 2) * scale;
    onView({ camera: { scale: scale, x: x, y: TOP_PAD - top * scale } });
    focusId = id;
    return true;
  }
  function waitFrame(n, then) { if (frameWidth()) { if (then) then(); } else if (n < 60) requestAnimationFrame(function () { waitFrame(n + 1, then); }); }
  if (model) waitFrame(0);
  // 用户动过镜头（滚轮、拖动、缩放键）后，画布尺寸变化不再重新取景
  const markMoved = function () { userMoved = true; };
  host.addEventListener('wheel', markMoved, { passive: true });
  let downAt = null;
  host.addEventListener('pointerdown', function (e) { downAt = [e.clientX, e.clientY]; });
  host.addEventListener('pointermove', function (e) { if (downAt && e.buttons && Math.abs(e.clientX - downAt[0]) + Math.abs(e.clientY - downAt[1]) > 4) markMoved(); });
  host.addEventListener('pointerup', function () { downAt = null; });
  // 画布变宽变窄（详情窗停靠、拖源码栏、窗口缩放）：图元会自己重新适配，随后按当前焦点再取一次景
  let resizeTimer = null;
  if (window.ResizeObserver) new ResizeObserver(function () {
    if (!ctl || userMoved) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () { if (!(focusId && frameFocus(focusId))) frameWidth(); }, 120);
  }).observe(host);
  document.addEventListener('click', function (e) {
    const z = e.target.closest('[data-lens-zoom]');
    if (z) { markMoved(); const b = host.querySelector('[data-viz2-graph-zoom="' + z.dataset.lensZoom + '"]'); if (b) b.click(); return; }
    if (e.target.closest('[data-lens-fit]')) { markMoved(); const f = host.querySelector('[data-viz2-graph-fit]'); if (f) f.click(); }
  });

  function neighbors(ids) {
    const out = [];
    model.edges.forEach(function (e) {
      if (ids.indexOf(e.s) >= 0) out.push(e.t);
      if (ids.indexOf(e.t) >= 0) out.push(e.s);
    });
    return out;
  }
  function matches(list, id) { return list.indexOf(id) >= 0 || list.indexOf(String(id).split('/')[0]) >= 0; }
  function paintHot() {
    const sc = scene(), lb = labels();
    if (!sc) return;
    const near = hot.length ? neighbors(hot.map(function (id) { return String(id).split('/')[0]; })) : [];
    sc.dataset.lensHot = String(hot.length > 0);
    if (lb) lb.dataset.lensHot = String(hot.length > 0);
    sc.querySelectorAll('[data-pypa-node]').forEach(function (el) {
      el.classList.toggle('lens-hot', matches(hot, el.dataset.pypaNode));
      el.classList.toggle('lens-near', matches(near, el.dataset.pypaNode));
    });
    if (lb) lb.querySelectorAll('[data-viz2-graph-label]').forEach(function (el) {
      el.classList.toggle('lens-hot', matches(hot, el.dataset.viz2GraphLabel));
      el.classList.toggle('lens-near', matches(near, el.dataset.viz2GraphLabel));
    });
    sc.querySelectorAll('[data-pypa-edge]').forEach(function (el) {
      const e = el.dataset.pypaEdge || '', s = e.split('>')[0], t = (e.split('>')[1] || '').split(':')[0];
      el.classList.toggle('lens-hot', hot.length > 0 && (matches(hot, s) || matches(hot, t)));
    });
  }
  // 节点不在可视区时平移过去，缩放保持不变
  function reveal(id) {
    const el = nodeEl(id), vp = host.querySelector('.viz2-graph-viewport');
    if (!el || !vp || !view.camera) return;
    const r = el.getBoundingClientRect(), v = vp.getBoundingClientRect(), pad = 32;
    const inX = r.left >= v.left + pad && r.right <= v.right - pad, inY = r.top >= v.top + pad && r.bottom <= v.bottom - pad;
    if (inX && inY) return;
    const c = view.camera, x = parseFloat(el.style.left) || 0, y = parseFloat(el.style.top) || 0;
    onView({ camera: { scale: c.scale,
      x: inX ? c.x : v.width / 2 - (x + el.offsetWidth / 2) * c.scale,
      y: inY ? c.y : v.height / 2 - (y + el.offsetHeight / 2) * c.scale } });
  }

  window.LensGraph = {
    highlight: function (ids) { hot = (ids || []).filter(Boolean); paintHot(); if (hot.length === 1) reveal(hot[0]); },
    clear: function () { hot = []; paintHot(); },
    select: function (id) {
      const el = nodeEl(id);
      if (!el) return;
      silent = true;
      // 走图元自身的点击路径：保留选中样式，且不触发“聚焦相关节点”的镜头移动
      el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, detail: 1 }));
      silent = false;
      // 首次选中（进页默认焦点）且用户还没动过镜头：按阶段取景；之后只做最小平移
      if (!focusId && !userMoved && frameFocus(id)) return;
      reveal(id);
    },
    fit: function () { const b = host.querySelector('[data-viz2-graph-fit]'); if (b) b.click(); }
  };

  // 悬停与下钻：只读图元已有的 data 属性，不改渲染器
  let hoverId = null;
  host.addEventListener('pointermove', function (e) {
    const t = e.target.closest('[data-pypa-node], [data-viz2-graph-label]');
    const id = t ? (t.dataset.pypaNode || t.dataset.viz2GraphLabel) : null;
    if (id === hoverId) return;
    hoverId = id;
    send('hover', id);
  });
  host.addEventListener('pointerleave', function () { hoverId = null; send('hover', null); });
  // 双击节点直接进 L2：模块、展开后的子节点，或数据 / 状态节点（去产生它的模块）；能否下钻由外层判断
  host.addEventListener('dblclick', function (e) {
    const t = e.target.closest('[data-pypa-node], [data-viz2-graph-label]');
    if (!t) return;
    const id = t.dataset.pypaNode || t.dataset.viz2GraphLabel;
    const node = nodeEl(id) || t, rect = node.getBoundingClientRect();
    send('drill', id, { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
  });
  window.addEventListener('message', function (e) {
    const m = e.data;
    if (e.source !== window.parent || !m || m.lens !== 'graph') return;
    if (m.type === 'chain' && CG) {
      const next = CG.build(m.module);
      if (!next) return;
      mountModel(next);
      waitFrame(0, function () { if (m.id) window.LensGraph.select(toInner(m.id)); });
      return;
    }
    if (!ctl) return;
    // 展开 / 收起一个可展开的模块：走图元自带的 ＋ / − 切换
    if (m.type === 'expand') {
      // open：只展开不收起（双击数据节点时展开产生它的模块）
      if (m.open && host.querySelector('[data-pypa-node^="' + m.id + '/"]')) return;
      const tg = host.querySelector('[data-pypa-expand-toggle="' + m.id + '"]'); if (tg) tg.click(); return;
    }
    if (m.type === 'highlight') window.LensGraph.highlight((m.ids || []).map(toInner));
    else if (m.type === 'clear') window.LensGraph.clear();
    else if (m.type === 'select') window.LensGraph.select(toInner(m.id));
    else if (m.type === 'fit') window.LensGraph.fit();
  });
  send('ready');
})();
