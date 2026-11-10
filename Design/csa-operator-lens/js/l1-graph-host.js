/* L1 图宿主：用 Explorer-UI 的 PtoVisualization2Graph.mount 渲染 CSA 语义图，并与外层页面互通。
 * 两个方向都走 postMessage：用 file:// 直接打开时浏览器把 iframe 当成不同源，不能直接调用对方的函数。
 * 外层 → 本页：{ lens: 'graph', type: 'highlight' | 'clear' | 'select' | 'fit', ids, id }
 * 本页 → 外层：{ lens: 'l1', type: 'ready' | 'hover' | 'select' | 'drill', id } */
(function () {
  'use strict';
  const model = window.LENS_CSA_GRAPH;
  window.PTO_PYPA_SEMANTIC_GRAPH = model; // compute-graph-scene 取 overlay 时的默认模型
  const host = document.getElementById('viz2-graph-inter');
  function send(type, id) {
    if (window.parent && window.parent !== window) window.parent.postMessage({ lens: 'l1', type: type, id: id == null ? null : id }, '*');
  }

  const scene = function () { return host.querySelector('[data-viz2-graph-scene]'); };
  const labels = function () { return host.querySelector('.viz2-graph-labels'); };
  const nodeEl = function (id) { return [].filter.call(host.querySelectorAll('[data-pypa-node]'), function (el) { return el.dataset.pypaNode === id; })[0]; };
  let view = { camera: null, expanded: [], nodeId: null, inspectorOpen: false, overlays: {} };
  let silent = false, ctl = null, hot = [];

  function onView(patch) {
    const prev = view.nodeId;
    view = Object.assign({}, view, patch, { inspectorOpen: false }); // 详情放在外层右栏，不用图内抽屉
    if (Object.prototype.hasOwnProperty.call(patch, 'nodeId') && patch.nodeId && patch.nodeId !== prev && !silent) {
      send('select', patch.nodeId);
    }
    ctl.update(null, view, { active: true });
    paintHot();
    const out = document.getElementById('lensScale');
    if (out && view.camera) out.textContent = Math.round(view.camera.scale * 100) + '%';
  }
  ctl = window.PtoVisualization2Graph.mount(host, { onSelect: function () {}, onView: onView, execution: { blocks: 0, taskCount: 0 } });
  ctl.update(null, view, { active: true });

  // 首屏：按宽度适配、顶端对齐，节点名可读；“适应”再看全图
  function extent() {
    let w = 0, h = 0;
    host.querySelectorAll('[data-pypa-node], [data-pypa-group]').forEach(function (el) {
      w = Math.max(w, (parseFloat(el.style.left) || 0) + el.offsetWidth);
      h = Math.max(h, (parseFloat(el.style.top) || 0) + el.offsetHeight);
    });
    return { w: w, h: h };
  }
  function frameWidth() {
    const vp = host.querySelector('.viz2-graph-viewport'), ex = extent();
    if (!vp || !vp.clientWidth || !ex.w) return false;
    const scale = Math.max(.3, Math.min(.85, (vp.clientWidth - 48) / ex.w));
    onView({ camera: { scale: scale, x: (vp.clientWidth - ex.w * scale) / 2, y: 20 } });
    return true;
  }
  (function waitFrame(n) { if (!frameWidth() && n < 40) requestAnimationFrame(function () { waitFrame(n + 1); }); })(0);
  document.addEventListener('click', function (e) {
    const z = e.target.closest('[data-lens-zoom]');
    if (z) { const b = host.querySelector('[data-viz2-graph-zoom="' + z.dataset.lensZoom + '"]'); if (b) b.click(); return; }
    if (e.target.closest('[data-lens-fit]')) { const f = host.querySelector('[data-viz2-graph-fit]'); if (f) f.click(); }
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
    send('drill', t.dataset.pypaNode || t.dataset.viz2GraphLabel);
  });
  window.addEventListener('message', function (e) {
    const m = e.data;
    if (e.source !== window.parent || !m || m.lens !== 'graph') return;
    if (m.type === 'highlight') window.LensGraph.highlight(m.ids);
    else if (m.type === 'clear') window.LensGraph.clear();
    else if (m.type === 'select') window.LensGraph.select(m.id);
    else if (m.type === 'fit') window.LensGraph.fit();
  });
  send('ready');
})();
