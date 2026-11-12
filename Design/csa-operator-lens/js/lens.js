/* CSA Operator Lens 主控：层级切换、源码联动、详情与跨层说明。
 * L1 整层图与 L2 Kernel 链：iframe graph.html，用 PyPTO-Compute-Explorer-UI Graph 页签的图元渲染（经 postMessage 互通）。
 * L2 Kernel 工作区：上方计算逻辑图（qk_pv 的 Tile 计算图，或模块 Kernel 链）；
 *   下方三个视图 Assignment（Generation → Ordering → Assignment）/ Pipeline（Lane × 轮次）/ 硅片落点。
 * 计算图与硅片落点的渲染与三处联动在 lens-tile.js（复制自 compile-board-demo/path-csa.js）。
 * 旧的 #L3 / #L4 链接落到 L2 的 qk_pv：#L3 = 上方计算图，#L4 = 下方硅片落点。 */
(function () {
  'use strict';

  const D = window.LensData, SRC = window.LENS_SOURCE, F = D.FACTS, G = window.LENS_CSA_GRAPH;
  const CSA = D.FILES.CSA, SPARSE = D.FILES.SPARSE;
  const $ = function (id) { return document.getElementById(id); };
  const esc = function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); };
  const pad2 = function (n) { return String(n).padStart(2, '0'); };
  const MOD = {}; D.MODULES.forEach(function (m) { MOD[m.id] = m; });
  const STAGE = {}; D.STAGES.forEach(function (s) { STAGE[s.id] = s; });
  const KIND_LABEL = { op: '计算', comm: '跨 Rank 通信', state: 'Cache / 状态', sync: 'Task 依赖' };
  const SCOPE_LABEL = { local: 'Local · T 行', full: 'Gathered · 2T 行', group: 'Group 归属', cache: '跨 step 状态', dep: '只承载依赖' };
  const GTYPE_LABEL = { data: '数据', state: '状态 / Cache', operation: '计算', composite: '可展开模块' };

  const state = {
    level: 1,
    module: 'attention',      // 当前模块；决定 L2 的 Kernel 链
    gnode: 'attention',       // L1 图上选中的节点（可能是数据节点或展开后的子节点）
    kernel: 'qk_pv',          // L2 选中的 Kernel
    tab: 'assignment',        // L2 下方视图：assignment | pipeline | chip（硅片落点，目前只有 qk_pv）
    l2top: 'logic',           // L2 顶部：logic（Kernel 计算图）| chain（模块 Kernel 链）
    phase: 'assignment',      // L2 Assignment 的阶段：generation | ordering | assignment
    filter: { rank: 0, req: null, q: null, kvRows: false },
    sel: null,                // L2 Lane / Work Item：{ type:'lane'|'cell', lane, idx }
    line: null,               // 点选的源码行 { file, no }
    hover: null
  };
  let QK = D.buildQkScenario(0);

  // ════════════════════════════════════════════════════════════════════════
  // 源码：整文件（L1 / L2）
  // ════════════════════════════════════════════════════════════════════════
  const KW = {};
  'def return import from as with for in if elif else not and or is None True False pass raise lambda while class yield assert global'.split(' ').forEach(function (k) { KW[k] = 1; });
  const PTO_API = { spmd: 1, at: 1, scope: 1, range: 1, pipeline: 1, unroll: 1, get_block_idx: 1, task_dummy: 1, deps: 1, allow_early_resolve: 1, put: 1, notify: 1, name_hint: 1, level: 1, create_l1: 1, gather_row: 1 };
  function hl(text) {
    const re = /(#.*$)|("[^"]*"|'[^']*')|(\d+(?:\.\d+)?(?:e[+-]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\s+)|(.)/g;
    let out = '', m, prev = '';
    while ((m = re.exec(text))) {
      const tok = m[0];
      if (m[1]) { out += '<span class="tk-cm">' + esc(tok) + '</span>'; break; }
      if (m[2]) out += '<span class="tk-str">' + esc(tok) + '</span>';
      else if (m[3]) out += '<span class="tk-num">' + tok + '</span>';
      else if (m[4]) {
        const next = text.slice(re.lastIndex);
        let cls = 'tk-id';
        if (KW[tok]) cls = 'tk-kw';
        else if (tok === 'pl' || tok === 'pld') cls = 'tk-mod';
        else if (/^\(/.test(next)) cls = 'tk-fn';
        else if (/^=(?!=)/.test(next)) cls = 'tk-kwarg';
        else if (/^[A-Z][A-Z0-9_]+$/.test(tok)) cls = 'tk-const';
        else if (prev === '.') cls = 'tk-attr';
        if (PTO_API[tok] && (prev === '.' || /^=(?!=)/.test(next))) cls += ' tk-hw';
        out += '<span class="' + cls + '">' + tok + '</span>';
      } else if (m[5]) out += tok;
      else out += '<span class="tk-punc">' + esc(tok) + '</span>';
      if (!m[5]) prev = tok;
    }
    return out;
  }
  const fileLines = {};
  function linesOf(file) { return fileLines[file] || (fileLines[file] = SRC.files[file].text.split('\n')); }
  let shownSrc = { file: null, a: 0, b: 0 };
  // 阶段高亮独立于悬停/选中，避免鼠标移开后丢失。
  let phaseFocus = null;
  const PHASE_SOURCE = {
    qk_pv: { generation: [140, 167], ordering: [182, 196], assignment: [208, 218] },
    cache_writeback: { generation: [325, 327], assignment: [323, 327] },
    merge_publish: { generation: [395, 396], assignment: [397, 405] }
  };
  function focusPhase() {
    const ranges = PHASE_SOURCE[state.kernel];
    const range = ranges && ranges[state.phase];
    phaseFocus = range ? { file: D.KERNELS[state.kernel].file, range: range } : null;
    applyFocus();
    if (phaseFocus && shownSrc.file === phaseFocus.file) revealLines(range[0], range[1]);
  }
  function renderSource(file, a, b) {
    if (shownSrc.file === file && shownSrc.a === a && shownSrc.b === b) return;
    shownSrc = { file: file, a: a, b: b };
    const lines = linesOf(file), tags = D.TAGS[file] || {};
    let html = a > 1 ? '<div class="cl-gap"><span></span><span></span><span>… 1–' + (a - 1) + ' 行：导入、常量与参数表</span></div>' : '';
    for (let no = a; no <= b; no++) {
      const t = lines[no - 1] || '';
      const depth = Math.floor(t.match(/^ */)[0].length / 4);
      const tag = tags[no];
      html += '<div class="cl" data-line="' + no + '"><span class="cl-mark"></span><span class="cl-no">' + no + '</span><span class="cl-text" style="--d:' + depth + '">'
        + hl(t) + (tag ? '<span class="cl-tag t-' + tag[1] + '" title="' + esc(tag[2]) + '">' + esc(tag[0]) + '</span>' : '') + '</span></div>';
    }
    if (b < lines.length - 1) html += '<div class="cl-gap"><span></span><span></span><span>… ' + (b + 1) + '–' + (lines.length - 1) + ' 行：测试入口、golden 与输入构造，不在本页范围</span></div>';
    if (window.workbenchCancelSourceScroll) window.workbenchCancelSourceScroll($('srcFile'));
    $('srcFile').innerHTML = html;
    $('srcFile').scrollTop = 0;
  }
  function srcRow(no) { return $('srcFile').querySelector('.cl[data-line="' + no + '"]'); }
  function revealLines(a, b, smooth) {
    const box = $('srcFile'), first = srcRow(a), last = srcRow(b || a) || first;
    if (!first) return;
    const top = first.offsetTop, bottom = last.offsetTop + last.offsetHeight;
    if (top >= box.scrollTop && bottom <= box.scrollTop + box.clientHeight) return;
    const target = bottom - top < box.clientHeight ? (top + bottom - box.clientHeight) / 2 : top - 24;
    if (smooth !== false && window.workbenchScrollSource) window.workbenchScrollSource(box, target);
    else {
      if (window.workbenchCancelSourceScroll) window.workbenchCancelSourceScroll(box);
      box.scrollTo({ top: Math.max(0, target), behavior: smooth === false ? 'auto' : 'smooth' });
    }
  }

  // ── 行 → 对象 ──
  function inRange(no, r) { return no >= r[0] && no <= r[1]; }
  function moduleForLine(file, no) {
    let best = null, span = Infinity;
    D.MODULES.forEach(function (m) {
      m.src.forEach(function (s) {
        if (s[0] === file && no >= s[1] && no <= s[2] && s[2] - s[1] < span) { best = m; span = s[2] - s[1]; }
      });
    });
    if (!best && file === SPARSE && no >= 105 && no <= 358) best = MOD.attention;
    if (!best && file !== CSA && file !== SPARSE) {
      // 子模块文件：看这一行落在哪个模块调用的函数里
      Object.keys(D.KERNELS).forEach(function (k) { const K = D.KERNELS[k]; if (!best && K.file === file && no >= K.fn[0] && no <= K.fn[1]) best = MOD[K.module]; });
    }
    return best;
  }
  // L1 图上的节点：最小的源码段胜出（数据节点的段通常比模块短）
  function graphNodeForLine(file, no) {
    let best = null, span = Infinity;
    G.nodes.forEach(function (n) {
      if (n.source.file !== file) return;
      n.source.ranges.forEach(function (r) { if (no >= r.start && no <= r.end && r.end - r.start < span) { best = n.id; span = r.end - r.start; } });
    });
    if (!best) { const m = moduleForLine(file, no); best = m ? m.id : null; }
    return best;
  }
  function graphNodeLines(id, file) {
    const info = window.LENS_GRAPH_INFO(id), out = [];
    if (!info) return out;
    if (info.module) return modSrcLines(info.module, file);
    if (info.node.source.file !== file) return out;
    info.node.source.ranges.forEach(function (r) { for (let i = r.start; i <= r.end; i++) out.push(i); });
    return out;
  }
  function kernelForLine(file, no) {
    let best = null;
    Object.keys(D.KERNELS).forEach(function (k) {
      const K = D.KERNELS[k];
      if (K.file === file && inRange(no, K.range) && (!best || K.range[1] - K.range[0] < best.range[1] - best.range[0])) best = K;
    });
    return best;
  }
  function qkRole(no) {
    const m = D.KERNELS.qk_pv.lineMap;
    if (inRange(no, m.lane)) return ['lane', '这条 Lane 的编号与要走的轮数：qk_lane_iters = ⌈(1280 − lane) / 24⌉。'];
    if (inRange(no, m.item)) return ['item', '按 qk_flat = lane + it × 24 从 qk_order 取一个 Work Item，拆出 token 与稀疏块。'];
    if (inRange(no, m.valid)) return ['valid', '读 bias 与块有效标记，决定走非空块还是空块分支。'];
    if (inRange(no, m.invalid)) return ['invalid', '空块分支：不做 matmul，只写初值。'];
    if (inRange(no, m.body)) return ['body', '非空块的 Tile 主体：拼 KV → 打分 → softmax → 加权。'];
    return null;
  }
  function modSrcLines(m, file) {
    const out = [];
    m.src.forEach(function (s) { if (s[0] === file) for (let i = s[1]; i <= s[2]; i++) out.push(i); });
    return out;
  }

  // ════════════════════════════════════════════════════════════════════════
  // L1：Explorer-UI Graph 图元（iframe）
  // ════════════════════════════════════════════════════════════════════════
  // 与 iframe 只用 postMessage 通信：file:// 打开时两边被当成不同源，直接调用对方函数会被浏览器拦下
  function postTo(frameId, msg) { const f = $(frameId); if (f && f.contentWindow) f.contentWindow.postMessage(Object.assign({ lens: 'graph' }, msg), '*'); }
  function postGraph(msg) { postTo('layerFrame', msg); }
  function postChain(msg) { postTo('chainFrame', msg); }
  const GRAPH = {
    highlight: function (ids) { postGraph({ type: 'highlight', ids: ids }); },
    clear: function () { postGraph({ type: 'clear' }); },
    select: function (id) { postGraph({ type: 'select', id: id }); },
    fit: function () { postGraph({ type: 'fit' }); }
  };
  const graphApi = function () { return GRAPH; };
  window.addEventListener('message', function (e) {
    const m = e.data, f1 = $('layerFrame'), f2 = $('chainFrame');
    if (!m) return;
    if (f1 && e.source === f1.contentWindow && m.lens === 'l1' && L1_BRIDGE[m.type]) L1_BRIDGE[m.type](m.id, m.point);
    else if (f2 && e.source === f2.contentWindow && m.lens === 'l2' && L2_BRIDGE[m.type]) L2_BRIDGE[m.type](m.id);
  });
  // L2 的 Kernel 链：同一套图元，模块变了才重新挂载，否则只同步选中
  let chainShown = null;
  function syncChain(force) {
    if (!chainOf(state.module)) return;
    // 模块里没有 Lane 视图的 Kernel 时，下方只有一张规则卡片：把高度让给 Kernel 链
    $('viewKernel').classList.toggle('is-rules-only', !chainOf(state.module).order.some(laneModelOf));
    if (force || chainShown !== state.module) { chainShown = state.module; postChain({ type: 'chain', module: state.module, id: state.kernel }); }
    else postChain({ type: 'select', id: state.kernel });
  }
  if ($('chainFrame')) $('chainFrame').addEventListener('load', function () { if (state.level === 2 && l2TopMode() === 'chain') syncChain(true); });

  // L2 顶部：所选 Kernel 有 Tile 计算图（qk_pv）时显示它；否则显示模块的 Kernel 链
  function l2TopMode() {
    const K = D.KERNELS[state.kernel];
    return K && K.deep && window.CsaTile && state.l2top !== 'chain' ? 'logic' : 'chain';
  }
  function renderL2Top() {
    const K = D.KERNELS[state.kernel], hasLogic = !!(K && K.deep && window.CsaTile), mode = l2TopMode(), M = MOD[state.module];
    document.querySelectorAll('#l2TopSwitch button').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.dataset.l2top === mode));
      b.disabled = b.dataset.l2top === 'logic' && !hasLogic;
      if (b.disabled) b.title = '这个 Kernel 的 Tile 计算图未接入（目前只有 qk_pv）'; else if (b.dataset.l2top === 'logic') b.title = '所选 Kernel 的循环嵌套与计算链';
    });
    const ch = chainOf(state.module);
    $('viewKernel').classList.toggle('is-rules-only', !!ch && !ch.order.some(laneModelOf));
    const wasChain = !$('chainFrame').hidden;
    $('hpLogic').hidden = mode !== 'logic';
    $('hpLogicMeta').hidden = mode !== 'logic';
    $('chainFrame').hidden = mode !== 'chain';
    $('l2TopTitle').textContent = mode === 'logic' ? K.name + ' · 计算图' : (M ? M.label : '') + ' · Kernel 链';
    if (mode === 'logic') {
      // 图由 lens-tile.js 渲染并维护（stage / UP_DOWN 切换、Work Item 上下文、与硅片落点的联动）
      if (!$('hpLogic').querySelector('svg')) window.CsaTile.renderLogic();
      paintL2Ctx();
    } else {
      $('l2TopNote').textContent = hasLogic ? '' : 'Tile 计算图目前只接入 qk_pv';
      syncChain(!wasChain);
    }
  }
  // 下方选中一个 Work Item 时，上方计算图只亮它会走的路径（变暗与实例标注由 lens-tile.js 的上下文完成）
  function paintL2Ctx() {
    const it = selItem(), plan = it ? itemPlan(it) : null;
    $('l2TopNote').textContent = plan ? 'Work Item #' + it.id + '（Q' + it.token + ' / SB' + it.sb + '）· ' + plan.kind + ' · ' + plan.short + (state.tab === 'chip' ? '' : ' · 双击切到硅片落点')
      : state.tab === 'chip' ? '悬停节点，下方硅片区域与左侧源码一起亮起' : '单击下方 Work Item，图上只亮它会走的路径';
  }
  function selItem() { return state.kernel === 'qk_pv' && state.sel && state.sel.type === 'cell' ? QK.items[state.sel.idx] : null; }
  const L2_BRIDGE = {
    ready: function () { chainShown = null; if (state.level === 2) syncChain(true); },
    hover: function (id) { if (state.level !== 2) return; state.hover = id && D.KERNELS[id] ? { type: 'kernel', id: id } : null; applyFocus(); },
    select: function (id) { if (state.level === 2 && D.KERNELS[id]) { if (id !== state.kernel) selectKernel(id); showDetail(); } },
    // 双击有 Tile 计算图的 Kernel：上方换成它的计算图
    drill: function (id) { if (state.level === 2 && D.KERNELS[id] && D.KERNELS[id].deep) { state.l2top = 'logic'; if (id !== state.kernel) selectKernel(id); else renderL2Top(); } }
  };
  let drillTimer = null;
  const L1_BRIDGE = {
    ready: function () { syncGraph(); },
    hover: function (id) { if (state.level !== 1) return; state.hover = id ? { type: 'gnode', id: id } : null; applyFocus(); },
    select: function (id) {
      if (state.level !== 1 || drillTimer !== null) return;
      state.gnode = id; state.line = null;
      showDetail();
      const root = String(id).split('/')[0];
      if (MOD[root]) state.module = root;
      renderInspector(); renderStatus(); applyFocus();
      const lines = graphNodeLines(id, shownSrc.file);
      if (lines.length) revealLines(Math.min.apply(null, lines), Math.max.apply(null, lines));
    },
    drill: function (id, point) { if (state.level === 1 && drillTimer === null) drillTo(id, point); }
  };
  // iframe 的 ready 消息可能早于本脚本注册监听，load 时再同步一次
  if ($('layerFrame')) $('layerFrame').addEventListener('load', function () { syncGraph(); });
  function syncGraph() {
    const g = graphApi();
    if (!g) return;
    if (state.gnode) g.select(state.gnode);
  }

  // ════════════════════════════════════════════════════════════════════════
  // L2：Kernel 链
  // ════════════════════════════════════════════════════════════════════════
  function chainOf(mod) { return D.CHAINS[mod] || null; }
  // 手工建模过 Lane 分配的 Kernel；其余 SPMD Kernel 只有派发规则卡片
  const LANE_MODELS = ['qk_pv', 'cache_writeback', 'merge_publish'];
  function laneModelOf(k) { return LANE_MODELS.indexOf(k) >= 0; }
  // L1 节点 → L2 的入口 Kernel：展开后的子节点直达对应 Kernel，模块默认取第一个有 Lane 视图的，再退到第一个 SPMD
  const CHILD_KERNEL = { 'attention/kv_touch': 'kv_touch', 'attention/qk_plan': 'qk_plan', 'attention/qk_order': 'qk_plan', 'attention/qk_pv': 'qk_pv', 'attention/rope_cs': 'rope_cs' };
  function entryKernel(id) {
    if (CHILD_KERNEL[id]) return CHILD_KERNEL[id];
    const root = String(id).split('/')[0], child = String(id).split('/')[1];
    if (child && D.KERNELS[child]) return child;   // 展开后的 Kernel 子节点
    if (root === 'publish') return 'merge_publish';
    const ch = chainOf(root);
    if (!ch) return null;
    return ch.order.filter(laneModelOf)[0] || ch.order.filter(function (k) { return D.KERNELS[k].dispatch === 'spmd'; })[0] || ch.order[0];
  }
  // 数据 / 状态节点：找产生它的模块（图上的入边），下钻到那个模块
  function producerOf(id) {
    const e = G.edges.filter(function (x) { return x.t === id && MOD[x.s] && chainOf(x.s); })[0];
    return e ? e.s : null;
  }
  // 派发多个 Kernel 的模块在 L1 是“容器”：双击展开，再双击其中的 Kernel 才进入 L2
  function isMultiKernel(mod) { const ch = chainOf(mod); return !!ch && ch.order.length > 1 && mod !== 'publish'; }
  function drillTo(id, point) {
    const rootId = String(id).split('/')[0], isChild = String(id).indexOf('/') > 0;
    if (!isChild && isMultiKernel(rootId)) { setDetailVisible(false); postGraph({ type: 'expand', id: rootId }); return false; }
    let k = entryKernel(id);
    // 只有数据 / 状态节点才退到产生它的模块；cache_ready_dep 这类依赖节点本身没有 L2
    const mk = MOD[rootId];
    if (!k && (!mk || mk.kind === 'state') && producerOf(id)) {
      const p = producerOf(id);
      if (isMultiKernel(p)) { setDetailVisible(false); postGraph({ type: 'expand', id: p, open: true }); return false; }
      k = entryKernel(p);
    }
    if (!k) return false;
    state.kernel = k; state.module = D.KERNELS[k].module; state.sel = null;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { setDetailVisible(false); setLevel(2); }
    else if (document.body.dataset.workbenchTransition === 'true' && document.startViewTransition) {
      const frameRect = $('layerFrame').getBoundingClientRect(), mid = $('lensMid'), midRect = mid.getBoundingClientRect();
      const x = frameRect.left - midRect.left + (point && Number.isFinite(point.x) ? point.x : frameRect.width / 2);
      const y = frameRect.top - midRect.top + (point && Number.isFinite(point.y) ? point.y : frameRect.height / 2);
      document.documentElement.style.setProperty('--wb-origin-x', Math.max(0, Math.min(midRect.width, x)) + 'px');
      document.documentElement.style.setProperty('--wb-origin-y', Math.max(0, Math.min(midRect.height, y)) + 'px');
      document.body.classList.add('is-workbench-drill');
      const transition = document.startViewTransition(function () { setDetailVisible(false); setLevel(2); });
      const finishTransition = function () { document.body.classList.remove('is-workbench-drill'); };
      transition.finished.then(finishTransition, finishTransition);
    }
    else {
      setDetailVisible(false);
      const frame = $('layerFrame'), rect = frame.getBoundingClientRect();
      // 图内坐标相对于 iframe；限制到可见画布，避免缩放焦点落在屏幕外。
      const x = point && Number.isFinite(point.x) ? Math.max(0, Math.min(rect.width, point.x)) : rect.width / 2;
      const y = point && Number.isFinite(point.y) ? Math.max(0, Math.min(rect.height, point.y)) : rect.height / 2;
      frame.style.setProperty('--drill-x', x + 'px');
      frame.style.setProperty('--drill-y', y + 'px');
      frame.style.setProperty('--drill-dx', (rect.width / 2 - x) + 'px');
      frame.style.setProperty('--drill-dy', (rect.height / 2 - y) + 'px');
      frame.style.setProperty('--wb-drill-dx', ((rect.width / 2 - x) * .28) + 'px');
      frame.style.setProperty('--wb-drill-dy', ((rect.height / 2 - y) * .28) + 'px');
      $('viewLayer').classList.add('is-drilling-out');
      drillTimer = window.setTimeout(function () {
        drillTimer = null;
        setLevel(2);
        const view = $('viewKernel');
        view.classList.add('is-drilling-in');
        view.addEventListener('animationend', function finish(e) {
          if (e.target !== view) return;
          view.classList.remove('is-drilling-in');
          view.removeEventListener('animationend', finish);
        });
      }, document.body.dataset.workbenchTransition === 'true' ? 320 : 420);
    }
    return true;
  }
  function renderChain() { renderL2Top(); }

  // ════════════════════════════════════════════════════════════════════════
  // L2：Work Item 模型（Assignment 与 Pipeline 共用）
  // 每个 item：key / cls / title / lane / iter / gen{x,y} / ord{x,y} / asg{x,y}；坐标是 Assignment 三个阶段的位置
  // ════════════════════════════════════════════════════════════════════════
  const models = {};
  function workModel(k) {
    const cacheKey = k + ':' + state.filter.rank;
    if (models[cacheKey]) return models[cacheKey];
    let M = null;
    if (k === 'qk_pv') {
      const S = 12, items = [];
      QK.items.forEach(function (it) {
        const r = it.req, bx = (r % 8) * 76, by = 18 + Math.floor(r / 8) * 112;
        const lx = (it.lane % 6) * 124, ly = 18 + Math.floor(it.lane / 6) * 112;
        items.push({ key: it.id, cls: !it.valid ? 'k-idle' : it.sb === 0 ? 'k-win' : 'k-cmp', lane: it.lane, iter: it.iter, req: it.req, token: it.token, kv: it.kv / 128, valid: it.valid,
          title: 'Work Item ' + it.id + ' · Q' + it.token + ' / SB' + it.sb + ' · qk_order[' + it.q + '] · Lane ' + it.lane + ' 第 ' + it.iter + ' 轮',
          gen: { x: bx + it.sb * S, y: by + (it.token % 8) * S },
          ord: { x: (it.q % 60) * S, y: 32 + Math.floor(it.q / 60) * S + (it.valid ? 0 : 14) },
          asg: { x: lx + (it.iter % 9) * S, y: ly + Math.floor(it.iter / 9) * S } });
      });
      const labels = { generation: [], ordering: [], assignment: [] };
      for (let r = 0; r < 32; r++) labels.generation.push([(r % 8) * 76, 14 + Math.floor(r / 8) * 112, 'Req ' + pad2(r)]);
      labels.generation.push([0, 466, '每个请求 8 个 Query（行）× 5 个稀疏块（列）：SB0 是滑窗块，SB1–4 是 Top-k 压缩块']);
      const vRows = Math.ceil(QK.validCount / 60);
      labels.ordering.push([0, 20, 'Pass 1 · 非空块按 token 顺序写进 qk_order · ' + QK.validCount + ' 项']);
      labels.ordering.push([0, 32 + vRows * S + 10, 'Pass 2 · 空块追加到尾部（不删除）· ' + (QK.items.length - QK.validCount) + ' 项']);
      for (let l = 0; l < 24; l++) labels.assignment.push([(l % 6) * 124, 14 + Math.floor(l / 6) * 112, 'Lane ' + pad2(l) + ' · ' + QK.lanes[l].items.length + ' 项']);
      labels.assignment.push([0, 466, 'Lane = qk_order 位置 mod 24，轮次 = ⌊位置 / 24⌋：前 41 轮几乎都是非空块，空块集中在最后 13 轮']);
      M = { cell: 10, vb: [744, 474], items: items, labels: labels, phases: ['generation', 'ordering', 'assignment'], lanes: 24, laneLabel: 'Lane',
        head: '256 Queries × 5 稀疏块 = ' + QK.items.length + ' Work Items · 非空 ' + QK.validCount + ' · 空 ' + (QK.items.length - QK.validCount) + ' · 24 Lanes × ' + QK.iters + ' 轮',
        legend: [['k-win', 'SB0 滑窗 · 读 Original'], ['k-cmp', 'SB1–4 压缩 · 读 Compressed'], ['k-idle', '空块 · 只写初值']],
        laneStat: function (l) { const its = QK.lanes[l].items; const o = its.filter(function (i) { return i.valid && i.sb === 0; }).length, c = its.filter(function (i) { return i.valid && i.sb > 0; }).length; return o + ' O · ' + c + ' C · ' + (its.length - o - c) + ' I'; } };
    } else if (k === 'cache_writeback') {
      const S = 12, items = [];
      for (let t = 0; t < F.KV_T; t++) {
        const b = Math.floor(t / F.WB_TILE), d = t % F.WB_TILE;
        items.push({ key: t, cls: t < F.T ? 'k-r0' : 'k-r1', lane: b, iter: d, title: 'token ' + t + '（来自 Rank ' + (t < F.T ? 0 : 1) + '）· block ' + b + ' · write_dt=' + d,
          gen: { x: (t % 32) * S, y: 32 + Math.floor(t / 32) * S + (t >= F.T ? 22 : 0) },
          asg: { x: (b % 8) * 74 + (d % 4) * S, y: 22 + Math.floor(b / 8) * 50 + Math.floor(d / 4) * S } });
      }
      const labels = { generation: [[0, 20, 'Rank 0 的 T 行 · token 0–255'], [0, 32 + 8 * S + 16, 'Rank 1 的 T 行 · token 256–511（AllGather 后 rank-major 排列）']], assignment: [] };
      for (let b = 0; b < 64; b++) labels.assignment.push([(b % 8) * 74, 18 + Math.floor(b / 8) * 50, 'Blk ' + pad2(b)]);
      M = { cell: 10, vb: [592, 420], items: items, labels: labels, phases: ['generation', 'assignment'], lanes: 64, laneLabel: 'Blk',
        head: '2T = 512 个 token ÷ 8 = 64 个 block · 每个 block 写连续 8 个 token（slot < 0 时运行期跳过）',
        legend: [['k-r0', '来自 Rank 0'], ['k-r1', '来自 Rank 1']], laneStat: function () { return '8 token'; } };
    } else if (k === 'merge_publish') {
      const S = 17, items = [], total = (F.T / F.PUBLISH_T_TILE) * (F.H / F.H_TILE);
      for (let p = 0; p < total; p++) {
        const tb = Math.floor(p / 4), hi = p % 4, dst = Math.floor((hi * 2) / F.LOCAL_O_GROUPS), w = p % F.PUBLISH_WORKERS, it = Math.floor(p / F.PUBLISH_WORKERS);
        items.push({ key: p, cls: dst === 0 ? 'k-r0' : 'k-r1', lane: w, iter: it,
          title: '打包任务 ' + p + ' · token ' + (tb * 8) + '–' + (tb * 8 + 7) + ' · head ' + (hi * 16) + '–' + (hi * 16 + 15) + ' → Group ' + (hi * 2) + '–' + (hi * 2 + 1) + ' → Rank ' + dst,
          gen: { x: 70 + tb * S, y: 36 + hi * S },
          asg: { x: (w % 8) * 74 + it * S, y: 22 + Math.floor(w / 8) * 40 } });
      }
      const labels = { generation: [[70, 24, 'token 块（每块 8 个 token）→ 32 块'], [0, 48, 'head 0–15'], [0, 65, 'head 16–31'], [0, 82, 'head 32–47'], [0, 99, 'head 48–63'], [0, 124, '颜色 = 目标 Rank：head 0–31 属于 Group 0–3（Rank 0），32–63 属于 Group 4–7（Rank 1）']], assignment: [] };
      for (let w = 0; w < 48; w++) labels.assignment.push([(w % 8) * 74, 18 + Math.floor(w / 8) * 40, 'Wkr ' + pad2(w)]);
      M = { cell: 14, vb: [600, 262], items: items, labels: labels, phases: ['generation', 'assignment'], lanes: 48, laneLabel: 'Wkr',
        head: '(T/8) × (64/16) = ' + total + ' 个打包任务 · 48 个 worker grid-stride 领取',
        legend: [['k-r0', '发往 Rank 0'], ['k-r1', '发往 Rank 1']],
        laneStat: function (l) { return Math.ceil((total - l) / 48) + ' 项'; } };
    }
    models[cacheKey] = M;
    return M;
  }
  const PHASE_NAME = { generation: 'Generation', ordering: 'Ordering', assignment: 'Assignment' };
  const PHASE_TIP = { generation: '源码先产生哪些工作', ordering: 'qk_plan 怎样排队', assignment: 'SPMD Lane 怎样领取' };

  function itemFiltered(it) {
    const f = state.filter;
    if (state.kernel !== 'qk_pv') return false;
    if (f.req != null && it.req !== f.req) return true;
    if (f.q != null && it.token !== f.q) return true;
    return false;
  }
  function renderKernelBar() {
    const K = D.KERNELS[state.kernel], M = K && K.dispatch === 'spmd' ? workModel(state.kernel) : null;
    document.querySelectorAll('#kernelTabs button').forEach(function (b) {
      b.setAttribute('aria-selected', String(b.dataset.tab === state.tab));
      b.disabled = b.dataset.tab === 'chip' ? state.kernel !== 'qk_pv' : b.dataset.tab === 'pipeline' ? !M : false;
      b.title = b.dataset.tab === 'chip' && b.disabled ? '硅片落点目前只接入 qk_pv' : ''; 
    });
    if (!M) { $('kernelBar').innerHTML = ''; return; }
    let s = '';
    if (state.tab === 'assignment') {
      const phases = M.phases.indexOf(state.phase) >= 0 ? state.phase : 'assignment';
      s += '<button class="lens-ic" type="button" data-l2="play" aria-label="' + (playing ? '暂停' : '播放三个阶段') + '" title="' + (playing ? '暂停' : '播放') + '">' + (playing ? '❚❚' : '▶') + '</button>'
        + '<div class="lens-seg" role="group" aria-label="Assignment 阶段">' + M.phases.map(function (p) {
          return '<button type="button" data-l2="phase" data-v="' + p + '" aria-pressed="' + (p === phases) + '" title="' + PHASE_TIP[p] + '">' + PHASE_NAME[p] + '</button>';
        }).join('') + '</div>';
    } else {
      s += '<span class="lens-bar-k">' + esc(M.head.split(' · ')[0]) + '</span>';
    }
    if (state.kernel === 'qk_pv') {
      const f = state.filter;
      s += '<label class="lens-field">Rank <select data-l2="rank"><option value="0"' + (f.rank === 0 ? ' selected' : '') + '>0</option><option value="1"' + (f.rank === 1 ? ' selected' : '') + '>1</option></select></label>'
        + '<label class="lens-field">Request <select data-l2="req"><option value="">全部</option>' + Array.from({ length: 32 }, function (_, i) { return '<option value="' + i + '"' + (f.req === i ? ' selected' : '') + '>' + i + '</option>'; }).join('') + '</select></label>'
        + '<label class="lens-field">Query <select data-l2="q"' + (f.req == null ? ' disabled' : '') + '><option value="">全部</option>' + (f.req == null ? '' : Array.from({ length: 8 }, function (_, i) { const t = f.req * 8 + i; return '<option value="' + t + '"' + (f.q === t ? ' selected' : '') + '>Q' + t + '</option>'; }).join('')) + '</select></label>'
        + (state.tab === 'pipeline' ? '<label class="lens-check"><input type="checkbox" data-l2="kv"' + (f.kvRows ? ' checked' : '') + '> Valid KV Rows</label>' : '')
        + '<span class="lens-chip is-syn">Synthetic 输入</span>';
    }
    $('kernelBar').innerHTML = s;
  }

  // ── Assignment：三个阶段共用同一批方块，切换阶段时移动位置（CSS transition） ──
  let asgKey = '';
  function renderAssignment() {
    const M = workModel(state.kernel);
    const phase = M.phases.indexOf(state.phase) >= 0 ? state.phase : 'assignment';
    const key = state.kernel + ':' + state.filter.rank;
    const host = $('kernelLanes');
    if (asgKey !== key || !host.querySelector('svg.lens-asg')) {
      asgKey = key;
      let s = '<svg class="lens-asg" viewBox="-4 -4 ' + (M.vb[0] + 8) + ' ' + (M.vb[1] + 8) + '" preserveAspectRatio="xMidYMin meet" role="group" aria-label="' + esc(D.KERNELS[state.kernel].name + ' 工作分配') + '">'
        + '<defs><pattern id="laneIdle" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="4" height="4" fill="#1d1d1d"/><rect width="1" height="4" fill="#4a4a4a"/></pattern></defs>';
      M.phases.forEach(function (p) {
        s += '<g class="asg-labels" data-phase="' + p + '">' + M.labels[p].map(function (l) {
          const laneMatch = /^(Lane|Blk|Wkr) (\d+)/.exec(l[2]);
          return '<text class="asg-label' + (laneMatch ? ' is-lane' : '') + '"' + (laneMatch ? ' data-lane="' + Number(laneMatch[2]) + '"' : '') + ' x="' + l[0] + '" y="' + l[1] + '">' + esc(l[2]) + '</text>';
        }).join('') + '</g>';
      });
      s += '<g class="asg-items">' + M.items.map(function (it) {
        return '<rect class="lane-cell asg-cell ' + it.cls + '" data-key="' + it.key + '" data-lane="' + it.lane + '" width="' + M.cell + '" height="' + M.cell + '" rx="1.5"><title>' + esc(it.title) + '</title></rect>';
      }).join('') + '</g></svg>';
      host.innerHTML = s;
    }
    const svg = host.querySelector('svg.lens-asg');
    svg.querySelectorAll('.asg-labels').forEach(function (g) { g.classList.toggle('is-on', g.dataset.phase === phase); });
    const byKey = {}; M.items.forEach(function (it) { byKey[it.key] = it; });
    const selKey = state.sel && state.sel.type === 'cell' ? state.sel.idx : null;
    const selLane = state.sel && state.sel.type === 'lane' ? state.sel.lane : null;
    const mateTok = selKey != null && state.kernel === 'qk_pv' ? QK.items[selKey].token : null;
    svg.querySelectorAll('.asg-cell').forEach(function (el) {
      const it = byKey[el.dataset.key], p = it[phase === 'ordering' ? 'ord' : phase === 'generation' ? 'gen' : 'asg'] || it.asg;
      el.style.transform = 'translate(' + p.x + 'px,' + p.y + 'px)';
      el.classList.toggle('is-sel', it.key === selKey);
      el.classList.toggle('is-mate', mateTok != null && it.key !== selKey && it.token === mateTok);
      el.classList.toggle('is-out', itemFiltered(it) || (selLane != null && it.lane !== selLane));
    });
    svg.querySelectorAll('.asg-label.is-lane').forEach(function (t) { t.classList.toggle('is-sel', Number(t.dataset.lane) === selLane); });
    $('kernelLegend').innerHTML = legendHtml(M);
  }
  function legendHtml(M) {
    return '<span class="lens-bar-k">' + esc(M.head) + '</span>' + M.legend.map(function (l) { return '<span class="lens-key"><i class="lane-sw ' + l[0] + '"></i>' + esc(l[1]) + '</span>'; }).join('');
  }
  let playing = false, playTimer = null;
  function stopPlay() { playing = false; clearTimeout(playTimer); }
  function play() {
    const M = workModel(state.kernel);
    if (playing) { stopPlay(); renderKernelBar(); return; }
    playing = true;
    let i = 0;
    const step = function () {
      if (!playing || state.tab !== 'assignment' || state.level !== 2) { stopPlay(); renderKernelBar(); return; }
      state.phase = M.phases[i];
      renderKernelBar(); renderAssignment(); focusPhase(); renderStatus();
      i += 1;
      if (i < M.phases.length) playTimer = setTimeout(step, 1600);
      else { playing = false; playTimer = setTimeout(renderKernelBar, 900); }
    };
    step();
  }

  // ── Pipeline：Lane × 轮次（列是领取顺序，不是时间） ──
  function renderPipeline() {
    const M = workModel(state.kernel), LX = 64, RX = 96;
    const lanes = []; for (let l = 0; l < M.lanes; l++) lanes.push([]);
    M.items.forEach(function (it) { lanes[it.lane][it.iter] = it; });
    const maxIt = Math.max.apply(null, lanes.map(function (l) { return l.length; }));
    const cw = state.kernel === 'merge_publish' ? 40 : state.kernel === 'cache_writeback' ? 14 : 12, gap = state.kernel === 'merge_publish' ? 4 : 2;
    const rowH = M.lanes > 40 ? 12 : 15, top = 24;
    const step = cw + gap, W = Math.max(640, LX + maxIt * step + RX), H = top + M.lanes * rowH + 6;
    const selKey = state.sel && state.sel.type === 'cell' ? state.sel.idx : null;
    const mateTok = selKey != null && state.kernel === 'qk_pv' ? QK.items[selKey].token : null;
    let s = '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMinYMin meet" role="group" aria-label="' + esc(D.KERNELS[state.kernel].name + ' Pipeline') + '">'
      + '<defs><pattern id="laneIdle" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="4" height="4" fill="#1d1d1d"/><rect width="1" height="4" fill="#4a4a4a"/></pattern></defs>'
      + '<text class="lane-axis" x="' + LX + '" y="10">' + (state.kernel === 'cache_writeback' ? 'write_dt →（块内逐行）' : '轮次 →') + '</text>';
    const tick = maxIt > 20 ? 5 : 1;
    for (let i = 0; i < maxIt; i += tick) s += '<text class="lane-axis" x="' + (LX + i * step) + '" y="' + (top - 4) + '">' + i + '</text>';
    lanes.forEach(function (L, li) {
      const y = top + li * rowH, rowSel = state.sel && state.sel.type === 'lane' && state.sel.lane === li;
      s += '<g class="lane-row' + (rowSel ? ' is-sel' : '') + '"><rect class="lane-bg" x="0" y="' + (y - 1) + '" width="' + W + '" height="' + rowH + '"/>'
        + '<text class="lane-label" data-lane="' + li + '" x="0" y="' + (y + rowH - 4) + '">' + M.laneLabel + ' ' + pad2(li) + '</text>';
      L.forEach(function (it, i) {
        if (!it) return;
        const x = LX + i * step, isSel = selKey === it.key, isMate = mateTok != null && !isSel && it.token === mateTok;
        const op = state.filter.kvRows && state.kernel === 'qk_pv' && it.valid ? Math.max(.2, it.kv) : null;
        s += '<rect class="lane-cell ' + it.cls + (isSel ? ' is-sel' : '') + (isMate ? ' is-mate' : '') + (itemFiltered(it) ? ' is-out' : '') + '" data-lane="' + li + '" data-key="' + it.key + '" x="' + x + '" y="' + y + '" width="' + cw + '" height="' + (rowH - 2) + '"' + (op != null ? ' style="fill-opacity:' + op.toFixed(2) + '"' : '') + '><title>' + esc(it.title) + '</title></rect>';
      });
      s += '<text class="lane-sub" x="' + (W - RX + 8) + '" y="' + (y + rowH - 4) + '">' + esc(M.laneStat(li)) + '</text></g>';
    });
    $('kernelLanes').innerHTML = s + '</svg>';
    asgKey = '';
    $('kernelLegend').innerHTML = legendHtml(M) + (state.filter.kvRows ? '<span class="lens-key">透明度 = 本块有效 KV 行 / 128</span>' : '');
  }
  function renderKernelBody() {
    const K = D.KERNELS[state.kernel];
    $('kernelMeta').textContent = K ? K.name + ' · ' + K.file + ' ' + K.range[0] + '–' + K.range[1] : '—';
    renderKernelBar();
    if (!K) { $('kernelLanes').innerHTML = ''; $('kernelLegend').innerHTML = ''; return; }
    if (K.dispatch !== 'spmd') {
      asgKey = '';
      $('kernelLanes').innerHTML = '<div class="lens-card"><p class="st-k">CORE_GROUP 单任务 · 无 Lane 分配</p>'
        + '<p>' + esc(K.what) + '</p>' + (K.loops ? '<p>' + esc(K.loops) + '</p>' : '')
        + '<p><code>pl.at</code> 不做 SPMD 拆分，所以没有 Assignment / Pipeline。可以对照本模块里的 SPMD Kernel 看区别。</p></div>';
      $('kernelLegend').innerHTML = '';
      return;
    }
    if (!laneModelOf(state.kernel)) {
      // 从源码扫描到的 SPMD Kernel：派发数是运行期表达式，这里只讲规则，不画 Lane
      asgKey = '';
      const others = (chainOf(state.module) || { order: [] }).order.filter(laneModelOf);
      $('kernelLanes').innerHTML = '<div class="lens-card"><p class="st-k">SPMD 派发 · Lane 视图未接入</p>'
        + '<p>' + esc(K.what) + '</p>'
        + '<p>派发数：<code>' + esc(K.laneExpr) + '</code>' + (K.deps.length ? '；等待 <code>' + esc(K.deps.join(', ')) + '</code>' : '') + (K.produces ? '；产生 <code>' + esc(K.produces) + '</code>' : '') + '。</p>'
        + '<p>派发数由运行期 token 数等标量算出；这一页只接入了 qk_pv、KV Writeback、Merge · Publish 三个 Kernel 的 Lane 分配。'
        + (others.length ? '本模块可看 <b>' + esc(others.map(function (k) { return D.KERNELS[k].name; }).join('、')) + '</b>。' : '') + '</p></div>';
      $('kernelLegend').innerHTML = '';
      return;
    }
    if (state.tab === 'assignment') renderAssignment(); else renderPipeline();
  }

  // ════════════════════════════════════════════════════════════════════════
  // 详情栏
  // ════════════════════════════════════════════════════════════════════════
  const TILE_NODE = { 'f-spmd': 'SPMD × 24 外框', 'f-iter': 'grid-stride 取块', 'f-pipe': 'PIPELINE · 2 个头块在飞', 'p-gather': '① 拼 KV 块', 'p-score': '② 打分', 'p-softmax': '③ 局部 softmax', 'p-weight': '④ 加权求和',
    kvl1: 'L1 开 KV 块', win: '滑窗按页成段搬', cmp: '压缩按 Top-k 逐行搬', empty: '空块分支', q: 'Q 块进 L1 / L0A', kvref: 'KV 两次复用', qk: 'Q·Kᵀ', scale: '缩放 + bias', max: 'row_max', exp: 'exp', sum: 'row_sum', cast: '转 BF16 回送', pv: 'P·V', out: '写回 mi / li / oi', split: 'AIV 切分' };
  const TILE_REGION = { cores: '24 个 Block', 'gm-q': 'GM · q', 'gm-ori': 'GM · Original KV', 'gm-cmp': 'GM · Compressed KV', 'gm-plan': 'GM · qk_plan 产物', 'gm-out': 'GM · 部分结果',
    aic: 'AIC', 'l1-slot': 'L1 · v2c 管道槽', 'l1-kv': 'L1 · KV 128 KB', 'l1-q': 'L1 · Q 32 KB', l0a: 'L0A', l0b: 'L0B', l0c: 'L0C', cube: 'Cube', pipe: 'C2V / V2C 管道', aiv0: 'AIV0', aiv1: 'AIV1' };
  const TILE_BUF = { slots: 'Vec · c2v 槽', bias: 'Vec · bias', tmp: 'Vec · 临时区', mi: 'Vec · mi', li: 'Vec · li', work: 'Vec · 工作区' };

  function chip(text, cls) { return '<span class="lens-chip' + (cls ? ' ' + cls : '') + '">' + esc(text) + '</span>'; }
  function srcLinks(list) {
    return '<div class="lens-src-links">' + list.map(function (s) {
      return '<button class="lens-src-link" type="button" data-lens-src="' + s[0] + ':' + s[1] + ':' + s[2] + '"><span>' + esc(s[0]) + '</span><b>' + s[1] + (s[2] !== s[1] ? '–' + s[2] : '') + '</b></button>';
    }).join('') + '</div>';
  }
  function xLevelCard(file, no) {
    // 行：L1 · L2 · 计算图 · 硅片（后两行都在 L2 里，对应上方计算图与下方硅片落点）
    const here = state.level === 1 ? 1 : tileOwns() ? 4 : 2;
    const gid = file === CSA ? graphNodeForLine(file, no) : null, gi = gid ? window.LENS_GRAPH_INFO(gid) : null;
    const m = moduleForLine(file, no), K = kernelForLine(file, no), role = K && K.id === 'qk_pv' ? qkRole(no) : null;
    const info = file === SPARSE && window.CsaTile ? window.CsaTile.lineInfo(no) : null;
    const text = (linesOf(file)[no - 1] || '').trim();
    const rows = [];
    const l1 = gi ? '<b>' + esc(gi.node.name) + '</b> · ' + esc(GTYPE_LABEL[gi.node.type]) + (m ? ' · ' + esc(STAGE[m.stage].label) : '') : m ? '<b>' + esc(m.label) + '</b> · ' + esc(STAGE[m.stage].label) : '';
    rows.push(['L1', l1 || '不属于任何模块（声明或空行）', !!(gi || m), (gi || m) ? '<button class="btn" type="button" data-lens-go="1" data-gnode="' + (gi ? gi.id : m.id) + '">看 L1</button>' : '']);
    let l2 = '';
    if (K) l2 = '<b>' + esc(K.name) + '</b> · ' + (K.dispatch === 'spmd' ? 'SPMD × ' + K.lanes : 'CORE_GROUP 单任务') + (role ? '<br>' + esc(role[1]) : '');
    else if (file === CSA && no >= 384 && no <= 393) l2 = '调用 <b>sparse_attn_csa</b>：展开成 kv_touch → qk_plan → qk_pv → rope_cs 四个 Kernel';
    else if (m && m.kernels) l2 = '<b>' + esc(m.label) + '</b> 含 ' + m.kernels.length + ' 个 Kernel，这一行在派发之外（准备参数或声明）';
    else if (m) l2 = 'Kernel 定义在被调用的子模块文件里，本 Demo 只接入了 decode_csa.py 与 decode_sparse_attn_csa.py 里直接写出的派发';
    const goKernel = K ? K.id : (file === CSA && no >= 384 && no <= 393 ? 'qk_pv' : null);
    rows.push(['L2', l2 || '—', !!(K || goKernel), goKernel ? '<button class="btn" type="button" data-lens-go="2" data-kernel="' + goKernel + '">看 L2</button>' : '']);
    if (info) {
      const tag = info.tag ? '<span class="cl-tag t-' + info.tag[1] + '">' + esc(info.tag[0]) + '</span>' : '';
      rows.push(['计算图', '<b>' + (info.nodes.length ? info.nodes.map(function (n) { return esc(TILE_NODE[n] || n); }).join(' · ') : 'qk_pv 循环体') + '</b>' + tag + (info.caption ? '<br>' + info.caption : ''), true,
        '<button class="btn" type="button" data-lens-go="3" data-line="' + no + '">看计算图</button>']);
      const places = info.regions.map(function (r) { return TILE_REGION[r] || r; }).concat(info.bufs.map(function (b) { return TILE_BUF[b] || b; }));
      rows.push(['硅片', places.length ? '<b>' + esc(places.join(' · ')) + '</b>' : '这一行不直接落到片上存储（标量计算或控制流）', places.length > 0,
        places.length ? '<button class="btn" type="button" data-lens-go="4" data-line="' + no + '">看硅片落点</button>' : '']);
    } else {
      const deep = (file === CSA && no >= 384 && no <= 393) || (K && K.deep);
      rows.push(['计算图', deep ? 'qk_pv 的计算图只映射 208–313 行的代码行；这一行不在其中' : '未接入：Tile 计算图与硅片落点目前只覆盖 qk_pv', false, '']);
      rows.push(['硅片', deep ? '同上' : '未接入', false, '']);
    }
    return '<p class="st-k">这一行在各层</p><p class="st-p"><code>' + esc(file) + ' · ' + no + '</code><br><code>' + esc(text.length > 90 ? text.slice(0, 88) + '…' : text) + '</code></p>'
      + '<ol class="lens-xlevel">' + rows.map(function (r, i) {
        return '<li class="' + (i + 1 === here ? 'is-here' : '') + (r[2] ? '' : ' is-none') + '"><span class="xl-k">' + r[0] + '</span><span class="xl-v">' + r[1] + '</span></li>';
      }).join('') + '</ol>';
  }
  // 右侧详情的算子卡片：功能摘要 / 数学表达式 / 输入输出示例 / 场景与限制（op-card-view.js 负责可视化）
  function cardHtml(card, op, caption) {
    if (!card || !window.LensCardView) return '';
    return (caption ? '<p class="oc-cap">' + esc(caption) + '</p>' : '') + window.LensCardView.html(card, { op: op });
  }
  function moduleDetail(m) {
    const kernels = m.kernels ? m.kernels.map(function (k) { return D.KERNELS[k]; }) : [];
    return '<p class="st-k">' + esc(STAGE[m.stage].label) + ' · ' + esc(STAGE[m.stage].en) + '</p>'
      + '<h2 class="st-h">' + esc(m.label) + '</h2>'
      + '<div class="lens-chips">' + chip(KIND_LABEL[m.kind]) + chip(SCOPE_LABEL[m.scope]) + chip('源码静态关系', 'is-real') + '</div>'
      + cardHtml(window.LensCards && window.LensCards.forModule(m.id), m.label)
      + '<p class="st-k">源码</p>' + srcLinks(m.src)
      + (kernels.length ? '<p class="st-k">Kernel · ' + kernels.length + ' 个</p><div class="lens-chips">' + kernels.map(function (K) { return chip(K.name); }).join('') + '</div>'
        + (isMultiKernel(m.id) ? '' : '<p class="lens-hint">它只派发这一个 Kernel：双击直接进入 L2。</p>') : '<p class="lens-empty">' + (m.kind === 'sync' ? 'task_dummy 只承载依赖、不派发 Kernel，所以没有 L2。' : producerOf(m.id) ? '双击这个节点下钻到写它的 ' + esc(MOD[producerOf(m.id)].label) + '（L2）。' : '这个模块没有可下钻的 Kernel。') + '</p>');
  }
  function graphNodeDetail(gi) {
    const n = gi.node, p = n.data || n.state || n.operation || n.composite || {};
    const shape = p.shape && p.shape.allocated ? '[' + p.shape.allocated.join(', ') + ']' : '';
    const parent = gi.id.indexOf('/') > 0 ? window.LENS_GRAPH_INFO(gi.rootId) : null;
    const src = n.source.ranges.map(function (r) { return [n.source.file, r.start, r.end]; });
    const kernel = parent && D.KERNELS[n.id] ? D.KERNELS[n.id] : null;
    return '<p class="st-k">' + esc(parent ? parent.node.name + ' · 内部' : GTYPE_LABEL[n.type]) + '</p>'
      + '<h2 class="st-h">' + esc(n.name) + '</h2>'
      + '<div class="lens-chips">' + chip(GTYPE_LABEL[n.type]) + (p.dtype ? chip(p.dtype) : '') + (shape ? chip(shape) : '') + (p.storage ? chip(p.storage.space) : '') + chip('源码静态关系', 'is-real') + '</div>'
      + '<dl class="lens-kv"><div><dt>是什么</dt><dd>' + esc(n.description) + '</dd></div>'
      + (n.state ? '<div><dt>初值</dt><dd>' + esc(n.state.initial) + '</dd></div><div><dt>本步更新</dt><dd>' + esc(n.state.update) + '</dd></div>' : '')
      + (n.source.symbols.length ? '<div><dt>符号</dt><dd><code>' + esc(n.source.symbols.join(', ')) + '</code></dd></div>' : '')
      + '</dl><p class="st-k">源码</p>' + srcLinks(src)
      + (entryKernel(gi.id) ? '<p class="lens-hint">双击这个节点进入 L2 · ' + esc(D.KERNELS[entryKernel(gi.id)].name) + '。</p>'
        : producerOf(n.id) ? '<p class="lens-hint">' + (isMultiKernel(producerOf(n.id)) ? '双击这个节点展开产生它的 ' + esc(MOD[producerOf(n.id)].label) + '，再双击其中的 Kernel 进入 L2。' : '双击这个节点进入产生它的 ' + esc(MOD[producerOf(n.id)].label) + '（L2）。') + '</p>' : '');
  }
  function kernelDetail(K) {
    const ch = chainOf(K.module);
    const own = window.LensCards && window.LensCards.isKernelCard(K);
    const card = window.LensCards ? cardHtml(window.LensCards.forKernel(K), own ? K.name : MOD[K.module].label, own ? '' : '下面是所属模块 ' + MOD[K.module].label + ' 的卡片；这个 Kernel 是它派发的其中一个') : '';
    let s = '<p class="st-k">' + esc(MOD[K.module].label) + ' · Kernel</p><h2 class="st-h">' + esc(K.name) + '</h2>'
      + '<div class="lens-chips">' + chip(K.dispatch === 'spmd' ? (K.auto ? 'SPMD' : 'SPMD × ' + K.lanes) : 'AT · CORE_GROUP') + (K.produces ? chip('→ ' + K.produces) : '')
      + chip(K.id === 'qk_pv' ? 'Synthetic 输入 · 规则取自源码' : '规则取自源码', K.id === 'qk_pv' ? 'is-syn' : 'is-real')
      + (K.compiled ? chip('pass 36 有同名 Kernel', 'is-real') : '') + '</div>'
      + card
      + '<p class="st-k">派发</p><dl class="lens-kv"><div><dt>派发</dt><dd>' + esc(K.what) + '</dd></div>'
      + (K.laneExpr ? '<div><dt>' + (K.auto ? '派发数' : 'Lane 数') + '</dt><dd><code>' + esc(K.laneExpr) + '</code></dd></div>' : '')
      + (K.fnName ? '<div><dt>所在函数</dt><dd><code>' + esc(K.fnName) + '()</code> · ' + esc(K.file) + '</dd></div>' : '')
      + (K.deps.length ? '<div><dt>等待</dt><dd><code>' + esc(K.deps.join(', ')) + '</code>：只画源码显式声明的依赖。</dd></div>' : '')
      + '</dl><p class="st-k">源码</p>' + srcLinks([[K.file, K.range[0], K.range[1]]]);
    if (K.deep) s += '<p class="lens-hint">上方是它的计算图：单击下方 Work Item，图上只亮它会走的路径；双击 Work Item 或图上节点，切到硅片落点看它落在 AIC / AIV 的哪里。</p>';
    if (ch && ch.order.length > 1) s += '<p class="note">模块 Kernel 链里，虚线是源码写出的 TaskId 依赖，实线是按张量读写推出的数据依赖。</p>';
    return s;
  }
  // ── Work Item → 计算图：一个 Work Item 在 qk_pv 循环体里会走哪条路。全部由源码规则 + 该项的 position / 有效行数推出 ──
  const CHAIN = ['f-pipe', 'p-score', 'p-softmax', 'p-weight', 'q', 'kvref', 'qk', 'scale', 'max', 'exp', 'sum', 'cast', 'pv', 'out'];
  function itemPlan(it) {
    const head = 'Work Item #' + it.id + ' · Q' + it.token + ' / SB' + it.sb;
    if (!it.valid) {
      const text = '空块：224 行判为无效，走 305 行的 else 分支，不拼 KV、不做 matmul；4 组头块各写 16 行 mi = −3e38、li = 0、oi = 0';
      return { kind: '空块', summary: text, short: '空块分支 · 只写初值',
        nodes: ['f-spmd', 'f-iter', 'empty'], badges: { empty: '4 组 × 16 行写初值' } };
    }
    if (it.sb === 0) {
      // 228–262：s0 = 0，窗口长度 = min(pos + 1, WIN)，页内起点决定成段数；不足 128 行的尾部用第 0 行占位、由 bias 屏蔽
      const len = Math.min(it.pos + 1, F.WIN), start = it.pos - len + 1, headOff = start % F.BLOCK_SIZE;
      const segs = Math.ceil((headOff + len) / F.BLOCK_SIZE), tail = F.ATTN_K_TILE - len;
      return { kind: '滑窗块', short: len + ' 行成段搬 · ' + segs + ' 段' + (tail ? ' · 占位 ' + tail : ''),
        summary: '滑窗块：窗口 ' + len + ' 行（position ' + it.pos + '），页内起点 ' + headOff + '，按 32 行一页成段搬 ' + segs + ' 段' + (tail ? '；其余 ' + tail + ' 行用占位行填满 128 行，由 bias 屏蔽' : '；128 行全部有效') + '。随后 2 个头块各做一次 Q·Kᵀ → softmax → P·V',
        nodes: ['f-spmd', 'f-iter', 'p-gather', 'kvl1', 'win'].concat(CHAIN),
        badges: { win: len + ' 行 · ' + segs + ' 段', kvl1: tail ? '占位 ' + tail + ' 行' : '128 行有效', 'f-pipe': '本块 2 个头块' } };
    }
    // 264–276：s0 ≥ 128 时窗口行数为 0，128 行全部走压缩路径逐行搬；Top-k 索引无效的行搬第 0 行占位
    const k = it.kv, tail = F.ATTN_K_TILE - k;
    return { kind: '压缩块', short: '逐行 128 次 · 有效 ' + k,
      summary: '压缩块：SB' + it.sb + ' 对应 Top-k 第 ' + ((it.sb - 1) * 128) + '–' + (it.sb * 128 - 1) + ' 位，窗口行数为 0，128 行全部逐行查压缩页表再搬；有效 ' + k + ' 行' + (tail ? '，另外 ' + tail + ' 行索引无效、搬占位行，由 bias 屏蔽' : '') + '。随后 2 个头块各做一次 Q·Kᵀ → softmax → P·V',
      nodes: ['f-spmd', 'f-iter', 'p-gather', 'kvl1', 'cmp'].concat(CHAIN),
      badges: { cmp: '逐行 128 次 · 有效 ' + k, kvl1: tail ? '占位 ' + tail + ' 行' : '128 行有效', 'f-pipe': '本块 2 个头块' } };
  }
  function tileContextFor(it) {
    const p = itemPlan(it);
    return { item: it, plan: p, ctx: { key: 'Work Item #' + it.id, nodes: p.nodes, badges: p.badges,
      caption: '<b>Lane ' + pad2(it.lane) + ' · 第 ' + it.iter + ' 轮 · Q' + it.token + ' / SB' + it.sb + '</b>：' + esc(p.summary) + '。其余节点已变暗；悬停任意部分照常联动。' } };
  }
  let tileCtx = null;
  function renderTileCtx() {
    const el = $('tileCtx');
    if (!tileCtx || !tileOwns()) { el.hidden = true; el.innerHTML = ''; return; }
    const it = tileCtx.item;
    el.hidden = false;
    el.innerHTML = '<span class="lens-ctx-k">Work Item</span>'
      + '<b>Lane ' + pad2(it.lane) + ' · 第 ' + it.iter + ' 轮 · Work Item #' + it.id + '（Q' + it.token + ' / SB' + it.sb + '）</b>'
      + '<span class="lens-ctx-v">' + esc(tileCtx.plan.kind) + ' · ' + esc(tileCtx.plan.short) + '</span>'
      + '<span class="lens-chip is-syn">Synthetic · Rank ' + it.rank + '</span>'
      + '<span class="lens-ctx-note">硅片分配对每个 Work Item 相同；Lane ' + pad2(it.lane) + ' 是逻辑 block，物理核由运行时调度</span>'
      + '<button class="lens-ctx-x" type="button" data-lens-ctx-clear aria-label="清除 Work Item，回到通用循环体" title="回到通用循环体">×</button>';
  }
  function setTileCtx(next) {
    tileCtx = next;
    if (window.CsaTile) window.CsaTile.setContext(next ? next.ctx : null);
    renderTileCtx();
  }
  // 计算图与硅片落点上的 Work Item 上下文始终跟着下方的选中走
  function syncCtx() {
    const it = selItem(), cur = tileCtx ? tileCtx.item : null;
    if (it === cur) { renderTileCtx(); return; }
    setTileCtx(it ? tileContextFor(it) : null);
  }

  function selectionDetail() {
    const sel = state.sel;
    if (!sel) return '';
    if (state.kernel === 'qk_pv') {
      if (sel.type === 'cell') {
        const it = QK.items[sel.idx], plan = itemPlan(it);
        const preview = '<p class="st-k">计算路径</p><div class="lens-preview"><b>' + esc(plan.kind) + ' · ' + esc(plan.short) + '</b><span>' + esc(plan.summary) + '。</span></div>'
          + '<p class="lens-hint">双击这个格子切换到硅片落点，看' + (it.valid ? '它怎么算' : '空块分支') + '。</p>';
        return '<p class="st-k">Work Item · Rank ' + it.rank + '</p><h2 class="st-h">#' + it.id + ' · Q' + it.token + ' / SB' + it.sb + '</h2>'
          + '<div class="lens-chips">' + chip(it.valid ? (it.sb === 0 ? '滑窗块' : '压缩块') : '空块') + chip('Synthetic', 'is-syn') + '</div>'
          + '<div class="lens-formula">workItemId = token × 5 + sb = <b>' + it.token + ' × 5 + ' + it.sb + ' = ' + it.id + '</b>\nqk_order 位置 = <b>' + it.q + '</b>\nLane = ' + it.q + ' mod 24 = <b>' + it.lane + '</b>\n轮次 = ⌊' + it.q + ' / 24⌋ = <b>' + it.iter + '</b></div>'
          + '<dl class="lens-kv"><div><dt>为什么在这里</dt><dd>' + esc(it.reason) + '。qk_plan 把非空块排在前面，所以' + (it.valid ? '它在前 ' + QK.validCount + ' 个位置里，24 条 Lane 先各领到重活。' : '它被追加到尾部，最后几轮才被领取。') + '</dd></div>'
          + '<div><dt>输入（示意）</dt><dd>请求 ' + it.req + ' · position ' + it.pos + ' · 本块有效 KV ' + it.kv + ' / 128 行。同一 token 的另外 4 个块已用细框标出。</dd></div></dl>'
          + preview;
      }
      const L = QK.lanes[sel.lane];
      const v = L.items.filter(function (i) { return i.valid; }).length;
      return '<p class="st-k">SPMD Lane</p><h2 class="st-h">Lane ' + pad2(L.index) + '</h2>'
        + '<div class="lens-formula">qk_lane_iters = ⌈(1280 − ' + L.index + ') / 24⌉ = <b>' + L.items.length + '</b>\n领取 qk_order[' + L.index + '], [' + (L.index + 24) + '], [' + (L.index + 48) + '], …\n非空 <b>' + v + '</b> · 空 <b>' + (L.items.length - v) + '</b></div>'
        + '<p class="note">24 条 Lane 是一次派发里的 24 个逻辑 block，不是 24 次 Submit；Lane 编号对应 a2a3 的 24 个 AIC。</p>';
    }
    if (state.kernel === 'cache_writeback') {
      if (sel.type === 'cell') { const t = sel.idx, b = Math.floor(t / 8);
        return '<p class="st-k">Token 行</p><h2 class="st-h">token ' + t + '</h2><div class="lens-formula">block = <b>' + b + '</b>，wb_t0 = ' + b + ' × 8 = ' + (b * 8) + '\nwrite_t = wb_t0 + ' + (t - b * 8) + ' = <b>' + t + '</b>\nrow = ori_slot_mapping[' + t + ']（< 0 跳过）</div><p class="note">AllGather 后按 rank-major 排列：token ' + t + ' 来自 Rank ' + (t < F.T ? 0 : 1) + '。两个 Rank 各自写一份完整的 Original KV Cache。</p>'; }
      return '<p class="st-k">Block</p><h2 class="st-h">Block ' + pad2(sel.lane) + '</h2><div class="lens-formula">负责 token <b>' + (sel.lane * 8) + '–' + (sel.lane * 8 + 7) + '</b>，块内 pl.range(8)</div>';
    }
    if (state.kernel === 'merge_publish') {
      if (sel.type === 'cell') { const p = sel.idx, tb = Math.floor(p / 4), hi = p % 4, g0 = hi * 2, dst = Math.floor(g0 / 4);
        return '<p class="st-k">打包任务</p><h2 class="st-h">#' + p + ' · token ' + (tb * 8) + '–' + (tb * 8 + 7) + ' · head ' + (hi * 16) + '–' + (hi * 16 + 15) + '</h2>'
          + '<div class="lens-formula">worker = ' + p + ' mod 48 = <b>' + (p % 48) + '</b>\ntoken_block = ' + p + ' // 4 = ' + tb + '，m_h_idx = ' + hi + '\nglobal_group0 = ' + (hi * 16) + ' // 8 = <b>' + g0 + '</b>\ndestination_rank = ' + g0 + ' // 4 = <b>' + dst + '</b></div>'
          + '<p class="note">合并 5 个稀疏块的 m / l / o → 归一化 → 输出 RoPE → 打包成 Group ' + g0 + '–' + (g0 + 1) + ' 的行，用 pld.tensor.put 直接写进 Rank ' + dst + ' 的 attention_window。</p>'; }
      const n = Math.ceil((128 - sel.lane) / 48);
      return '<p class="st-k">Worker</p><h2 class="st-h">Worker ' + pad2(sel.lane) + '</h2><div class="lens-formula">pl.range(' + sel.lane + ', 128, 48) → <b>' + n + '</b> 个打包任务</div><p class="note">128 不是 48 的倍数：worker 0–31 领 3 个，32–47 领 2 个。</p>';
    }
    return '';
  }
  function renderInspector() {
    let s = '', meta = '';
    if (state.line) s += xLevelCard(state.line.file, state.line.no);
    if (state.level === 1) {
      const gi = state.gnode ? window.LENS_GRAPH_INFO(state.gnode) : null;
      meta = gi ? GTYPE_LABEL[gi.node.type] : '—';
      s += gi ? (gi.module ? moduleDetail(gi.module) : graphNodeDetail(gi)) : '<p class="lens-empty">点图上的节点或左侧源码行。</p>';
    } else {
      const K = D.KERNELS[state.kernel];
      meta = K ? K.name : '—';
      s += selectionDetail() + (K ? kernelDetail(K) : '');
    }
    $('inspectorMeta').textContent = meta;
    $('inspector').innerHTML = s;
  }

  // ════════════════════════════════════════════════════════════════════════
  // 联动
  // ════════════════════════════════════════════════════════════════════════
  function applyFocus() {
    const h = state.hover;
    let hotLines = [], hotKernel = null;
    const file = shownSrc.file;
    if (state.level === 1) {
      const g = graphApi();
      if (h && h.type === 'gnode') hotLines = graphNodeLines(h.id, file);
      else if (h && h.type === 'line') { hotLines = [h.no]; const id = graphNodeForLine(file, h.no); if (g) { if (id) g.highlight([id]); else g.clear(); } }
      if (g && !(h && h.type === 'line')) g.clear();
    } else if (tileOwns()) {
      return; // 硅片落点视图：源码摘录、计算图、硅片三处联动由 lens-tile.js 负责
    } else if (state.level === 2) {
      const K = D.KERNELS[state.kernel];
      // 计算图节点：悬停节点本身，或悬停源码行时它落在的节点
      const lineInfo = h && h.type === 'line' && file === SPARSE && window.CsaTile ? window.CsaTile.lineInfo(h.no) : null;
      const hotNodes = h && h.type === 'lnode' ? [h.id] : lineInfo ? lineInfo.nodes : [];
      document.querySelectorAll('#hpLogic .lg-hit').forEach(function (g) { g.classList.toggle('is-hot', hotNodes.indexOf(g.dataset.node) >= 0); });
      if (h && h.type === 'lnode' && window.CsaTile) hotLines = window.CsaTile.nodeLines(h.id);
      else if (h && h.type === 'kernel') { hotKernel = h.id; const R = D.KERNELS[h.id].range; for (let i = R[0]; i <= R[1]; i++) hotLines.push(i); }
      else if (h && h.type === 'line') { const k = kernelForLine(file, h.no); hotKernel = k ? k.id : null; hotLines = [h.no]; }
      else if (h && (h.type === 'lane' || h.type === 'cell') && K && K.lineMap) {
        const lm = K.lineMap, add = function (r) { if (r) for (let i = r[0]; i <= r[1]; i++) hotLines.push(i); };
        add(lm.lane); if (h.type === 'cell') { add(lm.item); const it = state.kernel === 'qk_pv' ? QK.items[h.idx] : null; if (it && !it.valid) add(lm.invalid); else add(lm.valid); }
      }
      if (h && h.type === 'line' && hotKernel) postChain({ type: 'highlight', ids: [hotKernel] });
      else if (!(h && h.type === 'kernel')) postChain({ type: 'clear' });
    }
    const sel = state.line && state.line.file === file ? state.line.no : null;
    document.querySelectorAll('#srcFile .cl').forEach(function (el) {
      const no = Number(el.dataset.line);
      el.classList.toggle('is-hot', hotLines.indexOf(no) >= 0);
      el.classList.toggle('is-sel', no === sel);
      el.classList.toggle('is-phase', !!phaseFocus && phaseFocus.file === file && inRange(no, phaseFocus.range));
    });
    if (h && h.type !== 'line' && hotLines.length) revealLines(Math.min.apply(null, hotLines), Math.max.apply(null, hotLines));
  }

  // ════════════════════════════════════════════════════════════════════════
  // 层级切换
  // ════════════════════════════════════════════════════════════════════════
  function renderCrumb() {
    const K = D.KERNELS[state.kernel], m = MOD[state.module];
    const parts = [[1, 'Decode CSA']];
    if (state.level >= 2) parts.push([2, (m ? m.label : '') + ' / ' + (K ? K.name : '')]);
    $('lensCrumb').innerHTML = parts.map(function (p, i) {
      return (i ? '<i>/</i>' : '') + '<button type="button" data-lens-go="' + p[0] + '"' + (p[0] === state.level ? ' aria-current="true"' : '') + '>' + esc(p[1]) + '</button>';
    }).join('');
  }
  function renderStatus() {
    if (tileOwns()) return; // 硅片落点视图的状态栏由 lens-tile.js 写
    const main = $('hpStatus'), dim = $('hpStatusDim');
    main.className = 'hp-status-main';
    if (state.level === 1) {
      const gi = state.gnode ? window.LENS_GRAPH_INFO(state.gnode) : null;
      main.textContent = 'L1 整层 · 5 个阶段 · ' + G.nodes.length + ' 个节点 · 当前：' + (gi ? gi.node.name : '—') + ' · 双击模块展开，双击 Kernel 进入 L2';
      dim.textContent = '数据：decode_csa.py 源码静态关系 · 图元复用 PyPTO Compute Explorer · 位置不表示耗时';
    } else {
      const K = D.KERNELS[state.kernel];
      main.textContent = 'L2 Kernel · ' + (K ? K.name + ' · ' + (K.dispatch === 'spmd' ? 'SPMD × ' + K.lanes + ' · ' + (!laneModelOf(state.kernel) ? '派发规则（Lane 视图未接入）' : state.tab === 'assignment' ? 'Assignment / ' + PHASE_NAME[state.phase] : 'Pipeline') : 'CORE_GROUP 单任务') : '—');
      dim.textContent = state.kernel === 'qk_pv' ? '数据：分配规则取自源码 · 有效 / 空块为 Synthetic 输入 · 列是领取顺序，不是时间' : '数据：分配规则取自源码 · 默认 T=256';
    }
  }
  // L2 下方是硅片落点时：源码栏换成 qk_pv 带硬件标签的摘录，详情换成问题分析，三处联动交给 lens-tile.js
  function tileOwns() { return state.level === 2 && state.tab === 'chip'; }
  function applyView() {
    const n = state.level, chip = tileOwns();
    $('viewLayer').hidden = n !== 1;
    $('viewKernel').hidden = n !== 2;
    $('viewKernel').classList.toggle('is-chip', chip);
    $('kernelWork').hidden = chip;
    $('hpLensPane').hidden = !chip;
    $('srcFile').hidden = chip;
    $('hpCode').hidden = !chip;
    $('inspectorView').hidden = chip;
    $('hpStoryPane').hidden = !chip;
    if (window.CsaTile) { if (!chip) window.CsaTile.clearFocus(); window.CsaTile.setActive(chip); }
  }
  function setLevel(n, opts) {
    opts = opts || {};
    // 旧层级 L3 / L4（以及“这一行在各层”的跳转）都落到 L2 的 qk_pv：L3 = 上方计算图，L4 = 下方硅片落点
    if (n >= 3) {
      state.module = 'attention'; state.kernel = 'qk_pv'; state.l2top = 'logic';
      if (n === 4) state.tab = 'chip';
      n = 2;
    }
    if (drillTimer !== null) { window.clearTimeout(drillTimer); drillTimer = null; }
    $('viewLayer').classList.remove('is-drilling-out');
    $('viewKernel').classList.remove('is-drilling-in');
    stopPlay(); phaseFocus = null;
    if (n === 2 && !chainOf(state.module)) state.module = 'attention';
    if (n === 2 && (!D.KERNELS[state.kernel] || D.KERNELS[state.kernel].module !== state.module)) state.kernel = entryKernel(state.module);
    if (n === 2 && state.tab === 'chip' && state.kernel !== 'qk_pv') state.tab = 'assignment';
    const changed = state.level !== n;
    state.level = n; state.hover = null;
    if (changed && !opts.keepSel) state.sel = null;
    document.body.dataset.level = n;
    document.querySelectorAll('#lensLevels button').forEach(function (b) { b.setAttribute('aria-selected', String(Number(b.dataset.level) === n)); });
    applyView();
    if (n === 1) { renderSource(CSA, 151, 499); $('srcMeta').textContent = CSA + ' · decode_csa() 151–499'; }
    else { showKernelSource(D.KERNELS[state.kernel]); renderChain(); renderKernelBody(); }
    syncCtx();

    renderCrumb(); renderStatus(); renderInspector(); applyFocus();
    if (n === 1) syncGraph();
    if (opts.line && tileOwns() && window.CsaTile) window.CsaTile.focusLine(opts.line);
    else if (opts.line) revealLines(opts.line, opts.line, true);
    else if (n === 1 && state.gnode) { const ls = graphNodeLines(state.gnode, CSA); if (ls.length) revealLines(Math.min.apply(null, ls), Math.max.apply(null, ls), false); }
    else if (n === 2 && !tileOwns()) { const K = D.KERNELS[state.kernel]; revealLines(K.line, K.line, false); }
    syncResizeHandle();
  }
  // L2 下方视图切换：Assignment / Pipeline / 硅片落点（只有 qk_pv 有硅片落点）
  function setTab(tab) {
    if (tab === 'chip' && state.kernel !== 'qk_pv') return;
    stopPlay(); phaseFocus = null;
    state.tab = tab; state.hover = null;
    if (tab === 'chip') state.l2top = 'logic'; // 硅片落点与上方计算图联动
    applyView();
    showKernelSource(D.KERNELS[state.kernel]);
    renderL2Top(); refreshL2(); renderCrumb();
  }
  // L2 源码栏：显示所选 Kernel 所在函数（模块可能跨多个函数、多个文件）；硅片落点视图下显示 qk_pv 摘录
  function showKernelSource(K) {
    renderSource(K.file, K.fn[0], K.fn[1]);
    $('srcMeta').textContent = tileOwns() ? SPARSE + ' · qk_pv 208–313 · 只列代码行，带硬件标签'
      : K.file + ' · ' + (K.fnName ? K.fnName + '() ' : '') + K.fn[0] + '–' + K.fn[1];
  }
  function selectKernel(id) {
    stopPlay(); phaseFocus = null;
    state.kernel = id; state.sel = null; state.line = null;
    if (state.tab === 'chip' && id !== 'qk_pv') { state.tab = 'assignment'; applyView(); }
    showKernelSource(D.KERNELS[id]); renderL2Top(); refreshL2(); renderCrumb();
    if (!tileOwns()) revealLines(D.KERNELS[id].line, D.KERNELS[id].line);
  }
  function refreshL2() { syncCtx(); renderKernelBody(); renderInspector(); renderStatus(); applyFocus(); if (l2TopMode() === 'logic') paintL2Ctx(); }
  function openSrc(file, a, b) {
    // 源码链接：留在当前视图能显示的文件里；否则切到能显示它的那一层
    if (tileOwns()) { if (file === SPARSE && a >= 208 && a <= 313) { window.CsaTile.focusLine(a); return; } setTab('assignment'); }
    if (shownSrc.file !== file || a < shownSrc.a || a > shownSrc.b) {
      if (file === CSA) setLevel(1);
      else {
        const k = Object.keys(D.KERNELS).filter(function (id) { const K = D.KERNELS[id]; return K.file === file && a >= K.fn[0] && a <= K.fn[1]; })[0];
        if (k) { state.kernel = k; state.module = D.KERNELS[k].module; }
        else state.module = 'attention';
        if (state.level === 2 && k) { showKernelSource(D.KERNELS[k]); renderChain(); renderKernelBody(); renderCrumb(); } else setLevel(2);
      }
    }
    state.line = { file: file, no: a };
    renderInspector(); applyFocus(); revealLines(a, b);
  }

  // ════════════════════════════════════════════════════════════════════════
  // 事件
  // ════════════════════════════════════════════════════════════════════════
  document.addEventListener('click', function (e) {
    if (e.target.closest('.lens-detail-close')) { setDetailVisible(false); return; }
    if (e.target.closest('#lensDetailToggle')) { setDetailVisible($('lensDetail').hidden); return; }
    if (tileOwns() && e.target.closest('#hpSvg .sv-buf, #hpSvg .sv-region')) showDetail();
    const lv = e.target.closest('#lensLevels button');
    if (lv) { setLevel(Number(lv.dataset.level)); return; }
    const go = e.target.closest('[data-lens-go]');
    if (go) {
      if (go.dataset.mod) state.module = go.dataset.mod;
      if (go.dataset.gnode) { state.gnode = go.dataset.gnode; const r = go.dataset.gnode.split('/')[0]; if (MOD[r]) state.module = r; }
      if (go.dataset.kernel) { state.kernel = go.dataset.kernel; state.module = D.KERNELS[go.dataset.kernel].module; }
      setLevel(Number(go.dataset.lensGo), { line: go.dataset.line ? Number(go.dataset.line) : null, keepSel: true });
      return;
    }
    if (e.target.closest('[data-lens-ctx-clear]')) { state.sel = null; refreshL2(); return; }
    const sl = e.target.closest('[data-lens-src]');
    if (sl) { const p = sl.dataset.lensSrc.split(':'); openSrc(p[0], Number(p[1]), Number(p[2])); return; }
    const tab = e.target.closest('#kernelTabs button');
    if (tab && !tab.disabled) { if (tab.dataset.tab !== state.tab) setTab(tab.dataset.tab); return; }
    const l2 = e.target.closest('[data-l2]');
    if (l2 && l2.dataset.l2 === 'play') { play(); return; }
    if (l2 && l2.dataset.l2 === 'phase') { stopPlay(); state.phase = l2.dataset.v; renderKernelBar(); renderAssignment(); focusPhase(); renderStatus(); return; }
    // 双击上方计算图节点：切到硅片落点，看这一步落在哪
    const lnode = e.target.closest('#hpLogic .lg-hit');
    if (lnode && e.detail >= 2 && state.tab !== 'chip' && state.kernel === 'qk_pv') { setTab('chip'); return; }
    const cell = e.target.closest('#kernelLanes .lane-cell[data-key]');
    if (cell && e.detail >= 2 && state.kernel === 'qk_pv') {
      const it = QK.items[Number(cell.dataset.key)];
      state.sel = { type: 'cell', lane: it.lane, idx: it.id };
      setTab('chip');
      return;
    }
    if (cell) { showDetail(); state.sel = { type: 'cell', lane: Number(cell.dataset.lane), idx: Number(cell.dataset.key) }; state.line = null; refreshL2(); return; }
    const top = e.target.closest('#l2TopSwitch button');
    if (top && !top.disabled) { state.l2top = top.dataset.l2top; renderL2Top(); return; }

    const lane = e.target.closest('#kernelLanes .lane-label, #kernelLanes .asg-label.is-lane');
    if (lane) { showDetail(); state.sel = { type: 'lane', lane: Number(lane.dataset.lane) }; state.line = null; refreshL2(); return; }
    const row = e.target.closest('#srcFile .cl');
    if (row) {
      showDetail();
      const no = Number(row.dataset.line);
      state.line = { file: shownSrc.file, no: no };
      if (state.level === 1) {
        const id = graphNodeForLine(shownSrc.file, no);
        if (id) { state.gnode = id; if (MOD[id]) state.module = id; else { const m = moduleForLine(shownSrc.file, no); if (m) state.module = m.id; } const g = graphApi(); if (g) g.select(id); }
      }
      if (state.level === 2) { const k = kernelForLine(shownSrc.file, no); if (k && k.id !== state.kernel) { phaseFocus = null; state.kernel = k.id; state.sel = null; renderKernelBody(); renderCrumb(); } }
      renderInspector(); renderStatus(); applyFocus();
    }
  });
  document.addEventListener('change', function (e) {
    const c = e.target.closest('[data-l2]');
    if (!c) return;
    const f = state.filter, v = c.value === '' ? null : Number(c.value);
    if (c.dataset.l2 === 'rank') { f.rank = v || 0; QK = D.buildQkScenario(f.rank); state.sel = null; }
    else if (c.dataset.l2 === 'req') { f.req = v; f.q = null; }
    else if (c.dataset.l2 === 'q') f.q = v;
    else if (c.dataset.l2 === 'kv') f.kvRows = c.checked;
    refreshL2();
  });
  function hoverFrom(t) {
    const cell = t.closest('#kernelLanes .lane-cell[data-key]'); if (cell) return { type: 'cell', idx: Number(cell.dataset.key) };
    const lane = t.closest('#kernelLanes .lane-label, #kernelLanes .asg-label.is-lane'); if (lane) return { type: 'lane', lane: Number(lane.dataset.lane) };
    const ln = t.closest('#hpLogic .lg-hit'); if (ln) return { type: 'lnode', id: ln.dataset.node };
    const row = t.closest('#srcFile .cl'); if (row) return { type: 'line', no: Number(row.dataset.line) };
    return null;
  }
  document.addEventListener('mouseover', function (e) {
    if (tileOwns()) return; // 硅片落点视图的悬停联动由 lens-tile.js 处理
    const h = hoverFrom(e.target);
    if (!h && state.hover && state.hover.type === 'gnode') return; // L1 图的悬停由 iframe 管理
    if (JSON.stringify(h) === JSON.stringify(state.hover)) return;
    state.hover = h; applyFocus();
  });
  document.addEventListener('focusin', function (e) {
    if (tileOwns()) return;
    const h = hoverFrom(e.target);
    if (h) { state.hover = h; applyFocus(); }
  });
  $('hpTagToggle').addEventListener('click', function () {
    const btn = $('hpTagToggle'), on = btn.getAttribute('aria-pressed') === 'true';
    btn.setAttribute('aria-pressed', String(!on));
    $('hpCodePane').classList.toggle('is-tags-off', on);
    btn.title = on ? '显示行内标签' : '隐藏行内标签';
    btn.setAttribute('aria-label', btn.title);
  });

  // 详情窗保持在视口内；拖动标题栏，内容区域仍可滚动和交互。
  const detail = $('lensDetail'), detailToggle = $('lensDetailToggle');
  function setDetailVisible(visible) { detail.hidden = !visible; detailToggle.setAttribute('aria-expanded', String(visible)); if (visible) clampDetail(); }
  function showDetail() { if (document.body.dataset.manualDetail !== 'true') setDetailVisible(true); }
  function clampDetail(left, top) {
    if (detail.hidden) return;
    const rect = detail.getBoundingClientRect();
    const x = Math.max(0, Math.min(window.innerWidth - rect.width, left == null ? rect.left : left));
    const y = Math.max(0, Math.min(window.innerHeight - rect.height, top == null ? rect.top : top));
    detail.style.left = x + 'px'; detail.style.top = y + 'px'; detail.style.right = 'auto';
  }
  detail.querySelectorAll('.lens-drag-handle').forEach(function (handle) {
    handle.addEventListener('pointerdown', function (e) {
      if (e.button !== 0 || e.target.closest('button')) return;
      e.preventDefault();
      const rect = detail.getBoundingClientRect(), x = e.clientX, y = e.clientY;
      handle.setPointerCapture(e.pointerId); handle.classList.add('is-dragging');
      function move(ev) { clampDetail(rect.left + ev.clientX - x, rect.top + ev.clientY - y); }
      function end() { handle.classList.remove('is-dragging'); handle.removeEventListener('pointermove', move); handle.removeEventListener('pointerup', end); handle.removeEventListener('pointercancel', end); handle.removeEventListener('lostpointercapture', end); }
      handle.addEventListener('pointermove', move); handle.addEventListener('pointerup', end); handle.addEventListener('pointercancel', end); handle.addEventListener('lostpointercapture', end);
    });
  });
  window.addEventListener('resize', function () { clampDetail(); });

  // ── 拖动：源码栏宽度与 L2 上下分栏（改写自 compile-board-demo/resize.js） ──
  const codePane = $('hpCodePane'), resizeHandle = $('hpCodeResize'), mainPane = document.querySelector('.hp-main');
  function resizeBounds() {
    const width = mainPane.getBoundingClientRect().width;
    return window.innerWidth > 1100 ? { min: 320, max: Math.max(320, width - 560) } : { min: 260, max: Math.max(260, width - 360) };
  }
  function setCodeWidth(w) {
    const b = resizeBounds(), v = Math.round(Math.max(b.min, Math.min(b.max, w)));
    mainPane.style.setProperty('--code-width', v + 'px');
    syncResizeHandle();
  }
  function syncResizeHandle() {
    const b = resizeBounds();
    resizeHandle.setAttribute('aria-valuemin', b.min);
    resizeHandle.setAttribute('aria-valuemax', Math.round(b.max));
    resizeHandle.setAttribute('aria-valuenow', Math.round(codePane.getBoundingClientRect().width));
  }
  function dragWith(handle, cls, onMove) {
    handle.addEventListener('pointerdown', function (e) {
      if (e.button !== 0) return;
      e.preventDefault();
      const start = { x: e.clientX, y: e.clientY };
      const begin = onMove(null);
      handle.setPointerCapture(e.pointerId);
      handle.classList.add('is-dragging'); document.body.classList.add(cls);
      function move(ev) { onMove({ dx: ev.clientX - start.x, dy: ev.clientY - start.y, begin: begin }); }
      function end() {
        handle.classList.remove('is-dragging'); document.body.classList.remove(cls);
        ['pointermove', 'pointerup', 'pointercancel', 'lostpointercapture'].forEach(function (t) { handle.removeEventListener(t, t === 'pointermove' ? move : end); });
      }
      handle.addEventListener('pointermove', move);
      ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(function (t) { handle.addEventListener(t, end); });
    });
  }
  dragWith(resizeHandle, 'is-resizing', function (m) { if (!m) return codePane.getBoundingClientRect().width; setCodeWidth(m.begin + m.dx); });
  resizeHandle.addEventListener('keydown', function (e) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault(); setCodeWidth(codePane.getBoundingClientRect().width + (e.key === 'ArrowLeft' ? -20 : 20));
  });
  const l2Kernel = $('viewKernel'), l2Graph = $('l2GraphPane'), l2Split = $('l2Split');
  function setL2Height(h) {
    const max = Math.max(160, l2Kernel.clientHeight - 220), v = Math.round(Math.max(140, Math.min(max, h)));
    l2Kernel.style.setProperty('--l2-graph-height', v + 'px');
    l2Split.setAttribute('aria-valuemin', 140); l2Split.setAttribute('aria-valuemax', max); l2Split.setAttribute('aria-valuenow', v);
  }
  dragWith(l2Split, 'is-resizing-vertical', function (m) { if (!m) return l2Graph.getBoundingClientRect().height; setL2Height(m.begin + m.dy); });
  l2Split.addEventListener('keydown', function (e) {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault(); setL2Height(l2Graph.getBoundingClientRect().height + (e.key === 'ArrowUp' ? -20 : 20));
  });
  window.addEventListener('resize', function () { syncResizeHandle(); });

  // ── 启动：支持 #L1 / #L2 直达；L2 可带 :pipeline / :chip；旧的 #L3 / #L4 落到 L2 的 qk_pv ──
  const m = /^#L([1-4])(?::(assignment|pipeline|chip))?$/.exec(location.hash || '');
  if (m && m[2]) state.tab = m[2];
  setLevel(m ? Number(m[1]) : 1);
})();
