/* 无限画布：L1 → L2 的下钻改成同一块画布上的相机缩放；「工作怎么分」「放在哪」是 L2 计算图上某个对象关联出来的卡片。
 *
 * 世界坐标（以 L1 平面为基准，单位 = 画布像素 × 相机缩放 1）：
 *   - L1 平面：#viewLayer（整层计算图），占 [0, 0, SW, SH]；
 *   - L2 平面：#viewKernel，按 K = 0.1 缩小后嵌在 L1 里被下钻节点的位置 P（双击点）上；
 *     L2 平面内部（L2 坐标，缩放 1 时与画布像素一致）：
 *       计算图卡片  [0, 0, SW, SH]            —— Kernel 工作区标题 + 计算图（#l2GraphPane）
 *       工作怎么分  [-SW/2 - g/2, SH + gap]    —— 关联对象：SPMD × 24 / FOR qk_it 外框（#kernelWork）
 *       放在哪      [ SW/2 + g/2, SH + gap]    —— 关联对象：PIPELINE 内的阶段 ①–④（#hpLensPane，仅 qk_pv）
 *   - 相机：screen = world × s + t。L1：s = 1；L2 计算图：s = 1 / K，L2 平面恰好 1:1，文字清晰。
 *
 * 层级仍由 lens.js 管（body[data-level]、页签 is-chip、源码联动都不变），本文件只负责“看哪里”：
 *   层级 1 → 2：从双击点放大到计算图；2 → 1：缩回到 L1；聚焦卡片时同步切 lens.js 的页签（源码栏随之切换）。
 *   L2 里滚轮缩放、拖空白处平移；缩到很小自动回到 L1。底部导航条与计算图上的入口按钮切换聚焦对象。 */
(function () {
  'use strict';
  const $ = function (id) { return document.getElementById(id); };
  const stage = $('lensMid'), L1 = $('viewLayer'), L2 = $('viewKernel'), frame = $('layerFrame');
  if (!stage || !L1 || !L2) return;
  const head = L2.querySelector(':scope > .hp-pane-head'), graph = $('l2GraphPane'), work = $('kernelWork'), tile = $('hpLensPane');
  const tabs = $('kernelTabs'), logic = $('hpLogic');
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const K = 0.1;

  document.body.dataset.workbenchTransition = 'false';   // lens.js 不再自己做下钻动画（也不用 View Transition），由相机接管
  stage.classList.add('wb-canvas');
  L2.classList.add('wb-plane');

  // ── 画布上的附加元素：卡片标题、连线、入口按钮、底部导航条 ──
  const NS = 'http://www.w3.org/2000/svg';
  function mk(tag, cls, parent, html) { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; if (parent) parent.appendChild(n); return n; }
  const links = document.createElementNS(NS, 'svg'); links.setAttribute('class', 'wb-cv-links'); links.setAttribute('aria-hidden', 'true'); L2.appendChild(links);
  const workHead = mk('div', 'wb-card-h is-work', L2, '<b>工作怎么分</b><small>分工 · 轮次 —— 展开 SPMD × 24 · FOR qk_it</small>');
  const chipHead = mk('div', 'wb-card-h is-chip', L2, '<b>放在哪</b><small>硅片落点 —— 阶段 ①–④ 的数据落在哪</small>');
  const extra = document.querySelector('.wb-chip-extra'); if (extra) chipHead.appendChild(extra);
  const hWork = mk('button', 'wb-cv-handle is-work', L2, '工作怎么分<i aria-hidden="true">↓</i>');
  const hChip = mk('button', 'wb-cv-handle is-chip', L2, '放在哪<i aria-hidden="true">↓</i>');
  hWork.type = hChip.type = 'button';
  hWork.title = '分工与轮次：这个 SPMD × 24 · FOR qk_it 外框里的 Work Item 怎么产生、排队、分给 24 个核';
  hChip.title = '硅片落点：PIPELINE 里阶段 ①–④ 的数据落在 AIC / AIV 哪块存储';
  const nav = mk('div', 'wb-cv-nav', stage,
    '<button type="button" data-f="graph">计算图</button><button type="button" data-f="work">工作怎么分</button>' +
    '<button type="button" data-f="chip">放在哪</button><i aria-hidden="true"></i><button type="button" data-f="overview" title="看整个 Kernel 平面">全览</button>');
  nav.setAttribute('role', 'toolbar'); nav.setAttribute('aria-label', '画布视角');

  // ── 状态 ──
  let SW = 0, SH = 0, P = null, focus = 'graph', cam = { s: 1, tx: 0, ty: 0 }, anim = null, inited = false;
  let level = Number(document.body.dataset.level) || 1;
  const lay = {};
  const isQk = function () { return /qk_pv/.test(($('kernelMeta') || {}).textContent || '') || /qk_pv/.test(($('lensCrumb') || {}).textContent || ''); };
  const clamp = function (v, a, b) { return Math.max(a, Math.min(b, v)); };
  const smooth = function (a, b, x) { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  const ease = function (t) { return t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };

  function place(el, r) { if (!el) return; el.style.left = r[0] + 'px'; el.style.top = r[1] + 'px'; el.style.width = r[2] + 'px'; el.style.height = r[3] + 'px'; }
  function layout() {
    const r = stage.getBoundingClientRect();
    const w = Math.round(r.width), h = Math.round(r.height);
    if (!w || !h) return false;
    const changed = w !== SW || h !== SH;
    SW = w; SH = h;
    if (!P) P = { x: SW / 2, y: SH / 2 };
    [L1, L2].forEach(function (p) { p.style.width = SW + 'px'; p.style.height = SH + 'px'; });
    const HH = 46, gap = Math.round(SH * .3), g = Math.round(SW * .06), qk = isQk();
    lay.graphCard = [0, 0, SW, SH];
    lay.work = qk ? [Math.round(-SW / 2 - g / 2), SH + gap, SW, SH] : [0, SH + gap, SW, SH];
    lay.chip = [Math.round(SW / 2 + g / 2), SH + gap, SW, SH];
    const hh = head ? 42 : 0;
    place(head, [0, 0, SW, hh]);
    place(graph, [0, hh, SW, SH - hh]);
    place(workHead, [lay.work[0], lay.work[1], SW, HH]);
    place(work, [lay.work[0], lay.work[1] + HH, SW, SH - HH]);
    place(chipHead, [lay.chip[0], lay.chip[1], SW, HH]);
    place(tile, [lay.chip[0], lay.chip[1] + HH, SW, SH - HH]);
    chipHead.hidden = !qk; L2.classList.toggle('wb-no-chip', !qk);
    nav.querySelector('[data-f="chip"]').hidden = !qk;
    return changed;
  }

  // ── 相机 ──
  function planeT(c) { const ox = P.x - K * SW / 2, oy = P.y - K * SH / 2; return { s: c.s * K, x: c.tx + ox * c.s, y: c.ty + oy * c.s }; }
  function camFor(target) {
    if (target === 'l1') return { s: 1, tx: 0, ty: 0 };
    let R, pad = 0;
    if (target === 'work') R = lay.work;
    else if (target === 'chip') R = lay.chip;
    else if (target === 'overview') {
      const qk = isQk(), xs = [0, lay.work[0]].concat(qk ? [lay.chip[0]] : []), xe = [SW, lay.work[0] + SW].concat(qk ? [lay.chip[0] + SW] : []);
      const x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xe);
      R = [x0, 0, x1 - x0, lay.work[1] + SH]; pad = 36;
    } else R = lay.graphCard;
    const f = Math.min((SW - 2 * pad) / R[2], (SH - 2 * pad) / R[3]);
    const s = f / K, px = (SW - R[2] * f) / 2 - R[0] * f, py = (SH - R[3] * f) / 2 - R[1] * f;
    const ox = P.x - K * SW / 2, oy = P.y - K * SH / 2;
    return { s: s, tx: px - ox * s, ty: py - oy * s };
  }
  function apply() {
    const p = planeT(cam);
    L1.style.transform = 'translate(' + cam.tx + 'px,' + cam.ty + 'px) scale(' + cam.s + ')';
    L2.style.transform = 'translate(' + p.x + 'px,' + p.y + 'px) scale(' + p.s + ')';
    const u = clamp(Math.log(cam.s) / Math.log(1 / K), 0, 1.4);
    const a1 = 1 - smooth(.28, .62, u), a2 = level === 2 ? smooth(.38, .8, u) : 0;
    L1.style.opacity = a1; L1.style.visibility = a1 < .01 ? 'hidden' : '';
    L2.style.opacity = a2; L2.style.visibility = a2 < .01 ? 'hidden' : '';
    stage.classList.toggle('is-l2', level === 2);
    // 点阵背景跟着相机走，像一张无限画布
    const gs = 22 * (level === 2 ? p.s : cam.s), g = gs < 8 ? gs * Math.ceil(8 / gs) : gs;
    stage.style.backgroundSize = g + 'px ' + g + 'px';
    stage.style.backgroundPosition = (level === 2 ? p.x : cam.tx) + 'px ' + (level === 2 ? p.y : cam.ty) + 'px';
  }
  function go(target, opts) {
    opts = opts || {};
    const B = camFor(target), A = { s: cam.s, tx: cam.tx, ty: cam.ty };
    if (anim) { anim.stop = true; anim = null; }
    if (still || opts.jump) { cam = B; apply(); settle(); return; }
    const dur = opts.dur || (Math.abs(Math.log(B.s / A.s)) > .2 ? 900 : 620), t0 = performance.now();
    const zooming = Math.abs(A.s - B.s) / Math.max(A.s, B.s) > .02;
    let fx = 0, fy = 0, Sx = 0, Sy = 0;
    if (zooming) {                                   // 绕一个不动点缩放：A、B 两个相机下落在同一屏幕位置的世界点
      fx = (B.tx - A.tx) / (A.s - B.s); fy = (B.ty - A.ty) / (A.s - B.s);
      Sx = A.s * fx + A.tx; Sy = A.s * fy + A.ty;
    }
    const cA = { x: (SW / 2 - A.tx) / A.s, y: (SH / 2 - A.ty) / A.s }, cB = { x: (SW / 2 - B.tx) / B.s, y: (SH / 2 - B.ty) / B.s };
    const me = { stop: false }; anim = me;
    stage.classList.add('is-moving');
    // 每帧用 rAF 推进；rAF 不跑时（页面在后台）用 34ms 的定时器兜底，动画照样走完
    function next() { let done = false; const f = function () { if (done) return; done = true; frameAt(performance.now()); }; requestAnimationFrame(f); setTimeout(f, 34); }
    function frameAt(now) {
      if (me.stop) return;
      const e = ease(clamp((now - t0) / dur, 0, 1));
      if (zooming) {
        const s = A.s * Math.pow(B.s / A.s, e);
        cam = { s: s, tx: Sx - s * fx, ty: Sy - s * fy };
      } else {                                       // 同一缩放下平移：中途略微拉远，像镜头摇过去
        const s = A.s * (1 - .12 * Math.sin(Math.PI * e)), cx = cA.x + (cB.x - cA.x) * e, cy = cA.y + (cB.y - cA.y) * e;
        cam = { s: s, tx: SW / 2 - s * cx, ty: SH / 2 - s * cy };
      }
      apply();
      if (e < 1) { next(); return; }
      layout(); cam = camFor(target); apply(); anim = null; stage.classList.remove('is-moving'); settle();
    }
    next();
    setTimeout(function () { if (anim === me) { me.stop = true; layout(); cam = camFor(target); apply(); anim = null; stage.classList.remove('is-moving'); settle(); } }, dur + 400);   // 页面在后台时 rAF 不跑，到点直接落到终点
  }
  // 动画结束时按最新尺寸重新算终点（层级切换时画布高度可能变了）
  function settle() { drawLinks(); paintNav(); setTimeout(drawLinks, 350); }   // 计算图按新尺寸重排后再画一次引线

  // ── 聚焦对象：计算图 / 工作怎么分 / 放在哪 / 全览；卡片与 lens.js 页签同步 ──
  let tabByUs = false;
  function setTab(tab) {
    const b = tabs && tabs.querySelector('[data-tab="' + tab + '"]');
    if (!b) return;
    const now = L2.classList.contains('is-chip') ? 'chip' : 'assignment';
    if ((tab === 'chip') === (now === 'chip')) return;
    tabByUs = true; b.click(); tabByUs = false;
  }
  function focusOn(f, opts) {
    if (f === 'chip' && !isQk()) f = 'work';
    focus = f;
    if (f === 'work') setTab('assignment');
    if (f === 'chip') setTab('chip');
    L2.classList.toggle('wb-bottom-closed', f === 'graph');      // 只看计算图时，撤掉“下方展开哪几层”的高亮与提示
    L2.dataset.focus = f;
    if (level === 2) go(f === 'graph' ? 'graph' : f, opts);
    paintNav();
  }
  function paintNav() {
    nav.querySelectorAll('button[data-f]').forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.f === focus)); });
    [['work', work, workHead], ['chip', tile, chipHead]].forEach(function (x) {
      const on = focus === x[0] || focus === 'overview' || focus === 'free';
      x[1].classList.toggle('wb-card-dim', !on); x[2].classList.toggle('wb-card-dim', !on);
    });
  }
  nav.addEventListener('click', function (e) { const b = e.target.closest('button[data-f]'); if (b) focusOn(b.dataset.f); });
  hWork.addEventListener('click', function () { focusOn('work'); });
  hChip.addEventListener('click', function () { focusOn('chip'); });
  [[work, 'work'], [tile, 'chip'], [workHead, 'work'], [chipHead, 'chip']].forEach(function (x) {
    x[0].addEventListener('pointerdown', function () { if (level === 2 && focus !== x[1]) focusOn(x[1]); }, true);
  });
  // 页签被别处切换（深链、“这一行在各层”跳到硅片落点……）时，相机跟过去
  new MutationObserver(function () {
    if (tabByUs || level !== 2 || !inited) return;
    const chip = L2.classList.contains('is-chip');
    if (chip && focus !== 'chip') focusOn('chip');
    else if (!chip && focus === 'chip') focusOn('work');
  }).observe(L2, { attributes: true, attributeFilter: ['class'] });

  // ── 连线与入口：从计算图上的对象（SPMD 外框 / PIPELINE 框）引到下方卡片 ──
  function rectIn(el) {
    const pr = L2.getBoundingClientRect(), r = el.getBoundingClientRect(), s = planeT(cam).s || 1;
    return { x: (r.left - pr.left) / s, y: (r.top - pr.top) / s, w: r.width / s, h: r.height / s };
  }
  function drawLinks() {
    if (level !== 2 || !SW) return;
    const logicOn = logic && !logic.hidden;
    const spmd = logicOn && logic.querySelector('.lg-hit[data-node="f-spmd"] > .lg-frame');
    const pipe = logicOn && logic.querySelector('.lg-hit[data-node="f-pipe"] > .lg-frame');
    const qk = isQk();
    const aW = spmd ? rectIn(spmd) : null, aC = pipe ? rectIn(pipe) : null;
    const wx = aW ? aW.x + Math.min(70, aW.w * .12) : SW * .3, cx = aC ? aC.x + aC.w / 2 : SW * .7;
    // 入口按钮放在对象下方约 70px，且避开画布底部的视角导航条
    const low = Math.max(aW ? aW.y + aW.h : 0, aC ? aC.y + aC.h : 0), gb = Math.round(Math.min(SH - 96, (low || SH * .7) + 72));
    const wy = aW ? aW.y + aW.h : gb - 30, cy = aC ? aC.y + aC.h : gb - 30;
    hWork.style.left = wx + 'px'; hWork.style.top = gb + 'px';
    hChip.style.left = cx + 'px'; hChip.style.top = gb + 'px'; hChip.hidden = !qk;
    const tW = [lay.work[0] + SW / 2, lay.work[1]], tC = [lay.chip[0] + SW / 2, lay.chip[1]];
    const x0 = Math.min(lay.work[0], 0), x1 = Math.max(lay.chip[0] + SW, SW);
    links.setAttribute('viewBox', x0 + ' 0 ' + (x1 - x0) + ' ' + (lay.work[1] + 10));
    links.style.left = x0 + 'px'; links.style.width = (x1 - x0) + 'px'; links.style.height = (lay.work[1] + 10) + 'px';
    const curve = function (ax, ay, bx, by) { const m = (by - ay) * .55; return 'M' + ax + ' ' + ay + ' C ' + ax + ' ' + (ay + m) + ', ' + bx + ' ' + (by - m) + ', ' + bx + ' ' + by; };
    let html = '';
    if (aW) html += '<path class="lead" d="M' + wx + ' ' + wy + ' V ' + (gb - 14) + '"/><circle class="dot" cx="' + wx + '" cy="' + wy + '" r="3.5"/>';
    if (aC && qk) html += '<path class="lead" d="M' + cx + ' ' + cy + ' V ' + (gb - 14) + '"/><circle class="dot" cx="' + cx + '" cy="' + cy + '" r="3.5"/>';
    html += '<path class="link is-work" d="' + curve(wx, gb + 14, tW[0], tW[1] - 4) + '"/>';
    if (qk) html += '<path class="link is-chip" d="' + curve(cx, gb + 14, tC[0], tC[1] - 4) + '"/>';
    links.innerHTML = html;
  }

  // ── 层级变化：lens.js 改 body[data-level] → 相机放大 / 缩回 ──
  function capturePoint() {
    const x = parseFloat(frame && frame.style.getPropertyValue('--drill-x')), y = parseFloat(frame && frame.style.getPropertyValue('--drill-y'));
    if (Number.isFinite(x) && Number.isFinite(y)) P = { x: frame.offsetLeft + x, y: frame.offsetTop + y };
  }
  // 双击下钻时 lens.js 先给 L1 加 is-drilling-out（并写下双击点），约 0.4s 后才切层级：这时就开始放大
  new MutationObserver(function () {
    if (L1.classList.contains('is-drilling-out') && level === 1) { capturePoint(); focus = 'graph'; go('graph', { dur: 1000 }); }
  }).observe(L1, { attributes: true, attributeFilter: ['class'] });
  function onLevel() {
    const lv = Number(document.body.dataset.level) || 1;
    if (lv === level && inited) return;
    const prev = level; level = lv;
    stage.classList.toggle('is-l2', lv === 2);
    if (lv === 2) {
      layout();
      if (!inited) return;
      const tgt = L2.classList.contains('is-chip') ? 'chip' : 'graph';
      focus = tgt; L2.classList.toggle('wb-bottom-closed', tgt === 'graph'); L2.dataset.focus = tgt;
      if (!anim) go(tgt, { dur: 1000 }); else apply();
      setTimeout(function () { window.dispatchEvent(new Event('resize')); settle(); }, 60);
    } else if (prev === 2) {
      focus = 'graph'; L2.dataset.focus = 'graph'; L2.classList.add('wb-bottom-closed');
      go('l1', { dur: 900 });
    } else apply();
  }
  new MutationObserver(onLevel).observe(document.body, { attributes: true, attributeFilter: ['data-level'] });

  // ── L2 里自由缩放与平移；缩得很小就回到 L1 ──
  stage.addEventListener('wheel', function (e) {
    if (level !== 2 || anim) return;
    e.preventDefault();
    const r = stage.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const f = Math.exp(-(e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY) * (e.ctrlKey ? .01 : .0015));
    const s = clamp(cam.s * f, .15 / K * .8, 3 / K);
    const wx = (mx - cam.tx) / cam.s, wy = (my - cam.ty) / cam.s;
    cam = { s: s, tx: mx - s * wx, ty: my - s * wy };
    focus = 'free'; L2.dataset.focus = 'free'; apply(); paintNav();
    if (s * K < .22) { const b = document.querySelector('#lensLevels button[data-level="1"]'); if (b) b.click(); }
    clearTimeout(stage._wbT); stage._wbT = setTimeout(drawLinks, 120);
  }, { passive: false });
  let drag = null, space = false;
  window.addEventListener('keydown', function (e) { if (e.code === 'Space' && !/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) space = true; });
  window.addEventListener('keyup', function (e) { if (e.code === 'Space') space = false; });
  stage.addEventListener('pointerdown', function (e) {
    if (level !== 2 || anim || e.button > 1) return;
    const bg = e.target === stage || e.target === L2 || e.target === links || e.button === 1 || space;
    if (!bg) return;
    e.preventDefault();
    drag = { x: e.clientX, y: e.clientY, tx: cam.tx, ty: cam.ty, id: e.pointerId };
    stage.setPointerCapture(e.pointerId); stage.classList.add('is-panning');
  });
  stage.addEventListener('pointermove', function (e) {
    if (!drag || e.pointerId !== drag.id) return;
    cam = { s: cam.s, tx: drag.tx + e.clientX - drag.x, ty: drag.ty + e.clientY - drag.y };
    if (focus !== 'free') { focus = 'free'; L2.dataset.focus = 'free'; paintNav(); }
    apply();
  });
  const endDrag = function (e) { if (!drag || e.pointerId !== drag.id) return; drag = null; stage.classList.remove('is-panning'); drawLinks(); };
  stage.addEventListener('pointerup', endDrag); stage.addEventListener('pointercancel', endDrag);

  // ── 尺寸变化：重新布局，停在当前聚焦对象上 ──
  new ResizeObserver(function () {
    if (!layout()) return;
    if (anim) return;
    if (level === 1) { cam = camFor('l1'); apply(); return; }
    if (focus !== 'free') { cam = camFor(focus === 'graph' ? 'graph' : focus); apply(); }
    setTimeout(settle, 60);
  }).observe(stage);
  if (logic) new MutationObserver(function () { if (level === 2 && !anim) { clearTimeout(logic._wbT); logic._wbT = setTimeout(drawLinks, 80); } }).observe(logic, { childList: true, subtree: true, attributes: true, attributeFilter: ['hidden', 'viewBox', 'transform'] });

  // ── 初始：按当前层级与深链直接落位（不播动画） ──
  function init() {
    if (!layout()) { setTimeout(init, 60); return; }
    level = Number(document.body.dataset.level) || 1;
    if (level === 2) {
      const chip = L2.classList.contains('is-chip'), deepWork = /^#L2:(assignment|pipeline|generation|ordering|eval)$/.test(location.hash);
      focus = chip ? 'chip' : deepWork ? 'work' : 'graph';
      L2.classList.toggle('wb-bottom-closed', focus === 'graph'); L2.dataset.focus = focus;
      cam = camFor(focus === 'graph' ? 'graph' : focus);
    } else cam = camFor('l1');
    inited = true;
    apply(); paintNav();
    setTimeout(function () { window.dispatchEvent(new Event('resize')); settle(); }, 120);
  }
  init();
})();
